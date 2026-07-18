"""Guided-revision register fold for impact summaries (wired into every run).

Impact summaries are written inline by the assembler session
(prompts/assemble_dossier.txt), so the old batch "writer" is gone. This module is the
register-revise safety net that runs AFTER consolidation: collect_dossier.py calls
revise_impact_register(...) once the dossier is assembled, to measure each one-sentence summary
against the TSBM target and tighten the over-target ones (top-1 feature per round, offenders only,
in-register statements untouched).

Import convention (src/ is on sys.path at runtime):
    from revise_impact import revise_impact_register
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

from dossier_response import Dossier
from llm_retry import cached_tokens, create_response_with_backoff

log = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

@lru_cache(maxsize=None)
def _load_template(name: str) -> Template:
    return Template((_PROMPT_DIR / name).read_text())

_PILLARS = [
    ("clinical_impacts", ("name", "impact_type", "url", "year")),
    ("community_impacts", ("name", "impact_type", "target_population", "url", "year")),
    ("economic_impacts", ("name", "impact_type", "url", "year")),
    ("policy_impacts", ("body_name", "impact_type", "role", "document_name", "url", "year")),
]

def _pillar_items(dossier: Any, pillar: str) -> list:
    """The candidate list for an impact pillar. Each pillar is now an absence-section object
    that carries its candidates under `.items` (plus a confirmed_absent flag)."""
    section = getattr(dossier, pillar, None)
    return getattr(section, "items", None) or []

def _parse_statements(raw_text: str, expected: int) -> Dict[int, str]:
    """Parse the model's JSON reply into an index -> statement map."""
    text = (raw_text or "").strip()
    if not text:
        return {}

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {}

    statements = data.get("statements") if isinstance(data, dict) else data
    if not isinstance(statements, list):
        return {}

    result: Dict[int, str] = {}
    for i, entry in enumerate(statements):
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index", i)
        statement = entry.get("statement")
        if isinstance(idx, int) and isinstance(statement, str) and statement.strip():
            result[idx] = _one_sentence(statement)
    return result

def _one_sentence(text: str) -> str:
    """Collapse whitespace and keep only the first sentence (enforces the register)."""
    collapsed = " ".join(text.split())
    if not collapsed:
        return collapsed

    first = re.split(r"(?<=[.!?])\s+", collapsed, maxsplit=1)[0].strip()
    if first and first[-1] not in ".!?":
        first += "."
    return first

def _load_register_guidance() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in (_PROMPT_DIR / "revise_register_guidance.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, val = line.partition(": ")
            out[key] = val
    return out

_REGISTER_GUIDANCE = _load_register_guidance()

_REVISE_WORKERS = 8

def _clean_statement(raw):
    """Tidy a single-statement plain-text reply: strip whitespace and any stray wrapping quotes."""
    s = (raw or "").strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1].strip()
    return s

def _revise_one(text, guidance, *, llm, model, template, prompt_cache_key=None):
    """Revise ONE summary in its own call (small context: the single sentence + the guidance).
    Returns (revised_text_or_None, usage, seconds, error). Never raises."""
    prompt = template.substitute(guidance=guidance, statement=text)
    t0 = time.perf_counter()
    extra = {"prompt_cache_key": prompt_cache_key} if prompt_cache_key else {}
    try:
        resp = create_response_with_backoff(llm, model=model, input=prompt, **extra)
    except Exception as exc:
        return None, None, round(time.perf_counter() - t0, 2), str(exc)
    raw = getattr(resp, "output_text", None)

    out = _parse_statements(raw, 1).get(0) or _one_sentence(_clean_statement(raw))
    return (out or None), getattr(resp, "usage", None), round(time.perf_counter() - t0, 2), None

def _revise_for_register(dossier: Dossier, *, llm, model: str, target_stats: Dict[str, Any],
                         threshold: float = 0.5, rounds: int = 3, on_usage=None, on_trace=None,
                         net_id: str = "") -> None:
    """guided-revision: measure each summary as its OWN document (matching how the target was
    built), and while the register runs too complex, revise the top-1 over-target feature on the
    OFFENDING summaries INDIVIDUALLY (one call each, in parallel), re-measuring up to `rounds`
    times. Each summary converges on its own; in-register ones are left untouched (no overshoot).
    Top-1 per round because the 10 features are correlated. In place; never raises."""
    if not target_stats:
        return
    try:
        import style_metrics
        nlp = style_metrics._load_nlp()
    except Exception as exc:
        log.warning(f"[{net_id}] impact_writer: style metrics unavailable, skipping register fold: {exc}")
        return
    template = _load_template("revise_impact.txt")
    for round_i in range(rounds):

        refs: List = []
        texts: List[str] = []
        for pillar, _ in _PILLARS:
            for i, c in enumerate(_pillar_items(dossier, pillar)):
                s = getattr(c, "summary", None)
                if s:
                    refs.append((pillar, i)); texts.append(s)
        if not texts:
            return

        per_feats = [style_metrics.compute_features(t, nlp) for t in texts]
        rows = style_metrics.mean_divergences(texts, target_stats, threshold, nlp=nlp)
        flagged = [r for r in rows if r[4] >= threshold]
        if not flagged:
            if round_i:
                log.info(f"[{net_id}] impact_writer: register converged after {round_i} round(s)")
            return
        feature, _cur, t_mean, t_sd, z = flagged[0]
        guidance = _REGISTER_GUIDANCE.get(feature, _REGISTER_GUIDANCE["default"])

        offenders = [g for g, f in enumerate(per_feats)
                     if t_sd > 0 and (f[feature] - t_mean) / t_sd >= threshold]
        if not offenders:
            return
        log.info(f"[{net_id}] impact_writer: register round {round_i + 1}/{rounds} targeting "
                 f"{feature} (z=+{z:.2f}); revising {len(offenders)}/{len(texts)} summaries individually")

        with ThreadPoolExecutor(max_workers=min(len(offenders), _REVISE_WORKERS)) as ex:
            results = list(ex.map(
                lambda g: (g, *_revise_one(texts[g], guidance, llm=llm, model=model,
                                           template=template, prompt_cache_key=net_id or None)),
                offenders))
        n_revised = 0
        usages = []
        rewrites = []
        for g, new, u, _secs, _err in results:
            if u:
                usages.append(u)
            if new:
                pillar, idx = refs[g]
                rewrites.append({"index": g, "before": texts[g], "after": new})
                _pillar_items(dossier, pillar)[idx].summary = new
                n_revised += 1
        for u in usages:
            if on_usage:
                on_usage(u)
        if on_trace:
            tin = sum(u.input_tokens for u in usages)
            tout = sum(u.output_tokens for u in usages)
            tcached = sum(cached_tokens(u) for u in usages)
            on_trace({"turn": "impact_revise", "round": round_i + 1, "feature": feature,
                      "z": round(z, 2), "n_offenders": len(offenders), "n_revised": n_revised,
                      "rewrites": rewrites,
                      "llm_seconds": round(max((r[3] for r in results), default=0.0), 2),
                      "token_usage": {"context": tin, "generated": tout, "total": tin + tout, "cached": tcached}})
        if not n_revised:
            return

def revise_impact_register(
    dossier: Dossier,
    *,
    llm,
    model: str,
    target_stats: Optional[Dict[str, Any]] = None,
    on_usage=None,
    on_trace=None,
    net_id: str = "",
) -> Dossier:
    """Tighten impact summaries toward the TSBM register, in place (the register safety net).

    Measures each `summary` and, while the register runs too complex, revises the over-target ones
    (top-1 feature per round, offenders only, up to 3 rounds; in-register summaries untouched).
    Summaries are written inline by the assembler session, then this runs after consolidation to
    tighten any that drift off-register (wired into the run). Never raises."""
    _revise_for_register(dossier, llm=llm, model=model, target_stats=target_stats or {},
                         on_usage=on_usage, on_trace=on_trace, net_id=net_id)
    return dossier
