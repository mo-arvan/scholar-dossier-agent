"""Scholar dossier agent: a state-as-memory research loop feeding a gated entry-op
assembler that folds distilled findings into the dossier via append/patch/confirm_absent."""

import contextlib
import json
import lzma
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Literal, Optional

import dotenv
import hydra
import pandas as pd
import tiktoken
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient, UsageLimitExceededError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
)

from adapters import ADAPTER_TOOL_SCHEMAS, PURE_ADAPTERS
from adapters.extract import TOOL_SCHEMA as EXTRACT_TOOL_SCHEMA
from adapters.extract import structured_extract
from adapters.search import TOOL_SCHEMA as SEARCH_TOOL_SCHEMA
from adapters.search import search as run_search
from cache import ResponseCache
from dossier_edit import ChangeLog, run_assembler_session, section_size
from source_filter import Observations, load_cris
from dossier_response import ConfidenceScore, Dossier, Grant, GrantCandidate, Status
from revise_impact import revise_impact_register
from llm_retry import (
    cached_tokens,
    create_response_with_backoff,
    deadline_for,
    parse_response_with_backoff,
    usage_dict,
)

dotenv.load_dotenv()

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

BLOCKED_DOMAINS = {
    "reporter.nih.gov",
    "projectreporter.nih.gov",
    "clinicaltrials.gov",
    "www.clinicaltrials.gov",
    "orcid.org",
    "www.linkedin.com",
    "www.researchgate.net",
}

def initialize_dossier(
    cfg: DictConfig, scholars_df: pd.DataFrame, hydra_output_dir: str
) -> None:
    output_dir = Path(hydra_output_dir)

    def _col(row, name: str) -> str:
        """A seed column's value, or "" when the CSV omits it (only NetID and name are required)."""
        val = row.get(name)
        return "" if val is None or pd.isna(val) else str(val).strip()

    for _, row in scholars_df.iterrows():
        net_id = str(row["NetID"]).strip()
        dossier_file_path = output_dir / f"{net_id}.json"
        if dossier_file_path.exists():
            log.info(f"Dossier for {net_id} already exists. Skipping.")
            continue
        dossier = Dossier(
            net_id=net_id,
            first_name=_col(row, "First Name"),
            last_name=_col(row, "Last Name"),
            last_known_college=_col(row, "College"),
            last_known_department=_col(row, "Department"),
            degree=_col(row, "Degree"),
            initial_research_title=_col(row, "Research Project Title"),
            affiliation=_col(row, "Affiliation"),
        )
        dossier_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dossier_file_path, "w") as f:
            f.write(dossier.model_dump_json(indent=2))

_tavily_retry = retry(
    retry=retry_if_exception_type(UsageLimitExceededError),
    wait=wait_chain(wait_fixed(2), wait_fixed(8), wait_fixed(20), wait_fixed(45)),
    stop=stop_after_attempt(5),
    reraise=True,
)

_ENC = tiktoken.get_encoding("o200k_base")

def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text or ""))

def _truncate(s: Optional[str], n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 3] + "..."

def _distill_tool_result(name: str, args: dict, result: Any, llm=None, model=None, on_usage=None) -> dict:
    """Compact a tool result into the fragment that enters context next turn; publication
    abstracts are trimmed to their outcome via the cheap model (numbers preserved)."""
    if name == "openalex_author_works":
        works = [
            {"title": w.get("title"), "year": w.get("year"), "venue": w.get("venue"),
             "doi": w.get("doi"), "pmid": w.get("pmid"),
             "cited_by": w.get("cited_by_count"), "abstract": w.get("abstract")}
            for w in (result.get("results") or [])
        ]
        _distill_abstracts(works, "abstract", llm, model, on_usage)
        return {
            "tool": name,
            "author": (result.get("meta") or {}).get("author"),
            "match_reason": (result.get("meta") or {}).get("match_reason"),
            "count": result.get("count"),
            "works": works,
            "error": result.get("error"),
        }
    if name == "pubmed_author_articles":
        arts = [
            {"pmid": a.get("pmid"), "doi": a.get("doi"),
             "title": a.get("title"), "year": a.get("year"),
             "journal": a.get("journal"), "mesh": (a.get("mesh_terms") or [])[:8], "abstract": a.get("abstract")}
            for a in (result.get("results") or [])
        ]
        _distill_abstracts(arts, "abstract", llm, model, on_usage)
        return {
            "tool": name,
            "count": result.get("count"),
            "articles": arts,
            "error": result.get("error"),
        }
    if name == "orcid_profile":
        return {
            "tool": name,
            "meta": result.get("meta"),
            "count": result.get("count"),
            "records": result.get("results") or [],
            "error": result.get("error"),
        }
    if name == "clinicaltrials_by_investigator":
        return {
            "tool": name,
            "count": result.get("count"),
            "trials": result.get("results") or [],
            "error": result.get("error"),
        }
    if name == "structured_extract":
        return {
            "tool": name,
            "url": result.get("url"),
            "distilled": result.get("distilled"),
            "error": result.get("error"),
        }
    if name == "search":
        if isinstance(result, dict) and result.get("error"):
            return {"tool": name, "query": args.get("query"), "error": result["error"]}
        items = result if isinstance(result, list) else []
        return {
            "tool": name,
            "query": args.get("query"),
            "results": [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "score": r.get("score"),
                    "snippet": r.get("content"),
                }
                for r in items
            ],
        }
    if name == "nih_reporter_projects_search":
        if isinstance(result, dict) and result.get("error"):
            return {"tool": name, "error": result["error"]}

        projects = (result.get("results") or result.get("projects") or []) if isinstance(result, dict) else []
        return {
            "tool": name,
            "count": result.get("count") if isinstance(result, dict) else 0,
            "projects": [
                {
                    "project_title": p.get("project_title"),
                    "project_num": p.get("project_num"),
                    "contact_pi_name": p.get("contact_pi_name"),
                    "organization_name": p.get("organization_name"),
                    "fiscal_year": p.get("fiscal_year"),
                    "award_amount": p.get("award_amount"),
                    "project_detail_url": p.get("project_detail_url"),
                }
                for p in projects
            ],
        }
    return {"tool": name, "result": result}

def _log_entry(name: str, args: dict, result: Any) -> str:
    key = (
        args.get("author_name")
        or args.get("name_or_orcid")
        or args.get("investigator_name")
        or args.get("pi_name")
        or args.get("url")
        or args.get("query")
        or ""
    )
    if isinstance(result, dict):
        if result.get("error"):
            outcome = f"error: {_truncate(result['error'], 60)}"
        elif name == "structured_extract":
            outcome = "extracted" if (result.get("distilled") or "").strip() else "nothing relevant"
        elif "count" in result:
            outcome = f"{result['count']} results"
        else:
            outcome = "ok"
    elif isinstance(result, list):
        outcome = f"{len(result)} results"
    else:
        outcome = "ok"
    return f"{name}({_truncate(str(key), 60)}) -> {outcome}"

MAX_TOOL_WORKERS = 8

def _find_data(rel) -> Path:
    """Locate a repo data file by walking up: data/ sits beside code/ here and inside the repo in
    the released snapshot, so neither a cwd-relative path nor a fixed depth works in both."""
    rel = Path(rel)
    for base in Path(__file__).resolve().parents:
        if (base / rel).is_file():
            return base / rel
    return rel

def _empty_section_issues(dossier: Dossier) -> List[Dict[str, Any]]:
    """Deterministic completeness check (no LLM): a section left empty and not confirmed absent
    is a coverage gap, emitted as an issue that seeds the next run's search."""
    issues: List[Dict[str, Any]] = []
    for sec in _DOSSIER_SECTIONS:
        v = getattr(dossier, sec)
        if sec == "key_profiles":
            n = sum(len(getattr(v, s, []) or []) for s in _PROFILE_SUBLISTS)
            empty, absent = n == 0, False
        elif isinstance(v, list):
            empty, absent = not v, False
        else:
            empty, absent = not v.items, bool(getattr(v, "confirmed_absent", False))
        if empty and not absent:
            issues.append({
                "type": "gap", "section": sec,
                "what": f"{sec} is empty and never confirmed absent",
                "fix": "Run dedicated queries for this section; record what they return, or "
                       "confirm the absence with the queries run and what they ruled out.",
            })
    return issues

def _flagged_issues(dossier: Dossier) -> List[Dict[str, Any]]:
    """Entries the write gate accepted with source_flag set, as issue dicts that seed the next
    run's searches and assembler session."""
    issues: List[Dict[str, Any]] = []
    doc = json.loads(dossier.model_dump_json())

    def _scan(section, entry):
        flag = entry.get("source_flag") or ""
        if flag:
            label = (entry.get("data") or {}).get("title") if isinstance(entry.get("data"), dict) else None
            label = label or entry.get("title") or entry.get("name") or entry.get("body_name") \
                or entry.get("url") or entry.get("summary") or "?"
            issues.append({
                "type": "flagged_source", "section": section,
                "what": f'the entry "{str(label)[:90]}" carries source_flag: {flag[:120]}',
                "fix": "Search for the authoritative source for this entry (per the accepted-sources "
                       "ranking); once a tool observes it, patch the entry's source so the flag clears.",
            })

    for sub, items in (doc.get("key_profiles") or {}).items():
        for it in items or []:
            if isinstance(it, dict):
                _scan("key_profiles", it)
    for it in doc.get("career_trajectory") or []:
        _scan("career_trajectory", it)
    for it in doc.get("publications") or []:
        _scan("publications", it)
    for sec in ("grants", "media_mentions", "clinical_impacts", "community_impacts",
                "economic_impacts", "policy_impacts"):
        for it in ((doc.get(sec) or {}).get("items") or []):
            _scan(sec, it)
    return issues

def _load_tsbm_target_stats() -> Dict[str, Any]:
    """TSBM register target (feature -> [mean, std]) for the impact register-revise safety net;
    returns {} if unavailable (never fatal). Walks up for data/ since this file's depth from the
    repo root varies by checkout, so a fixed depth would silently miss it in one of them."""
    rel = Path("data") / "tsbm_style" / "tsbm_impact_statements.json"
    here = Path(__file__).resolve()
    for base in here.parents:
        path = base / rel
        if path.is_file():
            try:
                return json.load(open(path)).get("target_stats") or {}
            except Exception as exc:
                log.warning(f"TSBM target stats unreadable at {path}; register-revise skipped: {exc}")
                return {}
    log.warning(f"TSBM target stats not found ({rel} above {here.parent}); register-revise skipped")
    return {}

_TSBM_TARGET_STATS = _load_tsbm_target_stats()

class TokenTracker:
    """Accumulate input/output token usage across every LLM call in a run."""

    def __init__(self):
        self.by_model: Dict[str, Dict[str, int]] = {}
        self.tavily_credits = 0.0
        self._lock = (
            threading.Lock()
        )

    def add(self, model: str, usage) -> None:
        if usage is None:
            return
        with self._lock:
            m = self.by_model.setdefault(model, {"input": 0, "output": 0, "cached": 0, "calls": 0})
            m["input"] += int(getattr(usage, "input_tokens", 0) or 0)
            m["output"] += int(getattr(usage, "output_tokens", 0) or 0)
            m["cached"] += cached_tokens(usage)
            m["calls"] += 1

    def add_tavily(self, credits) -> None:
        """Accumulate Tavily credits reported via include_usage (extract reports in batches of 5,
        so a single call can read 0; the run total is still correct)."""
        if not credits:
            return
        with self._lock:
            self.tavily_credits += float(credits)

    def summary(self) -> Dict[str, Any]:
        ti = sum(m["input"] for m in self.by_model.values())
        to = sum(m["output"] for m in self.by_model.values())
        tc = sum(m.get("cached", 0) for m in self.by_model.values())
        return {
            "input_tokens": ti,
            "output_tokens": to,
            "total_tokens": ti + to,
            "cached_input_tokens": tc,
            "calls": sum(m["calls"] for m in self.by_model.values()),
            "by_model": dict(self.by_model),
            "tavily_credits": round(self.tavily_credits, 3),
        }

def _gv(obj, *attrs):
    """Nested getattr: _gv(candidate, 'data', 'award_number')."""
    for a in attrs:
        obj = getattr(obj, a, None)
        if obj is None:
            return None
    return obj

_PROFILE_SUBLISTS = (
    "institutional_homepage",
    "google_scholar",
    "linkedin",
    "orcid",
    "personal_website",
)

_DOSSIER_SECTIONS = (
    "key_profiles",
    "career_trajectory",
    "publications",
    "grants",
    "media_mentions",
    "clinical_impacts",
    "community_impacts",
    "economic_impacts",
    "policy_impacts",
)

def _dedup_pub_findings(turns: list) -> list:
    """Collapse an OpenAlex work and a PubMed article for the same paper (shared DOI/PMID),
    keeping the richer abstract; operates on a copy so the live turns stay byte-stable."""
    import copy as _copy
    turns = _copy.deepcopy(turns)
    best: Dict[str, tuple] = {}

    def _keys(rec):
        out = []
        for k in ("doi", "pmid"):
            v = rec.get(k)
            if v:
                out.append(f"{k}:{str(v).strip().lower()}")
        return out

    for t in turns:
        for f in (t.get("findings") or []):
            tool = f.get("tool")
            recs = f.get("works") if tool == "openalex_author_works" else (
                f.get("articles") if tool == "pubmed_author_articles" else None)
            if not recs:
                continue
            keep = []
            for rec in recs:
                ks = _keys(rec)
                prior = next((best[k] for k in ks if k in best), None)
                if prior is None:
                    keep.append(rec)
                    for k in ks:
                        best[k] = rec
                    continue

                if len((rec.get("abstract") or "")) > len((prior.get("abstract") or "")):
                    prior["abstract"] = rec.get("abstract")
                for extra in ("doi", "pmid"):
                    if not prior.get(extra) and rec.get(extra):
                        prior[extra] = rec[extra]
                        best[f"{extra}:{str(rec[extra]).strip().lower()}"] = prior
            if tool == "openalex_author_works":
                f["works"] = keep
            else:
                f["articles"] = keep
    return turns

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PARTIALS = {
    "source_preference": (_PROMPTS_DIR / "partials" / "source_preference.txt").read_text().strip(),
}

def _load_prompt(path) -> Template:
    """Load a prompt file and inline the shared partials (e.g. $source_preference), returning a
    Template for the caller's runtime placeholders."""
    injected = Template(Path(path).read_text()).safe_substitute(**_PARTIALS)
    return Template(injected)

_IMPACT_PILLARS = {"clinical_impacts", "community_impacts", "economic_impacts", "policy_impacts"}

_CRITIQUE_PROMPT = _load_prompt(_PROMPTS_DIR / "critique_evidence.txt")
_DISTILL_ABSTRACT_PROMPT = Template(
    (_PROMPTS_DIR / "distill_abstract.txt").read_text()
)

_STALL_NUDGE = (_PROMPTS_DIR / "stall_nudge.txt").read_text().strip()

def _distill_abstracts(records: list, key: str, llm=None, model=None, on_usage=None) -> None:
    """Trim each record[key] abstract to its outcome (numbers preserved) with one cheap-model call
    per abstract, run in parallel, in place; a record keeps its full abstract if the call fails."""
    idx = [i for i, r in enumerate(records) if (r.get(key) or "").strip()]
    if not idx or llm is None:
        return

    def _one(i):
        try:
            resp = create_response_with_backoff(
                llm, model=model,
                input=_DISTILL_ABSTRACT_PROMPT.safe_substitute(abstract=records[i][key]))
            return i, (getattr(resp, "output_text", "") or "").strip(), getattr(resp, "usage", None)
        except Exception as e:
            log.warning(f"abstract distill failed ({e}); keeping full abstract")
            return i, "", None

    with ThreadPoolExecutor(max_workers=min(len(idx), 8)) as ex:
        results = list(ex.map(_one, idx))
    for i, outcome, usage in results:
        if on_usage and usage:
            on_usage(usage)
        if outcome:
            records[i][key] = outcome

_ISSUE_SECTIONS = (
    "key_profiles", "career_trajectory", "publications", "grants", "media_mentions",
    "clinical_impacts", "community_impacts", "economic_impacts", "policy_impacts", "cross_section",
)

class _Issue(BaseModel):
    type: Literal["gap", "duplicate", "contradiction", "weak_source"]
    section: Literal[_ISSUE_SECTIONS]
    what: str
    fix: str

class _Critique(BaseModel):
    issues: list[_Issue]

_OBJECTIVES = (
    "key_profiles", "career_trajectory", "publications", "grants", "media_mentions",
    "clinical_impacts", "community_impacts", "economic_impacts", "policy_impacts",
)

FINISH_TOOL_SCHEMA = {
    "type": "function",
    "name": "finish",
    "description": (
        "End the research loop. Call this ONLY when the 'When to stop' checklist holds: every "
        "objective either has evidence, is genuinely absent, or had a dedicated query that came "
        "back empty. Account for EVERY objective in `coverage`. Do NOT call finish while any "
        "objective is still empty or thin and has not had a query aimed straight at it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "coverage": {
                "type": "object",
                "description": (
                    "One status per objective: 'covered' (evidence recorded), 'confirmed_none' "
                    "(the scholar genuinely has none), or 'searched_empty' (a dedicated query "
                    "found nothing)."
                ),
                "properties": {
                    o: {"type": "string", "enum": ["covered", "confirmed_none", "searched_empty"]}
                    for o in _OBJECTIVES
                },
                "required": list(_OBJECTIVES),
                "additionalProperties": False,
            },
            "summary": {
                "type": "string",
                "description": "One sentence: what you found and any gap you are deliberately leaving.",
            },
        },
        "required": ["coverage", "summary"],
    },
}

_KEEP_RECENT_TURNS = 2

def _render_turn(t: dict) -> str:
    """One append-only transcript block for a turn. Frozen once written, so the prompt prefix
    (instructions + dossier + all prior turns) stays byte-stable and the cache extends."""
    head = "Prior runs" if t.get("n") == 0 else f"Turn {t.get('n')}"
    parts = [f"### {head}"]
    if t.get("note"):
        parts.append(t["note"])
    if t.get("actions"):
        parts.append("Actions taken:")
        parts.extend(f"- {a}" for a in t["actions"])
    if t.get("findings"):
        parts.append("Distilled findings:")
        parts.append(json.dumps(t["findings"], ensure_ascii=False, indent=2))
    return "\n".join(parts)

def _render_transcript(turns: List[dict]) -> str:
    return "\n\n".join(_render_turn(t) for t in turns) if turns else "(no turns yet)"

def _assembler_transcript(turns: list, *, net_id: str, dossier: Dossier,
                          max_context_tokens: int) -> tuple:
    """The assembler's transcript (pub-twin-deduped, with a failsafe that drops the oldest turns
    if it would overflow the model context) plus the structured publication records it carries,
    for the session's publication-completeness backstop."""
    turns = _dedup_pub_findings(turns)
    transcript = _render_transcript(turns)
    base_tokens = _count_tokens(dossier.model_dump_json(exclude={"action_log", "last_updated"}))
    dropped = 0
    while turns and base_tokens + _count_tokens(transcript) > max_context_tokens:
        turns = turns[1:]
        transcript = _render_transcript(turns)
        dropped += 1
    if dropped:
        log.warning(
            f"[{net_id}] assembler context over {max_context_tokens} tok; "
            f"dropped {dropped} oldest turns as a failsafe"
        )
    pub_records = []
    for t in turns:
        for f in (t.get("findings") or []):
            if f.get("tool") in ("openalex_author_works", "pubmed_author_articles"):
                for r in (f.get("works") or f.get("articles") or []):
                    pub_records.append({k: r.get(k) for k in
                                        ("title", "year", "venue", "journal", "doi", "pmid")
                                        if r.get(k)})
    return transcript, pub_records

def run_research_agent(
    net_id: str,
    current_dossier: Dossier,
    prompt_template: Template,
    openai_client: OpenAI,
    model_name: str,
    cheap_model_name: str,
    max_turns_per_run: int,
    reasoning_effort: str,
    compaction_context_tokens: int,
    max_context_tokens: int,
    cache: Optional[ResponseCache],
    grant_context: Optional[List[Dict[str, Any]]],
    prior_issues: Optional[List[str]] = None,
    prior_issue_dicts: Optional[List[Dict[str, Any]]] = None,
    changelog: Optional[ChangeLog] = None,
    use_prompt_cache_key: bool = True,
    max_assembler_turns: int = 20,
    source_gate_attempts: int = 3,
    cris_props: Optional[set] = None,
    obs: Optional[Observations] = None,
    dispositions: Optional[Dict[str, Any]] = None,
    research_run: Optional[int] = None,
) -> tuple[Optional[Dossier], List[Dict[str, Any]], Dict[str, Any]]:
    """Run the state-as-memory loop and return (dossier, trace, token_summary).

    `prior_issues`/`prior_issue_dicts` seed a repair run's prompt and assembler issue list; `obs`
    carries the cross-run grounding keyset so the gate's "ungrounded" means never observed at all."""
    tavily_client = TavilyClient()
    tracker = TokenTracker()
    prompt_cache_key = net_id if use_prompt_cache_key else None
    cache_kwargs = {"prompt_cache_key": prompt_cache_key} if prompt_cache_key else {}
    obs = obs if obs is not None else Observations()

    @_tavily_retry
    def tavily_search(query, num_results):
        """Raw Tavily search hook for the search adapter; the SDK param is max_results, not
        num_results (which silently defaults to 5), and result count does not affect credit cost."""
        resp = tavily_client.search(query=query, max_results=num_results, include_usage=True)
        if isinstance(resp, dict):
            tracker.add_tavily((resp.get("usage") or {}).get("credits"))
        return resp

    @_tavily_retry
    def tavily_extract_text(url: str) -> str:
        """Cached Tavily extract -> raw text string (hook for structured_extract)."""
        if cache:
            cached = cache.get_extract(url)
            if cached is not None:
                cache.track_extraction(url, success=True)
                return json.dumps(cached)
        response = tavily_client.extract(url, include_usage=True)
        if isinstance(response, dict):
            tracker.add_tavily((response.get("usage") or {}).get("credits"))
        results = response.get("results", [])
        if not results:
            failed = response.get("failed_results", [])
            msg = (
                failed[0].get("error", "Extraction failed")
                if failed
                else "Extraction failed"
            )
            if cache:
                cache.track_extraction(url, success=False, error=msg)
            raise ValueError(msg)
        results = results[0] if len(results) == 1 else results
        if cache:
            cache.set_extract(url, results)
            cache.track_extraction(url, success=True)
        return json.dumps(results)

    def dispatch(name: str, args: dict) -> Any:
        try:
            if name in PURE_ADAPTERS:
                return PURE_ADAPTERS[name](cache=cache, **args)
            if name == "structured_extract":
                return structured_extract(
                    args.get("url", ""),
                    args.get("focus", ""),
                    args.get("scholar", ""),
                    llm=openai_client,
                    model=cheap_model_name,
                    cache=cache,
                    tavily_extract=tavily_extract_text,
                    on_usage=lambda u: tracker.add(cheap_model_name, u),
                )
            if name == "search":
                return run_search(**args, tavily_search=tavily_search, cache=cache)
            return {"error": f"unknown tool: {name}"}
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:
            log.error(f"Tool {name} failed: {e}")
            return {"error": str(e)}

    tools = list(ADAPTER_TOOL_SCHEMAS) + [EXTRACT_TOOL_SCHEMA, SEARCH_TOOL_SCHEMA, FINISH_TOOL_SCHEMA]

    def assemble(dossier: Dossier, turns: list, issues=None, label: str = "final"):

        transcript, pub_records = _assembler_transcript(
            turns, net_id=net_id, dossier=dossier, max_context_tokens=max_context_tokens)
        return run_assembler_session(
            dossier, transcript, issues, client=openai_client, model=model_name,
            net_id=net_id, obs=obs, cris=cris_props, reasoning_effort=reasoning_effort,
            changelog=changelog, max_turns=max_assembler_turns,
            source_gate_attempts=source_gate_attempts, expected_pubs=pub_records,
            dispositions=dispositions, run=research_run, label=label,
            on_usage=lambda u: tracker.add(model_name, u), prompt_cache_key=prompt_cache_key,
        )

    dossier = current_dossier

    if not dossier.grants.items:
        _seed_internal_grants(dossier, grant_context)
        if changelog and dossier.grants.items:
            changelog.emit("grants", "seed", None, after=len(dossier.grants.items))
    action_log: List[str] = list(
        dossier.action_log
    )
    turns: List[dict] = []
    if action_log:
        turns.append({"n": 0, "note": "Actions from prior runs (already done, do not repeat):",
                      "actions": list(action_log), "findings": []})
    if prior_issues:
        turns.append({"n": 0, "note": "Prior-round critique:",
                      "actions": list(prior_issues), "findings": []})
    trace: List[Dict[str, Any]] = []
    empty_streak = 0
    run_t0 = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        with contextlib.nullcontext():
            prev_id: Optional[str] = None
            pending_outputs: list = []
            last_ctx_tokens = 0
            for turn in range(max_turns_per_run):
                turn_t0 = time.perf_counter()

                def _build_base_prompt() -> str:

                    return prompt_template.safe_substitute(
                        current_dossier_json=dossier.model_dump_json(exclude={"action_log", "last_updated"}),
                        transcript=_render_transcript(turns),
                        max_turns_per_run=max_turns_per_run,
                        max_tool_calls=max_turns_per_run,
                    )

                compaction_event = None
                fold_turns = turns[:-_KEEP_RECENT_TURNS]
                n_fold = sum(len(t.get("findings") or []) for t in fold_turns)
                if prev_id is not None and n_fold and last_ctx_tokens >= compaction_context_tokens:
                    before = _section_counts(dossier)
                    dossier, asm_rec = assemble(dossier, fold_turns, label="compaction")
                    trace.append(asm_rec)
                    stats = asm_rec.get("stats") or {}
                    applied = stats.get("append", 0) + stats.get("patch", 0) + stats.get("absent", 0)
                    if stats.get("call_failed") and not applied:

                        log.warning(f"[{net_id}] in-loop assembly failed; keeping turns")
                    else:
                        turns = turns[-_KEEP_RECENT_TURNS:]
                        prev_id, pending_outputs = None, []
                        compaction_event = {
                            "findings_folded": n_fold,
                            "sections_before": before,
                            "sections_after": _section_counts(dossier),
                        }

                if prev_id is None:
                    call_input: Any = _build_base_prompt()
                    chain_kwargs: Dict[str, Any] = {"store": True}
                else:
                    call_input = pending_outputs
                    chain_kwargs = {"store": True, "previous_response_id": prev_id}

                ctx_est = last_ctx_tokens if "previous_response_id" in chain_kwargs else len(call_input) // 4
                try:
                    llm_t0 = time.perf_counter()
                    response = create_response_with_backoff(
                        client=openai_client,
                        model=model_name,
                        tools=tools,
                        input=call_input,
                        reasoning={"effort": reasoning_effort, "summary": "auto"},
                        deadline_s=deadline_for(ctx_est),
                        **cache_kwargs,
                        **chain_kwargs,
                    )
                except Exception as e:

                    log.error(
                        f"create() failed at turn {turn + 1}; ending loop early: {e}"
                    )
                    break
                llm_seconds = round(time.perf_counter() - llm_t0, 2)
                prev_id = response.id
                last_ctx_tokens = int(getattr(getattr(response, "usage", None), "input_tokens", 0) or 0)

                rec = {
                    "turn": turn + 1,
                    "response_id": response.id,
                    "chained": "previous_response_id" in chain_kwargs,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "llm_seconds": llm_seconds,
                    "compaction": compaction_event,
                    "input_prompt": call_input if isinstance(call_input, str)
                    else json.dumps(call_input, default=str)[:4000],
                    "output_text": getattr(response, "output_text", None),
                    "token_usage": usage_dict(getattr(response, "usage", None)),
                    "tool_calls": [],
                }
                tracker.add(model_name, response.usage)

                fcs = [o for o in response.output if o.type == "function_call"]
                finish_call = next((o for o in fcs if o.name == "finish"), None)
                tool_fcs = [o for o in fcs if o.name != "finish"]

                if finish_call:

                    try:
                        rec["finish"] = json.loads(finish_call.arguments) if finish_call.arguments else {}
                    except (ValueError, TypeError):
                        rec["finish"] = {"raw": finish_call.arguments}
                    rec["elapsed_seconds"] = round(time.perf_counter() - turn_t0, 2)
                    rec["dossier_after"] = json.loads(dossier.model_dump_json())
                    trace.append(rec)
                    fin_summary = rec["finish"].get("summary") if isinstance(rec["finish"], dict) else ""
                    log.info(f"[{net_id}] finished at turn {turn + 1}/{max_turns_per_run}: {fin_summary}")
                    break

                if not tool_fcs:

                    empty_streak += 1
                    rec["elapsed_seconds"] = round(time.perf_counter() - turn_t0, 2)
                    rec["dossier_after"] = json.loads(dossier.model_dump_json())
                    trace.append(rec)
                    if empty_streak >= 2:
                        log.info(
                            f"[{net_id}] stopped after {empty_streak} stalled turns (no tools, "
                            f"no finish) at turn {turn + 1}/{max_turns_per_run}"
                        )
                        break

                    pending_outputs = [{"role": "user", "content": _STALL_NUDGE}]
                    log.info(f"[{net_id}] stalled turn {turn + 1}; nudging to act or finish")
                    continue

                empty_streak = 0

                def _run_tool(fc):
                    args = json.loads(fc.arguments) if fc.arguments else {}
                    started_at = datetime.now(
                        timezone.utc
                    ).isoformat()
                    t0 = time.perf_counter()
                    try:
                        result = dispatch(fc.name, args)
                    except Exception as exc:
                        result = {"error": f"dispatch_error:{exc}"}
                    return (
                        fc,
                        args,
                        result,
                        round(time.perf_counter() - t0, 2),
                        started_at,
                    )

                if len(tool_fcs) == 1:
                    outcomes = [_run_tool(tool_fcs[0])]
                else:
                    with ThreadPoolExecutor(
                        max_workers=min(len(tool_fcs), MAX_TOOL_WORKERS)
                    ) as ex:
                        outcomes = list(
                            ex.map(_run_tool, tool_fcs)
                        )

                turn_actions: List[str] = []
                turn_findings: List[dict] = []
                next_outputs: List[dict] = []
                for fc, args, result, tool_seconds, started_at in outcomes:
                    distilled = _distill_tool_result(
                        fc.name, args, result,
                        llm=openai_client, model=cheap_model_name,
                        on_usage=lambda u: tracker.add(cheap_model_name, u))
                    log_str = _log_entry(fc.name, args, result)
                    turn_findings.append(distilled)
                    turn_actions.append(log_str)
                    action_log.append(log_str)
                    next_outputs.append({
                        "type": "function_call_output",
                        "call_id": getattr(fc, "call_id", None),
                        "output": json.dumps(distilled, default=str),
                    })
                    tc_rec = {
                        "function": fc.name,
                        "arguments": args,
                        "started_at": started_at,
                        "elapsed_seconds": tool_seconds,
                        "log": log_str,
                        "result_distilled": distilled,
                        "result_raw": result,
                    }
                    rec["tool_calls"].append(tc_rec)
                    obs.add_tool_call(tc_rec)

                turns.append({"n": turn + 1, "actions": turn_actions, "findings": turn_findings})
                pending_outputs = next_outputs
                rec["elapsed_seconds"] = round(time.perf_counter() - turn_t0, 2)
                rec["dossier_after"] = json.loads(dossier.model_dump_json())
                trace.append(rec)

        final_compaction = None
        touched: set = set()
        n_remaining = sum(len(t.get("findings") or []) for t in turns)
        if n_remaining or prior_issue_dicts:
            before = _section_counts(dossier)
            dossier, asm_rec = assemble(dossier, turns, prior_issue_dicts)
            trace.append(asm_rec)
            touched = {op.get("section", "").split(".")[0]
                       for op in (asm_rec.get("ops") or []) if op.get("ok")}
            final_compaction = {
                "findings_folded": n_remaining,
                "sections_before": before,
                "sections_after": _section_counts(dossier),
                "assembler_stats": asm_rec.get("stats"),
            }

        if changelog:
            for s in _DOSSIER_SECTIONS:
                if s not in touched:
                    n = section_size(dossier, s)
                    changelog.emit(s, "skip", None, before=n, after=n)

        dossier.action_log = action_log

        token_summary = tracker.summary()
        token_summary["elapsed_seconds"] = round(time.perf_counter() - run_t0, 2)
        token_summary["started_at"] = started_at
        token_summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        token_summary["dirty_sections"] = sorted(touched)

        last_asm = next((r for r in reversed(trace) if r.get("turn") == "assemble"), None)
        token_summary["pubs_missing_at_end"] = int(
            (last_asm or {}).get("stats", {}).get("pubs_missing_at_end", 0))
        trace.append(
            {
                "turn": "final",
                "timestamp": token_summary["finished_at"],
                "final_compaction": final_compaction,
                "section_counts": _section_counts(dossier),
                "output_parsed": dossier.model_dump_json(indent=2),
                "action_log": action_log,
                "token_summary": token_summary,
            }
        )
        log.info(
            f"[{net_id}] tokens: {token_summary['input_tokens']} in / {token_summary['output_tokens']} out "
            f"across {token_summary['calls']} calls in {token_summary['elapsed_seconds']}s; "
            f"Tavily {token_summary['tavily_credits']} credits; "
            f"by model: {token_summary['by_model']}"
        )
        return dossier, trace, token_summary
    except Exception as e:

        log.error(f"Error running research agent: {e}")
        summary = tracker.summary()
        summary["elapsed_seconds"] = round(time.perf_counter() - run_t0, 2)
        summary["started_at"] = started_at
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        return dossier, trace, summary

def _sum_by_model(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum a list of token summaries into one {input,output,cached,total,calls,by_model}."""
    agg: Dict[str, Dict[str, int]] = {}
    for s in summaries:
        for model, m in (s.get("by_model") or {}).items():
            a = agg.setdefault(model, {"input": 0, "output": 0, "cached": 0, "calls": 0})
            a["input"] += m["input"]
            a["output"] += m["output"]
            a["cached"] += m.get("cached", 0)
            a["calls"] += m["calls"]
    ti = sum(a["input"] for a in agg.values())
    to = sum(a["output"] for a in agg.values())
    tc = sum(a["cached"] for a in agg.values())
    elapsed = round(sum(s.get("elapsed_seconds", 0) or 0 for s in summaries), 2)
    credits = round(sum(s.get("tavily_credits", 0) or 0 for s in summaries), 3)
    return {
        "input_tokens": ti,
        "output_tokens": to,
        "total_tokens": ti + to,
        "cached_input_tokens": tc,
        "cache_hit_rate": round(tc / ti, 3) if ti else 0.0,
        "calls": sum(a["calls"] for a in agg.values()),
        "by_model": agg,
        "elapsed_seconds": elapsed,
        "tavily_credits": credits,
    }

_COUNT_SECTIONS = (
    "media_mentions",
    "career_trajectory",
    "publications",
    "clinical_impacts",
    "community_impacts",
    "economic_impacts",
    "policy_impacts",
    "grants",
)

def _section_items(value) -> list:
    """The list of entries in a section value: an absence-section object carries them under
    `.items`; publications / career_trajectory are bare lists."""
    items = getattr(value, "items", None)
    if items is not None:
        return items
    return value or []

def _section_counts(d: Dossier) -> Dict[str, int]:
    """Non-zero entry counts per dossier section, for compaction / per-turn deltas."""
    counts: Dict[str, int] = {}
    kp = d.key_profiles
    for sub in _PROFILE_SUBLISTS:
        n = len(getattr(kp, sub, []) or [])
        if n:
            counts[f"key_profiles.{sub}"] = n
    for sec in _COUNT_SECTIONS:
        n = len(_section_items(getattr(d, sec, None)))
        if n:
            counts[sec] = n
    return counts

def _counts_from_snapshot(snap: Any) -> Optional[Dict[str, int]]:
    """Section counts derived from a dossier JSON snapshot dict (for the slim trace)."""
    if not isinstance(snap, dict):
        return None
    counts: Dict[str, int] = {}
    for sub, v in (snap.get("key_profiles") or {}).items():
        if isinstance(v, list) and v:
            counts[f"key_profiles.{sub}"] = len(v)
    for sec in _COUNT_SECTIONS:
        v = snap.get(sec)
        if isinstance(v, dict):
            v = v.get("items")
        if isinstance(v, list) and v:
            counts[sec] = len(v)
    return counts

def _slim_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lean, readable view of the full trace: distilled tool results, section-count deltas,
    and compaction summaries, dropping raw API payloads and full per-turn dossier snapshots."""
    slim: List[Dict[str, Any]] = []
    for rec in trace:
        if rec.get("turn") == "final":
            slim.append(
                {
                    k: rec.get(k)
                    for k in (
                        "turn",
                        "timestamp",
                        "final_compaction",
                        "section_counts",
                        "action_log",
                        "token_summary",
                    )
                }
            )
            continue
        if rec.get("turn") == "assemble":
            slim.append({k: rec.get(k) for k in
                         ("turn", "label", "run", "end_reason", "timestamp", "stats",
                          "counts_before", "counts_after", "nudge_cycles", "skipped_records",
                          "auto_skipped_labels", "filter_log", "token_usage", "ops")})
            continue
        if rec.get("turn") in ("impact_writer", "impact_revise"):
            slim.append(
                {
                    k: rec.get(k)
                    for k in (
                        "turn",
                        "timestamp",
                        "llm_seconds",
                        "token_usage",
                        "findings_folded",
                        "attempts",
                        "section",
                        "n_items",
                        "n_revised",
                        "feature",
                        "z",
                        "error",
                    )
                    if k in rec
                }
            )
            continue
        s = {
            k: rec.get(k)
            for k in (
                "turn",
                "response_id",
                "timestamp",
                "llm_seconds",
                "compaction",
                "token_usage",
                "elapsed_seconds",
            )
        }
        s["dossier_after"] = _counts_from_snapshot(rec.get("dossier_after"))
        s["tool_calls"] = [
            {
                "function": tc.get("function"),
                "arguments": tc.get("arguments"),
                "elapsed_seconds": tc.get("elapsed_seconds"),
                "log": tc.get("log"),
                "result_distilled": tc.get("result_distilled"),
            }
            for tc in rec.get("tool_calls", [])
        ]
        slim.append(s)
    return slim

def _seed_internal_grants(
    dossier: Dossier, records: Optional[List[Dict[str, Any]]]
) -> Dossier:
    """Deterministically seed internal CRIS proposals into dossier.grants from the authoritative
    export; idempotent on Proposal Number, so re-running across research runs does not duplicate."""
    if not records:
        return dossier

    def _year(date_str):

        try:
            return int(str(date_str).strip().split("/")[-1])
        except (ValueError, TypeError, IndexError):
            return None

    def _float(v):
        try:
            return float(str(v).replace(",", "").replace("$", "").strip())
        except (ValueError, TypeError):
            return None

    def _s(v):
        return (str(v).strip() or None) if v is not None else None

    existing = {_gv(c, "data", "proposal_number") for c in dossier.grants.items}
    for rec in records:
        pn = _s(rec.get("Proposal Number"))
        if pn and pn in existing:
            continue
        grant = Grant(
            proposal_number=pn,
            title=_s(rec.get("Proposal Title")),
            award_number=_s(rec.get("Award Grant Code")),
            sponsor=_s(rec.get("Sponsor Name")),
            pi_net_id=_s(rec.get("Proposal PI Net ID")),
            pi_name=_s(rec.get("Proposal PI Full Name")),
            status_label=_s(rec.get("Proposal Status Description")),
            start_date=_s(rec.get("Proposal Start Date")),
            end_date=_s(rec.get("Proposal End Date")),
            start_year=_year(rec.get("Proposal Start Date")),
            end_year=_year(rec.get("Proposal End Date")),
            amount=_float(rec.get("Proposal Total Amount")),
            institution=_s(rec.get("Dept Level 5 Title")),
        )
        dossier.grants.items.append(
            GrantCandidate(
                data=grant,
                reasoning="Internal CRIS proposal record (authoritative).",
                confidence=ConfidenceScore.HIGH,
                status=Status.CONFIRMED,
            )
        )
        existing.add(pn)
    return dossier

def critique_dossier(net_id, dossier, action_log, *, client, model, reasoning_effort,
                     on_trace=None, on_usage=None, prompt_cache_key=None):
    """Isolated dossier critic (fresh call, not chained to the loop): returns issue dicts
    {type, section, what, fix}, or None if the critic failed (never treat None as clean)."""
    prompt = _CRITIQUE_PROMPT.safe_substitute(
        dossier_json=dossier.model_dump_json(exclude={"action_log", "last_updated"}),
        action_log=("\n".join(action_log) if action_log else "(none recorded)"),
    )
    try:
        parsed = parse_response_with_backoff(
            client=client, model=model, input=prompt, text_format=_Critique,
            reasoning={"effort": reasoning_effort, "summary": "auto"},
            deadline_s=deadline_for(len(prompt) // 4), store=True,
            **({"prompt_cache_key": prompt_cache_key} if prompt_cache_key else {}),
        )
        if on_usage and getattr(parsed, "usage", None):
            on_usage(parsed.usage)
        out = parsed.output_parsed
        issues = [i.model_dump() for i in out.issues] if out and out.issues else []
    except Exception as e:
        log.warning(f"[{net_id}] dossier critic failed ({e}); state UNKNOWN, the loop continues "
                    f"on the deterministic checks")
        issues = None
    log.info(f"[{net_id}] critique: "
             + ("unavailable" if issues is None else f"{len(issues)} issue(s) {[i['type'] for i in issues]}"))
    if on_trace:
        on_trace({"step": "critique", "timestamp": datetime.now(timezone.utc).isoformat(), "issues": issues})
    return issues

def run_app(cfg: DictConfig, hydra_output_dir: str) -> None:
    log.info("Application started.")

    cache = ResponseCache()
    cache.clear_expired()

    scholars_df = pd.read_csv(cfg.dossier.scholars_path)
    selected_scholars = scholars_df

    only = str(cfg.dossier.get("only", "") or "").strip()
    if only:
        wanted = {x.strip() for x in only.split(",") if x.strip()}
        selected_scholars = scholars_df[
            scholars_df["NetID"].astype(str).str.strip().isin(wanted)
        ]
        log.info(
            f"dossier.only -> {sorted(wanted)}; running {len(selected_scholars)} scholar(s)"
        )
    log.info(f"{len(selected_scholars)} scholar(s) from {cfg.dossier.scholars_path}")
    initialize_dossier(cfg, selected_scholars, hydra_output_dir)

    grant_path = _find_data("data/grant_data.csv")
    grants_by_netid: Dict[str, List[Dict[str, Any]]] = {}
    if grant_path.is_file():

        grants_df = pd.read_csv(grant_path, header=1, dtype=str)
        selected_netids = selected_scholars["NetID"].astype(str).str.strip()
        grants_df["Proposal PI Net ID"] = (
            grants_df["Proposal PI Net ID"].astype(str).str.strip()
        )

        status = grants_df["Proposal Status Description"].astype(str).str.strip()
        kept = grants_df[status.isin(["Funded", "Pending"])]
        filtered = kept[kept["Proposal PI Net ID"].isin(selected_netids)].fillna("")
        grants_by_netid = {
            nid: grp.to_dict(orient="records")
            for nid, grp in filtered.groupby("Proposal PI Net ID")
        }
    else:
        log.warning(
            f"{grant_path} not found; running without seeded internal grant context."
        )

    prompt_template = _load_prompt(Path(cfg.dossier.prompt_path))

    model_name = cfg.model.name
    cheap_model_name = cfg.dossier.get("cheap_model_name", model_name)
    azure_region = cfg.model.get("azure_region", "")
    region_suffix = f"_{azure_region}" if azure_region else ""
    endpoint = os.environ.get(f"AZURE_OPENAI_ENDPOINT{region_suffix}")
    api_key = os.environ.get(f"AZURE_OPENAI_API_KEY{region_suffix}")
    if not endpoint or not api_key:
        raise ValueError(
            f"Missing Azure OpenAI credentials for region '{azure_region or 'default'}'."
        )

    base = endpoint.rstrip("/")
    if not base.endswith("/openai/v1"):
        base += "/openai/v1"
    log.info(
        f"Model: {model_name} (cheap: {cheap_model_name}) on {azure_region or 'default'}"
    )

    openai_client = OpenAI(
        base_url=base + "/", api_key=api_key, timeout=1200.0, max_retries=0
    )

    max_research_runs = cfg.dossier.get("max_research_runs", 3)
    max_turns_per_run = cfg.dossier.get("max_turns_per_run", 25)
    reasoning_effort = cfg.dossier.get("reasoning_effort", "medium")
    compaction_context_tokens = cfg.dossier.get("compaction_context_tokens", 16000)
    max_context_tokens = cfg.dossier.get("max_context_tokens", 300000)
    scholar_workers = cfg.dossier.get("scholar_workers", 10)
    use_prompt_cache_key = cfg.dossier.get("prompt_cache_key", True)
    max_assembler_turns = cfg.dossier.get("max_assembler_turns", 20)
    source_gate_attempts = cfg.dossier.get("source_gate_attempts", 3)

    cris_props = load_cris()

    output_dir = Path(hydra_output_dir)
    token_runs: Dict[str, List[Dict[str, Any]]] = {}
    scholar_wall: Dict[str, float] = {}
    run_t0 = time.perf_counter()
    run_started_at = datetime.now(timezone.utc).isoformat()

    def process_scholar(file: Path, net_id: str, grant_context: List[Dict[str, Any]]):
        """One scholar end-to-end (research run(s) + dossier/trace writes), run in its own
        thread. Returns (net_id, [token_summary], wall_seconds)."""
        scholar_t0 = time.perf_counter()
        log.info(f"[{net_id}] processing")
        summaries: List[Dict[str, Any]] = []
        prior_issues: Optional[List[str]] = None
        prior_issue_dicts: Optional[List[Dict[str, Any]]] = None
        aux_tracker = TokenTracker()
        dirty_impacts: set = set()
        cache_key = net_id if use_prompt_cache_key else None
        changes_path = output_dir / f"{net_id}.changes.jsonl"

        obs = Observations()
        obs_path = output_dir / f"{net_id}.obs.json"
        if obs_path.exists():
            try:
                obs.load_keys(json.loads(obs_path.read_text()))
            except (OSError, ValueError) as e:
                log.warning(f"[{net_id}] unreadable observations sidecar ({e}); starting empty")

        dispositions: Dict[str, Any] = {"skipped_records": []}
        disp_path = output_dir / f"{net_id}.disp.json"
        if disp_path.exists():
            try:
                dispositions = json.loads(disp_path.read_text())
            except (OSError, ValueError) as e:
                log.warning(f"[{net_id}] unreadable dispositions sidecar ({e}); starting empty")
        if dispositions.get("skipped_records"):
            log.info(f"[{net_id}] offer ledger: {len(dispositions['skipped_records'])} "
                     f"record(s) already settled as skipped")
        for research_run in range(1, max_research_runs + 1):
            log.info(f"[{net_id}] research run {research_run}/{max_research_runs}")
            with open(file) as f:
                current_dossier = Dossier.model_validate_json(f.read())
            changelog = ChangeLog(changes_path, net_id, research_run)
            updated_dossier, trace, token_summary = run_research_agent(
                net_id,
                current_dossier,
                prompt_template,
                openai_client,
                model_name,
                cheap_model_name,
                max_turns_per_run,
                reasoning_effort,
                compaction_context_tokens,
                max_context_tokens,
                cache,
                grant_context,
                prior_issues=prior_issues,
                prior_issue_dicts=prior_issue_dicts,
                changelog=changelog,
                use_prompt_cache_key=use_prompt_cache_key,
                max_assembler_turns=max_assembler_turns,
                source_gate_attempts=source_gate_attempts,
                cris_props=cris_props,
                obs=obs,
                dispositions=dispositions,
                research_run=research_run,
            )
            try:
                obs_path.write_text(json.dumps(obs.export_keys()))
                disp_path.write_text(json.dumps(dispositions, ensure_ascii=False, indent=2))
            except OSError as e:
                log.warning(f"[{net_id}] could not write sidecars: {e}")
            summaries.append(token_summary)
            dirty_impacts |= {s for s in token_summary.get("dirty_sections", []) if s in _IMPACT_PILLARS}
            if not updated_dossier:
                log.warning(
                    f"[{net_id}] agent returned no dossier; stopping iterations."
                )
                break
            updated_dossier.last_updated = datetime.now(timezone.utc)
            with open(file, "w") as f:
                f.write(updated_dossier.model_dump_json(indent=2))
            trace_dir = file.parent / "traces" / net_id
            trace_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            meta = {
                "research_run": research_run,
                "net_id": net_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            with lzma.open(trace_dir / f"run_{research_run}_{ts}.full.json.xz", "wt", encoding="utf-8") as f:
                json.dump({**meta, "interactions": trace}, f, indent=2)
            with lzma.open(trace_dir / f"run_{research_run}_{ts}.slim.json.xz", "wt", encoding="utf-8") as f:
                json.dump({**meta, "interactions": _slim_trace(trace)}, f, indent=2)
            log.info(f"[{net_id}] dossier updated (run {research_run})")

            if research_run < max_research_runs:
                crit = critique_dossier(
                    net_id, updated_dossier, list(updated_dossier.action_log),
                    client=openai_client, model=model_name, reasoning_effort=reasoning_effort,
                    on_usage=lambda u: aux_tracker.add(model_name, u), prompt_cache_key=cache_key,
                )

                gaps = _empty_section_issues(updated_dossier)
                flagged = _flagged_issues(updated_dossier)
                n_left = int(token_summary.get("pubs_missing_at_end") or 0)
                if n_left:
                    gaps.append({
                        "type": "gap", "section": "publications",
                        "what": f"{n_left} publication records the tools returned this run were "
                                f"never appended (the assembly ended early)",
                        "fix": "Re-query OpenAlex and PubMed (cached, no new cost) so the records "
                               "re-enter the transcript, then append every one of them.",
                    })
                if gaps:
                    log.info(f"[{net_id}] {len(gaps)} empty unmarked section(s) seed the next run: "
                             f"{[g['section'] for g in gaps]}")
                if flagged:
                    log.info(f"[{net_id}] {len(flagged)} flagged source(s) seed the next run")
                issues = (crit or []) + gaps + flagged
                with lzma.open(trace_dir / f"critique_run_{research_run}_{ts}.json.xz", "wt", encoding="utf-8") as f:
                    json.dump(issues, f, indent=2)
                if not issues:
                    log.info(f"[{net_id}] dossier clean after run {research_run}"
                             + (" (critic unavailable; deterministic checks clean)" if crit is None else "")
                             + "; stopping")
                    break
                prior_issues = [f"[{i['type']}] ({i['section']}) {i['what']} -> fix: {i['fix']}" for i in issues]
                prior_issue_dicts = issues

        if _TSBM_TARGET_STATS and dirty_impacts:
            with open(file) as f:
                final_dossier = Dossier.model_validate_json(f.read())
            rtracker = TokenTracker()
            revise_trace: List[Dict[str, Any]] = []
            revise_impact_register(
                final_dossier, llm=openai_client, model=model_name,
                target_stats=_TSBM_TARGET_STATS,
                on_usage=lambda u: rtracker.add(model_name, u),
                on_trace=revise_trace.append, net_id=net_id,
            )
            final_dossier.last_updated = datetime.now(timezone.utc)
            with open(file, "w") as f:
                f.write(final_dossier.model_dump_json(indent=2))
            summaries.append(rtracker.summary())
            if revise_trace:
                trace_dir = file.parent / "traces" / net_id
                trace_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                with lzma.open(trace_dir / f"revise_{ts}.json.xz", "wt", encoding="utf-8") as f:
                    json.dump(revise_trace, f, indent=2)
        summaries.append(aux_tracker.summary())

        wall = round(time.perf_counter() - scholar_t0, 2)
        log.info(f"[{net_id}] done in {wall}s")
        return net_id, summaries, wall

    work = []
    for file in output_dir.glob("*.json"):
        if (
            file.name == "token_summary.json"
        ):
            continue
        nid = file.stem.strip()
        work.append((file, nid, grants_by_netid.get(nid, [])))

    n_workers = max(1, min(len(work), scholar_workers))
    log.info(f"Processing {len(work)} scholars, up to {n_workers} concurrently")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_scholar, f, nid, gc): nid for f, nid, gc in work
        }
        for fut in as_completed(futures):
            nid = futures[fut]
            try:
                rid, summaries, wall = fut.result()
                token_runs[rid] = summaries
                scholar_wall[rid] = wall
            except Exception as e:
                log.error(f"[{nid}] failed: {e}")

    per_scholar = {
        nid: {**_sum_by_model(lst), "wall_clock_seconds": scholar_wall.get(nid)}
        for nid, lst in token_runs.items()
    }
    aggregate = _sum_by_model([s for lst in token_runs.values() for s in lst])
    run_elapsed = round(time.perf_counter() - run_t0, 2)
    n = len(per_scholar) or 1
    aggregate["wall_clock_seconds"] = run_elapsed
    aggregate["n_scholars"] = len(per_scholar)
    aggregate["avg_seconds_per_scholar"] = round(sum(scholar_wall.values()) / n, 2)
    aggregate["started_at"] = run_started_at
    aggregate["finished_at"] = datetime.now(timezone.utc).isoformat()
    with open(output_dir / "token_summary.json", "w") as f:
        json.dump({"per_scholar": per_scholar, "aggregate": aggregate}, f, indent=2)
    log.info(
        f"Total: {aggregate.get('input_tokens', 0)} in / {aggregate.get('output_tokens', 0)} out, "
        f"{aggregate.get('calls', 0)} calls; wall-clock {run_elapsed}s "
        f"({aggregate['avg_seconds_per_scholar']}s/scholar avg) across {len(per_scholar)} scholars"
    )
    log.info(
        "Per-scholar: "
        + ", ".join(
            f"{nid}={s['total_tokens']}tok/{s.get('wall_clock_seconds')}s"
            for nid, s in per_scholar.items()
        )
    )

    final = cache.get_stats()
    log.info(
        f"Cache: {final['search_entries']} search, {final['extract_entries']} extract, "
        f"{final['api_entries']} api entries"
    )

@hydra.main(version_base=None, config_path="../conf", config_name="dossier")
def main(cfg: DictConfig) -> None:
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    run_app(cfg, hydra_output_dir)

if __name__ == "__main__":
    main()
