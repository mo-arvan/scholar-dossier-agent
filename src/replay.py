"""Replay assembly on a saved trace's tool-call log, without re-running the search.

Reconstructs the seed dossier, transcript, and Observations from the trace, then re-runs the
assembler over them.
"""
import argparse
import json
import lzma
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_dossier import _IMPACT_PILLARS, _assembler_transcript, TokenTracker
from dossier_edit import run_assembler_session
from dossier_response import Dossier
from source_filter import Observations, load_cris

def turns_from_trace(trace: dict) -> list:
    """Reconstruct the append-only transcript from the trace's loop turns."""
    turns = []
    for r in trace.get("interactions", []):
        if not isinstance(r, dict):
            continue
        tcs = r.get("tool_calls")
        if not tcs:
            continue
        actions = [tc.get("log") for tc in tcs if tc.get("log")]
        findings = [tc.get("result_distilled") for tc in tcs if tc.get("result_distilled")]
        turns.append({"n": r.get("turn"), "actions": actions, "findings": findings})
    return turns

def obs_from_trace(trace: dict, sidecar: Path) -> Observations:
    """Rebuild what the run's tools observed from the trace, plus the cross-run sidecar
    when one sits beside the run's dossiers."""
    obs = Observations()
    for r in trace.get("interactions", []):
        if isinstance(r, dict):
            for tc in (r.get("tool_calls") or []):
                obs.add_tool_call(tc)
    if sidecar.exists():
        try:
            obs.load_keys(json.loads(sidecar.read_text()))
        except (OSError, ValueError):
            pass
    return obs

def seed_from_trace(trace: dict):
    """The pre-assembly seed dossier: the first loop turn's dossier_after (identity + seeded
    grants; other sections empty since assembly hasn't run yet)."""
    for r in trace.get("interactions", []):
        if isinstance(r, dict) and r.get("tool_calls") and r.get("dossier_after"):
            return Dossier.model_validate(r["dossier_after"])
    return None

def make_client(cfg):
    region = cfg.model.get("azure_region", "")
    suf = f"_{region}" if region else ""
    endpoint = os.environ.get(f"AZURE_OPENAI_ENDPOINT{suf}")
    api_key = os.environ.get(f"AZURE_OPENAI_API_KEY{suf}")
    if not endpoint or not api_key:
        raise ValueError(f"Missing Azure OpenAI credentials for region '{region or 'default'}'.")
    base = endpoint.rstrip("/")
    if not base.endswith("/openai/v1"):
        base += "/openai/v1"
    return OpenAI(base_url=base + "/", api_key=api_key, timeout=1200.0, max_retries=0)

def load_trace(path) -> dict:
    """Load a trace, transparently handling xz-compressed (.json.xz) and raw (.json)."""
    p = str(path)
    if p.endswith(".xz"):
        with lzma.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(p) as f:
        return json.load(f)

def _dedup_traces(paths):
    """Keep one file per logical trace when both a raw .json and its .json.xz are on disk
    (the raw is a gitignored local convenience); prefer the compressed canonical form."""
    paths = list(paths)
    xz_stems = {str(p)[:-3] for p in paths if str(p).endswith(".xz")}
    return [p for p in paths if str(p).endswith(".xz") or str(p) not in xz_stems]

def find_traces(target: str, net_ids: set) -> list:
    p = Path(target)
    if p.is_file():
        return [p]
    out = []
    for d in sorted((p / "traces").glob("*")):
        if net_ids and d.name not in net_ids:
            continue
        full = sorted(_dedup_traces(d.glob("run_*.full.json*")))
        if full:
            out.append(full[-1])
    return out

def preview(dossier: Dossier) -> None:
    for pillar in ("clinical_impacts", "community_impacts", "economic_impacts", "policy_impacts"):
        items = getattr(dossier, pillar).items
        if items:
            print(f"    {pillar} ({len(items)}):")
            for c in items[:2]:
                print(f"      - {c.summary}")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="a run dir (outputs/<ts>) or a single trace .full.json")
    ap.add_argument("net_ids", nargs="*", help="optional net_id subset")
    ap.add_argument("--out", default=None, help="output dir (default: alongside the run)")
    ap.add_argument("--id", default=None, dest="label",
                    help="label this replay; output becomes <net_id>.replay.<id>.json so reruns "
                         "(e.g. one per prompt version) do not overwrite each other")
    ap.add_argument("--impacts-only", action="store_true",
                    help="reassemble ONLY the 4 impact pillars (seed from the run's dossier so the "
                         "other sections are present for url context, with the pillars cleared), "
                         "for fast iteration on the impact guidance")
    ap.add_argument("--dry", action="store_true", help="reconstruct only; no LLM call")
    args = ap.parse_args()

    traces = find_traces(args.target, set(args.net_ids))
    if not traces:
        print("no traces found under", args.target)
        return

    cfg = client = None
    if not args.dry:
        load_dotenv()
        with initialize_config_dir(
            config_dir=str(Path(__file__).resolve().parent.parent / "conf"), version_base=None
        ):
            cfg = compose(config_name="dossier")
        client = make_client(cfg)
    cris = load_cris()

    for tf in traces:
        trace = load_trace(tf)
        net_id = trace.get("net_id") or tf.parent.name
        run_root = tf.parent.parent.parent
        turns = turns_from_trace(trace)
        n_find = sum(len(t["findings"]) for t in turns)
        if args.impacts_only:

            final_path = run_root / f"{net_id}.json"
            if not final_path.exists():
                print(f"{net_id}: --impacts-only needs the run's dossier at {final_path}; skipping")
                continue
            seed = Dossier.model_validate(json.load(open(final_path)))
            for p in _IMPACT_PILLARS:
                getattr(seed, p).items.clear()
                getattr(seed, p).confirmed_absent = False
        else:
            seed = seed_from_trace(trace)
        if seed is None or not turns:
            print(f"{net_id}: cannot reconstruct (turns={len(turns)}, seed={'ok' if seed else 'missing'})")
            continue
        scope = "4 impact pillars" if args.impacts_only else "all sections"
        print(f"{net_id}: {len(turns)} turns, {n_find} findings"
              + ("  [dry: reconstruct only]" if args.dry else f", reassembling {scope}..."))
        if args.dry:
            continue
        obs = obs_from_trace(trace, run_root / f"{net_id}.obs.json")
        tracker = TokenTracker()
        transcript, pub_records = _assembler_transcript(
            turns, net_id=net_id, dossier=seed,
            max_context_tokens=cfg.dossier.get("max_context_tokens", 300000))
        dossier, rec = run_assembler_session(
            seed, transcript, None, client=client, model=cfg.model.name, net_id=net_id,
            obs=obs, cris=cris, expected_pubs=pub_records,
            reasoning_effort=cfg.dossier.get("reasoning_effort", "medium"),
            max_turns=cfg.dossier.get("max_assembler_turns", 20),
            source_gate_attempts=cfg.dossier.get("source_gate_attempts", 3),
            on_usage=lambda u: tracker.add(cfg.model.name, u),
        )
        out_dir = Path(args.out) if args.out else run_root
        name = f"{net_id}.replay.{args.label}.json" if args.label else f"{net_id}.replay.json"
        out_path = out_dir / name
        out_path.write_text(dossier.model_dump_json(indent=2))
        n_imp = sum(len(getattr(dossier, p).items) for p in _IMPACT_PILLARS)
        s = tracker.summary()
        print(f"  -> {out_path} ({n_imp} impacts; {s['input_tokens']} in / {s['output_tokens']} out; "
              f"stats {rec.get('stats')})")
        preview(dossier)

if __name__ == "__main__":
    main()
