"""Gated entry-op dossier assembly: append/patch/confirm_absent, each validated, checked for
identity collisions, and source-gated before it is committed to the Dossier."""

import json
import logging
import re
import threading
import typing
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, TypeAdapter, ValidationError

from dossier_response import Dossier
from finding_id import finding_id
from llm_retry import create_response_with_backoff, deadline_for

log = logging.getLogger(__name__)

_PROFILE_SUBLISTS = ("institutional_homepage", "google_scholar", "linkedin", "orcid", "personal_website")
_LIST_SECTIONS = ("career_trajectory", "publications")
ABSENCE_SECTIONS = ("grants", "media_mentions", "clinical_impacts",
                    "community_impacts", "economic_impacts", "policy_impacts")
SECTIONS: Tuple[str, ...] = tuple(
    [f"key_profiles.{s}" for s in _PROFILE_SUBLISTS] + list(_LIST_SECTIONS) + list(ABSENCE_SECTIONS)
)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())

def _field_of(section: str) -> str:
    """The Dossier field a section name lives on ('key_profiles.orcid' -> 'key_profiles')."""
    return section.split(".")[0]

@lru_cache(maxsize=None)
def _section_adapter(fieldname: str) -> TypeAdapter:
    """TypeAdapter for one Dossier field's annotation (shared schema with dossier_response)."""
    return TypeAdapter(Dossier.model_fields[fieldname].annotation)

@lru_cache(maxsize=None)
def _entry_fields(section: str) -> str:
    """One line of entry-shape hint for a section, derived from the entry model so it can't
    drift from dossier_response."""
    base = _field_of(section)
    ann = Dossier.model_fields[base].annotation
    if base == "key_profiles":
        model = ann.model_fields[section.split(".")[1]].annotation
        model = typing.get_args(model)[0]
    else:
        if not isinstance(ann, type) or not issubclass(ann, BaseModel):
            model = typing.get_args(ann)[0]
        else:
            model = typing.get_args(ann.model_fields["items"].annotation)[0]
    parts = []
    for name, f in model.model_fields.items():
        if name == "source_flag":
            continue
        sub = f.annotation
        sub = next((a for a in typing.get_args(sub) if a is not type(None)), sub)
        if isinstance(sub, type) and issubclass(sub, BaseModel):
            parts.append(f"{name}{{{', '.join(sub.model_fields)}}}")
        elif isinstance(sub, type) and issubclass(sub, Enum):
            parts.append(f"{name}({'|'.join(m.value for m in sub)})")
        else:
            parts.append(name)
    return ", ".join(parts)

def _payload(entry: dict) -> dict:
    return entry.get("data") if isinstance(entry.get("data"), dict) else entry

def _entry_identity(section: str, entry: dict) -> Optional[str]:
    """Normalized identity key for an entry (doi/pmid/openalex_id for publications, an
    award/proposal/nih number for grants, url for media); None elsewhere, so append never
    blocks there."""
    if not isinstance(entry, dict):
        return None
    payload = _payload(entry)
    base = _field_of(section)
    if base == "publications":
        for k in ("doi", "pmid", "openalex_id"):
            if payload.get(k):
                return f"{k}={_norm(payload[k])}"
    elif base == "grants":
        for k in ("award_number", "proposal_number", "nih_project_num"):
            if payload.get(k):
                return f"{k}={_norm(payload[k])}"
    elif base == "media_mentions":
        if entry.get("url"):
            return f"url={_norm(entry['url'])}"
    return None

def _id_parts(section: str, entry: dict) -> list:
    """The finding_id key parts for one entry, mirroring make_review_workbook.flatten's keys so
    an assembler ID joins to the review workbook's row for the same finding."""
    d = _payload(entry)
    base = _field_of(section)
    if base == "key_profiles":
        return [entry.get("url") or section]
    if base == "career_trajectory":
        return [d.get("title"), d.get("institution")]
    if base == "publications":
        return [d.get("doi") or d.get("pmid") or d.get("openalex_id") or d.get("title")]
    if base == "grants":
        return [d.get("award_number") or d.get("proposal_number")
                or d.get("nih_project_num") or d.get("title")]
    if base == "media_mentions":
        return [entry.get("url")] if entry.get("url") else [entry.get("source"), entry.get("title")]

    return [entry.get("identifier") or entry.get("name") or entry.get("body_name"),
            entry.get("year")]

def entry_id(section: str, entry: dict, salt: str = "") -> str:
    parts = [p for p in _id_parts(section, entry) if p not in (None, "")]
    if not parts:
        parts = [json.dumps(entry, sort_keys=True, ensure_ascii=False)]
    if salt:
        parts.append(salt)
    return finding_id(_field_of(section), *parts)

def _entry_label(entry: dict, width: int = 50) -> str:
    d = _payload(entry)
    label = (d.get("title") or d.get("name") or entry.get("name") or entry.get("body_name")
             or entry.get("title") or entry.get("url") or d.get("url") or entry.get("summary") or "?")
    label = str(label)
    return label[: width - 3] + "..." if len(label) > width else label

def _entries_of(fieldname: str, data) -> list:
    if fieldname in _LIST_SECTIONS:
        return data or []
    if fieldname == "key_profiles":
        return [e for sub in _PROFILE_SUBLISTS for e in ((data or {}).get(sub) or [])]
    return (data or {}).get("items") or []

def find_collision(section: str, value) -> Optional[str]:
    """The first duplicated identity key in a section value, or None (a keyed duplicate is a
    loud write-time error, not a silent dup)."""
    fieldname = _field_of(section)
    data = _section_adapter(fieldname).dump_python(value, mode="json")
    seen = set()
    for e in _entries_of(fieldname, data):
        k = _entry_identity(fieldname, e)
        if k is None:
            continue
        if k in seen:
            return k
        seen.add(k)
    return None

def section_size(dossier: Dossier, fieldname: str) -> int:
    """Entry count for one Dossier field (profiles summed across sub-lists)."""
    v = getattr(dossier, fieldname)
    if fieldname == "key_profiles":
        return sum(len(getattr(v, s, []) or []) for s in _PROFILE_SUBLISTS)
    items = getattr(v, "items", None)
    if items is not None:
        return len(items)
    return len(v or [])

def _dump_field(dossier: Dossier, fieldname: str):
    return _section_adapter(fieldname).dump_python(getattr(dossier, fieldname), mode="json")

def _render_entry(e: dict) -> str:
    """One compact line for one entry, with `source_flag` stripped so the model never sees or
    tries to manage its own flag."""
    e = {k: v for k, v in e.items() if k != "source_flag"}
    return json.dumps(e, separators=(",", ":"), sort_keys=True, ensure_ascii=False)

@dataclass
class OpResult:
    ok: bool
    op: str
    section: str
    message: str
    entry_id: Optional[str] = None
    flagged: bool = False
    gate_verdict: Optional[str] = None
    error_kind: Optional[str] = None

class AssemblerState:
    """Session working state: per-section ordered (id, entry) pairs over one Dossier.

    IDs stay stable for the session once assigned, so the model's handle never goes stale; a
    validator that would silently drop an entry surfaces as a loud rejection instead.
    """

    def __init__(self, dossier: Dossier):
        self.dossier = dossier
        self.pairs: Dict[str, List[list]] = {}
        self.attempts: Counter = Counter()
        for section in SECTIONS:
            self.pairs[section] = []
            seen: Counter = Counter()
            for e in self._current_entries(section):
                eid = entry_id(section, e)
                seen[eid] += 1
                if seen[eid] > 1:
                    eid = entry_id(section, e, salt=str(seen[eid]))
                self.pairs[section].append([eid, e])

    def _current_entries(self, section: str) -> list:
        base = _field_of(section)
        data = _dump_field(self.dossier, base)
        if base == "key_profiles":
            return (data or {}).get(section.split(".")[1]) or []
        if base in _LIST_SECTIONS:
            return data or []
        return (data or {}).get("items") or []

    def ids_of(self, section: str) -> List[str]:
        return [p[0] for p in self.pairs[section]]

    def counts(self) -> Dict[str, int]:
        return {s: len(self.pairs[s]) for s in SECTIONS if self.pairs[s]}

    def render(self) -> str:
        """The full ID-stamped dossier render for the base prompt; deterministic byte-for-byte
        for a given state (prompt-cache invariant)."""
        blocks = []
        for section in SECTIONS:
            pairs = self.pairs[section]
            head = f"## {section} ({len(pairs)})  entry fields: {_entry_fields(section)}"
            base = _field_of(section)
            if base in ABSENCE_SECTIONS:
                sec = getattr(self.dossier, base)
                if getattr(sec, "confirmed_absent", False):
                    head += f"  [confirmed_absent: {sec.confirmed_absent_reason or 'no reason recorded'}]"
            lines = [head]
            if not pairs:
                lines.append("(none)")
            lines.extend(f"{eid} {_render_entry(e)}" for eid, e in pairs)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _rebuild_value(self, base: str):
        """The Dossier field value implied by the current pairs, validated. Raises ValidationError."""
        if base == "key_profiles":
            data = {sub: [e for _i, e in self.pairs[f"key_profiles.{sub}"]] for sub in _PROFILE_SUBLISTS}
        elif base in _LIST_SECTIONS:
            data = [e for _i, e in self.pairs[base]]
        else:
            current = _dump_field(self.dossier, base) or {}
            data = {**current, "items": [e for _i, e in self.pairs[base]]}
        return _section_adapter(base).validate_python(data)

    def _commit(self, section: str, value) -> None:
        """Write the validated value onto the dossier and refresh each pair's entry dict from
        the validated dump, preserving its session ID by position."""
        base = _field_of(section)
        setattr(self.dossier, base, value)
        data = _dump_field(self.dossier, base)
        if base == "key_profiles":
            for sub in _PROFILE_SUBLISTS:
                sec = f"key_profiles.{sub}"
                fresh = (data or {}).get(sub) or []
                for pair, e in zip(self.pairs[sec], fresh):
                    pair[1] = e
        else:
            fresh = data if base in _LIST_SECTIONS else (data or {}).get("items") or []
            for pair, e in zip(self.pairs[base], fresh):
                pair[1] = e

    def _survived(self, section: str, value) -> bool:
        """True when validation kept every working entry (a keep-rule drop = a loud reject)."""
        base = _field_of(section)
        data = _section_adapter(base).dump_python(value, mode="json")
        if base == "key_profiles":
            return all(
                len(((data or {}).get(sub) or [])) == len(self.pairs[f"key_profiles.{sub}"])
                for sub in _PROFILE_SUBLISTS
            )
        got = data if base in _LIST_SECTIONS else (data or {}).get("items") or []
        return len(got) == len(self.pairs[base])

def _err_unknown_section(section: str) -> str:
    return f"Unknown section '{section}'. Valid sections: {', '.join(SECTIONS)}."

def _err_unknown_id(state: AssemblerState, section: str, eid: str) -> str:
    listing = "; ".join(f"{i} ({_entry_label(e)})" for i, e in state.pairs[section]) or "(section is empty)"
    return (f"Unknown id '{eid}' in '{section}'. Nothing was changed. Current entries: {listing}. "
            f"Use one of these ids, or append_entry if the entry does not exist yet.")

def _err_schema(e, section: str) -> str:
    return (f"The write was rejected by the schema, so nothing was changed: {str(e)[:300]}. "
            f"Entry fields for '{section}': {_entry_fields(section)}. Fix the fields and retry.")

def _err_dropped(section: str) -> str:
    return (f"The write was rejected: the entry violates a keep-rule of '{section}' (for example a "
            f"profile needs a url, a grant needs an identifying number or title, an impact needs a "
            f"name and summary). Nothing was changed. Add the missing field and retry.")

def _err_collision(section: str, key: str, state: AssemblerState) -> str:
    base = _field_of(section)
    holder = next((i for s in SECTIONS if _field_of(s) == base
                   for i, e in state.pairs[s] if _entry_identity(s, e) == key), None)
    hint = f" That identity already exists as {holder}; patch that entry instead." if holder else ""
    return f"Rejected: this would leave two entries with the same identity ({key}) in '{section}'.{hint}"

_GATE_GUIDE = {
    "empty": "the entry has no source. Give it the source the finding came from (a URL from the "
             "transcript, or the internal reference for a CRIS grant).",
    "malformed": "the entry's source is not a usable URL or internal reference. Use the well-formed "
                 "source the transcript carries for this finding.",
    "ungrounded": "the entry's source was never returned by any tool (this run or a prior one), so "
                  "it cannot be verified. Use the source that actually appears in the transcript "
                  "for this finding; do not invent one.",
}

Gate = Optional[Callable[[str, dict], Tuple[str, str]]]

def _gate_check(gate: Gate, section: str, entry: dict) -> Tuple[str, str]:
    return gate(section, entry) if gate is not None else ("ok", "")

def _parse_json_object(raw: str, what: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError) as e:
        return None, f"{what} is not valid JSON: {str(e)[:200]}. Nothing was changed."
    if not isinstance(obj, dict):
        return None, f"{what} must be a single JSON object. Nothing was changed."
    return obj, None

def apply_append(state: AssemblerState, section: str, entry_json: str, *,
                 gate: Gate = None, max_attempts: int = 3,
                 changelog: Optional["ChangeLog"] = None) -> OpResult:
    """Append ONE validated entry.

    The source gate runs inside the write: a failing source rejects the append until the last
    allowed attempt, when the entry is accepted with source_flag set instead of dropped.
    """
    if section not in SECTIONS:
        return OpResult(False, "append", section, _err_unknown_section(section), error_kind="usage")
    entry, err = _parse_json_object(entry_json, "entry")
    if err:
        return OpResult(False, "append", section, err, error_kind="schema")
    if entry.get("source_flag"):
        entry["source_flag"] = ""
    eid = entry_id(section, entry)
    taken = set(state.ids_of(section))
    salt = 1
    while eid in taken:
        salt += 1
        eid = entry_id(section, entry, salt=str(salt))

    state.pairs[section].append([eid, entry])
    try:
        value = state._rebuild_value(_field_of(section))
        if not state._survived(section, value):
            raise _KeepRuleDrop()
        dup = find_collision(section, value)
    except ValidationError as e:
        state.pairs[section].pop()
        return OpResult(False, "append", section, _err_schema(e, section), error_kind="schema")
    except _KeepRuleDrop:
        state.pairs[section].pop()
        return OpResult(False, "append", section, _err_dropped(section), error_kind="schema")
    if dup:
        state.pairs[section].pop()
        msg = _err_collision(section, dup, state)
        return OpResult(False, "append", section, msg, error_kind="collision")

    verdict, src = _gate_check(gate, section, entry)
    flagged = False
    if verdict != "ok":
        key = (section, eid)
        state.attempts[key] += 1
        if state.attempts[key] < max_attempts:
            state.pairs[section].pop()
            msg = (f"Rejected ({verdict} source, attempt {state.attempts[key]}/{max_attempts}): "
                   f"{_GATE_GUIDE[verdict]} Source given: {src or '(blank)'}. Nothing was written; "
                   f"re-append with a grounded source.")
            if changelog:
                changelog.emit(section, "reject", eid, before=verdict, after=(src or "(blank)"))
            return OpResult(False, "append", section, msg, entry_id=eid,
                            gate_verdict=verdict, error_kind="gate")

        flagged = True
        entry["source_flag"] = f"{verdict} source at write time: {src or '(blank)'}"
        value = state._rebuild_value(_field_of(section))

    state._commit(section, value)
    n = len(state.pairs[section])
    if changelog:
        changelog.emit(section, "append", eid, after=n)
        if flagged:
            changelog.emit(section, "flag", eid, after=entry["source_flag"])
    ack = f"OK {eid} appended to {section} ({n} entries)."
    if flagged:
        ack = (f"OK {eid} appended to {section} ({n} entries); its source could not be verified, "
               f"so the entry is flagged for the human reviewer. Move on.")
    return OpResult(True, "append", section, ack, entry_id=eid, flagged=flagged,
                    gate_verdict=None if not flagged else verdict)

class _KeepRuleDrop(Exception):
    pass

def apply_patch(state: AssemblerState, section: str, eid: str, fields_json: str, *,
                gate: Gate = None, changelog: Optional["ChangeLog"] = None) -> OpResult:
    """Merge named fields onto ONE existing entry (a key inside the nested `data` object merges
    in directly).

    A patch may not break an already-grounded source; one whose source now passes clears
    source_flag.
    """
    if section not in SECTIONS:
        return OpResult(False, "patch", section, _err_unknown_section(section), error_kind="usage")
    idx = next((i for i, (pid, _e) in enumerate(state.pairs[section]) if pid == eid), None)
    if idx is None:
        return OpResult(False, "patch", section, _err_unknown_id(state, section, eid),
                        error_kind="unknown_id")
    fields, err = _parse_json_object(fields_json, "fields")
    if err:
        return OpResult(False, "patch", section, err, error_kind="schema")
    flag_ignored = False
    if "source_flag" in fields:

        fields = {k: v for k, v in fields.items() if k != "source_flag"}
        flag_ignored = True
        if not fields:
            return OpResult(False, "patch", section,
                            "Rejected: source_flag is harness-managed and cannot be patched "
                            "directly. It clears automatically when the entry's source passes "
                            "the gate; patch the source itself. Nothing was changed.",
                            error_kind="usage")

    old = state.pairs[section][idx][1]
    patched = json.loads(json.dumps(old))
    data = patched.get("data") if isinstance(patched.get("data"), dict) else None
    changed: Dict[str, tuple] = {}
    for k, v in fields.items():
        if k == "data" and isinstance(v, dict) and data is not None:
            for dk, dv in v.items():
                if dk not in data:
                    return OpResult(False, "patch", section,
                                    f"Unknown data field '{dk}' for '{section}'. Valid data fields: "
                                    f"{', '.join(sorted(data))}. Nothing was changed.",
                                    error_kind="unknown_field")
                changed[f"data.{dk}"] = (data.get(dk), dv)
                data[dk] = dv
        elif k in patched and k != "data":
            changed[k] = (patched.get(k), v)
            patched[k] = v
        elif data is not None and k in data:
            changed[f"data.{k}"] = (data.get(k), v)
            data[k] = v
        else:
            valid = sorted(set(patched) - {"data", "source_flag"}) + (
                [f"data.{d}" for d in sorted(data)] if data is not None else [])
            return OpResult(False, "patch", section,
                            f"Unknown field '{k}' for '{section}'. Valid fields: {', '.join(valid)}. "
                            f"Nothing was changed.", error_kind="unknown_field")
    if not changed:
        return OpResult(False, "patch", section, "No fields given; nothing was changed.",
                        error_kind="usage")

    flagged = False
    if gate is not None:
        pre_verdict, _pre_src = _gate_check(gate, section, old)
        post_verdict, post_src = _gate_check(gate, section, patched)
        if post_verdict != "ok" and pre_verdict == "ok":
            return OpResult(False, "patch", section,
                            f"Rejected: this patch would leave the entry with an {post_verdict} "
                            f"source ({post_src or '(blank)'}), breaking a source that currently "
                            f"verifies. Nothing was changed.",
                            gate_verdict=post_verdict, error_kind="gate")
        if post_verdict == "ok":
            patched["source_flag"] = ""
        else:

            flagged = True
            patched["source_flag"] = f"{post_verdict} source at write time: {post_src or '(blank)'}"

    state.pairs[section][idx][1] = patched
    try:
        value = state._rebuild_value(_field_of(section))
        if not state._survived(section, value):
            raise _KeepRuleDrop()
        dup = find_collision(section, value)
    except ValidationError as e:
        state.pairs[section][idx][1] = old
        return OpResult(False, "patch", section, _err_schema(e, section), error_kind="schema")
    except _KeepRuleDrop:
        state.pairs[section][idx][1] = old
        return OpResult(False, "patch", section, _err_dropped(section), error_kind="schema")
    if dup:
        state.pairs[section][idx][1] = old
        return OpResult(False, "patch", section, _err_collision(section, dup, state),
                        error_kind="collision")
    state._commit(section, value)

    if changelog:
        before = {k: v[0] for k, v in changed.items()}
        after = {k: v[1] for k, v in changed.items()}
        changelog.emit(section, "patch", eid,
                       before=json.dumps(before, ensure_ascii=False, default=str),
                       after=json.dumps(after, ensure_ascii=False, default=str))
        if flagged:
            changelog.emit(section, "flag", eid, after=patched["source_flag"])
    fresh = state.pairs[section][idx][1]
    ack = f"OK {eid} patched: {_render_entry(fresh)}"
    if flag_ignored:
        ack += " (source_flag is harness-managed and was ignored)"
    return OpResult(True, "patch", section, ack, entry_id=eid, flagged=flagged)

def apply_skip(dispositions: dict, expected_pubs: List[dict], record: str, reason: str, *,
               run: Optional[int] = None,
               changelog: Optional["ChangeLog"] = None) -> OpResult:
    """Record that an offered publication record does not belong in this dossier, durably
    (persisted per scholar) so no later run re-litigates it. `record` is the record's doi,
    pmid, or exact title."""
    if not (reason or "").strip():
        return OpResult(False, "skip", "publications",
                        "Rejected: skip_record needs the reason (why this record is not this "
                        "scholar's). Nothing was recorded.", error_kind="usage")
    key = _norm(record)
    if not key:
        return OpResult(False, "skip", "publications",
                        "Rejected: give the record's doi, pmid, or exact title as listed. "
                        "Nothing was recorded.", error_kind="usage")
    match = next((r for r in expected_pubs or [] if key in _record_keys(r)), None)
    if match is None:
        return OpResult(False, "skip", "publications",
                        f"No offered record matches '{record}'. Use the doi, pmid, or exact "
                        f"title exactly as listed in the coverage check. Nothing was recorded.",
                        error_kind="usage")
    keys = _record_keys(match)
    if set(keys) & _disposed_keys(dispositions):
        return OpResult(True, "skip", "publications",
                        f"Already recorded as skipped: {match.get('title') or record}.")
    dispositions.setdefault("skipped_records", []).append({
        "keys": keys, "label": match.get("title") or record,
        "reason": reason.strip(), "run": run, "auto": False,
    })
    if changelog:
        changelog.emit("publications", "skip_record", keys[0], after=reason.strip()[:200])
    return OpResult(True, "skip", "publications",
                    f"OK, recorded as skipped for this and every later run: "
                    f"{match.get('title') or record}.")

def apply_absent(state: AssemblerState, section: str, reason: str, *,
                 changelog: Optional["ChangeLog"] = None) -> OpResult:
    """Mark an absence-capable section as confirmed absent (searched, none exist), with the
    queries run and what they ruled out as the reason. Only valid while the section is empty."""
    if section not in ABSENCE_SECTIONS:
        return OpResult(False, "absent", section,
                        f"'{section}' cannot be confirmed absent. Only these sections can: "
                        f"{', '.join(ABSENCE_SECTIONS)}.", error_kind="usage")
    if state.pairs[section]:
        return OpResult(False, "absent", section,
                        f"Rejected: '{section}' has {len(state.pairs[section])} entries; a section "
                        f"with entries cannot be confirmed absent. Nothing was changed.",
                        error_kind="usage")
    if not (reason or "").strip():
        return OpResult(False, "absent", section,
                        "Rejected: confirm_absent needs the reason (the queries run and what they "
                        "ruled out). Nothing was changed.", error_kind="usage")
    sec = getattr(state.dossier, section)
    sec.confirmed_absent = True
    sec.confirmed_absent_reason = reason.strip()
    value = state._rebuild_value(section)
    state._commit(section, value)
    if changelog:
        changelog.emit(section, "absent", None, after=reason.strip()[:200])
    return OpResult(True, "absent", section, f"OK {section} confirmed absent.")

class ChangeLog:
    """Append one JSONL row per applied op to the run dir.

    Thread-safe: guards concurrent writes to one scholar's file.
    """

    def __init__(self, path, net_id: str, source_run: int):
        self.path = Path(path)
        self.net_id = net_id
        self.source_run = source_run
        self._lock = threading.Lock()

    @staticmethod
    def _clip(v):
        if isinstance(v, str) and len(v) > 800:
            return v[:797] + "..."
        return v

    def emit(self, section: str, op: str, key_or_anchor=None, before=None, after=None) -> None:
        rec = {
            "when": datetime.now(timezone.utc).isoformat(),
            "net_id": self.net_id,
            "source_run": self.source_run,
            "section": section,
            "op": op,
            "key_or_anchor": self._clip(key_or_anchor),
            "before": self._clip(before),
            "after": self._clip(after),
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

_SECTION_ENUM = list(SECTIONS)
_ABSENCE_ENUM = list(ABSENCE_SECTIONS)

APPEND_TOOL = {
    "type": "function", "name": "append_entry",
    "description": ("Append one new entry (a single JSON object matching the section's entry "
                    "schema) to a section. The write is validated, checked for keyed duplicates, "
                    "and its source is verified against what the tools actually observed; a "
                    "failing write returns the reason and nothing is changed."),
    "parameters": {"type": "object", "properties": {
        "section": {"type": "string", "enum": _SECTION_ENUM},
        "entry": {"type": "string", "description": "the new entry as a JSON object string"}},
        "required": ["section", "entry"], "additionalProperties": False},
}
PATCH_TOOL = {
    "type": "function", "name": "patch_entry",
    "description": ("Set named fields on ONE existing entry, addressed by the id shown at the "
                    "start of its line in the dossier. Only the fields you pass change; a key "
                    "inside the entry's nested `data` object may be passed directly. Same "
                    "validation and source check as append_entry."),
    "parameters": {"type": "object", "properties": {
        "section": {"type": "string", "enum": _SECTION_ENUM},
        "id": {"type": "string", "description": "the entry's id, e.g. F-3f9c2a71e0"},
        "fields": {"type": "string", "description": "the fields to set, as a JSON object string"}},
        "required": ["section", "id", "fields"], "additionalProperties": False},
}
ABSENT_TOOL = {
    "type": "function", "name": "confirm_absent",
    "description": ("Mark a section as confirmed absent (the scholar genuinely has none) with the "
                    "queries run and what they ruled out. Only valid while the section is empty; "
                    "appending to it later clears the flag automatically."),
    "parameters": {"type": "object", "properties": {
        "section": {"type": "string", "enum": _ABSENCE_ENUM},
        "reason": {"type": "string"}},
        "required": ["section", "reason"], "additionalProperties": False},
}
SKIP_TOOL = {
    "type": "function", "name": "skip_record",
    "description": ("Record that an offered publication record is not this scholar's work, with "
                    "the reason. Durable: the record stays settled for this and every later run, "
                    "so nothing re-raises it. Identify it by its doi, pmid, or exact title as "
                    "listed in the coverage check."),
    "parameters": {"type": "object", "properties": {
        "record": {"type": "string", "description": "the record's doi, pmid, or exact title"},
        "reason": {"type": "string"}},
        "required": ["record", "reason"], "additionalProperties": False},
}
ASSEMBLER_TOOLS = [APPEND_TOOL, PATCH_TOOL, ABSENT_TOOL, SKIP_TOOL]

@lru_cache(maxsize=1)
def _assemble_prompt() -> Template:
    raw = (_PROMPTS_DIR / "assemble_dossier.txt").read_text()
    partial = (_PROMPTS_DIR / "partials" / "source_preference.txt").read_text().strip()
    return Template(Template(raw).safe_substitute(source_preference=partial))

def _record_keys(r: dict) -> List[str]:
    """Normalized identity keys of one offered record (doi/pmid/openalex_id/title); dispositions
    store all of them so a later match hits on any."""
    return [_norm(r[k]) for k in ("doi", "pmid", "openalex_id", "title") if r.get(k)]

def _disposed_keys(dispositions: Optional[dict]) -> set:
    return {k for d in (dispositions or {}).get("skipped_records", []) for k in d.get("keys", [])}

def _missing_pubs(state: AssemblerState, expected: List[dict],
                  disposed: frozenset = frozenset()) -> List[dict]:
    """Expected publication records not yet in the dossier and not disposed of, matched by
    doi/pmid/openalex_id/normalized title.

    Publications are the one section where the transcript is structured, so completeness here
    is a mechanical diff rather than a judgment call.
    """
    have = set()
    for _i, e in state.pairs["publications"]:
        d = _payload(e)
        for k in ("doi", "pmid", "openalex_id"):
            if d.get(k):
                have.add(_norm(d[k]))
        if d.get("title"):
            have.add(_norm(d["title"]))
    out = []
    for r in expected or []:
        keys = _record_keys(r)
        if keys and not any(k in have or k in disposed for k in keys):
            out.append(r)
    return out

def _missing_doi_items(state: AssemblerState, expected: List[dict]) -> List[dict]:
    """Publications entries missing a doi whose offered record has one, matched by pmid or
    normalized title; fires only when the value is actually known."""
    by_key: Dict[str, dict] = {}
    for r in expected or []:
        for k in _record_keys(r):
            by_key.setdefault(k, r)
    items = []
    for eid, e in state.pairs["publications"]:
        d = _payload(e)
        if d.get("doi"):
            continue
        match = None
        for k in ("pmid", "title"):
            if d.get(k) and _norm(d[k]) in by_key:
                match = by_key[_norm(d[k])]
                break
        if match and match.get("doi"):
            items.append({"id": eid, "doi": match["doi"], "title": d.get("title") or ""})
    return items

def _run_filters(state: AssemblerState, expected_pubs: Optional[List[dict]],
                 dispositions: Optional[dict]) -> Dict[str, list]:
    """The deterministic filter registry, run when the model stops; each filter returns the
    items it wants fixed in this session. Empty everywhere means the session may end."""
    out: Dict[str, list] = {}
    if expected_pubs:
        disposed = frozenset(_disposed_keys(dispositions))
        missing = _missing_pubs(state, expected_pubs, disposed)
        if missing:
            out["missing_records"] = missing
        doi_items = _missing_doi_items(state, expected_pubs)
        if doi_items:
            out["missing_doi"] = doi_items
    return out

_NUDGE_CAP = 80

_NUDGE_PARTS = (_PROMPTS_DIR / "filters_nudge.txt").read_text().split("\n---\n")
_NUDGE_PREAMBLE = _NUDGE_PARTS[0].strip()
_NUDGE_MISSING = Template(_NUDGE_PARTS[1].strip())
_NUDGE_DOI = Template(_NUDGE_PARTS[2].strip())

def _filters_nudge(items: Dict[str, list]) -> str:
    parts = [_NUDGE_PREAMBLE]
    missing = items.get("missing_records") or []
    if missing:
        lines = "\n".join(
            json.dumps({k: r[k] for k in ("title", "year", "venue", "doi", "pmid") if r.get(k)},
                       ensure_ascii=False)
            for r in missing[:_NUDGE_CAP])
        parts.append(_NUDGE_MISSING.substitute(count=len(missing), lines=lines))
    doi_items = items.get("missing_doi") or []
    if doi_items:
        lines = "\n".join(f'{it["id"]} <- doi "{it["doi"]}"  ({it["title"][:60]})'
                          for it in doi_items[:_NUDGE_CAP])
        parts.append(_NUDGE_DOI.substitute(count=len(doi_items), lines=lines))
    return "\n\n".join(parts)

_MAX_NUDGE_CYCLES = 3

def run_assembler_session(dossier: Dossier, transcript: str, issues: Optional[List[dict]], *,
                          client, model: str, net_id: str, obs=None, cris=None,
                          reasoning_effort: str = "medium", changelog: Optional[ChangeLog] = None,
                          max_turns: int = 20, source_gate_attempts: int = 3,
                          expected_pubs: Optional[List[dict]] = None,
                          dispositions: Optional[dict] = None, run: Optional[int] = None,
                          label: str = "final",
                          on_usage=None, prompt_cache_key: Optional[str] = None):
    """One assembler session: folds this run's findings and issue list into the dossier via the
    gated entry ops, until the mechanical filters pass or the nudge cap is hit.

    `dispositions` is mutated in place; returns (dossier, trace_record); never raises.
    """
    from source_filter import check_entry

    gate: Gate = None
    if obs is not None:
        cris_set = cris if cris is not None else set()
        gate = lambda sec, e: check_entry(sec, e, obs, cris_set)

    state = AssemblerState(dossier)
    counts_before = state.counts()
    issues_txt = "\n".join(
        f"- [{i.get('type')}] ({i.get('section')}) {i.get('what')}  FIX: {i.get('fix')}"
        for i in (issues or [])
    ) or "(none)"
    prompt = _assemble_prompt().safe_substitute(
        transcript=transcript or "(no findings this run)",
        issues=issues_txt,
        dossier=state.render(),
    )

    dispositions = dispositions if dispositions is not None else {"skipped_records": []}
    tag = f"{label}" + (f" r{run}" if run else "")
    stats = Counter()
    events: list = []
    filter_log: list = []
    nudges: list = []
    usages: list = []
    end_reason = "turn_budget"
    extra = {"prompt_cache_key": prompt_cache_key} if prompt_cache_key else {}
    prev_id = None
    pending = None
    nudge_cycles = 0
    last_items_n = float("inf")
    last_ctx = len(prompt) // 4

    for turn_i in range(1, max_turns + 1):
        try:
            if prev_id is None:
                resp = create_response_with_backoff(
                    client=client, model=model, tools=ASSEMBLER_TOOLS, input=prompt,
                    reasoning={"effort": reasoning_effort, "summary": "auto"},
                    deadline_s=deadline_for(last_ctx), store=True, **extra)
            else:
                resp = create_response_with_backoff(
                    client=client, model=model, tools=ASSEMBLER_TOOLS, input=pending,
                    reasoning={"effort": reasoning_effort, "summary": "auto"},
                    deadline_s=deadline_for(last_ctx), store=True,
                    previous_response_id=prev_id, **extra)
        except Exception as e:
            log.warning(f"[{net_id}] assemble[{tag}]: call failed ({e}); ending session")
            stats["call_failed"] += 1
            end_reason = "call_failed"
            break
        usage = getattr(resp, "usage", None)
        if on_usage and usage:
            on_usage(usage)
        if usage is not None:
            usages.append(usage)
            last_ctx = int(getattr(usage, "input_tokens", 0) or 0) or last_ctx
        prev_id = resp.id
        fcs = [o for o in resp.output if getattr(o, "type", None) == "function_call"]
        if not fcs:

            items = _run_filters(state, expected_pubs, dispositions)
            filter_log.append({
                "turn": turn_i,
                "items": {k: [({f: r[f] for f in ("title", "doi", "pmid") if r.get(f)}
                               if k == "missing_records" else r) for r in v]
                          for k, v in items.items()},
            })
            n_items = sum(len(v) for v in items.values())
            if not n_items:
                end_reason = "clean"
                break
            if nudge_cycles >= _MAX_NUDGE_CYCLES:
                end_reason = "cycle_cap"
                break
            if n_items >= last_items_n:
                end_reason = "stalled"
                break
            nudge_cycles += 1
            last_items_n = n_items
            stats["completeness_nudge"] += n_items
            log.info(f"[{net_id}] assemble[{tag}]: filter nudge {nudge_cycles}, "
                     f"{ {k: len(v) for k, v in items.items()} }")
            msg = _filters_nudge(items)
            nudges.append(msg)
            pending = [{"role": "user", "content": msg}]
            continue
        outputs = []
        for fc in fcs:
            try:
                args = json.loads(fc.arguments) if fc.arguments else {}
            except (ValueError, TypeError):
                args = {}
            section = args.get("section", "")
            if fc.name == "append_entry":
                r = apply_append(state, section, args.get("entry", ""), gate=gate,
                                 max_attempts=source_gate_attempts, changelog=changelog)
            elif fc.name == "patch_entry":
                r = apply_patch(state, section, args.get("id", ""), args.get("fields", ""),
                                gate=gate, changelog=changelog)
            elif fc.name == "confirm_absent":
                r = apply_absent(state, section, args.get("reason", ""), changelog=changelog)
            elif fc.name == "skip_record":
                r = apply_skip(dispositions, expected_pubs or [], args.get("record", ""),
                               args.get("reason", ""), run=run, changelog=changelog)
            else:
                r = OpResult(False, fc.name, section, f"unknown tool: {fc.name}", error_kind="usage")
            stats[r.op if r.ok else f"reject_{r.error_kind or 'other'}"] += 1
            if r.ok and r.flagged:
                stats["flagged"] += 1
            if not r.ok and r.error_kind == "gate":
                stats[f"gate_{r.gate_verdict}"] += 1
            events.append({"turn": turn_i, "op": r.op, "section": r.section, "ok": r.ok,
                           "id": r.entry_id, "flagged": r.flagged,
                           "error": None if r.ok else r.message[:300]})
            outputs.append({"type": "function_call_output",
                            "call_id": getattr(fc, "call_id", None), "output": r.message})
        pending = outputs
    else:
        log.warning(f"[{net_id}] assemble[{tag}]: hit the {max_turns}-turn budget; ending session")
        stats["turn_budget_hit"] += 1

    died = bool(stats.get("call_failed") or stats.get("turn_budget_hit"))
    left_items = _run_filters(state, expected_pubs, dispositions)
    left = left_items.get("missing_records") or []
    auto_skipped_labels: list = []
    if left:
        if died:
            stats["pubs_missing_at_end"] = len(left)
            log.warning(f"[{net_id}] assemble[{tag}]: DIED with {len(left)} publication "
                        f"record(s) unrecorded; they seed the next run")
        else:
            for r in left:
                dispositions.setdefault("skipped_records", []).append({
                    "keys": _record_keys(r), "label": r.get("title") or "?",
                    "reason": "left unrecorded after the coverage nudge (no reason given)",
                    "run": run, "auto": True,
                })
                auto_skipped_labels.append((r.get("title") or "?")[:60])
                if changelog:
                    changelog.emit("publications", "skip_record", _record_keys(r)[0],
                                   after="(auto) left unrecorded after the coverage nudge")
            stats["auto_skipped"] = len(left)
            log.info(f"[{net_id}] assemble[{tag}]: {len(left)} offered record(s) auto-recorded "
                     f"as skipped: {auto_skipped_labels[:5]}"
                     + (" ..." if len(auto_skipped_labels) > 5 else ""))
    if left_items.get("missing_doi") and not died:
        log.info(f"[{net_id}] assemble[{tag}]: {len(left_items['missing_doi'])} publications "
                 f"still lack an offered doi after nudging (left to the critique)")

    counts_after = state.counts()
    n_in = sum(int(getattr(u, "input_tokens", 0) or 0) for u in usages)
    n_cached = sum(int(getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0)
                   for u in usages)
    n_out = sum(int(getattr(u, "output_tokens", 0) or 0) for u in usages)
    rejects = {k[len("reject_"):]: v for k, v in stats.items() if k.startswith("reject_")}
    log.info(
        f"[{net_id}] assemble[{tag}]: +{stats['append']} appended, {stats['patch']} patched, "
        f"{stats['absent']} absent, {stats['skip']} skipped ({stats['auto_skipped']} auto), "
        f"{sum(rejects.values())} rejected"
        f"{' ' + json.dumps(rejects) if rejects else ''}, {stats['flagged']} flagged; "
        f"{nudge_cycles} nudge cycle(s); end={end_reason}; "
        f"{len(usages)} turns, {n_in} in ({round(100 * n_cached / n_in) if n_in else 0}% cached) "
        f"/ {n_out} out"
    )
    rec = {"turn": "assemble", "label": label, "run": run, "end_reason": end_reason,
           "timestamp": datetime.now(timezone.utc).isoformat(),
           "stats": dict(stats), "counts_before": counts_before, "counts_after": counts_after,
           "nudge_cycles": nudge_cycles,
           "skipped_records": len((dispositions or {}).get("skipped_records", [])),
           "auto_skipped_labels": auto_skipped_labels,
           "filter_log": filter_log, "nudges": nudges,
           "ops": events,
           "token_usage": {"context": n_in, "generated": n_out, "total": n_in + n_out,
                           "cached": n_cached}}
    return state.dossier, rec
