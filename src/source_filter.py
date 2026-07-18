"""Deterministic check of whether an entry's source is present, well-formed, and grounded
(empty / malformed / ungrounded).

Grounding checks only tool-result evidence and the CRIS export, never the model's own text.
"""

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>)?#]+", re.I)
PMID_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.I)
PMC_RE = re.compile(r"(PMC\d+)", re.I)
NCT_RE = re.compile(r"(NCT\d{8})", re.I)
API_TOOLS = {"openalex_author_works", "pubmed_author_articles", "orcid_profile",
             "clinicaltrials_by_investigator", "nih_reporter_projects_search"}
CRIS_REF = re.compile(r"CRIS proposal #(\S+)", re.I)
REPORTER_REF = re.compile(r"NIH RePORTER project (\S+)", re.I)

def normalize_url(url: str) -> str:
    """Normalize URL for matching: lowercase, strip, remove trailing slash, http/https, www, sort query params."""
    from urllib.parse import parse_qsl, urlencode, urlparse
    url = url.strip().lower()
    url = url.rstrip("/")
    url = re.sub(r"^https://", "http://", url)
    url = re.sub(r"^http://www\.", "http://", url)
    parsed = urlparse(url)
    if parsed.query:
        sorted_query = urlencode(sorted(parse_qsl(parsed.query)))
        url = parsed._replace(query=sorted_query).geturl()
    return url

def url_core(url):
    """Normalized netloc+path (no scheme/www/query), the stable token to match a URL in the trace."""
    if not url:
        return ""
    n = re.sub(r"^https?://(www\.)?", "", normalize_url(url).lower())
    return n.split("?")[0].rstrip("/")

def url_ids(url):
    """Identifier tokens a URL can be grounded by (DOI/PMID/PMC/NCT)."""
    ids = set()
    for m in DOI_RE.finditer(url):
        ids.add(m.group(0).rstrip("/.").lower())
    for m in PMID_RE.finditer(url):
        ids.add(m.group(1))
    for m in PMC_RE.finditer(url):
        ids.add(m.group(1).lower())
    for m in NCT_RE.finditer(url):
        ids.add(m.group(1).lower())
    return ids

def find_data(rel) -> Path:
    """Locate a repo data file by walking up, since data/ sits at a different depth in this
    repo than in the released snapshot."""
    rel = Path(rel)
    for base in Path(__file__).resolve().parents:
        if (base / rel).is_file():
            return base / rel
    return rel

def load_cris(path=None) -> set:
    """Load the CRIS proposal-number set; warns (not raises) since a missing export is a
    legitimate state but must not fail silently."""
    import csv
    path = Path(path) if path else find_data("data/grant_data.csv")
    props = set()
    if not path.is_file():
        log.warning(f"CRIS export not found at {path}; CRIS-sourced grants will read as ungrounded")
        return props
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 3:
        return props
    pi = {h.strip(): i for i, h in enumerate(rows[1])}.get("Proposal Number")
    if pi is None:
        return props
    for r in rows[2:]:
        if pi < len(r) and r[pi].strip():
            props.add(r[pi].strip())
    return props

def extract_ok(tc) -> bool:
    """A structured_extract that returned page content (log tail 'extracted' + non-empty distilled)."""
    entry = str(tc.get("log") or "")
    tail = entry.split("->")[-1].strip() if "->" in entry else ""
    rr = tc.get("result_raw")
    return tail.startswith("extracted") and isinstance(rr, dict) and bool(rr.get("distilled"))

URL_RE = re.compile(r"https?://[^\s\"'<>\\)\]}]+", re.I)

NIH_PROJECT_RE = re.compile(r"\b\d?[A-Z]\d{2}[A-Z]{2}\d{6}\b")

class Observations:
    """Grounding evidence accumulated from tool RESULTS only, never from arguments, model
    text, or failed calls.

    Cross-run: export_keys/load_keys persist a compact keyset so evidence observed in an
    earlier run still grounds a later one.
    """

    def __init__(self):
        self.extracted_ok = set()
        self.extract_failed = set()
        self.seeded_keys = set()
        self._parts = []
        self._blob = None

    def add_tool_call(self, tc) -> None:
        """Record one tool call's RESULT. Shape matches a trace's tool_call dict."""
        fn = tc.get("function")
        if fn == "structured_extract":
            core = url_core((tc.get("arguments") or {}).get("url"))
            if extract_ok(tc):
                self.extracted_ok.add(core)
                self._parts.append(json.dumps(tc.get("result_raw")).lower())
            elif core:
                self.extract_failed.add(core)
        elif fn in API_TOOLS or fn == "search":
            self._parts.append(json.dumps([tc.get("result_raw"), tc.get("result_distilled")]).lower())
        self._blob = None

    @property
    def blob(self) -> str:
        if self._blob is None:
            self._blob = "".join(self._parts)
        return self._blob

    def as_tuple(self):
        return self.extracted_ok, self.extract_failed, self.blob

    def _blob_keys(self) -> set:
        """The blob distilled to its groundable tokens: every URL's core, plus every DOI / PMID / PMC / NCT / NIH-project identifier."""
        blob = self.blob
        keys = {url_core(m.group(0)) for m in URL_RE.finditer(blob)}
        for m in DOI_RE.finditer(blob):
            keys.add(m.group(0).rstrip("/.").lower())
        for m in PMC_RE.finditer(blob):
            keys.add(m.group(1).lower())
        for m in NCT_RE.finditer(blob):
            keys.add(m.group(1).lower())
        for m in NIH_PROJECT_RE.finditer(blob.upper()):
            keys.add(m.group(0).lower())
        keys.discard("")
        return keys

    def export_keys(self) -> dict:
        """Compact, JSON-serializable grounding keyset: this run's distilled blob tokens plus whatever was seeded from prior runs (cumulative)."""
        return {
            "extracted_ok": sorted(self.extracted_ok),
            "extract_failed": sorted(self.extract_failed),
            "keys": sorted(self._blob_keys() | self.seeded_keys),
        }

    def load_keys(self, d: dict) -> None:
        """Seed prior-run grounding evidence from an export_keys dict (missing/empty is fine)."""
        if not isinstance(d, dict):
            return
        self.extracted_ok |= set(d.get("extracted_ok") or [])
        self.extract_failed |= set(d.get("extract_failed") or [])
        self.seeded_keys |= {str(k).lower() for k in (d.get("keys") or [])}

def source_verdict(src, obs, cris):
    """Verdict (ok | empty | malformed | ungrounded) for a source field, plus broke_on_open.

    obs may be an Observations or the legacy 3-tuple; broke_on_open flags a grounded URL that
    failed to open during the run (a softer signal, not a failure).
    """
    if isinstance(obs, Observations):
        extracted_ok, extract_failed, blob = obs.as_tuple()
        seeded = obs.seeded_keys
    else:
        extracted_ok, extract_failed, blob = obs
        seeded = set()
    s = (src or "").strip()
    if not s:
        return "empty", False
    m = CRIS_REF.match(s)
    if m:
        return ("ok" if m.group(1) in cris else "ungrounded"), False
    m = REPORTER_REF.match(s)
    if m:
        p = m.group(1).lower()
        return ("ok" if (p in blob or p in seeded) else "ungrounded"), False
    if not s.lower().startswith(("http://", "https://")):
        return "malformed", False
    host = re.sub(r"^https?://", "", s).split("/")[0]
    core = url_core(s)
    if not core or "." not in host or " " in s:
        return "malformed", False
    tokens = [core] + list(url_ids(s))
    grounded = (core in extracted_ok) or any(t and (t in blob or t in seeded) for t in tokens)
    if not grounded:
        return "ungrounded", False
    return "ok", (core in extract_failed)

IMPACT_SECTIONS = ("clinical_impacts", "community_impacts", "economic_impacts", "policy_impacts")

def entry_source(section: str, entry: dict) -> str:
    """The clickable verification source for one entry; section-specific, and not simply the
    `source` field.

    A media_mentions entry has both `url` (link) and `source` (outlet name); using `source`
    uniformly would mark every media row malformed.
    """
    data = entry.get("data") if isinstance(entry.get("data"), dict) else entry
    if section == "grants":
        return (data.get("nih_project_detail_url") or data.get("url")
                or (f"NIH RePORTER project {data['nih_project_num']}" if data.get("nih_project_num") else None)
                or (f"CRIS proposal #{data['proposal_number']}" if data.get("proposal_number") else "")) or ""
    return data.get("url") or ""

def check_entry(section: str, entry: dict, obs, cris) -> tuple:
    """Per-write source gate: (verdict, source) for one entry at the moment it is written.

    `section` may be a sub-list name ("key_profiles.orcid"); publications are never
    source-gated (always pass).
    """
    base = section.split(".")[0]
    if base == "publications":
        return "ok", ""
    src = entry_source(base, entry)
    verdict, _broke = source_verdict(src, obs, cris)
    return verdict, src

def iter_sources(d: dict):
    """Yield (section, label, source) for every entry that carries a verifiable source.

    Shared by the in-loop check and the retrospective scorer, so both agree on what a source is.
    """
    for ptype, v in (d.get("key_profiles") or {}).items():
        items = v if isinstance(v, list) else ([v] if v else [])
        for it in items:
            if isinstance(it, dict):
                yield "key_profiles", ptype.replace("_", " "), entry_source("key_profiles", it)
    for c in d.get("career_trajectory") or []:
        data = c.get("data") or {}
        yield "career_trajectory", str(data.get("title") or "?"), entry_source("career_trajectory", c)
    for sec in ("grants", "media_mentions", *IMPACT_SECTIONS):
        v = d.get(sec)
        items = (v or {}).get("items") if isinstance(v, dict) else (v or [])
        for it in items or []:
            if not isinstance(it, dict):
                continue
            data = it.get("data") if isinstance(it.get("data"), dict) else it
            label = data.get("title") or data.get("name") or data.get("summary") or it.get("summary") or "?"
            yield sec, str(label), entry_source(sec, it)
