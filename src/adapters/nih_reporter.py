"""NIH RePORTER adapter: a PI's funded projects via the public RePORTER API.

Pure-API adapter (httpx + stdlib only), same contract as the other pure adapters:
exposes ``fetch(...)`` and ``TOOL_SCHEMA`` and takes a ``cache=`` kwarg, so the loop
wires it through ``PURE_ADAPTERS``. Keyless; no auth.

Search by PI name and/or by grant number. A name search is fuzzy and misses grants the
scholar holds as a co-investigator (RePORTER returns them under the contact PI); a
grant-number search is exact, so pass every award number you have (from ORCID fundings,
CRIS records, publications) as its own thread.

Import convention (src/ is on sys.path at runtime):
    from adapters.nih_reporter import fetch, TOOL_SCHEMA
"""

import logging
import re
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger(__name__)

_API = "https://api.reporter.nih.gov/v2/projects/search"

_FIELDS = ["ApplId", "ProjectTitle", "ProjectNum", "ContactPiName", "PrincipalInvestigators",
           "Organization", "FiscalYear", "AwardAmount", "ProjectDetailUrl",
           "ProjectStartDate", "ProjectEndDate", "AgencyIcAdmin", "ActivityCode"]

TOOL_SCHEMA = {
    "type": "function",
    "name": "nih_reporter_projects_search",
    "description": (
        "Search NIH RePORTER for NIH-funded projects (title, project number, the full PI list, "
        "organization, fiscal year, award amount, start/end dates, agency, and the RePORTER detail "
        "URL). Search by pi_name AND/OR by project_num. A project_num search is exact and is the "
        "reliable way to find a grant the scholar holds as a co-investigator, so pass any award "
        "number you have (e.g. R01HL146615). RePORTER is the authoritative source for an NIH grant; "
        "prefer its ProjectDetailUrl over an aggregator or profile page."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pi_name": {"type": "string", "description": "A principal investigator's name."},
            "project_num": {"type": "string", "description": "A grant / core project number to match exactly, e.g. R01HL146615."},
            "org_name": {"type": "string", "description": "Optional organization to disambiguate a common name."},
            "limit": {"type": "integer", "description": "Max projects to return (default 25)."},
        },
    },
}

def _g(p: dict, *keys):
    for k in keys:
        if p.get(k) is not None:
            return p[k]
    return None

def _pi_list(p: dict) -> List[Dict[str, Any]]:
    """The project's full PI list, so the caller can see whether the scholar is the (contact) PI
    or one of several investigators."""
    out = []
    for pi in _g(p, "principal_investigators", "PrincipalInvestigators") or []:
        name = " ".join(x for x in [pi.get("first_name"), pi.get("middle_name"), pi.get("last_name")] if x).strip()
        out.append({"name": name or pi.get("full_name"), "is_contact_pi": bool(pi.get("is_contact_pi"))})
    return out

def _core_num(project_num: str) -> str:
    """Strip the application-type prefix and support-year suffix so a core number matches
    (e.g. '5R01HL146615-03' -> 'R01HL146615')."""
    m = re.search(r"([A-Z]\d{2}[A-Z]{2}\d{6})", (project_num or "").upper())
    return m.group(1) if m else (project_num or "")

def fetch(pi_name: Optional[str] = None, project_num: Optional[str] = None,
          org_name: Optional[str] = None, limit: int = 25, *, cache=None) -> Dict[str, Any]:
    """Return the shared envelope ``{source, query, count, results, meta, error}``; ``results`` are
    the projects. Search by pi_name and/or project_num (at least one). Never raises."""
    query = {"pi_name": pi_name, "project_num": project_num, "org_name": org_name, "limit": limit}
    if cache is not None:
        cached = cache.get_api("nih_reporter", pi_name=pi_name, project_num=project_num,
                               org_name=org_name, limit=limit)
        if cached is not None:
            return cached
    try:
        criteria: Dict[str, Any] = {}
        if project_num:
            criteria["project_nums"] = [_core_num(project_num), project_num]
        if pi_name:
            parts = [p for p in (pi_name or "").strip().split() if p]
            first = parts[0] if parts else ""
            last = parts[-1] if len(parts) > 1 else ""
            criteria["pi_names"] = [{"first_name": first, "last_name": last,
                                     "any_name": "" if (first or last) else pi_name}]
        if org_name:
            criteria["org_names"] = [org_name]
        if not criteria:
            return {"source": "nih_reporter", "query": query, "count": 0, "results": [],
                    "meta": {}, "error": "provide pi_name or project_num"}
        payload = {"criteria": criteria, "offset": 0, "limit": max(1, min(int(limit), 50)),
                   "include_fields": _FIELDS}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(_API, json=payload,
                               headers={"Accept": "application/json", "Content-Type": "application/json"})
        if resp.status_code >= 400:
            log.error(f"NIH RePORTER HTTP {resp.status_code}: {resp.text[:200]}")
            return {"source": "nih_reporter", "query": query, "count": 0,
                    "results": [], "meta": {}, "error": f"HTTP {resp.status_code}"}
        results = resp.json().get("results", []) or []
        normalized = []
        for p in results:
            appl_id = _g(p, "appl_id", "ApplId")
            detail = _g(p, "project_detail_url", "ProjectDetailUrl") or (
                f"https://reporter.nih.gov/project-details/{appl_id}" if appl_id else None)
            agency = _g(p, "agency_ic_admin", "AgencyIcAdmin") or {}
            normalized.append({
                "project_title": _g(p, "project_title", "ProjectTitle"),
                "project_num": _g(p, "project_num", "ProjectNum"),
                "contact_pi_name": _g(p, "contact_pi_name", "ContactPiName"),
                "principal_investigators": _pi_list(p),
                "organization_name": _g(p, "organization_name", "Organization"),
                "fiscal_year": _g(p, "fiscal_year", "FiscalYear"),
                "award_amount": _g(p, "award_amount", "AwardAmount"),
                "project_start_date": _g(p, "project_start_date", "ProjectStartDate"),
                "project_end_date": _g(p, "project_end_date", "ProjectEndDate"),
                "agency": (agency.get("abbreviation") or agency.get("name")) if isinstance(agency, dict) else agency,
                "activity_code": _g(p, "activity_code", "ActivityCode"),
                "project_detail_url": detail,
            })
        result = {"source": "nih_reporter", "query": query, "count": len(normalized),
                  "results": normalized, "meta": {}, "error": None}
        if cache is not None:
            cache.set_api("nih_reporter", result, pi_name=pi_name, project_num=project_num,
                          org_name=org_name, limit=limit)
        return result
    except Exception as e:
        log.error(f"NIH RePORTER error: {e}")
        return {"source": "nih_reporter", "query": query, "count": 0,
                "results": [], "meta": {}, "error": str(e)}
