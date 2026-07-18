"""ClinicalTrials.gov v2 adapter (tool: clinicaltrials_by_investigator).

Pure-API: httpx + stdlib only. Queries the ClinicalTrials.gov API v2
(/api/v2/studies) for studies where a person is listed as an overall
official (PI / sub-investigator / study director), and parses
protocolSection into compact normalized records.
"""

from typing import Optional

import httpx

API_URL = "https://clinicaltrials.gov/api/v2/studies"

def _structs_date(struct: Optional[dict]) -> Optional[str]:
    """Pull the 'date' string out of a *DateStruct block (may be YYYY-MM or YYYY-MM-DD)."""
    if not struct:
        return None
    return struct.get("date")

def _norm_name(name: str) -> str:
    """Lowercase, strip credential suffixes (MD, PhD, ...) and punctuation for matching."""
    n = name.lower()
    for sep in (",", ";", "("):
        if sep in n:
            n = n.split(sep)[0]
    return " ".join(n.split())

def _name_core_tokens(name: str) -> list:
    """First + last name tokens for a ctgov investigator query, dropping initials and
    credential suffixes ('Jane Smith' / 'Smith, Jane' -> ['Jane', 'Smith'];
    'Smith J' -> ['Smith'])."""
    raw = (name or "").strip()
    if "," in raw:
        a, b = (raw.split(",", 1) + [""])[:2]
        raw = f"{b.strip()} {a.strip()}"
    cred = {"md", "phd", "mph", "ms", "msc", "do", "rn", "pharmd", "mba", "dds", "dnp"}
    toks = [t.strip(".") for t in raw.replace(",", " ").split()]
    toks = [t for t in toks if t.isalpha() and len(t) >= 2 and t.lower() not in cred]
    return [toks[0], toks[-1]] if len(toks) >= 2 else toks

def _match_role(officials: list, investigator_name: str) -> Optional[str]:
    """Return the role of the official whose name matches the investigator, else None.

    Matching is loose: the official record often appends a credential
    ("Jane Smith, PharmD"), so we compare on a normalized core name and
    fall back to a last-name / token overlap check.
    """
    target = _norm_name(investigator_name)
    target_tokens = set(target.split())
    for off in officials:
        name = off.get("name") or ""
        core = _norm_name(name)
        if not core:
            continue
        if core == target:
            return off.get("role")

        core_tokens = set(core.split())
        if target_tokens and core_tokens:
            if target_tokens <= core_tokens or core_tokens <= target_tokens:
                return off.get("role")
    return None

def _normalize_study(study: dict, investigator_name: str) -> dict:
    """Turn one API study object into the contracted normalized record."""
    ps = study.get("protocolSection", {}) or {}

    idm = ps.get("identificationModule", {}) or {}
    nct_id = idm.get("nctId")
    title = idm.get("briefTitle") or idm.get("officialTitle")

    status_mod = ps.get("statusModule", {}) or {}
    overall_status = status_mod.get("overallStatus")
    start_date = _structs_date(status_mod.get("startDateStruct"))

    design = ps.get("designModule", {}) or {}
    phases = design.get("phases") or []
    phase = ", ".join(phases) if phases else None

    conditions = (ps.get("conditionsModule", {}) or {}).get("conditions") or []

    arms = ps.get("armsInterventionsModule", {}) or {}
    interventions = [
        i.get("name")
        for i in (arms.get("interventions") or [])
        if i.get("name")
    ]

    completion_date = _structs_date(status_mod.get("completionDateStruct")
                                    or status_mod.get("primaryCompletionDateStruct"))
    study_type = design.get("studyType")
    enrollment = (design.get("enrollmentInfo") or {}).get("count")

    sponsor_mod = ps.get("sponsorCollaboratorsModule", {}) or {}
    lead_sponsor = (sponsor_mod.get("leadSponsor") or {}).get("name")
    collaborators = [c.get("name") for c in (sponsor_mod.get("collaborators") or []) if c.get("name")]

    officials = (ps.get("contactsLocationsModule", {}) or {}).get(
        "overallOfficials"
    ) or []
    role = _match_role(officials, investigator_name)

    ctgov_url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else None

    return {
        "nct_id": nct_id,
        "title": title,
        "overall_status": overall_status,
        "phase": phase,
        "study_type": study_type,
        "enrollment": enrollment,
        "conditions": conditions,
        "interventions": interventions,
        "lead_sponsor": lead_sponsor,
        "collaborators": collaborators,
        "role": role,
        "start_date": start_date,
        "completion_date": completion_date,
        "ctgov_url": ctgov_url,
    }

def fetch(
    investigator_name: str,
    lead_sponsor: Optional[str] = None,
    *,
    limit: int = 100,
    cache=None,
) -> dict:
    """Find ClinicalTrials.gov studies for an investigator.

    Queries the AREA[OverallOfficialName] field (precise: matches people
    listed as an overall official, not full-text), optionally narrowed by
    lead sponsor. Returns a JSON-serializable dict; never raises.
    """
    query = {
        "investigator_name": investigator_name,
        "lead_sponsor": lead_sponsor,
        "limit": limit,
    }

    if cache is not None:
        cached = cache.get_api("clinicaltrials", **query)
        if cached is not None:
            return cached

    def _result(count, results, meta, error=None):
        out = {
            "source": "clinicaltrials",
            "query": query,
            "count": count,
            "results": results,
            "meta": meta,
            "error": error,
        }
        if cache is not None and error is None:
            cache.set_api("clinicaltrials", out, **query)
        return out

    page_size = max(1, min(int(limit), 100))

    core = _name_core_tokens(investigator_name)
    if len(core) >= 2:
        inner = f"{core[0]} AND {core[1]}"
    elif core:
        inner = core[0]
    else:
        inner = investigator_name
    term = (f"AREA[OverallOfficialName]({inner}) "
            f"OR AREA[ResponsiblePartyInvestigatorFullName]({inner})")
    params = {
        "query.term": term,
        "countTotal": "true",
        "pageSize": page_size,
    }
    if lead_sponsor:
        params["query.lead"] = lead_sponsor

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                API_URL,
                params=params,
                headers={"Accept": "application/json"},
            )
        if resp.status_code >= 400:
            return _result(
                0, [], {}, error=f"HTTP {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
    except Exception as e:
        return _result(0, [], {}, error=str(e))

    studies = data.get("studies", []) or []
    total = data.get("totalCount")

    results = []
    for study in studies[:limit]:
        try:
            results.append(_normalize_study(study, investigator_name))
        except Exception as e:

            results.append({"nct_id": None, "error": str(e)})

    meta = {
        "investigator_name": investigator_name,
        "lead_sponsor": lead_sponsor,
        "total_matches": total,
        "query_field": "AREA[OverallOfficialName]",
    }
    return _result(len(results), results, meta)

TOOL_SCHEMA = {
    "type": "function",
    "name": "clinicaltrials_by_investigator",
    "description": (
        "Search ClinicalTrials.gov (official registry) for clinical trials where a "
        "scholar is listed as an overall official (principal investigator, "
        "sub-investigator, or study director). Use this to find clinical-trial impact "
        "evidence for a clinician-researcher. Optionally narrow by lead sponsor "
        "(e.g. the scholar's institution). Results are already structured (nct_id, "
        "title, overall_status, phase, conditions, interventions, lead_sponsor, role, "
        "start_date, ctgov_url) - no extraction needed. An empty result means the "
        "person is not registered as a trial official, which is common and not an error."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "investigator_name": {
                "type": "string",
                "description": "Full name of the investigator.",
            },
            "lead_sponsor": {
                "type": "string",
                "description": "Optional lead-sponsor name to narrow results, e.g. 'University of Illinois'.",
            },
            "limit": {
                "type": "integer",
                "description": "Max studies to return (max 100); omit for the default of 100.",
            },
        },
        "required": ["investigator_name"],
    },
}
