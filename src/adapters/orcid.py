"""ORCID public-API adapter (tool: orcid_profile).

Pulls a scholar's employments, educations, works, and fundings from the ORCID
public API (no key). If given an ORCID iD, fetches /{id}/record directly; otherwise
resolves via expanded-search filtered by name (+ affiliation when given), picks the
best match, then reads the record. Pure-API: imports only httpx + stdlib.
"""

import re

import httpx

BASE = "https://pub.orcid.org/v3.0"
HEADERS = {"Accept": "application/json", "User-Agent": "scholar-dossier-agent/1.0"}
TIMEOUT = 60.0

ORCID_RE = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]")

def _is_orcid_id(value):
    """True if value contains a well-formed ORCID iD (16-digit, dash-separated)."""
    if not value:
        return False
    return bool(ORCID_RE.search(value.replace("https://orcid.org/", "")))

def _extract_orcid_id(value):
    """Pull the bare iD out of a raw iD or an orcid.org URL."""
    m = ORCID_RE.search(value)
    return m.group(0) if m else None

def _date_year(date_obj):
    """Pull an int year out of an ORCID fuzzy-date dict, or None."""
    if not date_obj:
        return None
    year = (date_obj.get("year") or {}).get("value")
    if year is None:
        return None
    try:
        return int(year)
    except (TypeError, ValueError):
        return None

def _org_name(org):
    """Organization display name from an ORCID organization dict."""
    if not org:
        return None
    return org.get("name")

def _value(field):
    """Unwrap an ORCID {value: ...} wrapper, or None."""
    if not field:
        return None
    return field.get("value")

def _affiliation_record(summary, kind):
    """Normalize an employment-summary / education-summary into a typed record."""
    org = summary.get("organization") or {}
    return {
        "kind": kind,
        "org": _org_name(org),
        "role": summary.get("role-title"),
        "title": summary.get("role-title"),
        "department": summary.get("department-name"),
        "start_year": _date_year(summary.get("start-date")),
        "end_year": _date_year(summary.get("end-date")),
        "url": _value(summary.get("url")),
    }

def _external_ids(summary):
    """Map each external-id type -> (value, url). ORCID stores a work's DOI/PMID/PMC and a
    funding's grant number here, with the aggregator's per-item URL, so never skip it."""
    out = {}
    for eid in ((summary.get("external-ids") or {}).get("external-id") or []):
        t = (eid.get("external-id-type") or "").lower()
        val = (eid.get("external-id-normalized") or {}).get("value") or eid.get("external-id-value")
        url = (eid.get("external-id-url") or {}).get("value")
        if t and t not in out:
            out[t] = (val, url)
    return out

def _work_record(summary):
    """Normalize a work-summary into a typed record (DOI, PMID, and PMC when ORCID has them)."""
    title = ((summary.get("title") or {}).get("title") or {}).get("value")
    journal = _value(summary.get("journal-title"))
    ext = _external_ids(summary)
    doi = ext.get("doi", (None, None))[0]
    pmid = ext.get("pmid", (None, None))[0]
    pmc = ext.get("pmc", (None, None))[0]
    url = _value(summary.get("url")) or ext.get("doi", (None, None))[1]
    if not url and doi:
        url = "https://doi.org/" + doi
    return {
        "kind": "work",
        "org": journal,
        "title": title,
        "role": summary.get("type"),
        "start_year": _date_year(summary.get("publication-date")),
        "end_year": None,
        "doi": doi,
        "pmid": pmid,
        "pmc": pmc,
        "url": url,
    }

def _funding_record(summary):
    """Normalize a funding-summary into a typed record. The grant number, the aggregator's
    per-grant URL (e.g. Dimensions), and the amount live in external-ids/amount; surface all three
    so the loop can look the grant up by number in NIH RePORTER and the writer can cite an
    authoritative page instead of the bare profile."""
    org = summary.get("organization") or {}
    title = ((summary.get("title") or {}).get("title") or {}).get("value")
    ext = _external_ids(summary)
    grant_number = ext_url = None
    for t in ("grant_number", "grant-number", "funded-grant"):
        if t in ext:
            grant_number, ext_url = ext[t]
            break
    amt = summary.get("amount") or {}
    amount = None
    if isinstance(amt, dict) and amt.get("value"):
        amount = " ".join(str(x) for x in (amt.get("value"), amt.get("currency-code")) if x)
    return {
        "kind": "funding",
        "org": _org_name(org),
        "title": title,
        "role": summary.get("type"),
        "start_year": _date_year(summary.get("start-date")),
        "end_year": _date_year(summary.get("end-date")),
        "grant_number": grant_number,
        "amount": amount,
        "url": _value(summary.get("url")) or ext_url,
    }

def _collect_affiliations(activities, section_key, summary_key, kind):
    """Walk activities-summary affiliation groups for employments / educations."""
    records = []
    section = activities.get(section_key) or {}
    for group in section.get("affiliation-group") or []:
        for entry in group.get("summaries") or []:
            summary = entry.get(summary_key)
            if summary:
                records.append(_affiliation_record(summary, kind))
    return records

def _collect_works(activities, limit):
    """Walk activities-summary work groups (one record per group, deduped by ORCID)."""
    records = []
    works = activities.get("works") or {}
    for group in works.get("group") or []:
        summaries = group.get("work-summary") or []
        if not summaries:
            continue
        records.append(_work_record(summaries[0]))
        if len(records) >= limit:
            break
    return records

def _collect_fundings(activities):
    """Walk activities-summary funding groups."""
    records = []
    fundings = activities.get("fundings") or {}
    for group in fundings.get("group") or []:
        for summary in group.get("funding-summary") or []:
            records.append(_funding_record(summary))
    return records

def _resolve_orcid_id(client, name, affiliation):
    """Resolve a name (+ affiliation) to a single best-match ORCID iD via expanded-search.

    Returns (orcid_id, match_note) or (None, reason-string).
    """
    parts = name.split()
    query_terms = []
    if len(parts) >= 2:
        given = parts[0]
        family = parts[-1]
        query_terms.append(f"given-names:{given} AND family-name:{family}")
    else:
        query_terms.append(f"family-name:{name}")
    q = query_terms[0]
    if affiliation:
        q = f"({q}) AND affiliation-org-name:{affiliation}"

    resp = client.get(
        f"{BASE}/expanded-search/",
        params={"q": q, "rows": 20},
        headers=HEADERS,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("expanded-result") or []

    if not results and affiliation:
        resp = client.get(
            f"{BASE}/expanded-search/",
            params={"q": query_terms[0], "rows": 20},
            headers=HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("expanded-result") or []

    if not results:
        return None, "no expanded-search match"

    if affiliation:
        aff_lower = affiliation.lower()
        for res in results:
            institutions = [i.lower() for i in (res.get("institution-name") or [])]
            if any(aff_lower in inst for inst in institutions):
                return res.get("orcid-id"), f"matched affiliation '{affiliation}'"

    best = results[0]
    note = f"top of {len(results)} expanded-search results"
    return best.get("orcid-id"), note

def fetch(name_or_orcid, affiliation=None, *, limit=100, cache=None):
    """Fetch an ORCID profile (employments, educations, works, fundings).

    Args:
        name_or_orcid: an ORCID iD ("0000-0002-7744-3518" or the orcid.org URL),
            or a scholar name to resolve via expanded-search.
        affiliation: optional institution to disambiguate a name resolution.
        limit: max works to return (employments/educations/fundings are unbounded).
        cache: optional ResponseCache; uses get_api/set_api keyed by query.

    Returns a JSON-serializable dict (never raises): source, query, count, results
    (typed list with a `kind` field), meta (orcid_id, name, match note), error.
    """
    query = {"name_or_orcid": name_or_orcid, "affiliation": affiliation, "limit": limit}
    result = {
        "source": "orcid",
        "query": query,
        "count": 0,
        "results": [],
        "meta": {},
        "error": None,
    }

    if cache is not None:
        cached = cache.get_api("orcid", **query)
        if cached is not None:
            return cached

    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
            match_note = "given iD"
            if _is_orcid_id(name_or_orcid):
                orcid_id = _extract_orcid_id(name_or_orcid)
            else:
                orcid_id, match_note = _resolve_orcid_id(
                    client, name_or_orcid, affiliation
                )
                if not orcid_id:
                    result["error"] = f"could not resolve ORCID iD ({match_note})"
                    return result

            resp = client.get(f"{BASE}/{orcid_id}/record", headers=HEADERS)
            resp.raise_for_status()
            record = resp.json()

            person = record.get("person") or {}
            name = person.get("name") or {}
            display_name = " ".join(
                p
                for p in [
                    _value(name.get("given-names")),
                    _value(name.get("family-name")),
                ]
                if p
            ) or None

            activities = record.get("activities-summary") or {}
            records = []
            records += _collect_affiliations(
                activities, "employments", "employment-summary", "employment"
            )
            records += _collect_affiliations(
                activities, "educations", "education-summary", "education"
            )
            records += _collect_works(activities, limit)
            records += _collect_fundings(activities)

            counts = {}
            for r in records:
                counts[r["kind"]] = counts.get(r["kind"], 0) + 1

            result["results"] = records
            result["count"] = len(records)
            result["meta"] = {
                "orcid_id": orcid_id,
                "orcid_url": f"https://orcid.org/{orcid_id}",
                "name": display_name,
                "match": match_note,
                "counts_by_kind": counts,
            }
    except httpx.HTTPStatusError as exc:
        result["error"] = f"HTTP {exc.response.status_code} from ORCID"
    except httpx.HTTPError as exc:
        result["error"] = f"network error: {exc}"
    except Exception as exc:
        result["error"] = f"unexpected error: {exc}"

    if cache is not None and result["error"] is None:
        cache.set_api("orcid", result, **query)
    return result

TOOL_SCHEMA = {
    "type": "function",
    "name": "orcid_profile",
    "description": (
        "Fetch a scholar's ORCID profile (employments, educations, publications, and "
        "fundings) from the ORCID public API. Pass an ORCID iD when known (e.g. "
        "'0000-0002-7744-3518') for a direct lookup; otherwise pass the scholar's full "
        "name plus their affiliation to disambiguate. Results are already structured "
        "(no extraction needed): each item has a 'kind' field (employment, education, "
        "work, funding) with org, role/title, start_year, end_year, and url. Use this "
        "for authoritative career-timeline and identity data, and to obtain a verified "
        "ORCID iD for the scholar."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name_or_orcid": {
                "type": "string",
                "description": (
                    "An ORCID iD (or orcid.org URL) for a direct lookup, or the "
                    "scholar's full name to resolve via search."
                ),
            },
            "affiliation": {
                "type": "string",
                "description": (
                    "Institution name to disambiguate when resolving by name "
                    "(e.g. 'University of Illinois Chicago'). Ignored when an iD is given."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max publications (works) to return; omit for the default of 100. Employments / educations / fundings are always complete.",
            },
        },
        "required": ["name_or_orcid"],
        "additionalProperties": False,
    },
}
