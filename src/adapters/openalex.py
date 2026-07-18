"""OpenAlex adapter: scholar publication list + abstracts.

Resolves an author by name (preferring an affiliation match) via /authors?search,
then pages /works?filter=author.id:<id>. Abstracts arrive as an inverted index and
are reconstructed to plain text. Pure-API: httpx + stdlib only; no intra-package
imports so it runs standalone.
"""

import os
import logging
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"
def contact_email() -> str:
    """The address OpenAlex's polite pool should use to reach whoever is running this.

    Required, with no default. A shipped default would make every deployment's traffic
    identify as one caller and route OpenAlex's contact to a stranger.
    """
    email = (os.environ.get("CONTACT_EMAIL") or "").strip()
    if not email:
        raise RuntimeError(
            "CONTACT_EMAIL is not set. OpenAlex's polite pool asks callers for a mailto so "
            "they can reach you about usage. Set CONTACT_EMAIL in .env to an address you "
            "monitor."
        )
    return email
PER_PAGE = 200

def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Rebuild plain-text abstract from an abstract_inverted_index.

    The index maps each word to the list of positions where it appears. We place
    every word at its positions, then join in position order.
    """
    if not inverted_index:
        return None
    positions: Dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return None
    ordered = [positions[i] for i in sorted(positions)]
    return " ".join(ordered)

def _short_id(openalex_id: Optional[str]) -> Optional[str]:
    """Turn a full OpenAlex URL id into its short form (e.g. A5012345678)."""
    if not openalex_id:
        return None
    return openalex_id.rstrip("/").split("/")[-1]

def _name_tokens(name: str) -> List[str]:
    """Lowercase alphabetic tokens of length >= 2 (drops middle initials / punctuation)."""
    cleaned = "".join(ch if ch.isalpha() or ch.isspace() else " " for ch in name.lower())
    return [t for t in cleaned.split() if len(t) >= 2]

def _name_matches(author: Dict[str, Any], wanted: List[str]) -> bool:
    """True if the author's PRIMARY display name covers every query token.

    OpenAlex /authors?search matches associated content, and display_name_alternatives
    is polluted with co-authors' names (an unrelated author can carry the queried name
    as an alternative). So gate on display_name only: the real author's canonical
    name contains the queried tokens, decoys do not.
    """
    if not wanted:
        return False
    have = set(_name_tokens(author.get("display_name") or ""))
    return all(tok in have for tok in wanted)

def _affiliation_score(author: Dict[str, Any], affiliation: str) -> int:
    """Count case-insensitive substring hits of affiliation in an author's institutions."""
    target = affiliation.lower()
    hits = 0
    last = author.get("last_known_institution") or {}
    if last and target in (last.get("display_name") or "").lower():
        hits += 2
    for inst in author.get("last_known_institutions") or []:
        if target in (inst.get("display_name") or "").lower():
            hits += 2

    for aff in author.get("affiliations") or []:
        inst = aff.get("institution") or {}
        if target in (inst.get("display_name") or "").lower():
            hits += 1
    return hits

def _resolve_author(
    client: httpx.Client, author_name: str, affiliation: Optional[str]
) -> Dict[str, Any]:
    """Pick the best matching author. Returns {"author": <raw>, "why": str} or {"error": ...}."""
    params = {
        "search": author_name,
        "per-page": 25,
        "mailto": contact_email(),
    }
    resp = client.get(f"{BASE_URL}/authors", params=params)
    if resp.status_code >= 400:
        return {"error": f"author search HTTP {resp.status_code}: {resp.text[:200]}"}
    all_candidates = resp.json().get("results", []) or []
    if not all_candidates:
        return {"error": f"no OpenAlex author found for '{author_name}'"}

    wanted = _name_tokens(author_name)
    named = [c for c in all_candidates if _name_matches(c, wanted)]
    candidates = named if named else all_candidates
    name_note = "" if named else " (no exact name match; using best of all returned)"

    if affiliation:
        scored = [(c, _affiliation_score(c, affiliation)) for c in candidates]
        with_aff = [(c, s) for (c, s) in scored if s > 0]
        if with_aff:
            best = max(with_aff, key=lambda cs: (cs[1], cs[0].get("works_count", 0)))[0]
            why = (
                f"name matched '{author_name}'; affiliation '{affiliation}' matched the "
                f"author's known institutions; chosen over {len(candidates) - 1} other "
                f"name match(es) by affiliation, then works_count="
                f"{best.get('works_count', 0)}{name_note}"
            )
            return {"author": best, "why": why}

    best = max(candidates, key=lambda c: c.get("works_count", 0))
    why = (
        f"name matched '{author_name}'; no affiliation hit, so chose the most-published "
        f"of {len(candidates)} name match(es), works_count="
        f"{best.get('works_count', 0)}{name_note}"
    )
    return {"author": best, "why": why}

def _normalize_work(work: Dict[str, Any]) -> Dict[str, Any]:
    """Map an OpenAlex work to the contract's normalized record."""
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    venue = source.get("display_name")

    topics = [
        t.get("display_name")
        for t in (work.get("topics") or [])
        if t.get("display_name")
    ]

    authors = [a.get("author", {}).get("display_name")
               for a in (work.get("authorships") or []) if a.get("author")]
    grants = [{"funder": g.get("funder_display_name"), "award_id": g.get("award_id")}
              for g in (work.get("grants") or []) if g.get("funder_display_name") or g.get("award_id")]
    ids = work.get("ids") or {}
    oa = work.get("open_access") or {}
    return {
        "title": work.get("title"),
        "authors": authors,
        "year": work.get("publication_year"),
        "doi": work.get("doi"),
        "pmid": ids.get("pmid"),
        "pmc": ids.get("pmcid"),
        "openalex_url": work.get("id"),
        "venue": venue,
        "type": work.get("type"),
        "cited_by_count": work.get("cited_by_count"),
        "grants": grants,
        "oa_url": oa.get("oa_url"),
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "topics": topics,
    }

def fetch(
    author_name: str,
    affiliation: Optional[str] = None,
    *,
    limit: int = 100,
    cache=None,
) -> Dict[str, Any]:
    """Resolve an author and return their works with reconstructed abstracts.

    Args:
        author_name: scholar's full name.
        affiliation: optional institution to prefer when disambiguating, e.g.
            "University of Illinois Chicago".
        limit: max works to return.
        cache: optional ResponseCache; uses get_api/set_api keyed on the query.

    Returns a dict with source/query/count/results/meta/error (never raises).
    """
    query = {"author_name": author_name, "affiliation": affiliation, "limit": limit}

    if cache is not None:
        cached = cache.get_api("openalex", **query)
        if cached is not None:
            return cached

    result: Dict[str, Any] = {
        "source": "openalex",
        "query": query,
        "count": 0,
        "results": [],
        "meta": {},
        "error": None,
    }

    try:
        with httpx.Client(
            timeout=30.0, headers={"Accept": "application/json"}, follow_redirects=True
        ) as client:
            resolved = _resolve_author(client, author_name, affiliation)
            if "error" in resolved:
                result["error"] = resolved["error"]
                return result

            author = resolved["author"]
            author_id = _short_id(author.get("id"))
            last_inst = author.get("last_known_institution") or {}
            if not last_inst:
                insts = author.get("last_known_institutions") or []
                last_inst = insts[0] if insts else {}

            result["meta"] = {
                "author": {
                    "openalex_id": author_id,
                    "display_name": author.get("display_name"),
                    "orcid": author.get("orcid"),
                    "last_known_institution": last_inst.get("display_name"),
                    "works_count": author.get("works_count"),
                },
                "match_reason": resolved["why"],
            }

            if not author_id:
                result["error"] = "resolved author has no OpenAlex id"
                return result

            works: List[Dict[str, Any]] = []
            cursor = "*"
            while len(works) < limit:
                params = {
                    "filter": f"author.id:{author_id}",
                    "per-page": min(PER_PAGE, limit - len(works)),
                    "cursor": cursor,
                    "sort": "cited_by_count:desc",
                    "mailto": contact_email(),
                }
                wresp = client.get(f"{BASE_URL}/works", params=params)
                if wresp.status_code >= 400:
                    result["error"] = (
                        f"works HTTP {wresp.status_code}: {wresp.text[:200]}"
                    )
                    break
                wdata = wresp.json()
                page = wdata.get("results", []) or []
                works.extend(page)
                cursor = (wdata.get("meta") or {}).get("next_cursor")
                if not page or not cursor:
                    break

            works = works[:limit]
            result["results"] = [_normalize_work(w) for w in works]
            result["count"] = len(result["results"])

    except Exception as e:
        log.error(f"OpenAlex adapter error: {e}")
        result["error"] = str(e)
        return result

    if cache is not None and result["error"] is None:
        cache.set_api("openalex", result, **query)

    return result

TOOL_SCHEMA = {
    "type": "function",
    "name": "openalex_author_works",
    "description": (
        "Look up a scholar in OpenAlex and return their publication list with "
        "reconstructed abstracts, venues, years, DOIs, citation counts, and topics. "
        "Use this to build the scholar's full body of work and as raw material for "
        "impact evidence. Provide the affiliation to disambiguate common names. "
        "Results are already structured; do NOT call extract() on them."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "author_name": {
                "type": "string",
                "description": "Scholar full name.",
            },
            "affiliation": {
                "type": "string",
                "description": (
                    "Optional institution to prefer when disambiguating, e.g. "
                    "'University of Illinois Chicago'."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max works to return; omit for the default of 100.",
            },
        },
        "required": ["author_name"],
    },
}
