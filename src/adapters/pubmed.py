"""PubMed adapter for the scholar dossier agent.

Replaces flaky PubMed/PMC scraping with the official NCBI E-utilities API:
esearch (find PMIDs for an author) then efetch (pull abstracts + metadata as XML).
Pure-API: imports only httpx + stdlib. Exposes fetch(...) and TOOL_SCHEMA.
"""

import os
import re
import xml.etree.ElementTree as ET

import httpx

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

DEFAULT_TOOL = "scholar-dossier-agent"

def contact_email() -> str:
    """The address NCBI should use to reach whoever is running this.

    Required, with no default. A shipped default would make every deployment's traffic
    identify as one caller and route NCBI's rate-limit contact to a stranger.
    """
    email = (os.environ.get("CONTACT_EMAIL") or "").strip()
    if not email:
        raise RuntimeError(
            "CONTACT_EMAIL is not set. NCBI E-utilities asks callers to identify themselves "
            "so they can reach you about usage. Set CONTACT_EMAIL in .env to an address you "
            "monitor."
        )
    return email

def _author_term(author_name):
    """Build a PubMed [Author] term from a free-form display name.

    PubMed expects 'Lastname Initials' (e.g. 'Smith JA'). Two input forms occur:
      - 'Jane Smith' (forename first) -> reformat to 'Smith J'
      - 'Smith J' / 'Smith JA' (already surname + initials) -> pass through unchanged
    The earlier version reformatted blindly, turning 'Smith J' into 'J S' (it took the
    initial as the surname) and returning zero results. Returns the raw author string
    (caller adds the [Author] tag).
    """
    name = (author_name or "").strip()
    if not name:
        return ""

    if "[" in name:
        name = name.split("[", 1)[0].strip()
    parts = name.split()
    if len(parts) == 1:
        return parts[0]

    last_tok = parts[-1].replace(".", "")
    if last_tok.isalpha() and (len(last_tok) == 1 or (len(last_tok) <= 3 and last_tok.isupper())):
        return name

    last = parts[-1]
    initials = "".join(
        sub[0]
        for part in parts[:-1]
        for sub in re.split(r"[-\s]", part)
        if sub
    )
    return f"{last} {initials}"

def _text(elem):
    """Plain text of an element including nested tags (e.g. <i>, <sup>), or ''."""
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()

def _join_abstract(article_elem):
    """Concatenate AbstractText sections (labeled, e.g. BACKGROUND/METHODS) to text."""
    parts = []
    for ab in article_elem.findall(".//Abstract/AbstractText"):
        label = ab.get("Label")
        body = _text(ab)
        if not body:
            continue
        if label:
            parts.append(f"{label}: {body}")
        else:
            parts.append(body)
    return " ".join(parts).strip()

def _pub_year(article_elem):
    """Best-effort publication year as int, preferring the article date then journal."""
    for path in (
        ".//Article/Journal/JournalIssue/PubDate/Year",
        ".//Article/ArticleDate/Year",
        ".//PubMedPubDate/Year",
    ):
        node = article_elem.find(path)
        if node is not None and node.text:
            try:
                return int(node.text.strip()[:4])
            except ValueError:
                pass

    md = article_elem.find(".//Journal/JournalIssue/PubDate/MedlineDate")
    if md is not None and md.text:
        digits = "".join(c for c in md.text[:4] if c.isdigit())
        if len(digits) == 4:
            return int(digits)
    return None

def _parse_article(article_elem):
    """Normalize one PubmedArticle XML element to the record dict."""
    pmid_node = article_elem.find(".//MedlineCitation/PMID")
    pmid = _text(pmid_node) if pmid_node is not None else ""

    title = _text(article_elem.find(".//Article/ArticleTitle"))
    abstract = _join_abstract(article_elem)

    journal = _text(article_elem.find(".//Article/Journal/Title"))
    if not journal:
        journal = _text(article_elem.find(".//Article/Journal/ISOAbbreviation"))

    mesh_terms = [
        _text(d)
        for d in article_elem.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
        if _text(d)
    ]
    publication_types = [
        _text(pt)
        for pt in article_elem.findall(".//Article/PublicationTypeList/PublicationType")
        if _text(pt)
    ]

    authors = []
    for au in article_elem.findall(".//Article/AuthorList/Author"):
        last = _text(au.find("LastName"))
        fore = _text(au.find("ForeName")) or _text(au.find("Initials"))
        name = " ".join(x for x in (fore, last) if x)
        if name:
            authors.append(name)
    doi = pmc = None
    for aid in article_elem.findall(".//ArticleIdList/ArticleId"):
        idt = (aid.get("IdType") or "").lower()
        if idt == "doi":
            doi = _text(aid)
        elif idt == "pmc":
            pmc = _text(aid)

    return {
        "pmid": pmid,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "journal": journal,
        "year": _pub_year(article_elem),
        "doi": doi,
        "pmc": pmc,
        "mesh_terms": mesh_terms,
        "publication_types": publication_types,
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
    }

def fetch(
    author_name,
    affiliation=None,
    *,
    limit=100,
    cache=None,
    tool=DEFAULT_TOOL,
    email=None,
):
    """Find an author's PubMed articles with abstracts via NCBI E-utilities.

    esearch builds the term '<Last FM>[Author]' (plus '<affiliation>[AD]' when given),
    then a single efetch batch pulls abstracts + metadata. Returns the standard dict;
    network/HTTP/parse failures come back as {"error": ...} rather than raising. The one
    exception is an unset CONTACT_EMAIL, which raises: that is a configuration error, not a
    runtime one, and it fails before any request goes out.
    """
    author_term = _author_term(author_name)
    query = {
        "author_name": author_name,
        "affiliation": affiliation,
        "limit": limit,
    }

    if not author_term:
        return {
            "source": "pubmed",
            "query": query,
            "count": 0,
            "results": [],
            "meta": {},
            "error": "empty author_name",
        }

    if cache is not None:
        cached = cache.get_api("pubmed", **query)
        if cached is not None:
            return cached

    term = f"{author_term}[Author]"
    if affiliation:
        term += f" AND {affiliation}[AD]"

    common = {"tool": tool, "email": email or contact_email()}

    _api_key = os.environ.get("NCBI_API_KEY")
    if _api_key:
        common["api_key"] = _api_key

    try:
        with httpx.Client(timeout=30.0) as client:
            esearch_resp = client.get(
                f"{EUTILS_BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": term,
                    "retmax": str(limit),
                    "retmode": "json",
                    "sort": "date",
                    **common,
                },
            )
            esearch_resp.raise_for_status()
            esearch_data = esearch_resp.json()
            esearchresult = esearch_data.get("esearchresult", {}) or {}
            pmids = esearchresult.get("idlist", []) or []

            meta = {
                "author_term": term,
                "total_found": esearchresult.get("count"),
            }

            if not pmids:
                result = {
                    "source": "pubmed",
                    "query": query,
                    "count": 0,
                    "results": [],
                    "meta": meta,
                    "error": None,
                }
                if cache is not None:
                    cache.set_api("pubmed", result, **query)
                return result

            efetch_resp = client.get(
                f"{EUTILS_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "rettype": "abstract",
                    "retmode": "xml",
                    **common,
                },
            )
            efetch_resp.raise_for_status()
            root = ET.fromstring(efetch_resp.content)

        results = [_parse_article(a) for a in root.findall(".//PubmedArticle")]

        result = {
            "source": "pubmed",
            "query": query,
            "count": len(results),
            "results": results,
            "meta": meta,
            "error": None,
        }
        if cache is not None:
            cache.set_api("pubmed", result, **query)
        return result

    except httpx.HTTPError as e:
        return {
            "source": "pubmed",
            "query": query,
            "count": 0,
            "results": [],
            "meta": {},
            "error": f"http error: {e}",
        }
    except ET.ParseError as e:
        return {
            "source": "pubmed",
            "query": query,
            "count": 0,
            "results": [],
            "meta": {},
            "error": f"xml parse error: {e}",
        }
    except Exception as e:
        return {
            "source": "pubmed",
            "query": query,
            "count": 0,
            "results": [],
            "meta": {},
            "error": str(e),
        }

TOOL_SCHEMA = {
    "type": "function",
    "name": "pubmed_author_articles",
    "description": (
        "Look up a scholar's PubMed articles via the official NCBI E-utilities API. "
        "Returns already-structured records (pmid, title, abstract, journal, year, "
        "mesh_terms, publication_types, pubmed_url) - no page extraction needed. Use "
        "this to get the scholar's biomedical publication list and abstracts as raw "
        "material for impact evidence. Pass affiliation (e.g. 'University of Illinois "
        "Chicago') to disambiguate common names."
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
                "description": "Optional affiliation filter applied as an [AD] term.",
            },
            "limit": {
                "type": "integer",
                "description": "Max articles to return; omit for the default of 100.",
            },
        },
        "required": ["author_name"],
    },
}
