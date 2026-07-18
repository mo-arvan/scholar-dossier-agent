"""structured_extract: loop primitive A.

Fetch a page's text and run ONE cheap-model call that distills it down to a
compact, focus-scoped fragment. The raw page never leaves this module, so the
orchestrator context stays small.

Unlike the pure-API adapters, this module is LLM-coupled: the caller passes an
OpenAI Responses client (`llm`) and a cheap `model` (gpt-5-mini). Page text is
fetched with httpx for non-blocked domains; a Tavily-extract hook is left for
the integrator to wire the existing collect_dossier.extract through.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from string import Template
from urllib.parse import urlparse

import httpx
import tiktoken

from llm_retry import create_response_with_backoff

log = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_MAX_PAGE_TOKENS = 100_000

@lru_cache(maxsize=None)
def _load_template(name: str) -> Template:
    return Template((_PROMPT_DIR / name).read_text())

@lru_cache(maxsize=1)
def _encoder():
    return tiktoken.get_encoding("o200k_base")

def _fit_to_tokens(text: str, budget: int = _MAX_PAGE_TOKENS) -> str:
    """Trim text to at most `budget` tokens, keeping the front of the page. No-op when it fits."""
    try:
        enc = _encoder()
        toks = enc.encode(text)
        if len(toks) <= budget:
            return text
        log.info("page over %d tokens (%d); trimming to fit the reader's context", budget, len(toks))
        return enc.decode(toks[:budget])
    except Exception:
        return text

BLOCKED_DOMAINS = {
    "reporter.nih.gov",
    "projectreporter.nih.gov",
    "clinicaltrials.gov",
    "orcid.org",
    "linkedin.com",
    "researchgate.net",
}

TOOL_SCHEMA = {
    "type": "function",
    "name": "structured_extract",
    "description": (
        "Read a single webpage URL and return a compact, focus-scoped text fragment: "
        "the page content relevant to 'focus' (what you are looking for, e.g. 'impact "
        "evidence and roles for Dr. Jane Smith'), in plain text. The raw page is not "
        "returned, only the distilled fragment, so prefer this over a full-text extract. "
        "Skip blocked domains (NIH RePORTER, ORCID, ClinicalTrials, LinkedIn, ResearchGate) "
        "and use the matching API tool instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The webpage URL to read and distill.",
            },
            "focus": {
                "type": "string",
                "description": (
                    "What to look for on the page, e.g. 'community health "
                    "programs and clinical leadership for Dr. Jane Smith'."
                ),
            },
            "scholar": {
                "type": "string",
                "description": (
                    "One sentence identifying the scholar and their research area, so "
                    "the page can be matched to the RIGHT person and not a namesake."
                ),
            },
        },
        "required": ["url", "focus", "scholar"],
    },
}

def _empty_result(url, *, distilled="", error=None):
    """A fragment with the contract shape (raw page never included)."""
    return {"url": url, "distilled": distilled, "error": error}

def _fetch_page_text(url, tavily_extract=None):
    """Return (text, error). Never raises.

    Uses the Tavily extract hook when provided (lets the integrator reuse the
    cached extract() in collect_dossier.py); otherwise falls back to httpx for
    non-blocked domains. Blocked domains short-circuit with an error.
    """
    domain = urlparse(url).netloc.lower()

    if any(domain == b or domain.endswith("." + b) for b in BLOCKED_DOMAINS):
        return "", f"blocked_domain_for_extract:{domain}"

    if tavily_extract is not None:

        try:
            raw = tavily_extract(url)
        except Exception as e:
            return "", f"tavily_extract_error:{e}"
        text = raw if isinstance(raw, str) else json.dumps(raw)
        return text, None

    try:
        with httpx.Client(
            timeout=30.0, follow_redirects=True, headers={"User-Agent": "scholar-dossier/1.0"}
        ) as client:
            resp = client.get(url)
        if resp.status_code >= 400:
            return "", f"http_{resp.status_code}"
        return resp.text, None
    except Exception as e:
        return "", f"fetch_error:{e}"

def _call_llm(llm, model, url, focus, page_text, scholar="", on_usage=None):
    """One cheap-model call returning the distilled text (plain, faithful, focus-scoped).
    Returns (text, error). If on_usage is given, it is called with resp.usage. Never raises."""
    prompt = _load_template("extract.txt").substitute(
        scholar=scholar or "the scholar named in the focus",
        focus=focus, url=url, page=_fit_to_tokens(page_text))
    try:
        resp = create_response_with_backoff(llm, model=model, input=prompt)
    except Exception as e:
        return None, f"llm_error:{e}"

    if on_usage:
        on_usage(getattr(resp, "usage", None))
    text = getattr(resp, "output_text", None)
    if not text:

        parts = []
        for item in getattr(resp, "output", []) or []:
            for chunk in getattr(item, "content", []) or []:
                t = getattr(chunk, "text", None)
                if t:
                    parts.append(t)
        text = "".join(parts)

    if not text:
        return None, "empty_llm_output"
    return text.strip(), None

def structured_extract(url, focus, scholar="", *, llm, model, cache=None, tavily_extract=None, on_usage=None):
    """Read one URL and return a compact, focus-scoped fragment.

    Args:
        url: The webpage URL to read.
        focus: What to look for, e.g. "impact evidence and roles for Dr. X".
        scholar: One-sentence description of the scholar (name + research area), so the
            cheap reader matches the page to the right person and not a namesake.
        llm: An OpenAI Responses client (same one the main loop uses).
        model: The cheap model name (e.g. "gpt-5-mini").
        cache: Optional ResponseCache; keyed via get_api/set_api("extract2", ...).
        tavily_extract: Optional hook fn(url)->str to reuse the cached Tavily
            extract from collect_dossier.py instead of a direct httpx GET.

    Returns:
        {url, distilled, error} where distilled is the focus-scoped text. The raw page is
        never returned. Never raises; failures are reported in "error".
    """
    if cache is not None:
        cached = cache.get_api("extract2", url=url, focus=focus, scholar=scholar)
        if cached is not None:
            return cached

    page_text, fetch_error = _fetch_page_text(url, tavily_extract=tavily_extract)
    if fetch_error is not None:
        result = _empty_result(url, error=fetch_error)
        if cache is not None:
            cache.set_api("extract2", result, url=url, focus=focus, scholar=scholar)
        return result

    text, llm_error = _call_llm(llm, model, url, focus, page_text, scholar=scholar, on_usage=on_usage)
    if llm_error is not None:
        return _empty_result(url, error=llm_error)

    result = {"url": url, "distilled": text, "error": None}
    if cache is not None:
        cache.set_api("extract2", result, url=url, focus=focus, scholar=scholar)
    return result
