"""Web-search adapter (Tavily): discover URLs and non-scholarly signals.

Unlike the pure-API adapters, this one needs the Tavily client, so the loop injects
it as a ``tavily_search`` hook (the same way ``extract`` takes a ``tavily_extract``
hook). Exposes ``search(...)`` and ``TOOL_SCHEMA``. Caching uses the loop's search
cache when provided.

Import convention (src/ is on sys.path at runtime):
    from adapters.search import search, TOOL_SCHEMA
"""

import logging
from typing import Any, List

log = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "type": "function",
    "name": "search",
    "description": (
        "General web search (Tavily) to discover URLs and non-scholarly signals "
        "(media, community programs, policy). Follow up promising URLs with structured_extract."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query. This is a SEMANTIC search, not a boolean one: AND / OR "
                    "are treated as literal words, not operators, so cover multiple angles with "
                    "separate search calls rather than OR. Wrap a name in double quotes for an "
                    "exact-phrase match (e.g. '\"Jane Smith\" cardiology')."
                ),
            },
            "num_results": {"type": "integer", "description": "Number of results (default 10)."},
        },
        "required": ["query"],
    },
}

def search(query: str, num_results: int = 10, *, tavily_search, cache=None) -> Any:
    """Return a list of result dicts (title/url/content), or ``{error}``. Never raises.

    Args:
        query: the search query.
        num_results: how many results to request.
        tavily_search: hook ``fn(query=, num_results=) -> response dict`` (injected by the loop).
        cache: optional search cache (get_search / set_search).
    """
    if cache is not None:
        cached = cache.get_search(query, num_results=num_results)
        if cached is not None:
            return cached
    try:
        response = tavily_search(query=query, num_results=num_results)
    except Exception as e:
        log.error(f"Tavily search error: {e}")
        return {"error": str(e)}
    results: List[dict] = response.get("results", []) if isinstance(response, dict) else (response or [])
    if cache is not None:
        cache.set_search(query, results, num_results=num_results)
    return results
