"""Domain-API adapters for the scholar dossier agent.

Each adapter replaces flaky web scraping with a structured API call. Contracts:

each exposes fetch(...) and TOOL_SCHEMA. The loop registers the tools at integration.

This package also exposes two registries the loop uses to wire the tools:
- PURE_ADAPTERS: tool name -> fetch callable (each takes a `cache=` kwarg).
- ADAPTER_TOOL_SCHEMAS: the Responses-API tool definitions for those adapters.
structured_extract (adapters.extract) is wired separately because it needs the LLM.
"""
from . import clinicaltrials, extract, nih_reporter, openalex, orcid, pubmed

PURE_ADAPTERS = {
    openalex.TOOL_SCHEMA["name"]: openalex.fetch,
    pubmed.TOOL_SCHEMA["name"]: pubmed.fetch,
    orcid.TOOL_SCHEMA["name"]: orcid.fetch,
    clinicaltrials.TOOL_SCHEMA["name"]: clinicaltrials.fetch,
    nih_reporter.TOOL_SCHEMA["name"]: nih_reporter.fetch,
}

ADAPTER_TOOL_SCHEMAS = [
    openalex.TOOL_SCHEMA,
    pubmed.TOOL_SCHEMA,
    orcid.TOOL_SCHEMA,
    clinicaltrials.TOOL_SCHEMA,
    nih_reporter.TOOL_SCHEMA,
]
