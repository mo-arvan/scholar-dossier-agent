"""Unit tests for structured_extract with a MOCKED llm (no creds, no network).

Run: uv run pytest tests/test_extract.py
"""

from adapters.extract import structured_extract, TOOL_SCHEMA

RAW_PAGE = (
    "SECRET_RAW_MARKER Dr. Jane Smith leads the Healthy Habits school program "
    "serving low-income children in Chicago. She also directs the free clinic. "
) * 50

class _StubResponse:
    def __init__(self, text):
        self.output_text = text

class _StubResponses:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._text)

class _StubLLM:
    def __init__(self, text):
        self.responses = _StubResponses(text)

CANNED = "Dr. Jane Smith leads the Healthy Habits school program in Chicago and directs the free clinic."

def _fake_tavily(url):
    """Hook stub: returns raw page text without touching the network."""
    return RAW_PAGE

def test_shape_and_no_leak():
    llm = _StubLLM(CANNED)
    out = structured_extract(
        "https://example.com/jane",
        "impact evidence and roles for Dr. Jane Smith",
        llm=llm,
        model="gpt-5-mini",
        tavily_extract=_fake_tavily,
    )

    assert set(out.keys()) == {"url", "distilled", "error"}, out.keys()
    assert out["url"] == "https://example.com/jane"
    assert out["distilled"] == CANNED
    assert out["error"] is None

    blob = repr(out)
    assert "SECRET_RAW_MARKER" not in blob, "raw page leaked into result!"

    assert len(llm.responses.calls) == 1
    sent = llm.responses.calls[0]["input"]
    assert "SECRET_RAW_MARKER" in sent, "page text should be sent TO the model"
    print("PASS test_shape_and_no_leak")

def test_blocked_domain_no_fetch_no_llm():
    llm = _StubLLM(CANNED)
    out = structured_extract(
        "https://orcid.org/0000-0002-1825-0097",
        "employment history",
        llm=llm,
        model="gpt-5-mini",
    )
    assert set(out.keys()) == {"url", "distilled", "error"}
    assert out["error"] and out["error"].startswith("blocked_domain_for_extract")
    assert out["distilled"] == ""

    assert len(llm.responses.calls) == 0
    print("PASS test_blocked_domain_no_fetch_no_llm")

def test_empty_llm_output():
    llm = _StubLLM("")
    out = structured_extract(
        "https://example.com/jane",
        "focus",
        llm=llm,
        model="gpt-5-mini",
        tavily_extract=_fake_tavily,
    )
    assert out["error"] == "empty_llm_output"
    assert out["distilled"] == ""
    print("PASS test_empty_llm_output")

def test_cache_roundtrip_and_no_llm_on_hit():
    class _Cache:
        def __init__(self):
            self.store = {}

        def get_api(self, source, **params):
            return self.store.get((source, tuple(sorted(params.items()))))

        def set_api(self, source, result, **params):
            self.store[(source, tuple(sorted(params.items())))] = result

    cache = _Cache()
    llm = _StubLLM(CANNED)
    first = structured_extract(
        "https://example.com/jane", "focus", llm=llm, model="gpt-5-mini",
        cache=cache, tavily_extract=_fake_tavily,
    )
    assert llm.responses.calls and len(llm.responses.calls) == 1

    second = structured_extract(
        "https://example.com/jane", "focus", llm=llm, model="gpt-5-mini",
        cache=cache, tavily_extract=_fake_tavily,
    )
    assert second == first
    assert len(llm.responses.calls) == 1, "cache hit should skip the LLM call"
    print("PASS test_cache_roundtrip_and_no_llm_on_hit")

def test_tool_schema_shape():
    assert TOOL_SCHEMA["type"] == "function"
    assert TOOL_SCHEMA["name"] == "structured_extract"
    props = TOOL_SCHEMA["parameters"]["properties"]
    assert "url" in props and "focus" in props and "scholar" in props
    assert TOOL_SCHEMA["parameters"]["required"] == ["url", "focus", "scholar"]
    print("PASS test_tool_schema_shape")

if __name__ == "__main__":
    test_shape_and_no_leak()
    test_blocked_domain_no_fetch_no_llm()
    test_empty_llm_output()
    test_cache_roundtrip_and_no_llm_on_hit()
    test_tool_schema_shape()
    print("ALL TESTS PASSED")
