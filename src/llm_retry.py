"""Shared retry policy for model calls: transient errors (connection drop, 429, 5xx) retry
with backoff; httpx timeouts and terminal 4xx errors fail fast.

WallClockTimeout (our length-based deadline) retries only twice, since a fresh connection
usually clears a trickle.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout

from openai import (APIConnectionError, APITimeoutError, BadRequestError,
                    InternalServerError, RateLimitError)
from tenacity import (retry, retry_if_exception_type, retry_if_not_exception_type,
                      wait_chain, wait_fixed)

log = logging.getLogger(__name__)

class WallClockTimeout(Exception):
    """A model call exceeded its wall-clock deadline. httpx's read timeout is inter-byte, so
    it never bounds a connection trickling keepalive bytes; retry-eligible since a fresh
    connection usually clears the trickle."""

CALL_DEADLINE_S = int(os.environ.get("LLM_CALL_DEADLINE_S", "240"))
_DEADLINE_POOL = ThreadPoolExecutor(max_workers=64, thread_name_prefix="llm-deadline")

def deadline_for(context_tokens: int) -> int:
    """Length-based wall-clock deadline (~4s/1k context tokens, floored at 120s, capped at
    900s). The 900s cap stays below the httpx client's 1200s timeout, so the retryable
    WallClockTimeout always fires first."""
    return min(900, max(120, int(context_tokens) // 250))

def _with_wall_clock(fn, deadline_s=None, **kwargs):
    limit = deadline_s or CALL_DEADLINE_S
    fut = _DEADLINE_POOL.submit(fn, **kwargs)
    try:
        return fut.result(timeout=limit)
    except _FutureTimeout:
        raise WallClockTimeout(f"model call exceeded {limit}s wall-clock deadline")

def _stop(retry_state) -> bool:
    """A wall-clock timeout retries only ONCE (2 attempts): one fresh connection clears a trickle,
    more just multiplies the worst case. Other transients keep the full 4 attempts."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    return retry_state.attempt_number >= (2 if isinstance(exc, WallClockTimeout) else 4)

_TRANSIENT_ERRORS = (APIConnectionError, RateLimitError, InternalServerError, WallClockTimeout)
_responses_retry = retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS) & retry_if_not_exception_type(APITimeoutError),
    wait=wait_chain(wait_fixed(10), wait_fixed(30), wait_fixed(60)),
    stop=_stop,
    reraise=True,
)

_prompt_cache_key_supported = True

def _strip_cache_key_on_reject(core):
    def wrapper(client, *, deadline_s=None, **kwargs):
        global _prompt_cache_key_supported
        if not _prompt_cache_key_supported:
            kwargs.pop("prompt_cache_key", None)
        try:
            return core(client, deadline_s=deadline_s, **kwargs)
        except BadRequestError as e:
            if "prompt_cache_key" in kwargs and "prompt_cache_key" in str(e):
                _prompt_cache_key_supported = False
                kwargs.pop("prompt_cache_key", None)
                log.warning("endpoint rejected prompt_cache_key; disabling it and retrying without.")
                return core(client, deadline_s=deadline_s, **kwargs)
            raise
    return wrapper

@_responses_retry
def _create_core(client, *, deadline_s=None, **kwargs):
    return _with_wall_clock(client.responses.create, deadline_s=deadline_s, **kwargs)

@_responses_retry
def _parse_core(client, *, deadline_s=None, **kwargs):
    return _with_wall_clock(client.responses.parse, deadline_s=deadline_s, **kwargs)

create_response_with_backoff = _strip_cache_key_on_reject(_create_core)
parse_response_with_backoff = _strip_cache_key_on_reject(_parse_core)

def usage_dict(usage):
    """Trace token_usage for one response, including prompt-cache hits. `cached`
    (usage.input_tokens_details.cached_tokens) signals whether the static-prefix-first
    prompt ordering is actually being reused."""
    if usage is None:
        return None
    details = getattr(usage, "input_tokens_details", None)
    return {
        "context": getattr(usage, "input_tokens", None),
        "generated": getattr(usage, "output_tokens", None),
        "total": getattr(usage, "total_tokens", None),
        "cached": getattr(details, "cached_tokens", None),
    }

def cached_tokens(usage):
    """The cached input tokens for one response (0 if unavailable), for aggregation."""
    details = getattr(usage, "input_tokens_details", None)
    return int(getattr(details, "cached_tokens", 0) or 0)
