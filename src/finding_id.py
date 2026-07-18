"""Deterministic, content-derived finding IDs: a hash of stable identity fields, not a
per-run counter, so the same finding gets the same ID across runs.

Exact for keyed findings (grants/trials/pubs by their identifier); best-effort (name+year
or URL) for impacts/media.
"""
import hashlib
import re

def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())

def finding_id(section: str, *key_parts) -> str:
    """Stable ID from a section tag plus the finding's identity fields; empty parts are dropped."""
    key = "|".join(_norm(p) for p in key_parts if p not in (None, ""))
    digest = hashlib.sha256(f"{section}|{key}".encode()).hexdigest()[:10]
    return f"F-{digest}"
