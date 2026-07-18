"""File-based cache for search/extract/API responses, keyed by params, with 30-day expiry.

Stored in data/cache/ (version controlled): the results are valuable and expensive to regenerate.
"""

import functools
import hashlib
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

log = logging.getLogger(__name__)

def _synchronized(method):
    """Run a ResponseCache method under self._lock (reentrant) for thread safety."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper

class ResponseCache:
    """File-based cache for API responses with 30-day expiry and domain tracking."""

    def __init__(self, cache_dir: str = "data/cache"):

        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_absolute() and not self.cache_dir.exists():
            for base in Path(__file__).resolve().parents:
                if (base / cache_dir).parent.is_dir():
                    self.cache_dir = base / cache_dir
                    break
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.search_cache_file = self.cache_dir / "search_cache.json"
        self.extract_cache_file = self.cache_dir / "extract_cache.json"
        self.domain_stats_file = self.cache_dir / "domain_stats.json"
        self.api_cache_file = self.cache_dir / "api_cache.json"
        self.expiry_days = 30

        self._lock = threading.RLock()

        self.search_cache = self._load_cache(self.search_cache_file)
        self.extract_cache = self._load_cache(self.extract_cache_file)
        self.domain_stats = self._load_cache(self.domain_stats_file)
        self.api_cache = self._load_cache(self.api_cache_file)

    def _load_cache(self, cache_file: Path) -> Dict:
        """Load cache from file, or return empty dict if not exists."""
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                log.warning(f"Failed to load cache from {cache_file}, starting fresh")
                return {}
        return {}

    def _save_cache(self, cache: Dict, cache_file: Path) -> None:
        """Save cache to file."""
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

    def _make_cache_key(self, **kwargs) -> str:
        """Create a cache key from parameters."""
        sorted_items = sorted(kwargs.items())
        key_string = json.dumps(sorted_items, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _is_expired(self, timestamp_str: str) -> bool:
        """Check if cache entry has expired (> 30 days old)."""
        cached_time = datetime.fromisoformat(timestamp_str)
        expiry_time = datetime.now(timezone.utc) - timedelta(days=self.expiry_days)
        return cached_time < expiry_time

    @_synchronized
    def get_search(self, query: str, **kwargs) -> Optional[Dict]:
        """Get cached search results, or None if absent/expired."""
        cache_key = self._make_cache_key(query=query, **kwargs)

        if cache_key in self.search_cache:
            entry = self.search_cache[cache_key]

            if self._is_expired(entry["timestamp"]):
                log.debug(f"Cache expired for query: {query[:50]}...")
                del self.search_cache[cache_key]
                self._save_cache(self.search_cache, self.search_cache_file)
                return None

            log.info(f"✓ Cache hit for search: {query[:50]}...")
            return entry["result"]

        return None

    @_synchronized
    def set_search(self, query: str, result: Dict, **kwargs) -> None:
        """Cache search results."""
        cache_key = self._make_cache_key(query=query, **kwargs)

        self.search_cache[cache_key] = {
            "query": query,
            "params": kwargs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }

        self._save_cache(self.search_cache, self.search_cache_file)
        log.debug(f"Cached search: {query[:50]}...")

    @_synchronized
    def get_extract(self, url: str, **kwargs) -> Optional[Dict]:
        """Get cached extraction results, or None if absent/expired."""
        cache_key = self._make_cache_key(url=url, **kwargs)

        if cache_key in self.extract_cache:
            entry = self.extract_cache[cache_key]

            if self._is_expired(entry["timestamp"]):
                log.debug(f"Cache expired for URL: {url[:50]}...")
                del self.extract_cache[cache_key]
                self._save_cache(self.extract_cache, self.extract_cache_file)
                return None

            log.info(f"✓ Cache hit for extract: {url[:50]}...")
            return entry["result"]

        return None

    @_synchronized
    def set_extract(self, url: str, result: Dict, **kwargs) -> None:
        """Cache extraction results."""
        cache_key = self._make_cache_key(url=url, **kwargs)

        self.extract_cache[cache_key] = {
            "url": url,
            "params": kwargs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }

        self._save_cache(self.extract_cache, self.extract_cache_file)
        log.debug(f"Cached extract: {url[:50]}...")

    @_synchronized
    def get_api(self, source: str, **params) -> Optional[Dict]:
        """Get a cached domain-API result (OpenAlex/PubMed/ORCID/ClinicalTrials/etc.), or None if absent/expired."""
        cache_key = self._make_cache_key(source=source, **params)
        entry = self.api_cache.get(cache_key)
        if not entry:
            return None
        if self._is_expired(entry["timestamp"]):
            del self.api_cache[cache_key]
            self._save_cache(self.api_cache, self.api_cache_file)
            return None
        log.info(f"✓ Cache hit for {source}: {json.dumps(params)[:60]}...")
        return entry["result"]

    @_synchronized
    def set_api(self, source: str, result: Dict, **params) -> None:
        """Cache a domain-API result by (source, params)."""
        cache_key = self._make_cache_key(source=source, **params)
        self.api_cache[cache_key] = {
            "source": source,
            "params": params,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        self._save_cache(self.api_cache, self.api_cache_file)

    @_synchronized
    def clear_expired(self) -> None:
        """Remove all expired entries from caches."""
        initial_search_count = len(self.search_cache)
        initial_extract_count = len(self.extract_cache)

        expired_keys = [
            k for k, v in self.search_cache.items() if self._is_expired(v["timestamp"])
        ]
        for key in expired_keys:
            del self.search_cache[key]

        expired_keys = [
            k for k, v in self.extract_cache.items() if self._is_expired(v["timestamp"])
        ]
        for key in expired_keys:
            del self.extract_cache[key]

        if len(self.search_cache) < initial_search_count:
            self._save_cache(self.search_cache, self.search_cache_file)
            log.info(
                f"Removed {initial_search_count - len(self.search_cache)} expired search entries"
            )

        if len(self.extract_cache) < initial_extract_count:
            self._save_cache(self.extract_cache, self.extract_cache_file)
            log.info(
                f"Removed {initial_extract_count - len(self.extract_cache)} expired extract entries"
            )

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc or "unknown"
        except Exception:
            return "unknown"

    @_synchronized
    def track_extraction(
        self, url: str, success: bool, error: Optional[str] = None
    ) -> None:
        """Track one extraction attempt by domain, for analytics."""
        domain = self._extract_domain(url)

        if domain not in self.domain_stats:
            self.domain_stats[domain] = {
                "total_attempts": 0,
                "successful": 0,
                "failed": 0,
                "errors": [],
            }

        self.domain_stats[domain]["total_attempts"] += 1

        if success:
            self.domain_stats[domain]["successful"] += 1
        else:
            self.domain_stats[domain]["failed"] += 1
            if error and error not in self.domain_stats[domain]["errors"]:
                self.domain_stats[domain]["errors"].append(error)

        self._save_cache(self.domain_stats, self.domain_stats_file)

    @_synchronized
    def get_domain_stats(self) -> Dict[str, Dict]:
        """Get extraction statistics by domain, sorted by most-attempted first."""
        stats = {}
        for domain, data in self.domain_stats.items():
            total = data["total_attempts"]
            successful = data["successful"]
            stats[domain] = {
                "total_attempts": total,
                "successful": successful,
                "failed": data["failed"],
                "success_rate": (successful / total * 100) if total > 0 else 0,
                "errors": data.get("errors", []),
            }

        return dict(
            sorted(stats.items(), key=lambda x: x[1]["total_attempts"], reverse=True)
        )

    @_synchronized
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "search_entries": len(self.search_cache),
            "extract_entries": len(self.extract_cache),
            "domains_tracked": len(self.domain_stats),
            "api_entries": len(self.api_cache),
        }
