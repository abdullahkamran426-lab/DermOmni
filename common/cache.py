from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

CACHE_VERSION = "v2"


@dataclass(frozen=True)
class CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Small process-local TTL cache; intentionally not a cross-user persistent store."""

    def __init__(self, ttl_seconds: float = 300.0, max_entries: int = 256) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, CacheEntry[T]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            if len(self._entries) >= self.max_entries and key not in self._entries:
                oldest_key = min(self._entries, key=lambda item: self._entries[item].expires_at)
                self._entries.pop(oldest_key, None)
            self._entries[key] = CacheEntry(value=value, expires_at=time.monotonic() + self.ttl_seconds)


def build_search_cache_key(
    *,
    operation: str,
    query: str,
    model: str,
    search_depth: str,
    max_results: int,
    domains: list[str],
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "operation": operation,
        "query": " ".join(query.lower().split()),
        "model": model,
        "search_depth": search_depth,
        "max_results": max_results,
        "domains": sorted(domains),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
