"""Sub-call result cache — avoids redundant API calls for identical prompts.

The cache is keyed on ``(model, prompt, temperature)`` and lives for the
duration of a single generation run.  It is **not** persisted across runs.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheStats:
    """Snapshot of cache performance counters."""

    hits: int = 0
    misses: int = 0
    size: int = 0


class SubCallCache:
    """LRU cache for sub-call responses keyed on content hash.

    Parameters
    ----------
    max_size:
        Maximum number of entries.  When exceeded, the least-recently-used
        entry is evicted.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, str] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def key(model: str, prompt: str, temperature: float) -> str:
        """Deterministic cache key from the sub-call parameters."""
        raw = f"{model}\0{prompt}\0{temperature}".encode()
        return hashlib.sha256(raw).hexdigest()

    def get(self, cache_key: str) -> str | None:
        """Look up a cached response.  Returns ``None`` on miss."""
        if cache_key in self._store:
            self._hits += 1
            self._store.move_to_end(cache_key)  # mark as recently used
            return self._store[cache_key]
        self._misses += 1
        return None

    def put(self, cache_key: str, value: str) -> None:
        """Store a response.  Evicts the LRU entry if at capacity."""
        if cache_key in self._store:
            self._store.move_to_end(cache_key)
            self._store[cache_key] = value
            return
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)  # evict oldest
        self._store[cache_key] = value

    @property
    def stats(self) -> CacheStats:
        """Current performance counters."""
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            size=len(self._store),
        )
