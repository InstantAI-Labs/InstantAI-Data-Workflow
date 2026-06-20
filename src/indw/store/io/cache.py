from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from cachetools import LRUCache

T = TypeVar('T')


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def to_dict(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': round(self.hits / total, 4) if total else 0.0,
        }


class _StatsLRU(LRUCache):
    def __init__(self, maxsize: int, stats: CacheStats) -> None:
        super().__init__(maxsize=maxsize)
        self._stats = stats

    def __setitem__(self, key: Any, value: Any) -> None:
        if self.maxsize and len(self) >= self.maxsize and key not in self:
            self._stats.evictions += 1
        super().__setitem__(key, value)


class BoundedLRU(Generic[T]):
    __slots__ = ('_cache', '_lock', '_thread_safe', 'stats', '_base_maxsize')

    def __init__(
        self,
        maxsize: int,
        *,
        stats: CacheStats | None = None,
        thread_safe: bool = False,
    ) -> None:
        size = max(64, int(maxsize))
        self._base_maxsize = size
        self.stats = stats or CacheStats()
        self._thread_safe = thread_safe
        self._lock = threading.Lock() if thread_safe else None
        self._cache = _StatsLRU(size, self.stats)

    @property
    def _maxsize(self) -> int:
        return int(self._cache.maxsize or 0)

    @_maxsize.setter
    def _maxsize(self, value: int) -> None:
        self._cache.maxsize = max(64, int(value))

    def _locked(self):
        if self._lock is None:
            from contextlib import nullcontext
            return nullcontext()
        return self._lock

    def get(self, key: Any) -> T | None:
        with self._locked():
            try:
                self.stats.hits += 1
                return self._cache[key]
            except KeyError:
                self.stats.hits -= 1
                self.stats.misses += 1
                return None

    def put(self, key: Any, value: T) -> None:
        with self._locked():
            self._cache[key] = value

    def set(self, key: Any, value: T) -> None:
        self.put(key, value)

    def clear(self) -> None:
        with self._locked():
            self._cache.clear()

    def trim_to_ratio(self, keep_ratio: float = 0.5) -> None:
        keep_ratio = max(0.1, min(1.0, keep_ratio))
        with self._locked():
            target = max(1, int(len(self._cache) * keep_ratio))
            while len(self._cache) > target:
                self._cache.popitem()
                self.stats.evictions += 1

    def __len__(self) -> int:
        with self._locked():
            return len(self._cache)
