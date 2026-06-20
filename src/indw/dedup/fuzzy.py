from __future__ import annotations

import re
from collections import defaultdict
from typing import DefaultDict

import numpy as np

from indw.dedup.normalize import normalize_for_dedup, stable_token_hash

_WORD = re.compile(r'\w+', re.UNICODE)
_HASH_SEED = 11400714819323198485


class StreamingFuzzyDedup:

    def __init__(
        self,
        threshold: float = 0.82,
        num_perm: int = 128,
        num_bands: int = 16,
        max_candidates: int = 32,
        quality_margin: float = 0.05,
    ):
        self.threshold = threshold
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.band_size = max(1, num_perm // num_bands)
        self.max_candidates = max_candidates
        self.quality_margin = quality_margin
        self.buckets: DefaultDict[tuple, list[tuple[np.ndarray, float]]] = defaultdict(list)
        self.duplicates = 0
        self.kept = 0

    def _shingles(self, text: str, n: int = 5) -> set[int]:
        normalized = normalize_for_dedup(text)
        tokens = _WORD.findall(normalized)
        if len(tokens) < n:
            if not tokens:
                return set()
            return {stable_token_hash(' '.join(tokens))}
        return {
            stable_token_hash(' '.join(tokens[i:i + n]))
            for i in range(len(tokens) - n + 1)
        }

    def _signature(self, shingles: set[int]) -> np.ndarray:
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.uint64)
        arr = np.array(list(shingles), dtype=np.uint64)
        sig = np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        for i, s in enumerate(arr):
            h = np.uint64((int(s) * _HASH_SEED + i) % 2 ** 64)
            sig = np.minimum(sig, h)
        return sig

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(a == b))

    def _bands(self, sig: np.ndarray) -> list[tuple]:
        keys = []
        for b in range(self.num_bands):
            start = b * self.band_size
            end = min(start + self.band_size, self.num_perm)
            band = tuple(sig[start:end].tolist())
            keys.append((b, band))
        return keys

    def _best_match(self, sig: np.ndarray) -> tuple[bool, float]:
        matched = False
        best_score = 0.0
        for key in self._bands(sig):
            for prev_sig, prev_score in self.buckets[key][:self.max_candidates]:
                if self._similarity(sig, prev_sig) >= self.threshold:
                    matched = True
                    best_score = max(best_score, prev_score)
        return matched, best_score

    def matches_near_duplicate(self, text: str) -> bool:
        sig = self._signature(self._shingles(text))
        matched, _ = self._best_match(sig)
        return matched

    def should_keep(self, text: str, quality_score: float = 0.0) -> bool:
        sig = self._signature(self._shingles(text))
        matched, best_score = self._best_match(sig)
        if not matched:
            return True
        if quality_score > best_score + self.quality_margin:
            return True
        self.duplicates += 1
        return False

    def register(self, text: str, quality_score: float = 0.0) -> None:
        sig = self._signature(self._shingles(text))
        for key in self._bands(sig):
            bucket = self.buckets[key]
            bucket.append((sig, quality_score))
            if len(bucket) > self.max_candidates * 4:
                del bucket[:-self.max_candidates]
        self.kept += 1

    def is_near_duplicate(self, text: str, quality_score: float = 0.0) -> bool:
        if not self.should_keep(text, quality_score):
            return True
        self.register(text, quality_score)
        return False

    def summary(self) -> dict:
        return {'fuzzy_duplicates': self.duplicates, 'fuzzy_kept': self.kept, 'bucket_count': len(self.buckets)}
