from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from indw.config import defaults as D
from indw.dedup.normalize import stable_token_hash

_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)


def _simhash64(text: str) -> int:
    words = _WORD.findall(text.lower())
    if not words:
        return 0
    weights = defaultdict(int)
    for w in words[:2048]:
        h = stable_token_hash(w)
        for i in range(64):
            weights[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i in range(64):
        if weights[i] >= 0:
            out |= 1 << i
    return out


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _token_set(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(union, 1)


@dataclass
class StreamingSemanticDedup:
    max_bucket: int = D.DEDUP_SEMANTIC_MAX_BUCKET
    hamming_threshold: int = D.DEDUP_SEMANTIC_HAMMING
    jaccard_threshold: float = D.DEDUP_SEMANTIC_JACCARD
    recent_jaccard_threshold: float = D.DEDUP_SEMANTIC_RECENT_JACCARD
    require_both_signals: bool = True
    quality_margin: float = 0.05
    _buckets: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    _bucket_tokens: dict[int, list[set[str]]] = field(default_factory=lambda: defaultdict(list))
    _recent: list[tuple[int, set[str], float]] = field(default_factory=list)
    duplicates: int = 0
    kept: int = 0

    def _matches(self, sig: int, toks: set[str]) -> bool:
        key = (sig >> 56) & 0xFF
        bucket = self._buckets[key]
        tok_bucket = self._bucket_tokens[key]
        for i, prev in enumerate(bucket):
            hamming_match = _hamming(sig, prev) <= self.hamming_threshold
            jaccard_match = (
                i < len(tok_bucket)
                and _jaccard(toks, tok_bucket[i]) >= self.jaccard_threshold
            )
            if self.require_both_signals:
                if hamming_match and jaccard_match:
                    return True
            elif hamming_match or jaccard_match:
                return True
        for prev, prev_toks, _ in self._recent[-128:]:
            hamming_match = _hamming(sig, prev) <= self.hamming_threshold
            jaccard_match = _jaccard(toks, prev_toks) >= self.recent_jaccard_threshold
            if self.require_both_signals:
                if hamming_match and jaccard_match:
                    return True
            elif hamming_match or jaccard_match:
                return True
        return False

    def matches_near_duplicate(self, text: str) -> bool:
        sig = _simhash64(text)
        toks = _token_set(text)
        return self._matches(sig, toks)

    def should_keep(self, text: str, quality_score: float = 0.0) -> bool:
        sig = _simhash64(text)
        toks = _token_set(text)
        if not self._matches(sig, toks):
            return True
        self.duplicates += 1
        return False

    def register(self, text: str, quality_score: float = 0.0) -> None:
        sig = _simhash64(text)
        toks = _token_set(text)
        key = (sig >> 56) & 0xFF
        bucket = self._buckets[key]
        tok_bucket = self._bucket_tokens[key]
        bucket.append(sig)
        tok_bucket.append(toks)
        self._recent.append((sig, toks, quality_score))
        if len(self._recent) > 2048:
            del self._recent[:-1024]
        if len(bucket) > self.max_bucket:
            k = len(bucket) - self.max_bucket
            del bucket[:k]
            del tok_bucket[:k]
        self.kept += 1

    def is_near_duplicate(self, text: str, quality_score: float = 0.0) -> bool:
        if self._matches(_simhash64(text), _token_set(text)):
            self.duplicates += 1
            return True
        self.register(text, quality_score)
        return False

    def summary(self) -> dict[str, int]:
        return {
            'semantic_duplicates': int(self.duplicates),
            'semantic_kept': int(self.kept),
            'semantic_buckets': int(len(self._buckets)),
        }


def create_semantic_dedup(
    *,
    hamming_threshold: int = 4,
    jaccard_threshold: float = 0.72,
    recent_jaccard_threshold: float = 0.75,
    **_: object,
) -> StreamingSemanticDedup:
    return StreamingSemanticDedup(
        hamming_threshold=hamming_threshold,
        jaccard_threshold=jaccard_threshold,
        recent_jaccard_threshold=recent_jaccard_threshold,
    )
