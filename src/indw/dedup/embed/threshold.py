from __future__ import annotations

import numpy as np

class AdaptiveSimilarityThreshold:
    def __init__(
        self,
        *,
        min_threshold: float = 0.75,
        max_threshold: float = 0.99,
        reservoir: int = 2048,
    ):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.reservoir = reservoir
        self._match: list[float] = []
        self._non_match: list[float] = []

    def record_match(self, sim: float) -> None:
        self._match.append(float(sim))
        if len(self._match) > self.reservoir:
            del self._match[: len(self._match) - self.reservoir]

    def record_non_match(self, sim: float) -> None:
        self._non_match.append(float(sim))
        if len(self._non_match) > self.reservoir:
            del self._non_match[: len(self._non_match) - self.reservoir]

    def value(self) -> float:
        if len(self._match) < 8 or len(self._non_match) < 8:
            return (self.min_threshold + self.max_threshold) * 0.5
        pos = float(np.percentile(self._match, 30))
        neg = float(np.percentile(self._non_match, 95))
        t = (pos + neg) * 0.5
        return max(self.min_threshold, min(self.max_threshold, t))

    def distribution(self) -> dict[str, float]:
        samples = self._match + self._non_match
        if not samples:
            return {}
        arr = np.array(samples, dtype=np.float64)
        return {
            'p10': round(float(np.percentile(arr, 10)), 4),
            'p50': round(float(np.percentile(arr, 50)), 4),
            'p90': round(float(np.percentile(arr, 90)), 4),
            'mean': round(float(arr.mean()), 4),
        }
