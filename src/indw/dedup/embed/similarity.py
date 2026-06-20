from __future__ import annotations

import numpy as np

from indw.dedup.embed.contracts import SimilarityProvider

class VectorSimilarityProvider:
    def __init__(self, metric: str = 'cosine'):
        m = metric.strip().lower()
        if m not in ('cosine', 'dot', 'euclidean'):
            raise ValueError(f'unsupported similarity metric: {metric}')
        self.metric = m

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == 'cosine':
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na <= 0 or nb <= 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))
        if self.metric == 'dot':
            return float(np.dot(a, b))
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def pairwise_max(self, query: np.ndarray, candidates: list[np.ndarray]) -> tuple[float, int]:
        if not candidates:
            return 0.0, -1
        best = -1.0
        best_i = -1
        for i, cand in enumerate(candidates):
            s = self.score(query, cand)
            if s > best:
                best = s
                best_i = i
        return best, best_i
