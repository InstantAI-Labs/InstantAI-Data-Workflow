from __future__ import annotations

import math

from indw.tools.reports.fast.stats import wilson_ci

def histogram_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 1.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    max_ent = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return ent / max(max_ent, 1e-9)

def position_confidence(histogram: list[int], doc_frequency: int, docs_seen: int) -> float:
    ent = histogram_entropy(histogram)
    consistency = 1.0 - min(1.0, ent)
    ci = wilson_ci(doc_frequency, max(docs_seen, 1))
    freq = ci['low']
    return min(1.0, consistency * 0.65 + freq * 0.35)

class PositionalLearner:
    def score(self, histogram: list[int], doc_frequency: int, docs_seen: int) -> float:
        return position_confidence(histogram, doc_frequency, docs_seen)
