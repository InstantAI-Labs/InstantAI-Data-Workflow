from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

def _entropy(dist: dict[str, float]) -> float:
    if not dist:
        return 0.0
    ent = 0.0
    for p in dist.values():
        if p > 0:
            ent -= p * math.log2(p)
    max_ent = math.log2(len(dist)) if len(dist) > 1 else 1.0
    return min(1.0, ent / max(max_ent, 1e-9))

@dataclass
class DiversityResult:
    diversity_score: float = 0.0
    language_diversity: float = 0.0
    source_diversity: float = 0.0
    domain_diversity: float = 0.0
    topic_diversity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'diversity_score': round(self.diversity_score, 4),
            'language_diversity': round(self.language_diversity, 4),
            'source_diversity': round(self.source_diversity, 4),
            'domain_diversity': round(self.domain_diversity, 4),
            'topic_diversity': round(self.topic_diversity, 4),
        }

def compute_diversity(
    *,
    language_distribution: dict[str, float],
    source_distribution: dict[str, float],
    domain_distribution: dict[str, float],
) -> DiversityResult:
    lang = _entropy(language_distribution)
    src = _entropy(source_distribution)
    dom = _entropy(domain_distribution)
    topic = dom * 0.65 + lang * 0.35
    score = 0.35 * lang + 0.30 * src + 0.25 * dom + 0.10 * topic
    return DiversityResult(
        diversity_score=min(1.0, score),
        language_diversity=lang,
        source_diversity=src,
        domain_diversity=dom,
        topic_diversity=topic,
    )
