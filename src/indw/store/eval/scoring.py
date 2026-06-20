from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from indw.store.eval.config import CorpusEvaluationConfig, ScoringWeights
from indw.store.eval.diversity import DiversityResult
from indw.store.eval.metrics import CorpusMetrics

@dataclass
class CorpusScoreResult:
    corpus_score: int = 0
    components: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'corpus_score': self.corpus_score,
            'components': {k: round(v, 4) for k, v in (self.components or {}).items()},
        }


def compute_corpus_score(
    metrics: CorpusMetrics,
    diversity: DiversityResult,
    *,
    config: CorpusEvaluationConfig,
) -> CorpusScoreResult:
    w: ScoringWeights = config.scoring_weights
    quality = metrics.quality_score * 100.0
    knowledge = metrics.knowledge_density * 100.0
    div = diversity.diversity_score * 100.0
    safety = (1.0 - metrics.toxicity_rate) * (1.0 - metrics.pii_rate) * 100.0
    dedup = (1.0 - metrics.duplicate_rate) * 100.0
    raw = (
        w.quality * quality
        + w.knowledge_density * knowledge
        + w.diversity * div
        + w.safety * safety
        + w.deduplication * dedup
    )
    total_w = w.quality + w.knowledge_density + w.diversity + w.safety + w.deduplication
    score = int(round(raw / max(total_w, 1e-9)))
    score = max(0, min(100, score))
    return CorpusScoreResult(
        corpus_score=score,
        components={
            'quality': quality,
            'knowledge_density': knowledge,
            'diversity': div,
            'safety': safety,
            'deduplication': dedup,
        },
    )
