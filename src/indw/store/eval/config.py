from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from functools import lru_cache
from typing import Any, Optional

from indw.config.loader import ConfigRef, Resolver, thaw

DEFAULT_EVALUATION_SPEC = 'corpus/evaluation'

@dataclass
class PromotionPolicy:
    promote_min_score: int = 90
    review_min_score: int = 80

@dataclass
class ScoringWeights:
    quality: float = 0.32
    knowledge_density: float = 0.20
    diversity: float = 0.18
    safety: float = 0.15
    deduplication: float = 0.15

@dataclass
class ComparisonThresholds:
    min_quality_improvement: float = 0.02
    max_duplicate_regression: float = 0.03
    max_toxicity_regression: float = 0.01
    max_pii_regression: float = 0.005
    max_language_drift: float = 0.12
    max_source_drift: float = 0.18

@dataclass
class CorpusEvaluationConfig:
    enabled: bool = True
    lightweight: bool = False
    output_dir: str = 'artifacts/data/corpus_evaluation'
    promotion: PromotionPolicy = field(default_factory=PromotionPolicy)
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    comparison: ComparisonThresholds = field(default_factory=ComparisonThresholds)
    validation_min_promotion_accuracy: float = 0.95
    validation_max_false_promotion_rate: float = 0.05

    @classmethod
    def resolve(cls, spec: str = DEFAULT_EVALUATION_SPEC) -> CorpusEvaluationConfig:
        return deepcopy(_resolve_corpus_eval_cached(spec))

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> CorpusEvaluationConfig:
        if not raw:
            return cls()
        prom = raw.get('promotion') or {}
        weights = raw.get('scoring_weights') or {}
        cmp = raw.get('comparison') or {}
        val = raw.get('validation') or {}
        return cls(
            enabled=bool(raw.get('enabled', True)),
            lightweight=bool(raw.get('lightweight', False)),
            output_dir=str(raw.get('output_dir', 'artifacts/data/corpus_evaluation')),
            promotion=PromotionPolicy(
                promote_min_score=int(prom.get('promote_min_score', 90)),
                review_min_score=int(prom.get('review_min_score', 80)),
            ),
            scoring_weights=ScoringWeights(
                quality=float(weights.get('quality', 0.32)),
                knowledge_density=float(weights.get('knowledge_density', 0.20)),
                diversity=float(weights.get('diversity', 0.18)),
                safety=float(weights.get('safety', 0.15)),
                deduplication=float(weights.get('deduplication', 0.15)),
            ),
            comparison=ComparisonThresholds(
                min_quality_improvement=float(cmp.get('min_quality_improvement', 0.02)),
                max_duplicate_regression=float(cmp.get('max_duplicate_regression', 0.03)),
                max_toxicity_regression=float(cmp.get('max_toxicity_regression', 0.01)),
                max_pii_regression=float(cmp.get('max_pii_regression', 0.005)),
                max_language_drift=float(cmp.get('max_language_drift', 0.12)),
                max_source_drift=float(cmp.get('max_source_drift', 0.18)),
            ),
            validation_min_promotion_accuracy=float(val.get('min_promotion_accuracy', 0.95)),
            validation_max_false_promotion_rate=float(val.get('max_false_promotion_rate', 0.05)),
        )

@lru_cache(maxsize=8)
def _resolve_corpus_eval_cached(spec: str) -> CorpusEvaluationConfig:
    resolved = Resolver.default().resolve(ConfigRef(kind='corpus', id=spec))
    return CorpusEvaluationConfig.from_dict(thaw(resolved.raw))
