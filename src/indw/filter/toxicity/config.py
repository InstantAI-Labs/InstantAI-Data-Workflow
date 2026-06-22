from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from indw.config.loader import ConfigRef, Resolver, thaw

DEFAULT_TOXICITY_SPEC = 'safety/toxicity'
DEFAULT_CATEGORIES = (
    'hate',
    'harassment',
    'violence',
    'extremism',
    'self_harm',
    'sexual_abuse',
)

@dataclass
class RejectionPolicy:
    accept_max: float = 0.20
    review_max: float = 0.60
    reject_max: float = 0.80
    hard_reject_min: float = 0.95

    @classmethod
    def from_thresholds(cls, th: dict[str, Any]) -> RejectionPolicy:
        return cls(
            accept_max=float(th.get('accept', 0.20)),
            review_max=float(th.get('review', 0.60)),
            reject_max=float(th.get('reject', 0.80)),
            hard_reject_min=float(th.get('hard_reject', 0.95)),
        )

    def band(self, score: float) -> str:
        if score >= self.hard_reject_min:
            return 'hard_reject'
        if score > self.review_max:
            return 'reject'
        if score > self.accept_max:
            return 'review'
        return 'accept'

@dataclass
class ClassifierConfig:
    enabled: bool = False

@dataclass
class ContextConfig:
    enabled: bool = True
    educational_dampening: float = 0.75
    min_confidence: float = 0.42

@dataclass
class ToxicityPolicyConfig:
    enabled: bool = True
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    scoring_weights: dict[str, float] = field(
        default_factory=lambda: {
            'classifier': 0.0,
            'rule': 0.45,
            'pattern': 0.45,
            'context': 0.10,
        }
    )
    rejection: RejectionPolicy = field(default_factory=RejectionPolicy)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    fast_rules: dict[str, Any] = field(default_factory=dict)
    reporting_output_dir: str = 'artifacts/data/toxicity'
    max_toxicity_score: Optional[float] = None
    validation_min_detection_rate: float = 0.95
    validation_max_false_positive_rate: float = 0.05
    validation_max_false_negative_rate: float = 0.05
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def educational_dampening(self) -> float:
        return self.context.educational_dampening

    @property
    def educational_min_confidence(self) -> float:
        return self.context.min_confidence

    @property
    def ml(self) -> ClassifierConfig:
        return self.classifier

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> ToxicityPolicyConfig:
        if not raw:
            return cls()
        th = raw.get('thresholds') or {}
        clf = raw.get('classifier') or {}
        ctx = raw.get('context') or {}
        reporting = raw.get('reporting') or {}
        validation = raw.get('validation') or {}
        cats = raw.get('categories') or list(DEFAULT_CATEGORIES)
        weights = dict(raw.get('scoring_weights') or cls().scoring_weights)
        if 'ml' in weights and 'classifier' not in weights:
            weights['classifier'] = weights.pop('ml')
        return cls(
            enabled=bool(raw.get('enabled', True)),
            categories=tuple(str(c) for c in cats),
            scoring_weights=weights,
            rejection=RejectionPolicy.from_thresholds(th),
            classifier=ClassifierConfig(
                enabled=bool(clf.get('enabled', False)),
            ),
            context=ContextConfig(
                enabled=bool(ctx.get('enabled', True)),
                educational_dampening=float(ctx.get('educational_dampening', 0.75)),
                min_confidence=float(ctx.get('min_confidence', 0.42)),
            ),
            fast_rules=dict(raw.get('fast_rules') or {}),
            reporting_output_dir=str(reporting.get('output_dir', 'artifacts/data/toxicity')),
            max_toxicity_score=(
                float(raw['max_toxicity_score']) if raw.get('max_toxicity_score') is not None else None
            ),
            validation_min_detection_rate=float(validation.get('min_detection_rate', 0.95)),
            validation_max_false_positive_rate=float(validation.get('max_false_positive_rate', 0.05)),
            validation_max_false_negative_rate=float(validation.get('max_false_negative_rate', 0.05)),
            _raw=thaw(raw),
        )

    @classmethod
    def resolve(cls, spec: Optional[str] = None) -> ToxicityPolicyConfig:
        cached = _resolve_toxicity_cached(spec or DEFAULT_TOXICITY_SPEC)
        return cls.from_dict(thaw(cached._raw))

    def rules_raw(self) -> dict[str, Any]:
        return {'fast_rules': self.fast_rules, **self._raw}

@lru_cache(maxsize=8)
def _resolve_toxicity_cached(spec: str) -> ToxicityPolicyConfig:
    resolved = Resolver.default().resolve(ConfigRef(kind='safety', id=spec))
    return ToxicityPolicyConfig.from_dict(thaw(resolved.raw))
