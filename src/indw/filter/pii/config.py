from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from functools import lru_cache
from typing import Any, Optional

from orchestration.resolver.refs import ConfigRef
from orchestration.resolver.resolver import Resolver

DEFAULT_PII_SPEC = 'data/safety/pii'
DEFAULT_ENTITY_TYPES = (
    'PERSON',
    'ORGANIZATION',
    'ADDRESS',
    'EMAIL',
    'PHONE',
    'ACCOUNT_ID',
    'FINANCIAL_ID',
    'CREDENTIAL',
)

@dataclass
class PiiThresholds:
    accept: float = 0.20
    redact: float = 0.60
    reject: float = 0.90
    hard_reject: float = 0.95

    def band(self, score: float) -> str:
        if score >= self.hard_reject:
            return 'hard_reject'
        if score > self.redact:
            return 'reject'
        if score > self.accept:
            return 'redact'
        return 'accept'

@dataclass
class NerConfig:
    enabled: bool = False

@dataclass
class SecretConfig:
    min_entropy: float = 3.2
    min_token_len: int = 12
    max_token_len: int = 512
    min_secret_score: float = 0.55

@dataclass
class PiiContextConfig:
    enabled: bool = True
    example_dampening: float = 0.78
    documentation_dampening: float = 0.72
    min_confidence: float = 0.40

@dataclass
class PiiPolicyConfig:
    enabled: bool = True
    entity_types: tuple[str, ...] = DEFAULT_ENTITY_TYPES
    scoring_weights: dict[str, float] = field(
        default_factory=lambda: {
            'entities': 0.40,
            'secrets': 0.38,
            'context': 0.22,
        }
    )
    thresholds: PiiThresholds = field(default_factory=PiiThresholds)
    ner: NerConfig = field(default_factory=NerConfig)
    secrets: SecretConfig = field(default_factory=SecretConfig)
    context: PiiContextConfig = field(default_factory=PiiContextConfig)
    redaction_enabled: bool = True
    reporting_output_dir: str = 'artifacts/data/pii'
    max_pii_score: Optional[float] = None
    validation_min_detection_rate: float = 0.95
    validation_max_false_positive_rate: float = 0.03
    validation_max_false_negative_rate: float = 0.03

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> PiiPolicyConfig:
        if not raw:
            return cls()
        th = raw.get('thresholds') or {}
        ner = raw.get('ner') or {}
        sec = raw.get('secrets') or {}
        ctx = raw.get('context') or {}
        red = raw.get('redaction') or {}
        reporting = raw.get('reporting') or {}
        validation = raw.get('validation') or {}
        types = raw.get('entity_types') or list(DEFAULT_ENTITY_TYPES)
        return cls(
            enabled=bool(raw.get('enabled', True)),
            entity_types=tuple(str(t) for t in types),
            scoring_weights=dict(raw.get('scoring_weights') or cls().scoring_weights),
            thresholds=PiiThresholds(
                accept=float(th.get('accept', 0.20)),
                redact=float(th.get('redact', 0.60)),
                reject=float(th.get('reject', 0.90)),
                hard_reject=float(th.get('hard_reject', 0.95)),
            ),
            ner=NerConfig(
                enabled=bool(ner.get('enabled', False)),
            ),
            secrets=SecretConfig(
                min_entropy=float(sec.get('min_entropy', 3.2)),
                min_token_len=int(sec.get('min_token_len', 12)),
                max_token_len=int(sec.get('max_token_len', 512)),
                min_secret_score=float(sec.get('min_secret_score', 0.55)),
            ),
            context=PiiContextConfig(
                enabled=bool(ctx.get('enabled', True)),
                example_dampening=float(ctx.get('example_dampening', 0.78)),
                documentation_dampening=float(ctx.get('documentation_dampening', 0.72)),
                min_confidence=float(ctx.get('min_confidence', 0.40)),
            ),
            redaction_enabled=bool(red.get('enabled', True)),
            reporting_output_dir=str(reporting.get('output_dir', 'artifacts/data/pii')),
            max_pii_score=float(raw['max_pii_score']) if raw.get('max_pii_score') is not None else None,
            validation_min_detection_rate=float(validation.get('min_detection_rate', 0.95)),
            validation_max_false_positive_rate=float(validation.get('max_false_positive_rate', 0.03)),
            validation_max_false_negative_rate=float(validation.get('max_false_negative_rate', 0.03)),
        )

    @classmethod
    def resolve(cls, spec: Optional[str] = None) -> PiiPolicyConfig:
        return deepcopy(_resolve_pii_cached(spec or DEFAULT_PII_SPEC))

@lru_cache(maxsize=8)
def _resolve_pii_cached(spec: str) -> PiiPolicyConfig:
    resolved = Resolver.default().resolve(ConfigRef(kind='safety', id=spec))
    return PiiPolicyConfig.from_dict(dict(resolved.raw))
