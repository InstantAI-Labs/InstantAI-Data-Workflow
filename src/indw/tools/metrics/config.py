from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from functools import lru_cache
from typing import Any, Optional

from indw.config.loader import ConfigRef, Resolver, thaw

DEFAULT_OBSERVABILITY_SPEC = 'observability/default'

@dataclass
class ObservabilityThresholds:
    duplicate_rate_increase: float = 0.05
    quality_score_decrease: float = 0.05
    toxicity_rate_increase: float = 0.02
    pii_rate_increase: float = 0.01
    language_mix_l1: float = 0.15
    source_imbalance_l1: float = 0.20

@dataclass
class ObservabilitySeverityConfig:
    critical_regressions: int = 2
    warning_relative_delta: float = 0.08

@dataclass
class ObservabilityPolicyConfig:
    enabled: bool = True
    output_dir: str = 'artifacts/data/observability'
    thresholds: ObservabilityThresholds = field(default_factory=ObservabilityThresholds)
    severity: ObservabilitySeverityConfig = field(default_factory=ObservabilitySeverityConfig)
    validation_min_regression_detection_accuracy: float = 0.95
    validation_max_false_alert_rate: float = 0.05

    @classmethod
    def resolve(cls, spec: str = DEFAULT_OBSERVABILITY_SPEC) -> ObservabilityPolicyConfig:
        return deepcopy(_resolve_observability_cached(spec))

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> ObservabilityPolicyConfig:
        if not raw:
            return cls()
        th = raw.get('thresholds') or {}
        sev = raw.get('severity') or {}
        val = raw.get('validation') or {}
        return cls(
            enabled=bool(raw.get('enabled', True)),
            output_dir=str(raw.get('output_dir', 'artifacts/data/observability')),
            thresholds=ObservabilityThresholds(
                duplicate_rate_increase=float(th.get('duplicate_rate_increase', 0.05)),
                quality_score_decrease=float(th.get('quality_score_decrease', 0.05)),
                toxicity_rate_increase=float(th.get('toxicity_rate_increase', 0.02)),
                pii_rate_increase=float(th.get('pii_rate_increase', 0.01)),
                language_mix_l1=float(th.get('language_mix_l1', 0.15)),
                source_imbalance_l1=float(th.get('source_imbalance_l1', 0.20)),
            ),
            severity=ObservabilitySeverityConfig(
                critical_regressions=int(sev.get('critical_regressions', 2)),
                warning_relative_delta=float(sev.get('warning_relative_delta', 0.08)),
            ),
            validation_min_regression_detection_accuracy=float(
                val.get('min_regression_detection_accuracy', 0.95)
            ),
            validation_max_false_alert_rate=float(val.get('max_false_alert_rate', 0.05)),
        )

@lru_cache(maxsize=8)
def _resolve_observability_cached(spec: str) -> ObservabilityPolicyConfig:
    resolved = Resolver.default().resolve(ConfigRef(kind='observability', id=spec))
    return ObservabilityPolicyConfig.from_dict(thaw(resolved.raw))
