from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from indw.tools.metrics.config import ObservabilityPolicyConfig, ObservabilityThresholds
from indw.tools.metrics.snapshot import CorpusSnapshot

def _l1_dist(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)

def _relative_drop(current: float, previous: float) -> float:
    if previous <= 0:
        return 0.0
    return max(0.0, (previous - current) / previous)

def _absolute_increase(current: float, previous: float) -> float:
    return max(0.0, current - previous)

@dataclass
class RegressionSignal:
    reason: str
    severity: str
    current: float
    previous: float
    delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            'reason': self.reason,
            'severity': self.severity,
            'current': round(self.current, 4),
            'previous': round(self.previous, 4),
            'delta': round(self.delta, 4),
        }

@dataclass
class RegressionResult:
    regression_detected: bool = False
    reason: Optional[str] = None
    signals: list[RegressionSignal] = field(default_factory=list)
    severity: str = 'INFO'

    def to_dict(self) -> dict[str, Any]:
        return {
            'regression_detected': self.regression_detected,
            'reason': self.reason,
            'severity': self.severity,
            'signals': [s.to_dict() for s in self.signals],
        }

def analyze_regression(
    current: CorpusSnapshot,
    previous: Optional[CorpusSnapshot],
    *,
    policy: Optional[ObservabilityPolicyConfig] = None,
) -> RegressionResult:
    if previous is None:
        return RegressionResult(regression_detected=False, severity='INFO')
    th = (policy or ObservabilityPolicyConfig()).thresholds
    sev_cfg = (policy or ObservabilityPolicyConfig()).severity
    signals: list[RegressionSignal] = []

    dup_delta = _absolute_increase(current.duplicate_rate, previous.duplicate_rate)
    if dup_delta > th.duplicate_rate_increase:
        signals.append(
            RegressionSignal(
                'duplicate_rate_increase',
                'WARNING',
                current.duplicate_rate,
                previous.duplicate_rate,
                dup_delta,
            )
        )

    qual_drop = _relative_drop(current.quality_score_mean, previous.quality_score_mean)
    if qual_drop > th.quality_score_decrease:
        signals.append(
            RegressionSignal(
                'quality_score_decrease',
                'WARNING',
                current.quality_score_mean,
                previous.quality_score_mean,
                qual_drop,
            )
        )

    tox_delta = _absolute_increase(current.toxicity_rate, previous.toxicity_rate)
    if tox_delta > th.toxicity_rate_increase:
        signals.append(
            RegressionSignal(
                'toxicity_rate_increase',
                'WARNING',
                current.toxicity_rate,
                previous.toxicity_rate,
                tox_delta,
            )
        )

    pii_delta = _absolute_increase(current.pii_rate, previous.pii_rate)
    if pii_delta > th.pii_rate_increase:
        signals.append(
            RegressionSignal(
                'pii_rate_increase',
                'WARNING',
                current.pii_rate,
                previous.pii_rate,
                pii_delta,
            )
        )

    lang_l1 = _l1_dist(current.language_distribution, previous.language_distribution)
    if lang_l1 > th.language_mix_l1:
        signals.append(
            RegressionSignal(
                'language_drift',
                'WARNING',
                lang_l1,
                0.0,
                lang_l1,
            )
        )

    src_l1 = _l1_dist(current.source_distribution, previous.source_distribution)
    if src_l1 > th.source_imbalance_l1:
        signals.append(
            RegressionSignal(
                'source_imbalance',
                'INFO',
                src_l1,
                0.0,
                src_l1,
            )
        )

    for sig in signals:
        if sig.delta >= sev_cfg.warning_relative_delta and sig.reason != 'source_imbalance':
            sig.severity = 'WARNING'
    if len(signals) >= sev_cfg.critical_regressions:
        for sig in signals:
            if sig.reason in (
                'duplicate_rate_increase',
                'quality_score_decrease',
                'toxicity_rate_increase',
                'pii_rate_increase',
            ):
                sig.severity = 'CRITICAL'

    detected = len(signals) > 0
    primary = signals[0].reason if signals else None
    overall = 'INFO'
    if any(s.severity == 'CRITICAL' for s in signals):
        overall = 'CRITICAL'
    elif any(s.severity == 'WARNING' for s in signals):
        overall = 'WARNING'
    return RegressionResult(
        regression_detected=detected,
        reason=primary,
        signals=signals,
        severity=overall,
    )

def compare_corpora(a: CorpusSnapshot, b: CorpusSnapshot) -> dict[str, Any]:
    return {
        'quality_difference': {
            'score_mean': round(a.quality_score_mean - b.quality_score_mean, 4),
            'score_p10': round(a.quality_score_p10 - b.quality_score_p10, 4),
        },
        'duplication_difference': {
            'duplicate_rate': round(a.duplicate_rate - b.duplicate_rate, 4),
        },
        'language_difference': {
            'l1_distance': round(_l1_dist(a.language_distribution, b.language_distribution), 4),
            'delta': {
                k: round(a.language_distribution.get(k, 0.0) - b.language_distribution.get(k, 0.0), 4)
                for k in set(a.language_distribution) | set(b.language_distribution)
            },
        },
        'source_difference': {
            'l1_distance': round(_l1_dist(a.source_distribution, b.source_distribution), 4),
            'delta': {
                k: round(a.source_distribution.get(k, 0.0) - b.source_distribution.get(k, 0.0), 4)
                for k in set(a.source_distribution) | set(b.source_distribution)
            },
        },
        'safety_difference': {
            'toxicity_rate': round(a.toxicity_rate - b.toxicity_rate, 4),
            'pii_rate': round(a.pii_rate - b.pii_rate, 4),
        },
    }
