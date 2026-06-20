from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indw.tools.metrics.regression import RegressionResult, RegressionSignal

@dataclass
class Alert:
    severity: str
    message: str
    reason: str
    metric: str
    current: float
    previous: float

    def to_dict(self) -> dict[str, Any]:
        return {
            'severity': self.severity,
            'message': self.message,
            'reason': self.reason,
            'metric': self.metric,
            'current': round(self.current, 4),
            'previous': round(self.previous, 4),
        }

def alerts_from_regression(result: RegressionResult) -> list[Alert]:
    alerts: list[Alert] = []
    for sig in result.signals:
        msg = _message_for(sig)
        alerts.append(
            Alert(
                severity=sig.severity,
                message=msg,
                reason=sig.reason,
                metric=sig.reason,
                current=sig.current,
                previous=sig.previous,
            )
        )
    return alerts

def _message_for(sig: RegressionSignal) -> str:
    templates = {
        'duplicate_rate_increase': 'Duplicate rate increased from {previous:.2%} to {current:.2%}',
        'quality_score_decrease': 'Mean quality score dropped from {previous:.3f} to {current:.3f}',
        'toxicity_rate_increase': 'Toxicity rejection rate increased from {previous:.2%} to {current:.2%}',
        'pii_rate_increase': 'PII rejection rate increased from {previous:.2%} to {current:.2%}',
        'language_drift': 'Language mix drift detected (L1={current:.3f})',
        'source_imbalance': 'Source distribution imbalance detected (L1={current:.3f})',
    }
    tpl = templates.get(sig.reason, 'Regression: {reason}')
    return tpl.format(
        reason=sig.reason,
        current=sig.current,
        previous=sig.previous,
    )
