from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from indw.store.eval.config import CorpusEvaluationConfig
from indw.store.eval.metrics import CorpusMetrics
from indw.tools.metrics.regression import _absolute_increase, _l1_dist, _relative_drop

@dataclass
class ComparisonSignal:
    kind: str
    direction: str
    metric: str
    current: float
    previous: float
    delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            'kind': self.kind,
            'direction': self.direction,
            'metric': self.metric,
            'current': round(self.current, 4),
            'previous': round(self.previous, 4),
            'delta': round(self.delta, 4),
        }


@dataclass
class CorpusComparison:
    improvements: list[ComparisonSignal] = field(default_factory=list)
    regressions: list[ComparisonSignal] = field(default_factory=list)
    language_drift: float = 0.0
    source_drift: float = 0.0
    skipped: bool = False
    mode: str = 'full'

    def to_dict(self) -> dict[str, Any]:
        payload = {
            'improvements': [s.to_dict() for s in self.improvements],
            'regressions': [s.to_dict() for s in self.regressions],
            'language_drift': round(self.language_drift, 4),
            'source_drift': round(self.source_drift, 4),
        }
        if self.skipped:
            payload['skipped'] = True
            payload['mode'] = self.mode
        return payload


def compare_versions(
    current: CorpusMetrics,
    previous: Optional[CorpusMetrics],
    *,
    config: CorpusEvaluationConfig,
) -> CorpusComparison:
    if previous is None:
        return CorpusComparison()
    th = config.comparison
    out = CorpusComparison()
    out.language_drift = _l1_dist(current.language_distribution, previous.language_distribution)
    out.source_drift = _l1_dist(current.source_distribution, previous.source_distribution)

    qual_delta = current.quality_score - previous.quality_score
    if qual_delta >= th.min_quality_improvement:
        out.improvements.append(
            ComparisonSignal('quality', 'improved', 'quality_score', current.quality_score, previous.quality_score, qual_delta)
        )
    elif _relative_drop(current.quality_score, previous.quality_score) > 0.05:
        out.regressions.append(
            ComparisonSignal('quality', 'regressed', 'quality_score', current.quality_score, previous.quality_score, -qual_delta)
        )

    dup_delta = previous.duplicate_rate - current.duplicate_rate
    if dup_delta > 0.005:
        out.improvements.append(
            ComparisonSignal('dedup', 'improved', 'duplicate_rate', current.duplicate_rate, previous.duplicate_rate, dup_delta)
        )
    dup_reg = _absolute_increase(current.duplicate_rate, previous.duplicate_rate)
    if dup_reg > th.max_duplicate_regression:
        out.regressions.append(
            ComparisonSignal('dedup', 'regressed', 'duplicate_rate', current.duplicate_rate, previous.duplicate_rate, -dup_reg)
        )

    tox_reg = _absolute_increase(current.toxicity_rate, previous.toxicity_rate)
    if tox_reg > th.max_toxicity_regression:
        out.regressions.append(
            ComparisonSignal('safety', 'regressed', 'toxicity_rate', current.toxicity_rate, previous.toxicity_rate, -tox_reg)
        )
    elif tox_reg < -0.002:
        out.improvements.append(
            ComparisonSignal('safety', 'improved', 'toxicity_rate', current.toxicity_rate, previous.toxicity_rate, -tox_reg)
        )

    pii_reg = _absolute_increase(current.pii_rate, previous.pii_rate)
    if pii_reg > th.max_pii_regression:
        out.regressions.append(
            ComparisonSignal('safety', 'regressed', 'pii_rate', current.pii_rate, previous.pii_rate, -pii_reg)
        )

    if out.language_drift > th.max_language_drift:
        out.regressions.append(
            ComparisonSignal('language', 'drift', 'language_distribution', out.language_drift, 0.0, out.language_drift)
        )
    if out.source_drift > th.max_source_drift:
        out.regressions.append(
            ComparisonSignal('source', 'drift', 'source_distribution', out.source_drift, 0.0, out.source_drift)
        )

    know_delta = current.knowledge_density - previous.knowledge_density
    if know_delta >= 0.03:
        out.improvements.append(
            ComparisonSignal('knowledge', 'improved', 'knowledge_density', current.knowledge_density, previous.knowledge_density, know_delta)
        )
    elif know_delta <= -0.05:
        out.regressions.append(
            ComparisonSignal('knowledge', 'regressed', 'knowledge_density', current.knowledge_density, previous.knowledge_density, know_delta)
        )

    return out
