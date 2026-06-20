from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdaptiveCalibrationConfig:
    enabled: bool = True
    warmup_samples: int = 200
    reservoir_size: int = 10000
    recent_window: int = 512
    downrank_anchor_percentile: float = 30.0


def _percentile_value(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    xs = sorted(samples)
    pos = p * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _std_dev(samples: list[float]) -> float:
    n = len(samples)
    if n < 2:
        return 0.0
    mean = sum(samples) / n
    return (sum((s - mean) ** 2 for s in samples) / (n - 1)) ** 0.5


@dataclass
class AdaptiveCalibrator:
    config: AdaptiveCalibrationConfig = field(default_factory=AdaptiveCalibrationConfig)
    _composite_samples: list[float] = field(default_factory=list)
    _q10_samples: list[float] = field(default_factory=list)
    _recent_composite: list[float] = field(default_factory=list)
    _total_observed: int = 0
    _rng: random.Random = field(default_factory=random.Random)

    def record(self, composite: float, q10: float) -> None:
        self._total_observed += 1
        cap = self.config.reservoir_size
        if len(self._composite_samples) < cap:
            self._composite_samples.append(composite)
            self._q10_samples.append(q10)
        else:
            j = self._rng.randint(0, self._total_observed - 1)
            if j < cap:
                self._composite_samples[j] = composite
                self._q10_samples[j] = q10

        win = max(64, self.config.recent_window)
        self._recent_composite.append(composite)
        if len(self._recent_composite) > win:
            self._recent_composite.pop(0)

    @property
    def ready(self) -> bool:
        return self._total_observed >= self.config.warmup_samples

    def _percentile_rank(self, value: float, samples: list[float]) -> float:
        if not samples:
            return 0.5
        below = sum(1 for s in samples if s < value)
        equal = sum(1 for s in samples if s == value)
        return (below + 0.5 * equal) / len(samples)

    def composite_percentile(self, score: float) -> float:
        return self._percentile_rank(score, self._composite_samples)

    def q10_percentile(self, score: float) -> float:
        return self._percentile_rank(score, self._q10_samples)

    def downrank_weight(
        self,
        composite: float,
        q10: float,
        *,
        issue_count: int = 0,
        signal_penalty: float = 0.0,
        near_duplicate: bool = False,
    ) -> float:
        if self.config.enabled and self.ready:
            pct = self.q10_percentile(q10)
            anchor = self.config.downrank_anchor_percentile / 100.0
            relative = max(0.0, min(1.0, pct / max(anchor, 0.05)))
            base = 0.35 + 0.65 * relative
        else:
            base = max(0.35, min(1.0, q10 / 10.0))

        if issue_count:
            base *= max(0.5, 1.0 - 0.08 * issue_count)
        if signal_penalty > 0:
            base *= max(0.4, 1.0 - signal_penalty)
        if near_duplicate:
            base *= 0.55
        return round(max(0.15, min(1.0, base)), 3)

    def signal_confidence(self, signals: dict[str, bool], doc_signals: Any) -> float:
        penalty = 0.0
        if signals.get('boilerplate'):
            penalty += min(0.3, getattr(doc_signals, 'boilerplate_score', 0.0))
        if signals.get('spam'):
            penalty += min(0.35, max(
                getattr(doc_signals, 'seo_spam_score', 0.0),
                getattr(doc_signals, 'commercial_score', 0.0),
            ))
        if signals.get('low_information'):
            penalty += min(0.25, getattr(doc_signals, 'low_information_score', 0.0))
        if signals.get('invalid_code'):
            penalty += 0.4
        if signals.get('ai_verbosity'):
            penalty += min(0.25, getattr(doc_signals, 'ai_verbosity_score', 0.0))
        if signals.get('hallucination_risk'):
            penalty += min(0.30, getattr(doc_signals, 'hallucination_risk_score', 0.0))
        if getattr(doc_signals, 'template_synthetic_score', 0.0) > 0.35:
            penalty += min(0.20, doc_signals.template_synthetic_score * 0.4)
        return min(1.0, penalty)

    def _drift_stats(self, comp: list[float]) -> dict[str, Any]:
        n = len(comp)
        if n < 64 or len(self._recent_composite) < 32:
            return {'drift_ready': False}
        reservoir_mean = sum(comp) / n
        recent = self._recent_composite
        recent_mean = sum(recent) / len(recent)
        std = _std_dev(comp)
        shift = recent_mean - reservoir_mean
        threshold = max(0.005, std * (len(recent) ** -0.5) * 2.0)
        return {
            'drift_ready': True,
            'recent_mean': round(recent_mean, 4),
            'reservoir_mean': round(reservoir_mean, 4),
            'mean_shift': round(shift, 6),
            'drift_significant': abs(shift) > threshold,
            'drift_threshold': round(threshold, 6),
        }

    def distribution_stats(self) -> dict[str, Any]:
        comp = list(self._composite_samples)
        q10 = list(self._q10_samples)
        n = len(comp)

        def _pct(samples: list[float], p: float) -> float:
            return round(_percentile_value(samples, p), 4)

        pct_keys = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
        comp_pct = {f'composite_p{int(p * 100)}': _pct(comp, p) for p in pct_keys}
        q10_pct = {f'q10_p{int(p * 100)}': _pct(q10, p) for p in pct_keys}

        variance = 0.0
        std = 0.0
        if n > 1:
            mean = sum(comp) / n
            variance = sum((s - mean) ** 2 for s in comp) / (n - 1)
            std = variance ** 0.5

        q10_std = _std_dev(q10) if len(q10) > 1 else 0.0
        flat = n > 10 and std < 0.001

        out: dict[str, Any] = {
            'observed': self._total_observed,
            'reservoir_size': n,
            'ready': self.ready,
            'composite_mean': round(sum(comp) / n, 4) if n else 0.0,
            'composite_std': round(std, 6),
            'score_variance': round(variance, 6),
            'q10_mean': round(sum(q10) / len(q10), 4) if q10 else 0.0,
            'q10_std': round(q10_std, 6),
            'distribution_flat': flat,
            **comp_pct,
            **q10_pct,
        }
        out.update(self._drift_stats(comp))
        return out

    def export_state(self) -> dict[str, Any]:
        return {
            'composite_samples': list(self._composite_samples),
            'q10_samples': list(self._q10_samples),
            'recent_composite': list(self._recent_composite),
            'total_observed': int(self._total_observed),
            'rng_state': self._rng.getstate(),
        }

    def import_state(self, data: dict[str, Any] | None) -> None:
        if not data:
            return
        self._composite_samples = [float(x) for x in (data.get('composite_samples') or [])]
        self._q10_samples = [float(x) for x in (data.get('q10_samples') or [])]
        self._recent_composite = [float(x) for x in (data.get('recent_composite') or [])]
        self._total_observed = int(data.get('total_observed', 0))
        rng_state = data.get('rng_state')
        if rng_state is not None:
            if isinstance(rng_state, list):
                rng_state = tuple(
                    tuple(part) if isinstance(part, list) else part
                    for part in rng_state
                )
            self._rng.setstate(rng_state)
