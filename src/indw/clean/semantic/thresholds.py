from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

@dataclass
class CorpusThresholdCalibrator:
    warmup: int = 200
    reservoir_size: int = 5000
    _utilities: deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    _noise: deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    _n: int = 0

    def observe(self, utility: float, noise: float) -> None:
        self._utilities.append(utility)
        self._noise.append(noise)
        self._n += 1

    def _percentile(self, values: deque[float], pct: float, default: float) -> float:
        if not values:
            return default
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(len(ordered) * pct)))
        return ordered[idx]

    def remove_threshold(self) -> float:
        if self._n < self.warmup:
            return 0.18
        return self._percentile(self._utilities, 0.22, 0.18)

    def downweight_threshold(self) -> float:
        if self._n < self.warmup:
            return 0.32
        return self._percentile(self._utilities, 0.38, 0.32)

    def noise_ceiling(self) -> float:
        if self._n < self.warmup:
            return 0.62
        return self._percentile(self._noise, 0.72, 0.62)

    def snapshot(self) -> dict[str, float]:
        return {
            'observations': float(self._n),
            'remove_threshold': round(self.remove_threshold(), 4),
            'downweight_threshold': round(self.downweight_threshold(), 4),
            'noise_ceiling': round(self.noise_ceiling(), 4),
        }

_GLOBAL_CALIBRATOR = CorpusThresholdCalibrator()

def get_threshold_calibrator() -> CorpusThresholdCalibrator:
    return _GLOBAL_CALIBRATOR
