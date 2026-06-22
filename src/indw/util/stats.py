from __future__ import annotations

import math

_WILSON_Z = 1.96


def wilson_ci(successes: int, n: int, z: float = _WILSON_Z) -> dict[str, float]:
    if n <= 0:
        return {'rate': 0.0, 'low': 0.0, 'high': 0.0}
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {'rate': p, 'low': max(0.0, center - margin), 'high': min(1.0, center + margin)}
