from __future__ import annotations

import math
from dataclasses import dataclass

@dataclass
class ConfidenceEstimate:
    primary_language: str
    confidence: float
    primary_probability: float
    margin: float

def estimate_confidence(distribution: dict[str, float]) -> ConfidenceEstimate:
    if not distribution:
        return ConfidenceEstimate('und', 0.0, 0.0, 0.0)
    ranked = sorted(distribution.items(), key=lambda kv: -kv[1])
    primary, p1 = ranked[0]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = max(0.0, p1 - p2)
    calibrated = p1 * (0.5 + 0.5 * min(1.0, margin / 0.4))
    confidence = min(1.0, max(0.0, calibrated))
    return ConfidenceEstimate(
        primary_language=primary,
        confidence=confidence,
        primary_probability=p1,
        margin=margin,
    )

def fragmentation_score(distribution: dict[str, float], *, active_threshold: float = 0.08) -> float:
    if not distribution:
        return 1.0
    active = [p for p in distribution.values() if p >= active_threshold]
    if len(active) <= 1:
        return 0.0
    ent = 0.0
    total = sum(active) or 1.0
    for p in active:
        q = p / total
        if q > 0:
            ent -= q * math.log2(q)
    max_ent = math.log2(len(active))
    return min(1.0, ent / max(max_ent, 1e-9))
