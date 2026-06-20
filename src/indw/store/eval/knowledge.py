from __future__ import annotations

def compute_evaluation_knowledge_density(
    *,
    factual_density: float,
    educational_value: float,
    reasoning_density: float,
    synthetic_score: float,
    technical_fraction: float = 0.0,
) -> float:
    info = (
        0.30 * factual_density
        + 0.28 * educational_value
        + 0.27 * reasoning_density
        + 0.15 * technical_fraction
    )
    repetition_penalty = min(1.0, synthetic_score * 0.85)
    return max(0.0, min(1.0, info * (1.0 - repetition_penalty * 0.5)))
