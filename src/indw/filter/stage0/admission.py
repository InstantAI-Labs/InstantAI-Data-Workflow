from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from indw.schedule.config.policy import HUGE_SURVIVOR_CHARS, LARGE_SURVIVOR_CHARS

DOC_TIER_SMALL = 'small'
DOC_TIER_MEDIUM = 'medium'
DOC_TIER_LARGE = 'large'
DOC_TIER_HUGE = 'huge'

_MEDIUM_CHARS = 8_000


def classify_doc_tier(chars: int) -> str:
    if chars >= HUGE_SURVIVOR_CHARS:
        return DOC_TIER_HUGE
    if chars >= LARGE_SURVIVOR_CHARS:
        return DOC_TIER_LARGE
    if chars >= _MEDIUM_CHARS:
        return DOC_TIER_MEDIUM
    return DOC_TIER_SMALL


def estimate_processing_cost(*, chars: int, tier: str) -> float:
    base = {
        DOC_TIER_SMALL: 1.0,
        DOC_TIER_MEDIUM: 2.5,
        DOC_TIER_LARGE: 6.0,
        DOC_TIER_HUGE: 14.0,
    }.get(tier, 2.0)
    return base * (1.0 + min(chars, 200_000) / 120_000.0)


def heavy_pool_isolated(tier: str) -> bool:
    return tier in (DOC_TIER_LARGE, DOC_TIER_HUGE)


@dataclass
class AdmissionDecision:
    tier: str
    cost: float
    isolated: bool
    semantic_required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            'tier': self.tier,
            'cost': round(self.cost, 3),
            'isolated': self.isolated,
            'semantic_required': self.semantic_required,
        }


def evaluate_admission(
    *,
    meaningful_chars: int,
    language_reject: bool = False,
) -> AdmissionDecision:
    tier = classify_doc_tier(meaningful_chars)
    cost = estimate_processing_cost(chars=meaningful_chars, tier=tier)
    return AdmissionDecision(
        tier=tier,
        cost=cost,
        isolated=heavy_pool_isolated(tier),
        semantic_required=not language_reject,
    )
