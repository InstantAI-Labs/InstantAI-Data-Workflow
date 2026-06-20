from __future__ import annotations

from typing import Any

from indw.filter.stage0.admission import evaluate_admission


def route_admission(*, meaningful_chars: int) -> dict[str, Any]:
    decision = evaluate_admission(meaningful_chars=meaningful_chars)
    return {
        'tier': decision.tier,
        'admission': decision.to_dict(),
        'semantic_required': decision.semantic_required,
        'isolated': decision.isolated,
        'cost': decision.cost,
    }


def admission_lane(tier: str) -> str:
    if tier == 'huge':
        return 'huge'
    if tier == 'large':
        return 'large'
    return 'normal'
