from __future__ import annotations

TIER0 = 0
TIER1 = 1
TIER2 = 2
TIER3 = 3
TIER4 = 4

TIER_COST: dict[int, float] = {
    TIER0: 1.0,
    TIER1: 4.0,
    TIER2: 12.0,
    TIER3: 48.0,
    TIER4: 160.0,
}

_STAGE_TIER: dict[str, int] = {
    's1_fast_preprocess': TIER0,
    's2_fast_filter': TIER0,
    's2_doc_dedup': TIER1,
    's2_structural_filter': TIER1,
    's2_metadata': TIER1,
    's3_admission': TIER1,
    's3_intermediate': TIER2,
    's4_intel_preview': TIER3,
    's4_high_quality': TIER3,
    's5_final_validation': TIER3,
    's6_output': TIER1,
    'embed_dedup': TIER4,
    'knowledge_extraction': TIER4,
}


def stage_tier(stage: str) -> int:
    return _STAGE_TIER.get(stage, TIER3)
