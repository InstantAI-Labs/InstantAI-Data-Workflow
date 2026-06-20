from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from indw.schedule.admission.tiers import TIER0, TIER1, TIER2, TIER3, TIER4, stage_tier

COMMODITY = 'commodity'
INTELLIGENCE = 'intelligence'


@dataclass(frozen=True)
class StageClass:
    stage: str
    kind: str
    owner: str
    tier: int
    library: str = ''
    notes: str = ''


_REGISTRY: tuple[StageClass, ...] = (
    StageClass('s1_fast_preprocess', COMMODITY, 'data.schedule.stages.pools.preprocess', TIER0, 'orjson', 'JSONL parse + minimal normalize'),
    StageClass('s2_fast_filter', COMMODITY, 'data.schedule.stages.pools.filter', TIER0, '', 'size gates'),
    StageClass('s2_doc_dedup', COMMODITY, 'data.dedup.service.exact_shard', TIER1, 'sqlite3+tenacity', 'exact hash dedup'),
    StageClass('s2_structural_filter', INTELLIGENCE, 'data.filter.stage0.engine', TIER1, '', 'Stage0 policy — unique'),
    StageClass('s2_metadata', COMMODITY, 'data.schedule.read.ingest', TIER1, '', 'metadata validation'),
    StageClass('s3_admission', INTELLIGENCE, 'data.filter.stage0.admission', TIER1, '', 'doc tier routing'),
    StageClass('language_gate', COMMODITY, 'data.filter.language.detect', TIER1, 'langid', 'early language reject'),
    StageClass('s3_intermediate', INTELLIGENCE, 'data.schedule.intel.pci', TIER2, '', 'PCI fingerprints'),
    StageClass('s4_intel_preview', INTELLIGENCE, 'data.schedule.intel.session', TIER3, 'sqlite3', 'ACIM + LCI preview'),
    StageClass('s4_high_quality', INTELLIGENCE, 'data.clean.corpus', TIER3, 'trafilatura', 'semantic clean + gate'),
    StageClass('s5_final_validation', INTELLIGENCE, 'data.filter.gate.quality', TIER3, '', 'score + calibrate'),
    StageClass('s6_output', COMMODITY, 'data.schedule.apply.merge', TIER1, 'orjson', 'sink write + chunk dedup'),
    StageClass('embed_dedup', COMMODITY, 'data.dedup.embed.pipeline', TIER4, 'sentence-transformers', 'vector near-dedup'),
    StageClass('knowledge_extraction', INTELLIGENCE, 'data.extract.core.units', TIER4, '', 'KE + publication recovery'),
    StageClass('html_extract', COMMODITY, 'data.clean.document.html', TIER0, 'trafilatura', 'boilerplate removal'),
    StageClass('fuzzy_dedup', COMMODITY, 'data.dedup.fuzzy', TIER1, 'datasketch', 'MinHash LSH at apply'),
    StageClass('semantic_dedup', COMMODITY, 'data.dedup.semantic', TIER1, '', 'SimHash at apply'),
    StageClass('curriculum_mix', INTELLIGENCE, 'data.schedule.mix.curriculum', TIER0, '', 'source interleaving'),
)


def classify_stage(name: str) -> StageClass | None:
    for row in _REGISTRY:
        if row.stage == name:
            return row
    tier = stage_tier(name)
    kind = INTELLIGENCE if tier >= TIER2 else COMMODITY
    return StageClass(name, kind, '', tier)


def commodity_stages() -> list[StageClass]:
    return [r for r in _REGISTRY if r.kind == COMMODITY]


def intelligence_stages() -> list[StageClass]:
    return [r for r in _REGISTRY if r.kind == INTELLIGENCE]


def classification_summary() -> dict[str, Any]:
    return {
        'commodity': [r.stage for r in commodity_stages()],
        'intelligence': [r.stage for r in intelligence_stages()],
        'registry': [
            {
                'stage': r.stage,
                'kind': r.kind,
                'owner': r.owner,
                'tier': r.tier,
                'library': r.library,
                'notes': r.notes,
            }
            for r in _REGISTRY
        ],
    }
