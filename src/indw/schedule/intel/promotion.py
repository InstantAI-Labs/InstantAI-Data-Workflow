from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from indw.config.defaults import (
    LCI_DOMAIN_PROMOTE_MIN_OBS,
    LCI_FAMILY_PROMOTE_MIN_OBS,
    LCI_GENE_PROMOTE_MATCH_STREAK,
    LCI_GENE_PROMOTE_MIN_OBS,
    LCI_PROMOTION_FAMILY_CONF_MIN,
)
from indw.schedule.intel.fingerprints import IntelligenceBundle
from indw.schedule.intel.genome import GenomeProfile, StructuralGene
from indw.schedule.intel.store import structural_verify_hash


@dataclass(frozen=True)
class PromotionResult:
    genes_promoted: int
    genes_rejected: int
    family_promoted: bool
    domain_promoted: bool
    structural_hash: str
    quality_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            'genes_promoted': self.genes_promoted,
            'genes_rejected': self.genes_rejected,
            'family_promoted': self.family_promoted,
            'domain_promoted': self.domain_promoted,
            'structural_hash': self.structural_hash,
            'quality_ok': self.quality_ok,
        }


def line_quality_ok(line: dict[str, Any]) -> bool:
    if str(line.get('kind') or '') != 'processed':
        return False
    chunks = line.get('chunks') or []
    if not chunks:
        return False
    rejects = line.get('cleaning_rejects') or []
    return len(rejects) == 0


def genes_structurally_aligned(
    raw: GenomeProfile | None,
    cleaned: GenomeProfile,
) -> bool:
    if raw is None or not raw.genes or not cleaned.genes:
        return True
    raw_by_idx = {g.span_index: g for g in raw.genes}
    matched = 0
    checked = 0
    for cg in cleaned.genes:
        rg = raw_by_idx.get(cg.span_index)
        if rg is None:
            continue
        checked += 1
        if rg.kind == cg.kind:
            matched += 1
    if checked == 0:
        return True
    return (matched / checked) >= 0.55


def gene_shape_stable(gene: StructuralGene, last_shape: str) -> bool:
    if not last_shape:
        return True
    return gene.shape_sig == last_shape


def kinds_for_promotion(genome: GenomeProfile) -> list[str]:
    kinds: list[str] = []
    mapping = {
        'forum_question': 'forum',
        'forum_answer': 'forum',
        'publication_scaffold': 'publication',
        'educational': 'educational',
        'code': 'code',
        'documentation': 'layout',
        'scientific': 'layout',
        'prose': 'layout',
    }
    for gkind in genome.gene_kinds:
        mapped = mapping.get(gkind)
        if mapped and mapped not in kinds:
            kinds.append(mapped)
    if not kinds:
        kinds.append('layout')
    return kinds


def family_promotion_eligible(
    *,
    observation_count: int,
    family_confidence: float,
    genes_promoted: int,
    streak: int,
) -> bool:
    if genes_promoted <= 0:
        return False
    if observation_count < LCI_FAMILY_PROMOTE_MIN_OBS:
        return False
    if streak < LCI_GENE_PROMOTE_MATCH_STREAK:
        return False
    if family_confidence < LCI_PROMOTION_FAMILY_CONF_MIN:
        return False
    return True


def domain_promotion_eligible(observation_count: int, verified_count: int) -> bool:
    return (
        observation_count >= LCI_DOMAIN_PROMOTE_MIN_OBS
        and verified_count >= max(2, LCI_GENE_PROMOTE_MIN_OBS // 2)
    )


def build_cleaned_intel(
    cleaned_text: str,
    intel: IntelligenceBundle,
) -> IntelligenceBundle:
    from indw.schedule.intel.fingerprints import build_intelligence_bundle
    rec_verified = 0
    if intel.family_confidence >= LCI_PROMOTION_FAMILY_CONF_MIN:
        rec_verified = max(1, intel.observation_count // 4)
    return build_intelligence_bundle(
        cleaned_text,
        family_id=intel.family_id,
        observation_count=intel.observation_count,
        verified_count=rec_verified,
    )


def structural_hash_for_intel(intel: IntelligenceBundle) -> str:
    return structural_verify_hash(intel.fp)
