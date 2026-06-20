from __future__ import annotations

from dataclasses import replace
from typing import Any

from indw.config.defaults import (
    ACIM_FAST_PATH,
    ACIM_VERIFY_CONFIDENCE,
    LCI_GENE_VERIFY_CONFIDENCE,
    LCI_INHERITED_VERIFY_CONFIDENCE,
)
from indw.schedule.intel.fingerprints import IntelligenceBundle
from indw.schedule.intel.hardware import HardwareSnapshot, adapt_routing_params
from indw.schedule.intel.lci_graph import LCIContext
from indw.schedule.intel.router import ProcessingDepth, RouteDecision, route_document
from indw.schedule.intel.store import IntelligenceStore


def route_with_lci(
    intel: IntelligenceBundle,
    store: IntelligenceStore,
    lci: LCIContext,
    *,
    fast_path_enabled: bool = ACIM_FAST_PATH,
    verify_threshold: float = ACIM_VERIFY_CONFIDENCE,
    hw: HardwareSnapshot | None = None,
) -> RouteDecision:
    base = route_document(
        intel,
        store,
        fast_path_enabled=False,
        verify_threshold=verify_threshold,
    )
    thr, boost = adapt_routing_params(
        hw or HardwareSnapshot(),
        verify_threshold=verify_threshold,
        cache_boost=base.cache_boost,
    )
    if base.depth == ProcessingDepth.FULL and base.reason in (
        'high_complexity_or_entropy', 'high_novelty', 'acim_disabled',
    ):
        return _attach_lci(base, lci, boost)

    effective_conf = max(
        intel.family_confidence,
        lci.gene_confidence * 0.45 + lci.domain_confidence * 0.30 + lci.inherited_confidence * 0.25,
    )
    if lci.novel_gene_ratio >= 0.72:
        return _attach_lci(replace(
            base,
            depth=ProcessingDepth.FULL,
            reason='high_novel_gene_ratio',
            family_confidence=effective_conf,
        ), lci, boost)

    inc = getattr(lci, 'incremental', None)
    if inc is not None and inc.reuse_eligible:
        boost = min(4, boost + 1)

    if (
        lci.gene_confidence < LCI_GENE_VERIFY_CONFIDENCE
        and lci.inherited_confidence < LCI_INHERITED_VERIFY_CONFIDENCE
        and effective_conf < thr
    ):
        return _attach_lci(replace(
            base,
            depth=ProcessingDepth.FULL,
            reason='low_lci_confidence',
            family_confidence=effective_conf,
        ), lci, boost)

    verified_kinds = list(base.verified_nodes)

    if not verified_kinds and lci.overlap_ratio < 0.35:
        return _attach_lci(replace(
            base,
            depth=ProcessingDepth.FULL,
            reason='low_corpus_overlap',
            family_confidence=effective_conf,
            cache_boost=max(boost, 2 if lci.gene_confidence >= 0.4 else 1),
        ), lci, boost)

    if not fast_path_enabled:
        return _attach_lci(RouteDecision(
            depth=ProcessingDepth.VERIFY,
            reason='lci_verify_capable',
            family_id=intel.family_id,
            family_confidence=effective_conf,
            verified_nodes=tuple(verified_kinds),
            cache_boost=max(boost, 2),
        ), lci, boost)

    return _attach_lci(RouteDecision(
        depth=ProcessingDepth.FAST,
        reason='lci_verified_fast_path',
        family_id=intel.family_id,
        family_confidence=effective_conf,
        verified_nodes=tuple(verified_kinds),
        cache_boost=max(boost, 4),
    ), lci, boost)


def _attach_lci(decision: RouteDecision, lci: LCIContext, boost: int) -> RouteDecision:
    if decision.cache_boost < boost:
        decision = replace(decision, cache_boost=boost)
    return decision


def lci_route_dict(decision: RouteDecision, lci: LCIContext) -> dict[str, Any]:
    out = decision.to_dict()
    out['lci'] = lci.to_dict()
    return out
