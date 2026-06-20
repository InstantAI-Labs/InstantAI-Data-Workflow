from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from indw.config.defaults import ACIM_FAST_PATH, ACIM_VERIFY_CONFIDENCE
from indw.schedule.intel.fingerprints import IntelligenceBundle
from indw.schedule.intel.store import IntelligenceStore, structural_verify_hash


class ProcessingDepth(str, Enum):
    FULL = 'full'
    VERIFY = 'verify'
    FAST = 'fast'


@dataclass(frozen=True)
class RouteDecision:
    depth: ProcessingDepth
    reason: str
    family_id: str = ''
    family_confidence: float = 0.0
    verified_nodes: tuple[str, ...] = ()
    cache_boost: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            'depth': self.depth.value,
            'reason': self.reason,
            'family_id': self.family_id,
            'family_confidence': round(self.family_confidence, 4),
            'verified_nodes': list(self.verified_nodes),
            'cache_boost': self.cache_boost,
        }


_GRAPH_KINDS = ('layout', 'publication', 'forum', 'educational', 'code')


def verified_graph_kinds(
    intel: IntelligenceBundle,
    store: IntelligenceStore,
) -> tuple[tuple[str, ...], str | None]:
    verify_hash = structural_verify_hash(intel.fp)
    kinds: list[str] = []
    for kind in _GRAPH_KINDS:
        node = store.get_graph_node(intel.family_id, kind)
        if node is None or not node.get('verified'):
            continue
        if node.get('fp_hash') != verify_hash:
            return (), f'structural_mismatch_{kind}'
        kinds.append(kind)
    return tuple(kinds), None


def route_document(
    intel: IntelligenceBundle,
    store: IntelligenceStore,
    *,
    fast_path_enabled: bool = ACIM_FAST_PATH,
    verify_threshold: float = ACIM_VERIFY_CONFIDENCE,
) -> RouteDecision:
    if intel.complexity >= 0.82 or intel.entropy >= 0.92:
        return RouteDecision(
            depth=ProcessingDepth.FULL,
            reason='high_complexity_or_entropy',
            family_id=intel.family_id,
            family_confidence=intel.family_confidence,
        )
    if intel.novelty >= 0.88:
        return RouteDecision(
            depth=ProcessingDepth.FULL,
            reason='high_novelty',
            family_id=intel.family_id,
            family_confidence=intel.family_confidence,
        )
    if intel.family_confidence < verify_threshold:
        return RouteDecision(
            depth=ProcessingDepth.FULL,
            reason='low_family_confidence',
            family_id=intel.family_id,
            family_confidence=intel.family_confidence,
        )
    verified_kinds, mismatch = verified_graph_kinds(intel, store)
    if mismatch is not None:
        return RouteDecision(
            depth=ProcessingDepth.FULL,
            reason=mismatch,
            family_id=intel.family_id,
            family_confidence=intel.family_confidence,
        )
    if not verified_kinds:
        return RouteDecision(
            depth=ProcessingDepth.FULL,
            reason='no_verified_graph_nodes',
            family_id=intel.family_id,
            family_confidence=intel.family_confidence,
            cache_boost=2 if intel.family_confidence >= 0.5 else 1,
        )
    if not fast_path_enabled:
        return RouteDecision(
            depth=ProcessingDepth.VERIFY,
            reason='observe_only_verify_capable',
            family_id=intel.family_id,
            family_confidence=intel.family_confidence,
            verified_nodes=tuple(verified_kinds),
            cache_boost=2,
        )
    return RouteDecision(
        depth=ProcessingDepth.FAST,
        reason='verified_family_fast_path',
        family_id=intel.family_id,
        family_confidence=intel.family_confidence,
        verified_nodes=verified_kinds,
        cache_boost=4,
    )
