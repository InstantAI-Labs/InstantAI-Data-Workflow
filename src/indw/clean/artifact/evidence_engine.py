from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from indw.clean.artifact.evidence_util import _EPS, _saturate
from indw.clean.artifact.evidence_features import (
    PopulationAdaptiveScaler,
    RawDocumentFeatures,
    SemanticFeatureBundle,
)
from indw.clean.artifact.evidence_model import (
    AdaptiveBaselineEstimator, AdaptiveThresholdEstimator, ConfidenceEstimator,
    DecisionExplanation, DistributionAwareNormalizer, DynamicQualityEstimator,
    DynamicQualityScores, EvidenceAggregator, IntentDistribution, IntentDistributionEstimator,
    LatentSemanticRepresentation, SemanticEvidenceBundle, SemanticProfile,
    SemanticProfileDiscovery, SemanticSignalExtractor, SemanticUtilityEstimator,
    UncertaintyEstimator, shared_baseline_estimator,
)

def _structural_preserve(raw: RawDocumentFeatures | None, integrity: float, strength: float) -> bool:
    if integrity > strength:
        return True
    if raw is None:
        return False
    return raw.fence_char_ratio > 0 or raw.table_line_ratio > 0 or raw.structured_line_ratio > 0

def _discard_reason_from_evidence(evidence: SemanticEvidenceBundle) -> str:
    dom_neg = max(evidence.negative, key=lambda k: evidence.negative[k]) if evidence.negative else ''
    dom_pos = max(evidence.positive, key=lambda k: evidence.positive[k]) if evidence.positive else ''
    return f'{dom_neg}_over_{dom_pos}' if dom_neg else 'low_utility'

def _build_explanation(
    evidence: SemanticEvidenceBundle,
    *,
    preserve: bool,
    margin: float,
) -> DecisionExplanation:
    dom_pos = max(evidence.positive, key=lambda k: evidence.positive[k]) if evidence.positive else ''
    dom_neg = max(evidence.negative, key=lambda k: evidence.negative[k]) if evidence.negative else ''
    return DecisionExplanation(
        utility=evidence.utility,
        margin=margin,
        threshold=evidence.threshold,
        confidence=evidence.confidence,
        uncertainty=evidence.uncertainty,
        dominant_positive=dom_pos,
        dominant_negative=dom_neg,
        preserve=preserve,
        discard_code=evidence.discard_code,
        discard_reason=_discard_reason_from_evidence(evidence) if not preserve else '',
    )

def resolve_selection(
    evidence: SemanticEvidenceBundle,
    *,
    raw: RawDocumentFeatures | None = None,
    integrity: float = 0.0,
) -> tuple[bool, str]:
    q = evidence.quality
    rep = evidence.representation
    baseline = shared_baseline_estimator()
    substance_vals = [
        q.educational, q.technical, q.reference, q.code,
        evidence.information_density_per_token, evidence.semantic_strength,
    ]
    substance = baseline.baseline(substance_vals)
    noise_vals = [
        q.narrative_filler, q.entertainment, q.commercial,
        evidence.redundancy, evidence.corruption,
    ]
    noise = baseline.baseline(noise_vals)
    threshold = evidence.threshold or AdaptiveThresholdEstimator().estimate(
        evidence.positive, evidence.negative,
    )
    util = evidence.utility

    if _structural_preserve(raw, integrity, substance):
        return True, ''
    if q.code > noise and (q.educational > noise or q.technical > substance):
        return True, ''

    margin = util - threshold * (1.0 - evidence.uncertainty)
    commerce = max(
        evidence.negative.get('transactional', 0.0),
        evidence.negative.get('promotional', 0.0),
    )
    admin = evidence.negative.get('administrative', 0.0)

    if (
        rep.narrative > max(rep.factual, rep.referential, rep.reasoning)
        and q.reference < substance
        and margin < 0.0
    ):
        if evidence.negative:
            evidence.discard_code = max(evidence.negative, key=lambda k: evidence.negative[k])
        return False, _discard_reason_from_evidence(evidence)

    if commerce > substance and margin < 0.0:
        evidence.discard_code = 'transactional' if commerce == evidence.negative.get('transactional', 0.0) else 'promotional'
        return False, _discard_reason_from_evidence(evidence)

    if admin > substance and margin < 0.0 and admin > noise:
        evidence.discard_code = 'administrative'
        return False, _discard_reason_from_evidence(evidence)

    learning_peak = baseline.baseline([q.educational, q.reference, q.technical, q.code])
    if learning_peak >= noise and margin > -threshold * (1.0 - evidence.uncertainty):
        return True, ''

    if raw is not None and raw.fact_relation_hits >= 1:
        compact = PopulationAdaptiveScaler.rate(float(raw.word_count), raw.sentence_count, raw.line_count)
        if compact <= learning_peak or rep.factual >= rep.narrative:
            return True, ''

    if margin >= 0.0 and substance >= noise:
        return True, ''
    if margin >= 0.0 and evidence.semantic_strength > noise:
        return True, ''

    if evidence.negative:
        evidence.discard_code = max(evidence.negative, key=lambda k: evidence.negative[k])
    return False, _discard_reason_from_evidence(evidence)

class SemanticEvidenceSelector:
    def __init__(self) -> None:
        self._threshold = AdaptiveThresholdEstimator()
        self._uncertainty = UncertaintyEstimator()
        self._confidence = ConfidenceEstimator()

    def decide(
        self,
        evidence: SemanticEvidenceBundle,
        *,
        structural_integrity: float,
        enabled: bool = True,
        word_count: int = 0,
        raw: RawDocumentFeatures | None = None,
    ) -> SemanticEvidenceBundle:
        if not enabled:
            evidence.preserve = True
            evidence.discard_code = ''
            evidence.structural_integrity = structural_integrity
            return evidence

        evidence.threshold = self._threshold.estimate(evidence.positive, evidence.negative)
        evidence.uncertainty = self._uncertainty.estimate(
            evidence.positive, evidence.negative, evidence.coherence,
        )
        wc = word_count or max(evidence.quality.fact_count * 12, 1)
        evidence.confidence = self._confidence.estimate(
            evidence.positive, evidence.negative, evidence.coherence, wc,
        )
        evidence.structural_integrity = structural_integrity

        from indw.clean.artifact.evidence_util import evidence_margin

        margin = evidence_margin(evidence.utility, evidence.threshold, evidence.uncertainty)
        preserve, reason = resolve_selection(
            evidence, raw=raw, integrity=structural_integrity,
        )
        evidence.preserve = preserve
        evidence.discard_reason = reason
        if preserve:
            evidence.discard_code = ''
        else:
            evidence.discard_code = max(evidence.negative, key=lambda k: evidence.negative[k])
        evidence.explanation = _build_explanation(evidence, preserve=preserve, margin=margin)
        return evidence

def apply_tokenizer_telemetry(evidence: SemanticEvidenceBundle, metrics: Any) -> SemanticEvidenceBundle:
    baseline = shared_baseline_estimator()
    efficiency = baseline.baseline([
        getattr(metrics, 'replay_stability', 0.0),
        1.0 - getattr(metrics, 'token_inflation', 0.0),
        getattr(metrics, 'structured_output_stability', 0.0),
        min(1.0, getattr(metrics, 'chars_per_token', 0.0) / 4.0),
    ])
    noise = baseline.baseline([
        getattr(metrics, 'repeated_token_span_score', 0.0),
        getattr(metrics, 'unicode_fragmentation', 0.0),
        getattr(metrics, 'token_inflation', 0.0),
    ])
    blend = efficiency * (1.0 - noise)
    evidence.coherence = max(0.0, min(1.0, 0.6 * evidence.coherence + 0.4 * efficiency))
    evidence.quality.coherence = evidence.coherence
    evidence.utility = max(0.0, min(1.0, 0.65 * evidence.utility + 0.35 * blend))
    evidence.confidence = max(0.0, min(1.0, 0.7 * evidence.confidence + 0.3 * (1.0 - noise)))
    return evidence

def structural_integrity(raw: RawDocumentFeatures) -> float:
    parts = [raw.fence_char_ratio, raw.table_line_ratio, raw.structured_line_ratio]
    return _saturate(sum(parts), parts)

@dataclass
class _PipelineState:
    bundle: SemanticFeatureBundle
    rep: LatentSemanticRepresentation
    quality: DynamicQualityScores

class SemanticEvidencePipeline:
    def __init__(self) -> None:
        self._signals = SemanticSignalExtractor()
        self._normalizer = DistributionAwareNormalizer()
        self._quality = DynamicQualityEstimator()
        self._aggregator = EvidenceAggregator()
        self._utility = SemanticUtilityEstimator()
        self._intent = IntentDistributionEstimator()
        self._profile = SemanticProfileDiscovery()
        self._decision = SemanticEvidenceSelector()

    def run(
        self,
        text: str,
        *,
        filters: Any | None = None,
        duplicate_ratio: float = 0.0,
        bundle: SemanticFeatureBundle | None = None,
        enabled: bool = True,
    ) -> SemanticEvidenceBundle:
        sem = bundle or self._signals.extract(text, filters=filters)
        rep = self._normalizer.normalize(sem)
        quality = self._quality.estimate(sem, rep, duplicate_ratio=duplicate_ratio)
        positive, negative, utility, strength = self._aggregator.aggregate(rep, quality, sem.raw)
        integrity = structural_integrity(sem.raw)
        intent = self._intent.estimate(rep, quality, sem.raw)
        profile = self._profile.discover(rep, quality)
        evidence = SemanticEvidenceBundle(
            utility=utility,
            semantic_strength=strength,
            information_density=quality.information_density,
            information_density_per_token=quality.information_density_per_token,
            novelty=quality.novelty,
            coherence=quality.coherence,
            redundancy=rep.redundancy,
            corruption=rep.corruption,
            positive=positive,
            negative=negative,
            quality=quality,
            representation=rep,
            structural_integrity=integrity,
            intent=intent,
            profile=profile,
        )
        out = self._decision.decide(
            evidence,
            structural_integrity=integrity,
            enabled=enabled,
            word_count=sem.raw.word_count,
            raw=sem.raw,
        )
        out._raw = sem.raw
        return out


_PIPELINE = SemanticEvidencePipeline()


def evidence_raw_features(
    evidence: SemanticEvidenceBundle,
    text: str,
) -> RawDocumentFeatures:
    raw = evidence._raw
    if raw is not None:
        return raw
    from indw.clean.artifact.evidence_features import shared_feature_extractor
    return shared_feature_extractor().extract(text)


def compute_semantic_evidence(
    text: str,
    *,
    filters: Any | None = None,
    duplicate_ratio: float = 0.0,
    bundle: SemanticFeatureBundle | None = None,
    enabled: bool = True,
) -> SemanticEvidenceBundle:
    from indw.clean.artifact.evidence_cache import (
        evidence_cache_key,
        get_evidence_cache,
    )

    key = evidence_cache_key(
        text,
        filters=filters,
        duplicate_ratio=duplicate_ratio,
        enabled=enabled,
        bundle=bundle,
    )
    if key is not None:
        cache = get_evidence_cache()
        hit = cache.get(key)
        if hit is not None:
            return hit
        result = _PIPELINE.run(
            text, filters=filters, duplicate_ratio=duplicate_ratio, bundle=bundle, enabled=enabled,
        )
        cache.put(key, result)
        return result
    return _PIPELINE.run(
        text, filters=filters, duplicate_ratio=duplicate_ratio, bundle=bundle, enabled=enabled,
    )


def resolve_semantic_evidence(
    text: str,
    *,
    filters: Any | None = None,
    duplicate_ratio: float = 0.0,
    bundle: SemanticFeatureBundle | None = None,
    enabled: bool = True,
) -> SemanticEvidenceBundle:
    if filters is not None or bundle is not None or duplicate_ratio != 0.0 or not enabled:
        return compute_semantic_evidence(
            text,
            filters=filters,
            duplicate_ratio=duplicate_ratio,
            bundle=bundle,
            enabled=enabled,
        )
    from indw.extract.core.context import get_document_context

    dctx = get_document_context()
    if dctx is not None:
        return dctx.section_evidence(text, lambda: compute_semantic_evidence(text))
    return compute_semantic_evidence(text)
