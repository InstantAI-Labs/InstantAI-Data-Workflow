from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any
from indw.filter.content.filters import analyze_content_filters, count_transaction_signals
from indw.clean.artifact.evidence_util import _CITATION_FMT, _EPS, _mean, _saturate, _spread, peer_baseline
from indw.clean.artifact.evidence_features import (
    DocumentFeatureExtractor,
    PopulationAdaptiveScaler,
    RawDocumentFeatures,
    SemanticFeatureBundle,
    shared_feature_extractor,
)

class SemanticSignalExtractor:
    def __init__(self, feature_extractor: DocumentFeatureExtractor | None = None) -> None:
        self._features = feature_extractor or shared_feature_extractor()

    def extract(self, text: str, *, filters: Any | None = None) -> SemanticFeatureBundle:
        raw = self._features.extract(text)
        if filters is not None:
            filt = filters
        else:
            filt = analyze_content_filters(
                text,
                words=[w.lower() for w in raw.words],
                lines=raw.lines,
            )
        return SemanticFeatureBundle(raw=raw, filters=filt)

@dataclass
class LatentSemanticRepresentation:
    educational: float = 0.0
    technical: float = 0.0
    factual: float = 0.0
    reasoning: float = 0.0
    structural: float = 0.0
    procedural: float = 0.0
    referential: float = 0.0
    code: float = 0.0
    transactional: float = 0.0
    navigational: float = 0.0
    narrative: float = 0.0
    promotional: float = 0.0
    corruption: float = 0.0
    redundancy: float = 0.0
    coherence: float = 0.0
    temporal: float = 0.0
    opinion: float = 0.0

    def values(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}

    def dominant(self) -> str:
        vals = self.values()
        if not vals:
            return 'educational'

class DistributionAwareNormalizer:
    def normalize(self, bundle: SemanticFeatureBundle) -> LatentSemanticRepresentation:
        raw = bundle.raw
        sig = bundle.quality_signals()
        wc = max(raw.word_count, 1)
        lc = max(raw.line_count, 1)
        scale = PopulationAdaptiveScaler
        txn_hits = bundle.filters.transaction_signal_hits
        if txn_hits < 0:
            txn_hits = sum(count_transaction_signals(raw.text))

        channels = {
            'educational': sig.educational_value + scale.rate(raw.copula_def_hits, lc) + scale.rate(raw.step_line_hits, lc) + sig.reasoning_density,
            'technical': sig.code_density + sig.delimiter_density + raw.fence_char_ratio + raw.numeric_token_ratio,
            'factual': sig.factual_density + scale.rate(raw.citation_hits, lc, wc) + raw.anchor_density + scale.rate(raw.fact_relation_hits, lc) + raw.numeric_token_ratio,
            'reasoning': sig.reasoning_density + sig.coherence_score * raw.line_len_cv,
            'structural': raw.structured_line_ratio + raw.table_line_ratio + sig.structural_quality + raw.fence_char_ratio,
            'procedural': scale.rate(raw.step_line_hits, lc) + sig.reasoning_density,
            'referential': scale.rate(raw.citation_hits, lc) + raw.structured_line_ratio + scale.rate(raw.year_hits, wc, lc),
            'code': sig.code_density + raw.fence_char_ratio,
            'transactional': bundle.filters.commercial_score + scale.rate(txn_hits, wc, lc),
            'navigational': bundle.filters.boilerplate_score + raw.url_char_ratio + raw.nav_line_ratio + bundle.filters.seo_spam_score,
            'narrative': sig.low_information_score + raw.exclaim_line_ratio + sig.artificial_enthusiasm_score + (1.0 - sig.semantic_diversity),
            'promotional': bundle.filters.commercial_score + bundle.filters.artificial_enthusiasm_score + bundle.filters.seo_spam_score,
            'corruption': bundle.filters.truncation_score + sig.html_score + sig.injection_score + bundle.filters.hallucination_risk_score,
            'redundancy': max(sig.line_repetition, sig.char_repetition, sig.repeated_span_score),
            'coherence': sig.coherence_score * sig.semantic_diversity,
            'temporal': scale.rate(raw.year_hits, wc, lc) + raw.schedule_token_ratio,
            'opinion': raw.first_person_ratio,
        }
        peers = list(channels.values())
        return LatentSemanticRepresentation(**{k: _saturate(v, peers) for k, v in channels.items()})

class AdaptiveBaselineEstimator:
    def baseline(self, values: list[float]) -> float:
        return peer_baseline(values)

    def spread(self, values: list[float]) -> float:
        return _spread(values)


_SHARED_BASELINE: AdaptiveBaselineEstimator | None = None


def shared_baseline_estimator() -> AdaptiveBaselineEstimator:
    global _SHARED_BASELINE
    if _SHARED_BASELINE is None:
        _SHARED_BASELINE = AdaptiveBaselineEstimator()
    return _SHARED_BASELINE

class InformationDensityEstimator:
    def estimate(self, bundle: SemanticFeatureBundle, rep: LatentSemanticRepresentation) -> tuple[float, float, int]:
        raw = bundle.raw
        sig = bundle.quality_signals()
        wc = max(raw.word_count, 1)
        te = max(raw.token_estimate, 1)
        span_keys: set[str] = set()
        for sent in re.split(r'[.!?\n]+', raw.text):
            s = sent.strip().lower()
            if len(s) >= 12:
                span_keys.add(s[:48])
        key_len = max(12, int(raw.avg_line_len)) if raw.avg_line_len > 0 else 32
        span_keys.update(m.group(0).lower()[:key_len] for m in _CITATION_FMT.finditer(raw.text))
        if raw.fence_char_ratio > 0:
            span_keys.add('code_block')
        if raw.table_line_ratio > 0:
            span_keys.add('table')
        fact_count = len(span_keys) + int(raw.citation_hits) + min(raw.year_hits, raw.sentence_count)
        lexical = sig.word_diversity * sig.semantic_diversity
        structural = rep.structural + rep.factual
        blend = _mean([lexical, structural, rep.educational])
        density = (fact_count / wc) * blend
        per_token = fact_count / te * _mean([lexical, rep.reasoning])
        return density, per_token, fact_count

class NoveltyEstimator:
    def estimate(self, bundle: SemanticFeatureBundle, rep: LatentSemanticRepresentation, duplicate_ratio: float) -> float:
        sig = bundle.quality_signals()
        redundancy = max(rep.redundancy, duplicate_ratio, sig.reasoning_repetition)
        synthetic = max(sig.synthetic_score, sig.template_synthetic_score)
        damp = _saturate(rep.coherence, [redundancy, synthetic, rep.coherence])
        return max(0.0, min(1.0, 1.0 - redundancy * (1.0 - damp) - synthetic * damp))

class SemanticCoherenceEstimator:
    def estimate(self, bundle: SemanticFeatureBundle, rep: LatentSemanticRepresentation) -> float:
        sig = bundle.quality_signals()
        return max(0.0, min(1.0, (rep.coherence + sig.coherence_score + (1.0 - rep.redundancy)) / 3.0))

@dataclass
class DynamicQualityScores:
    educational: float = 0.0
    technical: float = 0.0
    reference: float = 0.0
    code: float = 0.0
    entertainment: float = 0.0
    commercial: float = 0.0
    storytelling: float = 0.0
    narrative_filler: float = 0.0
    overall: float = 0.0
    information_density: float = 0.0
    information_density_per_token: float = 0.0
    fact_count: int = 0
    novelty: float = 0.0
    coherence: float = 0.0
    corruption: float = 0.0

class DynamicQualityEstimator:
    def __init__(self) -> None:
        self._density = InformationDensityEstimator()
        self._novelty = NoveltyEstimator()
        self._coherence = SemanticCoherenceEstimator()

    def estimate(
        self,
        bundle: SemanticFeatureBundle,
        rep: LatentSemanticRepresentation,
        *,
        duplicate_ratio: float = 0.0,
        tokenizer_metrics: Any | None = None,
    ) -> DynamicQualityScores:
        density, per_token, fact_count = self._density.estimate(bundle, rep)
        novelty = self._novelty.estimate(bundle, rep, duplicate_ratio)
        coherence = self._coherence.estimate(bundle, rep)
        sig = bundle.quality_signals()

        commercial = _saturate(
            bundle.filters.commercial_score + rep.transactional + rep.promotional,
            [bundle.filters.commercial_score, rep.transactional, rep.promotional],
        )
        edu_base = _saturate(
            sig.educational_value + rep.educational + rep.reasoning + sig.reasoning_density,
            [sig.educational_value, rep.educational, rep.reasoning, sig.reasoning_density, per_token],
        )
        edu_anchor = _saturate(
            sig.factual_density + per_token + sig.reasoning_density,
            [sig.factual_density, per_token, sig.reasoning_density, rep.factual],
        )
        edu_weight = _saturate(
            edu_anchor + sig.reasoning_density + bundle.raw.fence_char_ratio + sig.code_density,
            [edu_anchor, sig.reasoning_density, bundle.raw.fence_char_ratio, sig.code_density],
        )
        educational = _saturate(
            edu_base * edu_weight * (1.0 - commercial) + per_token * coherence,
            [edu_base, edu_weight, per_token, coherence, sig.reasoning_density, rep.reasoning],
        )
        technical_code = _saturate(sig.code_density + bundle.raw.fence_char_ratio, [sig.code_density, bundle.raw.fence_char_ratio, rep.code])
        technical_dense = _saturate(
            sig.delimiter_density + sig.factual_density + rep.reasoning,
            [sig.delimiter_density, sig.factual_density, rep.reasoning, rep.factual],
        )
        technical = _saturate(
            max(technical_code, technical_dense * (1.0 - rep.narrative)),
            [technical_code, technical_dense, rep.narrative],
        )
        reference = _saturate(
            rep.referential + rep.factual,
            [rep.referential, rep.factual, bundle.raw.citation_hits / max(bundle.raw.line_count, 1)],
        )
        code = _saturate(sig.code_density + bundle.raw.fence_char_ratio, [sig.code_density, bundle.raw.fence_char_ratio, rep.code])
        entertainment = _saturate(
            rep.narrative + bundle.raw.exclaim_line_ratio + sig.low_information_score,
            [rep.narrative, bundle.raw.exclaim_line_ratio, sig.low_information_score, educational],
        ) * (1.0 - educational)
        storytelling = _saturate(
            entertainment + sig.artificial_enthusiasm_score,
            [entertainment, sig.artificial_enthusiasm_score, sig.low_information_score],
        )
        narrative_filler = _saturate(
            storytelling + entertainment - educational - per_token,
            [storytelling, entertainment, educational, per_token],
        )
        short_bound = PopulationAdaptiveScaler.short_doc_boundary(bundle.raw)
        if bundle.raw.word_count < short_bound and bundle.raw.fact_relation_hits >= 1:
            short_edu = _saturate(
                per_token + sig.factual_density + rep.factual + rep.educational,
                [per_token, sig.factual_density, rep.factual, rep.educational],
            )
            damp = _saturate(short_edu, [entertainment, narrative_filler, short_edu])
            educational = max(educational, short_edu)
            entertainment = min(entertainment, short_edu * damp)
            narrative_filler = min(narrative_filler, short_edu * damp)

        positive = [educational, technical, reference, code, per_token, novelty, coherence]
        negative = [narrative_filler, entertainment, commercial, rep.corruption, rep.redundancy, duplicate_ratio]
        pos = _mean(positive)
        neg = _mean(negative)
        signal = pos / max(pos + neg, _EPS)
        substance = _mean([educational, technical, reference, code])
        noise = _spread(negative) or neg
        overall = max(0.0, min(1.0, signal * substance * (1.0 - _saturate(neg, negative))))
        if noise > substance:
            overall = min(overall, substance)
        if bundle.raw.fence_char_ratio > 0 or bundle.raw.table_line_ratio > 0:
            overall = max(overall, _saturate(substance + bundle.raw.fence_char_ratio + bundle.raw.table_line_ratio, positive))
        if bundle.raw.fence_char_ratio > 0 and rep.procedural > 0:
            educational = max(
                educational,
                _saturate(
                    edu_base + rep.procedural + rep.code + bundle.raw.step_line_hits / max(bundle.raw.line_count, 1),
                    [edu_base, rep.procedural, rep.code, per_token],
                ),
            )

        if tokenizer_metrics is not None:
            tok_eff = _saturate(
                getattr(tokenizer_metrics, 'replay_stability', 0.0)
                + (1.0 - getattr(tokenizer_metrics, 'token_inflation', 0.0))
                + getattr(tokenizer_metrics, 'structured_output_stability', 0.0),
                [
                    getattr(tokenizer_metrics, 'replay_stability', 0.0),
                    1.0 - getattr(tokenizer_metrics, 'token_inflation', 0.0),
                    getattr(tokenizer_metrics, 'structured_output_stability', 0.0),
                ],
            )
            tok_noise = _saturate(
                getattr(tokenizer_metrics, 'repeated_token_span_score', 0.0)
                + getattr(tokenizer_metrics, 'unicode_fragmentation', 0.0),
                [
                    getattr(tokenizer_metrics, 'repeated_token_span_score', 0.0),
                    getattr(tokenizer_metrics, 'unicode_fragmentation', 0.0),
                ],
            )
            coherence = _saturate(coherence * tok_eff, [coherence, tok_eff, per_token])
            overall = _saturate(overall * (1.0 - tok_noise * 0.35) + tok_eff * 0.15, [overall, tok_eff, coherence])

        return DynamicQualityScores(
            educational=educational,
            technical=technical,
            reference=reference,
            code=code,
            entertainment=entertainment,
            commercial=commercial,
            storytelling=storytelling,
            narrative_filler=narrative_filler,
            overall=overall,
            information_density=density,
            information_density_per_token=per_token,
            fact_count=fact_count,
            novelty=novelty,
            coherence=coherence,
            corruption=rep.corruption,
        )

@dataclass
class IntentDistribution:
    knowledge: float = 0.0
    transactional: float = 0.0
    administrative: float = 0.0
    promotional: float = 0.0
    navigational: float = 0.0
    entertainment: float = 0.0
    half_life: float = 0.0
    transience: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}

@dataclass
class SemanticProfile:
    axes: dict[str, float] = field(default_factory=dict)
    primary: str = 'educational'
    secondary: str = 'educational'
    entropy: float = 0.0

    def to_dict(self) -> dict[str, float | str | dict[str, float]]:
        return {
            'primary': self.primary,
            'secondary': self.secondary,
            'entropy': round(self.entropy, 4),
            'axes': {k: round(v, 4) for k, v in self.axes.items()},
        }

@dataclass
class DecisionExplanation:
    utility: float = 0.0
    margin: float = 0.0
    threshold: float = 0.0
    confidence: float = 0.0
    uncertainty: float = 0.0
    dominant_positive: str = ''
    dominant_negative: str = ''
    preserve: bool = True
    discard_code: str = ''
    discard_reason: str = ''

    def to_dict(self) -> dict[str, float | bool | str]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}

class IntentDistributionEstimator:
    def estimate(
        self,
        rep: LatentSemanticRepresentation,
        quality: DynamicQualityScores,
        raw: RawDocumentFeatures,
    ) -> IntentDistribution:
        vals = rep.values()
        total = sum(vals.values()) or _EPS
        mix = {k: v / total for k, v in vals.items()}
        baseline = AdaptiveBaselineEstimator()
        knowledge = baseline.baseline([
            mix.get('educational', 0.0), mix.get('factual', 0.0),
            mix.get('reasoning', 0.0), mix.get('referential', 0.0), quality.overall,
        ])
        transactional = baseline.baseline([mix.get('transactional', 0.0), quality.commercial])
        administrative = baseline.baseline([
            mix.get('temporal', 0.0), raw.contact_token_ratio, raw.schedule_token_ratio, raw.uniform_line_ratio,
        ])
        promotional = baseline.baseline([mix.get('promotional', 0.0), mix.get('transactional', 0.0)])
        navigational = baseline.baseline([mix.get('navigational', 0.0), raw.nav_line_ratio, raw.url_char_ratio])
        edu_damp = baseline.baseline([knowledge, mix.get('procedural', 0.0), quality.code])
        entertainment = baseline.baseline([
            mix.get('narrative', 0.0), quality.entertainment, quality.storytelling,
        ]) * (1.0 - edu_damp)
        durable = baseline.baseline([mix.get('factual', 0.0), mix.get('referential', 0.0), quality.coherence])
        volatile = baseline.baseline([mix.get('temporal', 0.0), mix.get('narrative', 0.0), administrative])
        half_life = _saturate(durable * (1.0 - volatile) + quality.novelty * rep.coherence, [durable, volatile, quality.novelty])
        return IntentDistribution(
            knowledge=min(1.0, knowledge),
            transactional=min(1.0, transactional),
            administrative=min(1.0, administrative),
            promotional=min(1.0, promotional),
            navigational=min(1.0, navigational),
            entertainment=min(1.0, entertainment),
            half_life=half_life,
            transience=max(0.0, 1.0 - half_life),
        )

class SemanticProfileDiscovery:
    def discover(
        self,
        rep: LatentSemanticRepresentation,
        quality: DynamicQualityScores,
    ) -> SemanticProfile:
        merged = {**rep.values(), 'overall': quality.overall}
        ranked = sorted(merged.items(), key=lambda item: item[1], reverse=True)
        axes = dict(ranked)
        primary = ranked[0][0] if ranked else 'educational'
        secondary = ranked[1][0] if len(ranked) > 1 else primary
        probs = [v for _, v in ranked if v > 0]
        total = sum(probs) or _EPS
        entropy = -sum((p / total) * math.log(p / total + _EPS) for p in probs)
        return SemanticProfile(axes=axes, primary=primary, secondary=secondary, entropy=entropy)

def discover_semantic_profile(
    rep: LatentSemanticRepresentation,
    quality: DynamicQualityScores,
) -> SemanticProfile:
    return SemanticProfileDiscovery().discover(rep, quality)

@dataclass
class SemanticEvidenceBundle:
    utility: float = 0.0
    confidence: float = 0.0
    uncertainty: float = 0.0
    redundancy: float = 0.0
    corruption: float = 0.0
    semantic_strength: float = 0.0
    information_density: float = 0.0
    information_density_per_token: float = 0.0
    novelty: float = 0.0
    coherence: float = 0.0
    preserve: bool = True
    discard_code: str = ''
    discard_reason: str = ''
    threshold: float = 0.0
    positive: dict[str, float] = field(default_factory=dict)
    negative: dict[str, float] = field(default_factory=dict)
    quality: DynamicQualityScores = field(default_factory=DynamicQualityScores)
    representation: LatentSemanticRepresentation = field(default_factory=LatentSemanticRepresentation)
    structural_integrity: float = 0.0
    intent: IntentDistribution = field(default_factory=IntentDistribution)
    profile: SemanticProfile = field(default_factory=SemanticProfile)
    explanation: DecisionExplanation = field(default_factory=DecisionExplanation)
    _raw: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, float | bool | str | dict[str, float]]:
        out: dict[str, float | bool | str | dict[str, float]] = {
            'utility': round(self.utility, 4),
            'confidence': round(self.confidence, 4),
            'uncertainty': round(self.uncertainty, 4),
            'redundancy': round(self.redundancy, 4),
            'corruption': round(self.corruption, 4),
            'semantic_strength': round(self.semantic_strength, 4),
            'information_density': round(self.information_density, 4),
            'information_density_per_token': round(self.information_density_per_token, 4),
            'novelty': round(self.novelty, 4),
            'coherence': round(self.coherence, 4),
            'preserve': self.preserve,
            'discard_code': self.discard_code,
            'discard_reason': self.discard_reason,
            'threshold': round(self.threshold, 4),
            'positive': {k: round(v, 4) for k, v in self.positive.items()},
            'negative': {k: round(v, 4) for k, v in self.negative.items()},
            'structural_integrity': round(self.structural_integrity, 4),
        }
        return out

class EvidenceAggregator:
    def __init__(self) -> None:
        self._baseline = AdaptiveBaselineEstimator()

    def aggregate(
        self,
        rep: LatentSemanticRepresentation,
        quality: DynamicQualityScores,
        raw: RawDocumentFeatures | None = None,
    ) -> tuple[dict[str, float], dict[str, float], float, float]:
        positive = {
            'educational': quality.educational,
            'technical': quality.technical,
            'factual': rep.factual,
            'reasoning': rep.reasoning,
            'referential': quality.reference,
            'code': quality.code,
            'structural': rep.structural,
            'coherence': quality.coherence,
            'novelty': quality.novelty,
            'density': quality.information_density_per_token,
        }
        admin = 0.0
        if raw is not None and (raw.contact_token_ratio > 0 or raw.schedule_token_ratio > 0):
            admin = AdaptiveBaselineEstimator().baseline([
                raw.contact_token_ratio, raw.schedule_token_ratio, raw.uniform_line_ratio, rep.temporal,
            ])
        negative = {
            'redundancy': rep.redundancy,
            'corruption': rep.corruption,
            'noise': _mean([quality.narrative_filler, quality.entertainment, quality.commercial]),
            'synthetic': quality.corruption,
            'navigational': rep.navigational,
            'transactional': rep.transactional,
            'promotional': rep.promotional,
            'administrative': admin,
        }
        pos_vals = list(positive.values())
        neg_vals = list(negative.values())
        strength = self._baseline.baseline(pos_vals)
        noise = self._baseline.baseline(neg_vals)
        utility = _saturate(strength, pos_vals) * (1.0 - _saturate(noise, neg_vals))
        return positive, negative, utility, strength

class UncertaintyEstimator:
    def estimate(self, positive: dict[str, float], negative: dict[str, float], coherence: float) -> float:
        vals = list(positive.values()) + list(negative.values())
        spread = _spread(vals)
        return max(0.0, min(1.0, spread * (1.0 - coherence)))

class ConfidenceEstimator:
    def estimate(self, positive: dict[str, float], negative: dict[str, float], coherence: float, word_count: int) -> float:
        spread = _spread(list(positive.values()))
        agreement = 1.0 - spread
        length = _saturate(float(word_count), [float(word_count), spread, coherence])
        return max(0.0, min(1.0, agreement * length + coherence * (1.0 - spread)))

class AdaptiveThresholdEstimator:
    def __init__(self) -> None:
        self._baseline = AdaptiveBaselineEstimator()

    def estimate(self, positive: dict[str, float], negative: dict[str, float]) -> float:
        pos = list(positive.values())
        neg = list(negative.values())
        pos_b = self._baseline.baseline(pos)
        neg_b = self._baseline.baseline(neg)
        return pos_b * (1.0 - neg_b * _saturate(neg_b, pos + neg))

class SemanticUtilityEstimator:
    def __init__(self) -> None:
        self._aggregator = EvidenceAggregator()

    def estimate(
        self,
        rep: LatentSemanticRepresentation,
        quality: DynamicQualityScores,
    ) -> tuple[float, float]:
        _, _, utility, strength = self._aggregator.aggregate(rep, quality, raw=None)
        return utility, strength
