from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indw.filter.content.filters import analyze_content_filters
from indw.clean.semantic.spec import SemanticSelectionConfig
from indw.clean.artifact.evidence import (
    shared_baseline_estimator,
    DistributionAwareNormalizer,
    DocumentFeatureExtractor,
    LatentSemanticRepresentation,
    PopulationAdaptiveScaler,
    RawDocumentFeatures,
    SemanticEvidenceBundle,
    SemanticFeatureBundle,
    SemanticSignalExtractor,
    compute_semantic_evidence,
    structural_integrity,
)

ContentCategory = str

_SHARED_SEM_SIGNALS: SemanticSignalExtractor | None = None

def _shared_semantic_extractor() -> SemanticSignalExtractor:
    global _SHARED_SEM_SIGNALS
    if _SHARED_SEM_SIGNALS is None:
        _SHARED_SEM_SIGNALS = SemanticSignalExtractor()
    return _SHARED_SEM_SIGNALS

def _has_preserved_structure(text: str, raw: RawDocumentFeatures | None = None) -> bool:
    if not text:
        return False
    r = raw or DocumentFeatureExtractor().extract(text)
    return structural_integrity(r) > 0.0

@dataclass
class DocumentStructureProfile:
    fact_ratio: float = 0.0
    explanation_ratio: float = 0.0
    instruction_ratio: float = 0.0
    transaction_ratio: float = 0.0
    listing_ratio: float = 0.0
    date_ratio: float = 0.0
    address_ratio: float = 0.0
    price_ratio: float = 0.0
    contact_ratio: float = 0.0
    navigation_ratio: float = 0.0
    word_count: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}

def _structure_profile_from_rep(
    sem: SemanticFeatureBundle,
    rep: LatentSemanticRepresentation,
    *,
    instruction_density: float | None = None,
) -> DocumentStructureProfile:
    raw = sem.raw
    if instruction_density is None:
        from indw.clean.meta.foundation import instruction_wrapper_density
        instr = instruction_wrapper_density(raw.text)
    else:
        instr = instruction_density
    listing = raw.table_line_ratio
    listing_signal = PopulationAdaptiveScaler.rate(
        raw.contact_token_ratio + raw.schedule_token_ratio,
        raw.nav_line_ratio + raw.uniform_line_ratio,
        rep.transactional,
    )
    if listing_signal > rep.educational:
        listing += raw.uniform_line_ratio
    return DocumentStructureProfile(
        fact_ratio=rep.factual,
        explanation_ratio=rep.educational,
        instruction_ratio=max(rep.procedural, instr),
        transaction_ratio=rep.transactional,
        listing_ratio=min(1.0, listing),
        date_ratio=rep.temporal,
        address_ratio=rep.transactional * raw.numeric_token_ratio,
        price_ratio=PopulationAdaptiveScaler.rate(raw.anchor_density, raw.word_count),
        contact_ratio=raw.contact_token_ratio,
        navigation_ratio=rep.navigational,
        word_count=raw.word_count,
    )

def compute_structure_profile(
    text: str,
    *,
    instruction_density: float | None = None,
    bundle: SemanticFeatureBundle | None = None,
    evidence: SemanticEvidenceBundle | None = None,
) -> DocumentStructureProfile:
    if not text or not text.strip():
        return DocumentStructureProfile()

    def _compute() -> DocumentStructureProfile:
        if bundle is not None:
            rep = evidence.representation if evidence else DistributionAwareNormalizer().normalize(bundle)
            return _structure_profile_from_rep(bundle, rep, instruction_density=instruction_density)
        if evidence is not None:
            from indw.clean.artifact.evidence_engine import evidence_raw_features

            raw = evidence_raw_features(evidence, text)
            sem = SemanticFeatureBundle(raw=raw, filters=None)
            return _structure_profile_from_rep(
                sem, evidence.representation, instruction_density=instruction_density,
            )
        sem = _shared_semantic_extractor().extract(text)
        rep = DistributionAwareNormalizer().normalize(sem)
        return _structure_profile_from_rep(sem, rep, instruction_density=instruction_density)

    try:
        from indw.extract.core.context import get_document_context
        ctx = get_document_context()
        if ctx is not None:
            return ctx.structure_profile(text, _compute)
    except Exception:
        pass
    return _compute()

@dataclass
class DocumentAnalysisBundle:
    filters: Any
    profile: DocumentStructureProfile
    _bundle: SemanticFeatureBundle | None = field(default=None, repr=False, compare=False)
    _signals: Any | None = field(default=None, repr=False, compare=False)
    _metadata_noise: float | None = field(default=None, repr=False, compare=False)
    _instruction_density: float | None = field(default=None, repr=False, compare=False)
    _evidence: SemanticEvidenceBundle | None = field(default=None, repr=False, compare=False)

    def evidence(self, text: str = '') -> SemanticEvidenceBundle:
        if self._evidence is None:
            src = text or (self._bundle.raw.text if self._bundle else '')
            self._evidence = compute_semantic_evidence(src, filters=self.filters, bundle=self._bundle)
        return self._evidence

    def intent(
        self,
        cv: ContentValueSignals | None = None,
        *,
        text: str = '',
    ) -> DocumentIntentScores:
        del cv
        ev = self.evidence(text)
        return _intent_from_evidence(ev)

    def signals(self, text: str) -> Any:
        if self._signals is None:
            self._signals = (self._bundle or SemanticSignalExtractor().extract(text, filters=self.filters)).quality_signals()
        return self._signals

    def metadata_noise(self, text: str) -> float:
        if self._metadata_noise is None:
            from indw.clean.meta.foundation import metadata_noise_ratio
            self._metadata_noise = metadata_noise_ratio(text)
        return self._metadata_noise

    def instruction_density(self, text: str) -> float:
        if self._instruction_density is None:
            from indw.clean.meta.foundation import instruction_wrapper_density
            self._instruction_density = instruction_wrapper_density(text)
        return self._instruction_density

    @property
    def raw_features(self) -> RawDocumentFeatures | None:
        if self._bundle is None:
            return None
        return self._bundle.raw

def build_analysis_bundle(text: str) -> DocumentAnalysisBundle:
    if not text or not text.strip():
        empty = analyze_content_filters('')
        return DocumentAnalysisBundle(empty, DocumentStructureProfile())
    from indw.clean.meta.foundation import instruction_wrapper_density
    sem = _shared_semantic_extractor().extract(text)
    instr = instruction_wrapper_density(
        text,
        lines=sem.raw.lines,
        word_count=sem.raw.word_count,
    )
    evidence = compute_semantic_evidence(text, bundle=sem)
    profile = _structure_profile_from_rep(sem, evidence.representation, instruction_density=instr)
    return DocumentAnalysisBundle(
        sem.filters, profile, _bundle=sem, _instruction_density=instr, _evidence=evidence,
    )


def resolve_analysis_bundle(text: str) -> DocumentAnalysisBundle:
    from indw.extract.core.context import get_document_context

    dctx = get_document_context()
    if dctx is not None:
        return dctx.analysis_bundle(text, lambda: build_analysis_bundle(text))
    return build_analysis_bundle(text)

@dataclass
class DocumentIntentScores:
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

def _intent_from_evidence(evidence: SemanticEvidenceBundle) -> DocumentIntentScores:
    intent = evidence.intent
    return DocumentIntentScores(
        knowledge=intent.knowledge,
        transactional=intent.transactional,
        administrative=intent.administrative,
        promotional=intent.promotional,
        navigational=intent.navigational,
        entertainment=intent.entertainment,
        half_life=intent.half_life,
        transience=intent.transience,
    )

@dataclass
class ContentValueSignals:
    category: ContentCategory = 'educational'
    educational_score: float = 0.0
    technical_score: float = 0.0
    information_density: float = 0.0
    information_density_per_token: float = 0.0
    storytelling_score: float = 0.0
    commercial_score: float = 0.0
    entertainment_score: float = 0.0
    code_score: float = 0.0
    reference_score: float = 0.0
    duplicate_score: float = 0.0
    overall_value_score: float = 0.0
    fact_count: int = 0
    word_count: int = 0
    token_estimate: int = 0
    narrative_filler_score: float = 0.0
    semantic_profile: dict[str, float] = field(default_factory=dict)
    evidence: SemanticEvidenceBundle | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, float | int | str | dict[str, float]]:
        out: dict[str, float | int | str | dict[str, float]] = {
            'category': self.category,
            'educational_score': round(self.educational_score, 4),
            'technical_score': round(self.technical_score, 4),
            'information_density': round(self.information_density, 4),
            'information_density_per_token': round(self.information_density_per_token, 4),
            'storytelling_score': round(self.storytelling_score, 4),
            'commercial_score': round(self.commercial_score, 4),
            'entertainment_score': round(self.entertainment_score, 4),
            'code_score': round(self.code_score, 4),
            'reference_score': round(self.reference_score, 4),
            'duplicate_score': round(self.duplicate_score, 4),
            'overall_value_score': round(self.overall_value_score, 4),
            'narrative_filler_score': round(self.narrative_filler_score, 4),
            'fact_count': self.fact_count,
            'word_count': self.word_count,
            'token_estimate': self.token_estimate,
            'semantic_profile': {k: round(v, 4) for k, v in self.semantic_profile.items()},
        }
        return out


def _signals_from_evidence(evidence: SemanticEvidenceBundle, raw: RawDocumentFeatures, duplicate_ratio: float) -> ContentValueSignals:
    q = evidence.quality
    return ContentValueSignals(
        category=evidence.profile.primary,
        educational_score=q.educational,
        technical_score=q.technical,
        information_density=q.information_density,
        information_density_per_token=q.information_density_per_token,
        storytelling_score=q.storytelling,
        commercial_score=q.commercial,
        entertainment_score=q.entertainment,
        code_score=q.code,
        reference_score=q.reference,
        duplicate_score=duplicate_ratio,
        overall_value_score=q.overall,
        fact_count=q.fact_count,
        word_count=raw.word_count,
        token_estimate=raw.token_estimate,
        narrative_filler_score=q.narrative_filler,
        semantic_profile=dict(evidence.profile.axes),
        evidence=evidence,
    )

def is_information_rich(
    value: ContentValueSignals,
    *,
    text: str = '',
    cfg: SemanticSelectionConfig | None = None,
) -> bool:
    c = cfg or SemanticSelectionConfig()
    if value.evidence is not None:
        ev = value.evidence
        if ev.preserve and not c.section_mode:
            return True
        if ev.utility >= ev.threshold and ev.semantic_strength > ev.uncertainty:
            return True
        if ev.semantic_strength > ev.uncertainty:
            commerce = max(
                ev.negative.get('transactional', 0.0),
                ev.negative.get('promotional', 0.0),
            )
            if commerce < ev.semantic_strength:
                return True
        if text and structural_integrity(DocumentFeatureExtractor().extract(text)) > ev.semantic_strength and not c.section_mode:
            return True
        return False
    if text and _has_preserved_structure(text) and not c.section_mode:
        return True
    baseline = shared_baseline_estimator()
    substance = baseline.baseline([
        value.educational_score, value.technical_score, value.reference_score,
        value.code_score, value.information_density_per_token,
    ])
    noise = baseline.baseline([
        value.narrative_filler_score, value.entertainment_score, value.commercial_score,
    ])
    return substance > noise

def classify_content(
    text: str,
    *,
    source: str = '',
    bundle: DocumentAnalysisBundle | None = None,
) -> ContentCategory:
    del source
    if not text:
        return 'educational'
    ctx = bundle or build_analysis_bundle(text)
    return ctx.evidence(text).profile.primary

def analyze_content_value(
    text: str,
    *,
    source: str = '',
    duplicate_ratio: float = 0.0,
    bundle: DocumentAnalysisBundle | None = None,
) -> ContentValueSignals:
    del source
    if not text or not text.strip():
        return ContentValueSignals()
    ctx = bundle or build_analysis_bundle(text)
    sem = ctx._bundle or SemanticSignalExtractor().extract(text, filters=ctx.filters)
    evidence = ctx._evidence
    if evidence is None or duplicate_ratio != 0.0:
        evidence = compute_semantic_evidence(
            text, bundle=sem, duplicate_ratio=duplicate_ratio, filters=ctx.filters,
        )
        ctx._evidence = evidence
    return _signals_from_evidence(evidence, sem.raw, duplicate_ratio)

def passes_category_quality(
    value: ContentValueSignals,
    *,
    text: str = '',
    cfg: SemanticSelectionConfig | None = None,
) -> tuple[bool, str]:
    return evaluate_semantic_selection(value, text=text, cfg=cfg)

def evaluate_semantic_selection(
    value: ContentValueSignals,
    *,
    text: str = '',
    cfg: SemanticSelectionConfig | None = None,
) -> tuple[bool, str]:
    c = cfg or SemanticSelectionConfig()
    if not c.enabled:
        return True, ''
    evidence = value.evidence
    if evidence is None and text:
        evidence = compute_semantic_evidence(text, enabled=c.enabled)
    elif evidence is None:
        return True, ''
    if evidence.preserve and not c.section_mode:
        return True, ''
    return False, evidence.discard_reason

@dataclass
class TrainingUtilityEstimate:
    utility_score: float = 0.0
    confidence: float = 0.0
    novelty: float = 0.0
    information_density: float = 0.0
    educational_value: float = 0.0
    reasoning_quality: float = 0.0
    technical_quality: float = 0.0
    synthetic_penalty: float = 0.0
    redundancy_penalty: float = 0.0
    hallucination_risk: float = 0.0
    format_quality: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | dict[str, float]]:
        return {
            'utility_score': round(self.utility_score, 4),
            'confidence': round(self.confidence, 4),
            'novelty': round(self.novelty, 4),
            'information_density': round(self.information_density, 4),
            'educational_value': round(self.educational_value, 4),
            'reasoning_quality': round(self.reasoning_quality, 4),
            'technical_quality': round(self.technical_quality, 4),
            'synthetic_penalty': round(self.synthetic_penalty, 4),
            'redundancy_penalty': round(self.redundancy_penalty, 4),
            'hallucination_risk': round(self.hallucination_risk, 4),
            'format_quality': round(self.format_quality, 4),
            'components': {k: round(v, 4) for k, v in self.components.items()},
        }

def estimate_training_utility(
    text: str,
    signals: Any,
    *,
    content_value: ContentValueSignals | None = None,
    domain: str = 'web',
    duplicate_ratio: float = 0.0,
) -> TrainingUtilityEstimate:
    del domain
    cv = content_value or analyze_content_value(text, duplicate_ratio=duplicate_ratio)
    evidence = cv.evidence or compute_semantic_evidence(text, duplicate_ratio=duplicate_ratio)
    q = evidence.quality
    rep = evidence.representation
    format_q = getattr(signals, 'formatting_score', 0.0) * getattr(signals, 'structural_quality', 0.0)
    components = dict(evidence.positive)
    components['format'] = format_q
    return TrainingUtilityEstimate(
        utility_score=evidence.utility,
        confidence=evidence.confidence,
        novelty=evidence.novelty,
        information_density=evidence.information_density_per_token,
        educational_value=evidence.positive.get('educational', q.educational),
        reasoning_quality=evidence.positive.get('reasoning', rep.reasoning),
        technical_quality=max(q.technical, q.code),
        synthetic_penalty=evidence.negative.get('synthetic', rep.corruption),
        redundancy_penalty=evidence.redundancy,
        hallucination_risk=evidence.negative.get('corruption', rep.corruption),
        format_quality=format_q,
        components=components,
    )

@dataclass
class DocumentFoundationScores:
    educational_score: float = 0.0
    technical_score: float = 0.0
    scientific_score: float = 0.0
    code_score: float = 0.0
    historical_score: float = 0.0
    commercial_score: float = 0.0
    legal_score: float = 0.0
    navigation_score: float = 0.0
    metadata_score: float = 0.0
    forum_score: float = 0.0
    instruction_score: float = 0.0
    knowledge_score: float = 0.0
    transactional_score: float = 0.0
    administrative_score: float = 0.0
    transience_score: float = 0.0
    promotional_score: float = 0.0
    entertainment_score: float = 0.0
    half_life_score: float = 0.0
    address_ratio: float = 0.0
    listing_ratio: float = 0.0
    contact_ratio: float = 0.0
    explanation_ratio: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}

def compute_foundation_document_scores(
    text: str,
    *,
    source: str = '',
    content_value: ContentValueSignals | None = None,
    quality_signals: Any | None = None,
    bundle: DocumentAnalysisBundle | None = None,
) -> DocumentFoundationScores:
    del source, quality_signals
    if not text or not text.strip():
        return DocumentFoundationScores()

    ctx = bundle or build_analysis_bundle(text)
    sem = ctx._bundle or SemanticSignalExtractor().extract(text, filters=ctx.filters)
    cv = content_value or analyze_content_value(text, bundle=ctx)
    evidence = cv.evidence or ctx.evidence(text)
    rep = evidence.representation
    q = evidence.quality
    intent = evidence.intent
    profile = ctx.profile
    metadata = ctx.metadata_noise(text)
    instruction = max(ctx.instruction_density(text), PopulationAdaptiveScaler.rate(sem.raw.qa_line_hits, sem.raw.line_count))
    baseline = shared_baseline_estimator()

    from indw.clean.meta.clean import legal_boilerplate_hits
    legal_hits = legal_boilerplate_hits(text)
    legal = PopulationAdaptiveScaler.rate(legal_hits, sem.raw.word_count, sem.raw.line_count)
    legal = min(1.0, legal * baseline.baseline([legal, metadata]))

    scientific = evidence.positive.get('factual', rep.factual) * q.coherence
    historical = baseline.baseline([rep.temporal, rep.referential]) * rep.factual
    commercial = max(cv.commercial_score, intent.transactional)

    return DocumentFoundationScores(
        educational_score=cv.educational_score,
        technical_score=cv.technical_score,
        scientific_score=scientific,
        code_score=cv.code_score,
        historical_score=historical,
        commercial_score=commercial,
        legal_score=legal,
        navigation_score=baseline.baseline([rep.navigational, intent.navigational]),
        metadata_score=metadata,
        forum_score=PopulationAdaptiveScaler.rate(sem.raw.qa_line_hits, sem.raw.line_count),
        instruction_score=instruction,
        knowledge_score=evidence.semantic_strength,
        transactional_score=intent.transactional,
        administrative_score=intent.administrative,
        transience_score=intent.transience,
        promotional_score=intent.promotional,
        entertainment_score=intent.entertainment,
        half_life_score=intent.half_life,
        address_ratio=profile.address_ratio,
        listing_ratio=profile.listing_ratio,
        contact_ratio=profile.contact_ratio,
        explanation_ratio=profile.explanation_ratio,
    )

def foundation_knowledge_strong(
    scores: DocumentFoundationScores,
    cv: ContentValueSignals,
    *,
    source: str = '',
) -> bool:
    del scores, source
    if cv.evidence is None:
        return cv.overall_value_score > 0.0
    ev = cv.evidence
    return ev.preserve or (ev.utility >= ev.threshold and ev.semantic_strength > ev.uncertainty)

def foundation_discard_reason(
    scores: DocumentFoundationScores,
    cv: ContentValueSignals,
    *,
    source: str = '',
) -> str:
    del scores, source
    if cv.evidence is None:
        return ''
    if cv.evidence.preserve:
        return ''
    return cv.evidence.discard_reason
