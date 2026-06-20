from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from indw.config.resolve import PipelineConfigContext, ctx_from_gate_or_quality
from indw.filter.score.analysis import analyze_document
from indw.filter.score.types import CanonicalDocumentScore
from indw.filter.decide.engine import DecisionEngine, PipelineDecision
from indw.filter.spec.document import CorpusDocument
from indw.filter.score.builder import build_canonical_score
from indw.filter.spec.quality import QualityPipelineConfig

if TYPE_CHECKING:
    from indw.filter.gate.quality import QualityGate

def score_document_canonical(
    text: str,
    *,
    source: str = '',
    duplicate_ratio: float = 0.0,
    gate: Optional['QualityGate'] = None,
    gate_ctx: Optional[PipelineConfigContext] = None,
    quality_config: Optional[QualityPipelineConfig] = None,
    provenance: Optional[dict[str, Any]] = None,
    skip_expensive: bool = False,
    analysis_scan: Optional[str] = None,
    analysis_full_len: Optional[int] = None,
    analysis_bundle: Any = None,
    prechecked_language: Any = None,
    **detector_kwargs: Any,
) -> CanonicalDocumentScore:
    ctx = gate_ctx or ctx_from_gate_or_quality(gate, quality_config)
    cfg = ctx.quality
    analysis = analyze_document(
        text,
        source=source,
        duplicate_ratio=duplicate_ratio,
        thresholds=cfg.thresholds,
        multilingual_policy=gate.multilingual_policy if gate else None,
        tokenizer_encoder=gate.tokenizer_encoder if gate else detector_kwargs.get('tokenizer_encoder'),
        toxicity_policy=ctx.toxicity if gate is None else gate._toxicity_policy,
        toxicity_detector=gate._toxicity_detector if gate else detector_kwargs.get('toxicity_detector'),
        pii_policy=ctx.pii if gate is None else gate._pii_policy,
        pii_detector=gate._pii_detector if gate else detector_kwargs.get('pii_detector'),
        language_policy=ctx.language if gate is None else gate._language_policy,
        language_identifier=gate._language_identifier if gate else detector_kwargs.get('language_identifier'),
        license_policy=ctx.license if gate is None else gate._license_policy,
        license_detector=gate._license_detector if gate else detector_kwargs.get('license_detector'),
        provenance=provenance,
        skip_expensive=skip_expensive,
        analysis_scan=analysis_scan,
        analysis_full_len=analysis_full_len,
        analysis_bundle=analysis_bundle,
        prechecked_language=prechecked_language,
    )
    return build_canonical_score(analysis, policy=ctx.pipeline)

def decide_document_canonical(
    score: CanonicalDocumentScore,
    text: str,
    *,
    doc: CorpusDocument | None = None,
    gate: Optional['QualityGate'] = None,
    gate_ctx: Optional[PipelineConfigContext] = None,
    quality_config: Optional[QualityPipelineConfig] = None,
    exact_duplicate: bool = False,
    near_duplicate: bool = False,
) -> PipelineDecision:
    ctx = gate_ctx or ctx_from_gate_or_quality(gate, quality_config)
    engine = DecisionEngine(ctx, calibrator=gate.calibrator if gate else None)
    decision = engine.decide(
        score,
        text,
        doc=doc,
        exact_duplicate=exact_duplicate,
        near_duplicate=near_duplicate,
    )
    engine.apply_to_score(score, decision)
    return decision

def process_document_canonical(
    text: str,
    *,
    source: str = '',
    duplicate_ratio: float = 0.0,
    gate: Optional['QualityGate'] = None,
    gate_ctx: Optional[PipelineConfigContext] = None,
    quality_config: Optional[QualityPipelineConfig] = None,
    provenance: Optional[dict[str, Any]] = None,
    exact_duplicate: bool = False,
    near_duplicate: bool = False,
    doc: CorpusDocument | None = None,
) -> tuple[CanonicalDocumentScore, PipelineDecision]:
    ctx = gate_ctx or ctx_from_gate_or_quality(gate, quality_config)
    score = score_document_canonical(
        text,
        source=source,
        duplicate_ratio=duplicate_ratio,
        gate=gate,
        gate_ctx=ctx,
        provenance=provenance,
    )
    decision = decide_document_canonical(
        score,
        text,
        doc=doc,
        gate=gate,
        gate_ctx=ctx,
        exact_duplicate=exact_duplicate,
        near_duplicate=near_duplicate,
    )
    return score, decision
