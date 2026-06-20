from __future__ import annotations

from indw.filter.score.analysis import DocumentAnalysis
from indw.filter.score.types import CanonicalDocumentScore
from indw.config.validation import ConfigResolutionError
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.content.code import analyze_code_dump
from indw.filter.refine.corpus import DocumentAnalysisCache, compute_document_metrics
from indw.filter.refine.truncation import TruncationResult
from indw.filter.refine.truncation import analyze_truncation
from indw.filter.score.artifacts import analyze_artifact_signals
from indw.filter.score.continuous import build_continuous_scores
from indw.clean.artifact.evidence import structural_integrity
from indw.clean.artifact.evidence_features import shared_feature_extractor
from indw.clean.artifact.evidence import apply_tokenizer_telemetry
from indw.filter.score.adaptive import adaptive_document_score

def build_canonical_score(
    analysis: DocumentAnalysis,
    *,
    policy: PipelinePolicy,
) -> CanonicalDocumentScore:
    if policy is None:
        raise ConfigResolutionError('build_canonical_score requires pipeline policy from PipelineConfigContext')
    text = analysis.text
    cache = DocumentAnalysisCache()
    if analysis.bundle is not None:
        cache.bundle = analysis.bundle
        cache.signals = analysis.signals
        cache.content_value = analysis.content_value
    metrics = compute_document_metrics(
        text,
        signals=analysis.signals,
        source=analysis.source,
        cache=cache,
    )
    cv = analysis.content_value
    sig = analysis.signals
    evidence = cv.evidence if cv is not None else None
    trunc_res = analyze_truncation(text)
    if trunc_res.probability < metrics.truncation_probability:
        trunc_res = TruncationResult(probability=metrics.truncation_probability)
    dump_res = analyze_code_dump(text)
    art_sig = analyze_artifact_signals(text)
    continuous = build_continuous_scores(
        knowledge_density=metrics.knowledge_density,
        educational_value=metrics.educational_value,
        factual_density=metrics.factual_density,
        coherence=metrics.coherence,
        language_quality=metrics.language_quality,
        code_quality=metrics.code_quality,
        evidence=evidence,
        cv=cv,
        signals=sig,
        trunc=trunc_res,
        code_dump=dump_res,
        artifact_ratio=analysis.artifact_ratio,
        duplicate_ratio=analysis.duplicate_ratio,
        artifact_signals=art_sig,
        text_chars=len(text),
        chars_per_token=policy.scoring.chars_per_token,
        composite_weights=policy.scoring.weights,
        artifact_penalty=policy.scoring.artifact_penalty,
        noise_penalty=policy.scoring.noise_penalty,
        duplication_penalty=policy.scoring.duplication_penalty,
    )
    utility_norm = adaptive_document_score(
        sig,
        domain=analysis.domain,
        code=analysis.code_sig,
        content_value=cv,
        multilingual_quality=analysis.mlang.multilingual_quality,
        text=text,
    )
    if analysis.tokenizer_runtime is not None and evidence is not None:
        apply_tokenizer_telemetry(evidence, analysis.tokenizer_runtime)
        utility_norm = max(utility_norm, evidence.utility)
    elif analysis.utility.utility_score > 0:
        utility_norm = max(utility_norm, analysis.utility.utility_score)

    continuous_w = max(0.0, policy.scoring.continuous_weight)
    utility_w = max(0.0, policy.scoring.utility_weight)
    blend_denom = continuous_w + utility_w
    composite = continuous.composite
    if utility_norm > 0 and blend_denom > 0:
        composite = (
            continuous_w * continuous.composite + utility_w * utility_norm * 100.0
        ) / blend_denom
    composite = max(0.0, min(100.0, composite))

    raw = analysis.bundle.raw_features if analysis.bundle is not None else None
    if raw is None:
        raw = shared_feature_extractor().extract(analysis.scan or text)
    artifact = max(
        continuous.artifact_severity,
        analysis.artifact_ratio * 100.0,
        art_sig.severity * 100.0,
    )
    blend = policy.scoring.context_blend
    novelty = 0.0
    if evidence is not None and hasattr(evidence, 'novelty'):
        novelty = min(100.0, evidence.novelty * 100.0)
    structural = max(continuous.format_quality, structural_integrity(raw) * 100.0)
    context = max(
        0.0,
        min(
            100.0,
            metrics.coherence * blend.coherence
            + structural * blend.structural
            + (100.0 - artifact) * blend.artifact_inverse
            - metrics.truncation_probability * blend.truncation_penalty,
        ),
    )
    components = dict(analysis.artifact_components or {})
    components['synthetic'] = sig.synthetic_score
    components['seo'] = sig.seo_spam_score
    components['truncation'] = sig.truncation_score
    components['utility_normalized'] = utility_norm

    return CanonicalDocumentScore(
        knowledge=metrics.knowledge_density,
        educational_value=metrics.educational_value,
        technical_value=metrics.code_quality,
        artifact_contamination=artifact,
        coherence=metrics.coherence,
        information_density=continuous.information_density,
        novelty=novelty,
        structural_integrity=structural,
        context_consistency=context,
        composite=composite,
        components=components,
        signals=sig,
        domain=analysis.domain,
        language=analysis.lang,
        lang_fragmentation=analysis.lang_frag,
        language_confidence=analysis.lang_assessment.confidence if analysis.lang_assessment else 0.0,
        mixed_language=analysis.lang_assessment.mixed_language if analysis.lang_assessment else False,
        language_assessment=analysis.lang_assessment,
        script_profile=analysis.profile,
        multilingual_quality=analysis.mlang.multilingual_quality,
        chars_per_token=analysis.mlang.chars_per_token,
        token_inflation_risk=analysis.mlang.token_inflation_risk,
        tokenizer_runtime=analysis.tokenizer_runtime,
        tokenizer_ids=analysis.token_ids,
        code_signals=analysis.code_sig,
        toxicity_score=analysis.toxicity.final.final_toxicity_score if analysis.toxicity else 0.0,
        toxicity_reason=analysis.toxicity.final.toxicity_reason if analysis.toxicity else None,
        toxicity_assessment=analysis.toxicity,
        pii_score=analysis.pii.risk.pii_score if analysis.pii else 0.0,
        pii_entities=len(analysis.pii.entities.entities) if analysis.pii else 0,
        pii_secrets=len(analysis.pii.secrets.spans) if analysis.pii else 0,
        pii_reason=analysis.pii.risk.reason if analysis.pii else None,
        pii_assessment=analysis.pii,
        reject_reason=analysis.reject_reason,
        content_value=cv,
        content_category=cv.category if cv is not None else 'unknown',
        training_utility=analysis.utility,
        license=analysis.license_assessment.license if analysis.license_assessment else 'Unknown',
        license_confidence=analysis.license_assessment.license_confidence if analysis.license_assessment else 0.0,
        copyright_status=analysis.license_assessment.copyright_status if analysis.license_assessment else 'unknown',
        attribution_required=analysis.license_assessment.attribution_required if analysis.license_assessment else False,
        document_type=analysis.license_assessment.document_type if analysis.license_assessment else 'unknown',
        license_assessment=analysis.license_assessment,
        artifact_ratio=analysis.artifact_ratio,
        artifact_components=analysis.artifact_components,
        utility_normalized=utility_norm,
    )
