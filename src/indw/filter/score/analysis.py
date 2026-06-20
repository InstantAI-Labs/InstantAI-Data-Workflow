from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from indw.filter.language.config import LanguagePolicyConfig
from indw.filter.language.fast_detector import FastLanguageDetector
from indw.filter.language.detect import LanguageAssessment, LanguageIdentifier
from indw.filter.language.script import scan_script_text
from indw.filter.language.script_metrics import compute_multilingual_metrics
from indw.filter.language.script_policy import MultilingualPolicyConfig
from indw.filter.spec.quality import QualityThresholds
from indw.filter.content.code import CodeQualitySignals, analyze_code
from indw.filter.content.domain import domain_from_text
from indw.filter.score.signals import QualitySignals
from indw.clean.document.value import (
    ContentValueSignals,
    TrainingUtilityEstimate,
    analyze_content_value,
    build_analysis_bundle,
    estimate_training_utility,
)
from indw.filter.license.config import LicensePolicyConfig
from indw.filter.license.detector import LicenseAssessment, LicenseDetector
from indw.filter.pii.config import PiiPolicyConfig
from indw.filter.pii.detect import PiiAssessment, PiiDetector
from indw.filter.toxicity.config import ToxicityPolicyConfig
from indw.filter.toxicity.detect import ToxicityAssessment, ToxicityDetector

if TYPE_CHECKING:
    from indw.filter.language.bridge import LiveTokenizerEncoder

@dataclass
class DocumentAnalysis:
    text: str
    scan: str
    full_len: int
    source: str
    duplicate_ratio: float
    signals: QualitySignals
    content_value: ContentValueSignals
    domain: str
    profile: Any
    lang: str
    lang_frag: float
    lang_assessment: Optional[LanguageAssessment]
    mlang: Any
    code_sig: Optional[CodeQualitySignals]
    utility: TrainingUtilityEstimate
    toxicity: Optional[ToxicityAssessment]
    pii: Optional[PiiAssessment]
    license_assessment: Optional[LicenseAssessment]
    artifact_ratio: float
    artifact_components: dict[str, float]
    reject_reason: str
    tokenizer_runtime: Any = None
    token_ids: Optional[list[int]] = None
    bundle: Any = None

def _sample_text(text: str, *, min_chars: int, sample_limit: int) -> tuple[str, int]:
    full_len = len(text)
    if full_len <= sample_limit:
        return text, full_len
    if sample_limit > 4096:
        tail = min(2048, sample_limit // 4)
        return text[: sample_limit - tail] + text[-tail:], full_len
    return text[:sample_limit], full_len

def analyze_document(
    text: str,
    *,
    source: str = '',
    duplicate_ratio: float = 0.0,
    thresholds: Optional[QualityThresholds] = None,
    multilingual_policy: Optional[MultilingualPolicyConfig] = None,
    tokenizer_encoder: Optional[LiveTokenizerEncoder] = None,
    toxicity_policy: Optional[ToxicityPolicyConfig] = None,
    toxicity_detector: Optional[ToxicityDetector] = None,
    pii_policy: Optional[PiiPolicyConfig] = None,
    pii_detector: Optional[PiiDetector] = None,
    language_policy: Optional[LanguagePolicyConfig] = None,
    language_identifier: Optional[LanguageIdentifier] = None,
    license_policy: Optional[LicensePolicyConfig] = None,
    license_detector: Optional[LicenseDetector] = None,
    provenance: Optional[dict[str, Any]] = None,
    skip_expensive: bool = False,
    analysis_scan: Optional[str] = None,
    analysis_full_len: Optional[int] = None,
    analysis_bundle: Any = None,
    prechecked_language: Optional[LanguageAssessment] = None,
) -> DocumentAnalysis:
    th = thresholds or QualityThresholds()
    mpol = multilingual_policy or MultilingualPolicyConfig()
    if analysis_bundle is not None:
        scan = analysis_scan if analysis_scan is not None else text
        full_len = analysis_full_len if analysis_full_len is not None else len(text)
        ctx = analysis_bundle
    else:
        scan, full_len = _sample_text(
            text,
            min_chars=th.min_chars,
            sample_limit=max(th.min_chars, th.score_sample_chars),
        )
        ctx = build_analysis_bundle(scan)
    sig = ctx.signals(scan)
    if full_len != len(scan):
        sig.length = full_len
    content_value = analyze_content_value(
        scan, source=source, duplicate_ratio=duplicate_ratio, bundle=ctx,
    )
    domain = domain_from_text(scan, source_hint=source)
    lang_pol = language_policy or LanguagePolicyConfig.resolve()
    lang_ident = language_identifier
    if lang_ident is None and lang_pol.enabled and not (
        prechecked_language is not None and lang_pol.skip_post_clean_detection
    ):
        lang_ident = LanguageIdentifier(lang_pol)
    segment_min_chars = (
        lang_pol.mixed.min_segment_chars
        if lang_pol.enabled
        and lang_ident is not None
        and lang_pol.mixed.enabled
        and not lang_pol.english_only
        else None
    )
    script_scan = scan_script_text(scan, segment_min_chars=segment_min_chars)
    profile = script_scan.profile
    lang_assessment: Optional[LanguageAssessment] = None
    if prechecked_language is not None and lang_pol.skip_post_clean_detection:
        lang_assessment = prechecked_language
        lang = lang_assessment.primary_language
        lang_frag = lang_assessment.fragmentation
    elif lang_ident is not None and lang_pol.enabled:
        lang_assessment = lang_ident.assess(
            scan,
            domain=domain,
            script_segments=script_scan.segments,
        )
        lang = lang_assessment.primary_language
        lang_frag = lang_assessment.fragmentation
    else:
        det = FastLanguageDetector(lang_pol.detector)
        dist = det.predict_distribution(scan)
        lang = max(dist, key=dist.get) if dist else 'und'
        lang_frag = profile.fragmentation_risk
    runtime_metrics = None
    token_ids: Optional[list[int]] = None
    if tokenizer_encoder is not None:
        token_ids, runtime_metrics = tokenizer_encoder.encode_metrics(
            text,
            profile=profile,
            text_delimiter_density=sig.delimiter_density,
            text_reasoning_density=sig.reasoning_density,
            structural_quality=sig.structural_quality,
        )
    mlang = compute_multilingual_metrics(
        text,
        profile,
        reasoning_density=sig.reasoning_density,
        structural_quality=sig.structural_quality,
        semantic_diversity=sig.semantic_diversity,
        tokenizer_runtime=runtime_metrics,
        policy_target_cpt=mpol.target_chars_per_token,
    )
    if lang_assessment is None:
        lang_frag = mlang.fragmentation
    code_sig = analyze_code(text) if domain == 'code' else None
    utility = estimate_training_utility(
        scan,
        sig,
        content_value=content_value,
        domain=domain,
        duplicate_ratio=duplicate_ratio,
    )
    tox_policy = toxicity_policy or ToxicityPolicyConfig()
    detector = toxicity_detector
    if detector is None and tox_policy.enabled:
        detector = ToxicityDetector(tox_policy)
    toxicity: Optional[ToxicityAssessment] = None
    if not skip_expensive and detector is not None and tox_policy.enabled:
        toxicity = detector.assess(
            text,
            factual_density=sig.factual_density,
            educational_value=sig.educational_value,
        )
    pii_pol = pii_policy or PiiPolicyConfig()
    pii_det = pii_detector
    if pii_det is None and pii_pol.enabled:
        pii_det = PiiDetector(pii_pol)
    pii: Optional[PiiAssessment] = None
    if not skip_expensive and pii_det is not None and pii_pol.enabled:
        pii = pii_det.assess(text)
    reason = ''
    if sig.software_piracy_score > th.max_software_piracy_score:
        reason = 'software_piracy'
    if not reason and sig.injection_score > th.max_prompt_injection_score:
        reason = 'injection'
    if not reason and mpol.enabled and profile.mixed_script_score > mpol.max_mixed_script_score:
        reason = 'mixed_script'
    if not reason and mpol.enabled and mlang.fragmentation > mpol.max_fragmentation_risk:
        reason = 'script_fragmentation'
    if not reason and mpol.enabled and profile.unicode_instability > mpol.max_unicode_instability:
        reason = 'unicode_instability'
    if not reason and mpol.enabled and mlang.reasoning_stability < mpol.min_reasoning_stability and sig.reasoning_density > 0.05:
        reason = 'reasoning_instability'
    if not reason and mpol.enabled and mlang.repeated_token_span_score > 0.72:
        reason = 'repeated_token_spans'
    if not reason and mpol.enabled and mlang.replay_stability < 0.85:
        reason = 'replay_instability'
    if not reason and toxicity is not None and toxicity.final.should_reject:
        reason = toxicity.final.toxicity_reason or 'toxicity'
    if not reason and pii is not None and pii.risk.should_hard_reject:
        reason = pii.risk.reason or 'pii'
    if not reason and pii is not None and pii.risk.should_reject:
        reason = pii.risk.reason or 'pii_entities'
    if not reason and lang_assessment is not None and lang_assessment.should_reject:
        reason = lang_assessment.reject_reason or 'language'
    lic_pol = license_policy or LicensePolicyConfig()
    lic_det = license_detector
    if lic_det is None and lic_pol.enabled:
        lic_det = LicenseDetector(lic_pol)
    license_assessment: Optional[LicenseAssessment] = None
    if lic_det is not None and lic_pol.enabled:
        prov = provenance or {}
        license_assessment = lic_det.assess(
            text,
            source=source,
            url=str(prov.get('url') or ''),
            domain=str(prov.get('domain') or ''),
            language=lang,
            crawl_date=str(prov.get('crawl_date') or ''),
            declared_license=str(prov.get('license') or prov.get('declared_license') or ''),
            repo_license_text=str(prov.get('repo_license_text') or ''),
            hf_id=str(prov.get('hf_id') or ''),
            piracy_score=sig.software_piracy_score,
            meta=prov if isinstance(prov.get('meta'), dict) else prov,
        )
        if not reason and license_assessment.filter_action == 'REMOVE' and license_assessment.reject_reason:
            reason = license_assessment.reject_reason
    from indw.clean.gate.evaluate import compute_artifact_ratio

    artifact_ratio, artifact_components = compute_artifact_ratio(
        scan,
        signals=sig,
        content_value=content_value,
        bundle=ctx,
        include_discovery=True,
    )
    return DocumentAnalysis(
        text=text,
        scan=scan,
        full_len=full_len,
        source=source,
        duplicate_ratio=duplicate_ratio,
        signals=sig,
        content_value=content_value,
        domain=domain,
        profile=profile,
        lang=lang,
        lang_frag=lang_frag,
        lang_assessment=lang_assessment,
        mlang=mlang,
        code_sig=code_sig,
        utility=utility,
        toxicity=toxicity,
        pii=pii,
        license_assessment=license_assessment,
        artifact_ratio=artifact_ratio,
        artifact_components=artifact_components,
        reject_reason=reason,
        tokenizer_runtime=runtime_metrics,
        token_ids=token_ids,
        bundle=ctx,
    )
