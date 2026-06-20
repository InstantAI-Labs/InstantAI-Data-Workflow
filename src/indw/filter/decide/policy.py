from __future__ import annotations

import re
from typing import Literal, Optional

from indw.filter.score.types import CanonicalDocumentScore
from indw.filter.spec.pipeline import DecisionHeuristicsPolicy, PipelinePolicy
from indw.filter.content.code import CodeQualitySignals, code_passes
from indw.filter.spec.quality import CurriculumConfig, QualityThresholds, SyntheticDefenseConfig
from indw.clean.semantic.spec import SemanticSelectionConfig
from indw.clean.artifact.evidence import AdaptiveBaselineEstimator
from indw.clean.document.value import TrainingUtilityEstimate, is_information_rich

ContentType = Literal['text', 'code', 'mixed']

HARD_REJECT = frozenset({
    'toxicity', 'pii', 'injection', 'software_piracy', 'proprietary_license',
    'restricted_license', 'incompatible_repo_license', 'paywalled', 'drm_protected',
    'redistribution_prohibited', 'pirated_content', 'language', 'unknown_language',
    'low_language_confidence', 'language_fragmentation', 'synthetic_spam',
    'secret_exposure', 'production_secret', 'credential_leak', 'customer_data',
    'commercial_content', 'entertainment_content', 'narrative_filler',
})

SOFT_ISSUES = frozenset({
    'low_information', 'truncated', 'coherence', 'structural_quality',
    'semantic_low_diversity', 'repeated_spans', 'reasoning_repetition', 'token_spam',
    'curriculum_floor', 'length', 'pii_entities', 'minor_pii', 'pii',
    'entertainment_content', 'low_information_density', 'low_semantic_value',
    'narrative_filler', 'opinion_content', 'boilerplate', 'commercial_content',
    'seo_spam', 'low_alpha', 'low_entropy', 'repetition', 'html_spam',
    'low_code_quality', 'mixed_script', 'script_fragmentation', 'unicode_instability',
    'token_inflation', 'reasoning_instability', 'repeated_token_spans',
    'replay_instability', 'ai_verbosity', 'hallucination_risk', 'unknown_license',
    'attribution_required', 'discovery_artifact', 'website_artifact',
})

_STYLE_MARKETING = re.compile(
    r'(?i)\b(?:'
    r'game[\s-]changer|revolutionary|cutting[\s-]edge|world[\s-]class|'
    r'unlock your potential|transform your life|exclusive offer|'
    r'industry[\s-]leading|best[\s-]in[\s-]class'
    r')\b'
)

def _heuristics(policy: DecisionHeuristicsPolicy | None) -> DecisionHeuristicsPolicy:
    if policy is None:
        raise ValueError('decision heuristics policy required')
    return policy

def detect_content_type(
    text: str,
    *,
    domain: str,
    code: Optional[CodeQualitySignals],
    policy: DecisionHeuristicsPolicy | None = None,
) -> ContentType:
    h = _heuristics(policy)
    code_hits = len(re.findall(r'^\s*(def |class |import |function\s*\(|#include)', text, re.M))
    has_fence = '```' in text
    prose_chars = sum(c.isalpha() for c in text)
    if domain == 'code' or (
        code_hits >= h.code_hits_for_code
        and prose_chars < len(text) * h.prose_ratio_for_code
    ):
        return 'code'
    if (
        code_hits >= h.code_hits_for_mixed
        or has_fence
        or (code and code.educational_score > h.educational_score_mixed and prose_chars > len(text) * h.prose_ratio_mixed)
    ):
        return 'mixed'
    return 'text'

def _ocr_noise_score(text: str, policy: DecisionHeuristicsPolicy | None = None) -> float:
    h = _heuristics(policy)
    if len(text) < h.ocr_min_text_chars:
        return 0.0
    tokens = text.split()
    if not tokens:
        return 0.0
    single_char = sum(1 for t in tokens if len(t) == 1 and not t.isdigit()) / len(tokens)
    broken = text.count('\ufffd')
    pipe_noise = text.count('|') / max(len(text), 1)
    score = 0.0
    if single_char > h.ocr_single_char_ratio:
        score = max(score, min(1.0, single_char * h.ocr_single_char_mult))
    if broken > h.ocr_broken_min:
        score = max(score, min(1.0, broken / max(len(text) / h.ocr_broken_char_div, 1)))
    if pipe_noise > h.ocr_pipe_noise_ratio:
        score = max(score, min(1.0, pipe_noise * h.ocr_pipe_noise_mult))
    return score

def quality_score_10_from_doc(
    doc: CanonicalDocumentScore,
    text: str,
    *,
    policy: DecisionHeuristicsPolicy | None = None,
) -> float:
    baseline = AdaptiveBaselineEstimator()
    utility = doc.training_utility.utility_score if doc.training_utility else doc.score
    cv = doc.content_value
    evidence = cv.evidence if cv is not None else None

    if evidence is not None:
        from indw.clean.artifact.evidence_util import evidence_margin

        margin = evidence_margin(evidence.utility, evidence.threshold, evidence.uncertainty)
        q10 = (evidence.utility + max(0.0, margin)) * 10.0
        if cv is not None and (evidence.preserve or is_information_rich(cv, text=text)):
            q10 = max(q10, evidence.semantic_strength * evidence.coherence * 10.0)
        noise = baseline.baseline(list(evidence.negative.values())) if evidence.negative else 0.0
        q10 *= max(0.0, 1.0 - noise * evidence.uncertainty)
        return max(0.0, min(10.0, q10))

    sig = doc.signals
    substance = baseline.baseline([
        utility, doc.score, sig.factual_density, sig.educational_value, sig.reasoning_density,
    ])
    noise = baseline.baseline([
        sig.truncation_score, sig.boilerplate_score, sig.commercial_score,
        sig.seo_spam_score, sig.low_information_score, _ocr_noise_score(text, policy),
    ])
    return max(0.0, min(10.0, substance * (1.0 - noise) * 10.0))

def build_signals(
    doc: CanonicalDocumentScore,
    *,
    duplicate: bool,
    near_duplicate: bool,
    policy: DecisionHeuristicsPolicy | None = None,
) -> dict[str, bool]:
    h = _heuristics(policy)
    sig = doc.signals
    baseline = AdaptiveBaselineEstimator()
    spam_floor = baseline.baseline([sig.seo_spam_score, sig.commercial_score])
    return {
        'boilerplate': sig.boilerplate_score > baseline.baseline([sig.boilerplate_score, sig.low_information_score]),
        'spam': spam_floor > baseline.spread([sig.seo_spam_score, sig.commercial_score, sig.token_spam_score]),
        'pii': doc.pii_score > baseline.baseline([doc.pii_score, 0.0]) or doc.pii_entities > 0,
        'toxicity': doc.toxicity_score > baseline.baseline([doc.toxicity_score, sig.injection_score]),
        'software_piracy': sig.software_piracy_score > baseline.baseline([sig.software_piracy_score, sig.injection_score]),
        'low_information': sig.low_information_score > baseline.baseline([sig.low_information_score, sig.boilerplate_score]),
        'ai_verbosity': sig.ai_verbosity_score > baseline.baseline([sig.ai_verbosity_score, sig.low_information_score]),
        'hallucination_risk': sig.hallucination_risk_score > baseline.baseline([sig.hallucination_risk_score, sig.template_synthetic_score]),
        'duplicate': duplicate or near_duplicate,
        'invalid_code': bool(
            doc.code_signals
            and (
                doc.code_signals.syntax_balance < h.invalid_syntax_balance
                or doc.code_signals.generated_score > h.generated_score_max
            )
        ),
        'secret_detected': (
            doc.pii_score > h.secret_pii_score
            or doc.pii_reason in {'secret_exposure', 'production_secret', 'credential_leak'}
        ),
        'license_unknown': doc.license == 'Unknown',
        'license_attribution': doc.attribution_required,
        'license_restricted': doc.license in ('Proprietary', 'Restricted', 'GPL'),
    }

def soft_issues(
    doc: CanonicalDocumentScore,
    text: str,
    th: QualityThresholds,
    *,
    semantic_selection: Optional[SemanticSelectionConfig] = None,
    policy: DecisionHeuristicsPolicy | None = None,
) -> list[str]:
    sig = doc.signals
    baseline = AdaptiveBaselineEstimator()
    issues: list[str] = []
    if sig.boilerplate_score > th.warn_boilerplate_score:
        issues.append('boilerplate')
    if sig.commercial_score > th.warn_commercial_score:
        issues.append('commercial')
    if sig.seo_spam_score > th.warn_seo_spam_score:
        issues.append('seo_spam')
    if sig.low_information_score > th.warn_low_information_score:
        issues.append('low_information')
    if doc.artifact_ratio > th.warn_discovery_artifact_score:
        issues.append('website_artifact')
    if sig.truncation_score > th.warn_truncation_score:
        issues.append('truncated')
    pii_floor = baseline.baseline([doc.pii_score, sig.injection_score])
    if doc.pii_score > pii_floor and not doc.pii_reason:
        issues.append('minor_pii')
    domain_noise = baseline.baseline([sig.seo_spam_score, sig.commercial_score, sig.boilerplate_score])
    if doc.domain in ('web', 'conversation') and domain_noise > baseline.spread([sig.seo_spam_score, sig.commercial_score]):
        issues.append('low_quality_domain')
    if doc.code_signals and doc.code_signals.educational_score < baseline.baseline([
        doc.code_signals.educational_score, doc.code_signals.syntax_balance,
    ]):
        issues.append('trivial_code')
    if _STYLE_MARKETING.search(text):
        issues.append('marketing_tone')
    if doc.content_value is not None:
        cv = doc.content_value
        sem_cfg = semantic_selection or SemanticSelectionConfig()
        ent_floor = baseline.baseline([cv.entertainment_score, cv.narrative_filler_score, cv.storytelling_score])
        edu_floor = baseline.baseline([cv.educational_score, cv.technical_score, cv.reference_score])
        if (
            not is_information_rich(cv, text=text, cfg=sem_cfg)
            and cv.entertainment_score > ent_floor
            and cv.educational_score < edu_floor
        ):
            issues.append('entertainment_content')
        if cv.commercial_score > th.warn_commercial_score and not is_information_rich(cv, text=text, cfg=sem_cfg):
            issues.append('commercial')
    ocr = _ocr_noise_score(text, policy)
    if ocr > baseline.baseline([ocr, sig.html_score, sig.injection_score]):
        issues.append('ocr_noise')
    if doc.mixed_language and doc.lang_fragmentation > baseline.baseline([doc.lang_fragmentation, sig.char_repetition]):
        issues.append('language_mixing')
    if sig.ai_verbosity_score > baseline.baseline([sig.ai_verbosity_score, sig.low_information_score]):
        issues.append('ai_verbosity')
    if sig.hallucination_risk_score > baseline.baseline([sig.hallucination_risk_score, sig.template_synthetic_score]):
        issues.append('hallucination_risk')
    return issues

def collect_adaptive_quality_issues(
    doc: CanonicalDocumentScore,
    text: str,
    th: QualityThresholds,
    *,
    semantic_selection: Optional[SemanticSelectionConfig] = None,
    synthetic_defense: Optional[SyntheticDefenseConfig] = None,
    curriculum: Optional[CurriculumConfig] = None,
    policy: DecisionHeuristicsPolicy | None = None,
) -> list[str]:
    h = _heuristics(policy)
    sig = doc.signals
    baseline = AdaptiveBaselineEstimator()
    issues: list[str] = []
    cv = doc.content_value
    evidence = cv.evidence if cv is not None else None
    sem_cfg = semantic_selection or SemanticSelectionConfig()
    syn = synthetic_defense or SyntheticDefenseConfig()
    cur = curriculum or CurriculumConfig()

    informative = baseline.baseline([sig.factual_density, sig.educational_value, sig.reasoning_density])
    has_useful = informative > baseline.spread([
        sig.factual_density, sig.educational_value, sig.reasoning_density,
    ]) or (doc.code_signals is not None and code_passes(doc.code_signals))
    if evidence is not None:
        has_useful = has_useful or (evidence.preserve and not sem_cfg.section_mode) or is_information_rich(cv, text=text, cfg=sem_cfg)
    elif cv is not None:
        has_useful = has_useful or is_information_rich(cv, text=text, cfg=sem_cfg)
    high_value = informative > baseline.baseline([informative, doc.score]) or (
        doc.code_signals is not None and code_passes(doc.code_signals)
    )
    if evidence is not None:
        high_value = high_value or (evidence.preserve and not sem_cfg.section_mode and evidence.utility >= evidence.threshold)

    if sig.length < th.min_chars and not has_useful:
        issues.append('length')
    elif sig.length > th.max_chars and not high_value:
        issues.append('length')

    if sem_cfg.enabled and evidence is not None and not evidence.preserve:
        issues.append(evidence.discard_reason or 'low_semantic_value')
    elif sem_cfg.enabled and evidence is not None and evidence.preserve and sem_cfg.section_mode:
        if evidence.utility < th.min_informative_density:
            issues.append(evidence.discard_reason or 'low_semantic_value')

    trunc_peer = baseline.baseline([sig.truncation_score, sig.boilerplate_score, sig.coherence_score])
    if (
        sig.truncation_score > trunc_peer
        and sig.truncation_score > baseline.spread([sig.truncation_score, sig.coherence_score])
        and not (
            doc.domain in h.truncation_exempt_domains
            and sig.truncation_score < h.truncation_wiki_exempt
        )
    ):
        issues.append('truncated')

    noise_peer = baseline.baseline([sig.boilerplate_score, sig.commercial_score, sig.seo_spam_score])
    if sig.boilerplate_score > noise_peer:
        issues.append('boilerplate')
    if sig.commercial_score > noise_peer:
        issues.append('commercial_content')
    if sig.seo_spam_score > noise_peer:
        issues.append('seo_spam')

    if doc.artifact_ratio > th.max_discovery_artifact_score and not has_useful:
        issues.append('discovery_artifact')
    elif doc.artifact_ratio > th.warn_discovery_artifact_score:
        issues.append('website_artifact')

    if (
        sig.low_information_score > baseline.baseline([sig.low_information_score, sig.boilerplate_score])
        and informative < baseline.baseline([informative, th.min_informative_density])
        and (cv is None or (
            cv.information_density < th.min_informative_density
            and cv.overall_value_score < baseline.baseline([cv.overall_value_score, cv.entertainment_score])
        ))
    ):
        issues.append('low_information')

    technical = doc.domain in h.technical_domains
    if (
        sig.alpha_ratio < baseline.baseline([sig.alpha_ratio, sig.structural_quality])
        and not technical
        and not (doc.code_signals and doc.code_signals.syntax_balance >= h.alpha_syntax_floor)
        and sig.reasoning_density < baseline.spread([sig.reasoning_density, sig.factual_density])
    ):
        issues.append('low_alpha')

    if sig.char_entropy < baseline.baseline([sig.char_entropy / 5.0, sig.word_diversity]) * 5.0:
        issues.append('low_entropy')
    if max(sig.line_repetition, sig.char_repetition) > baseline.baseline([sig.line_repetition, sig.char_repetition]):
        issues.append('repetition')
    if sig.html_score > baseline.baseline([sig.html_score, sig.injection_score]) and doc.domain != 'code':
        issues.append('html_spam')
    if sig.token_spam_score > baseline.baseline([sig.token_spam_score, sig.seo_spam_score]):
        issues.append('token_spam')
    if sig.structural_quality < baseline.baseline([sig.structural_quality, sig.coherence_score]):
        issues.append('structural_quality')
    if sig.coherence_score < baseline.baseline([sig.coherence_score, sig.structural_quality]):
        issues.append('coherence')
    if sig.reasoning_repetition > baseline.baseline([sig.reasoning_repetition, sig.line_repetition]):
        issues.append('reasoning_repetition')

    if syn.enabled and sig.synthetic_score > baseline.baseline([sig.synthetic_score, sig.template_synthetic_score]):
        issues.append('synthetic_spam')
    if syn.enabled and sig.semantic_diversity < baseline.baseline([sig.semantic_diversity, sig.word_diversity]):
        issues.append('semantic_low_diversity')
    if syn.enabled and sig.repeated_span_score > baseline.baseline([sig.repeated_span_score, sig.line_repetition]):
        issues.append('repeated_spans')
    if cur.enabled and doc.score < baseline.baseline([doc.score, cur.min_stage_score]):
        issues.append('curriculum_floor')

    if doc.domain == 'code' and doc.code_signals and not code_passes(doc.code_signals):
        if not (
            doc.code_signals.syntax_balance >= h.code_syntax_floor
            and doc.code_signals.generated_score < h.generated_score_max
        ):
            issues.append('low_code_quality')

    if doc.mixed_language and doc.lang_fragmentation > baseline.baseline([doc.lang_fragmentation, sig.char_repetition]):
        issues.append('script_fragmentation')
    if doc.token_inflation_risk > baseline.baseline([doc.token_inflation_risk, 1.0 - doc.multilingual_quality]):
        issues.append('token_inflation')
    if doc.chars_per_token > 0 and doc.chars_per_token < baseline.baseline([doc.chars_per_token, h.chars_per_token_floor]):
        issues.append('token_inflation')

    return issues

def decision_confidence(
    doc: CanonicalDocumentScore,
    *,
    utility: TrainingUtilityEstimate | None,
    signal_penalty: float,
    issue_count: int,
    policy: DecisionHeuristicsPolicy | None = None,
) -> float:
    h = _heuristics(policy)
    base = utility.confidence if utility is not None else h.confidence_base
    base *= max(h.signal_penalty_floor, 1.0 - signal_penalty * h.signal_penalty_weight)
    if issue_count >= h.issue_count_penalty:
        base *= h.issue_confidence_mult
    if (
        utility is not None
        and utility.educational_value > h.synthetic_edu_threshold
        and utility.synthetic_penalty > h.synthetic_penalty_threshold
    ):
        base *= h.synthetic_confidence_mult
    if doc.language_confidence > 0:
        base = min(
            h.confidence_max,
            base * h.language_blend_base + doc.language_confidence * h.language_blend_weight,
        )
    return max(h.confidence_min, min(h.confidence_max, base))
