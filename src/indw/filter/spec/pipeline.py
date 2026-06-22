from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

from indw.config.defaults import (
    DECISION_ALPHA_SYNTAX_FLOOR,
    DECISION_CHARS_PER_TOKEN_FLOOR,
    DECISION_CODE_HITS_FOR_CODE,
    DECISION_CODE_HITS_FOR_MIXED,
    DECISION_CODE_SYNTAX_FLOOR,
    DECISION_CONFIDENCE_BASE,
    DECISION_CONFIDENCE_MAX,
    DECISION_CONFIDENCE_MIN,
    DECISION_EDUCATIONAL_SCORE_MIXED,
    DECISION_GENERATED_SCORE_MAX,
    DECISION_INVALID_SYNTAX_BALANCE,
    DECISION_ISSUE_CONFIDENCE_MULT,
    DECISION_ISSUE_COUNT_PENALTY,
    DECISION_LANGUAGE_BLEND_BASE,
    DECISION_LANGUAGE_BLEND_WEIGHT,
    DECISION_OCR_BROKEN_CHAR_DIV,
    DECISION_OCR_BROKEN_MIN,
    DECISION_OCR_MIN_TEXT_CHARS,
    DECISION_OCR_PIPE_NOISE_MULT,
    DECISION_OCR_PIPE_NOISE_RATIO,
    DECISION_OCR_SINGLE_CHAR_MULT,
    DECISION_OCR_SINGLE_CHAR_RATIO,
    DECISION_PROSE_RATIO_FOR_CODE,
    DECISION_PROSE_RATIO_MIXED,
    DECISION_SECRET_PII_SCORE,
    DECISION_SIGNAL_PENALTY_FLOOR,
    DECISION_SIGNAL_PENALTY_WEIGHT,
    DECISION_SOFT_SIGNAL_PENALTY,
    DECISION_SYNTHETIC_CONFIDENCE_MULT,
    DECISION_SYNTHETIC_EDU_THRESHOLD,
    DECISION_SYNTHETIC_PENALTY_THRESHOLD,
    DECISION_TECHNICAL_DOMAINS,
    DECISION_TRUNCATION_EXEMPT_DOMAINS,
    DECISION_TRUNCATION_WIKI_EXEMPT,
    DEFAULT_PIPELINE_SPEC,
    MIN_CHARS_AFTER_REPAIR,
    STRUCT_CODE_DUMP_PROB,
    STRUCT_EDUCATIONAL_CODE_SCORE,
    STRUCT_EDUCATIONAL_SYNTAX_BALANCE,
    STRUCT_TRUNC_REMOVE_PROB,
    STRUCT_TRUNC_REPAIR_PROB,
)
from indw.config.loader import ConfigRef, Resolver

@dataclass
class StageToggle:
    enabled: bool = True

@dataclass
class CompositeWeights:
    knowledge: float = 0.22
    educational: float = 0.17
    technical: float = 0.14
    coherence: float = 0.14
    information_density: float = 0.07
    novelty: float = 0.05

@dataclass
class ContextBlendWeights:
    coherence: float = 0.5
    structural: float = 0.3
    artifact_inverse: float = 0.2
    truncation_penalty: float = 20.0

@dataclass
class ScoringPolicy:
    chars_per_token: float = 3.8
    weights: CompositeWeights = field(default_factory=CompositeWeights)
    context_blend: ContextBlendWeights = field(default_factory=ContextBlendWeights)
    continuous_weight: float = 0.70
    utility_weight: float = 0.30
    artifact_penalty: float = 0.30
    noise_penalty: float = 0.10
    duplication_penalty: float = 0.08

@dataclass
class CuratorBand:
    min_composite: float = 0.0
    min_knowledge: float = 0.0
    max_artifact_contamination: float = 100.0
    min_words: int = 0

@dataclass
class CuratorSalvagePolicy:
    technical_value_floor: float = 45.0
    educational_value_floor: float = 42.0
    salvage_flags: frozenset[str] = field(default_factory=lambda: frozenset({'synthetic_spam', 'low_value_news'}))

@dataclass
class CodeRewritePolicy:
    min_structural_integrity: float = 85.0
    min_technical_value: float = 45.0

@dataclass
class ClassificationThresholds:
    content_commercial: float = 0.55
    signals_commercial: float = 0.45
    max_overall_value: float = 0.22
    max_educational: float = 0.20
    max_technical: float = 0.25
    low_value_overall: float = 0.22
    government_low_value: float = 0.15
    license_min_hits: int = 2
    license_max_overall: float = 0.22
    scaffold_max_words: int = 35
    metadata_only_max_words: int = 40
    metadata_only_max_facts: int = 2
    metadata_only_max_educational: float = 0.15
    commercial_edu_floor: float = 0.25
    structured_code_syntax: float = 0.85
    structured_code_educational: float = 0.15
    code_density_text: float = 0.18
    code_density_mixed: float = 0.06

@dataclass
class ClassificationPolicy:
    enabled: bool = True
    thresholds: ClassificationThresholds = field(default_factory=ClassificationThresholds)

@dataclass
class CuratorPolicy:
    keep: CuratorBand = field(default_factory=lambda: CuratorBand(
        min_composite=48.0, min_knowledge=40.0, max_artifact_contamination=42.0, min_words=12,
    ))
    rewrite: CuratorBand = field(default_factory=lambda: CuratorBand(
        min_composite=25.0, min_knowledge=25.0, max_artifact_contamination=68.0, min_words=12,
    ))
    drop: CuratorBand = field(default_factory=CuratorBand)
    rewrite_sample_weight: float = 0.85
    salvage: CuratorSalvagePolicy = field(default_factory=CuratorSalvagePolicy)
    code_rewrite: CodeRewritePolicy = field(default_factory=CodeRewritePolicy)
    hard_reject_flags: frozenset[str] = field(default_factory=lambda: frozenset({
        'toxicity', 'pii', 'injection', 'software_piracy', 'proprietary_license',
        'credential_leak', 'synthetic_spam', 'commercial_content', 'code_dump', 'truncated',
        'context_mismatch', 'low_value_news', 'license_or_metadata_only',
        'entertainment_clickbait', 'instruction_scaffold_only', 'vendor_sdk_dump',
    }))

@dataclass
class RewritePolicy:
    normalize_qa: bool = True
    strip_qa_tail: bool = True
    strip_artifact_lines: bool = True
    max_padding_ratio: float = 0.20
    max_synthetic_score: float = 0.18
    max_seo_score: float = 0.10
    prefer_compression: bool = True

@dataclass
class ValidationPolicy:
    score_bounds: tuple[float, float] = (0.0, 100.0)
    require_doc_id: bool = False
    reject_conflicting_action: bool = True

@dataclass
class StructuralRepairThresholds:
    trunc_repair_probability: float = STRUCT_TRUNC_REPAIR_PROB
    trunc_remove_probability: float = STRUCT_TRUNC_REMOVE_PROB
    code_dump_probability: float = STRUCT_CODE_DUMP_PROB
    educational_code_score: float = STRUCT_EDUCATIONAL_CODE_SCORE
    educational_syntax_balance: float = STRUCT_EDUCATIONAL_SYNTAX_BALANCE

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> StructuralRepairThresholds:
        if not raw:
            return cls()
        d = cls()
        return cls(
            trunc_repair_probability=float(raw.get('trunc_repair_probability', d.trunc_repair_probability)),
            trunc_remove_probability=float(raw.get('trunc_remove_probability', d.trunc_remove_probability)),
            code_dump_probability=float(raw.get('code_dump_probability', d.code_dump_probability)),
            educational_code_score=float(raw.get('educational_code_score', d.educational_code_score)),
            educational_syntax_balance=float(raw.get('educational_syntax_balance', d.educational_syntax_balance)),
        )

@dataclass
class DecisionHeuristicsPolicy:
    code_hits_for_code: int = DECISION_CODE_HITS_FOR_CODE
    prose_ratio_for_code: float = DECISION_PROSE_RATIO_FOR_CODE
    code_hits_for_mixed: int = DECISION_CODE_HITS_FOR_MIXED
    educational_score_mixed: float = DECISION_EDUCATIONAL_SCORE_MIXED
    prose_ratio_mixed: float = DECISION_PROSE_RATIO_MIXED
    ocr_min_text_chars: int = DECISION_OCR_MIN_TEXT_CHARS
    ocr_single_char_ratio: float = DECISION_OCR_SINGLE_CHAR_RATIO
    ocr_single_char_mult: float = DECISION_OCR_SINGLE_CHAR_MULT
    ocr_broken_min: int = DECISION_OCR_BROKEN_MIN
    ocr_broken_char_div: int = DECISION_OCR_BROKEN_CHAR_DIV
    ocr_pipe_noise_ratio: float = DECISION_OCR_PIPE_NOISE_RATIO
    ocr_pipe_noise_mult: float = DECISION_OCR_PIPE_NOISE_MULT
    invalid_syntax_balance: float = DECISION_INVALID_SYNTAX_BALANCE
    generated_score_max: float = DECISION_GENERATED_SCORE_MAX
    secret_pii_score: float = DECISION_SECRET_PII_SCORE
    truncation_wiki_exempt: float = DECISION_TRUNCATION_WIKI_EXEMPT
    truncation_exempt_domains: frozenset[str] = DECISION_TRUNCATION_EXEMPT_DOMAINS
    technical_domains: frozenset[str] = DECISION_TECHNICAL_DOMAINS
    code_syntax_floor: float = DECISION_CODE_SYNTAX_FLOOR
    alpha_syntax_floor: float = DECISION_ALPHA_SYNTAX_FLOOR
    chars_per_token_floor: float = DECISION_CHARS_PER_TOKEN_FLOOR
    confidence_base: float = DECISION_CONFIDENCE_BASE
    signal_penalty_floor: float = DECISION_SIGNAL_PENALTY_FLOOR
    signal_penalty_weight: float = DECISION_SIGNAL_PENALTY_WEIGHT
    soft_signal_penalty: float = DECISION_SOFT_SIGNAL_PENALTY
    issue_count_penalty: int = DECISION_ISSUE_COUNT_PENALTY
    issue_confidence_mult: float = DECISION_ISSUE_CONFIDENCE_MULT
    synthetic_edu_threshold: float = DECISION_SYNTHETIC_EDU_THRESHOLD
    synthetic_penalty_threshold: float = DECISION_SYNTHETIC_PENALTY_THRESHOLD
    synthetic_confidence_mult: float = DECISION_SYNTHETIC_CONFIDENCE_MULT
    language_blend_base: float = DECISION_LANGUAGE_BLEND_BASE
    language_blend_weight: float = DECISION_LANGUAGE_BLEND_WEIGHT
    confidence_min: float = DECISION_CONFIDENCE_MIN
    confidence_max: float = DECISION_CONFIDENCE_MAX

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> DecisionHeuristicsPolicy:
        if not raw:
            return cls()
        d = cls()
        exempt = raw.get('truncation_exempt_domains')
        technical = raw.get('technical_domains')
        return cls(
            code_hits_for_code=int(raw.get('code_hits_for_code', d.code_hits_for_code)),
            prose_ratio_for_code=float(raw.get('prose_ratio_for_code', d.prose_ratio_for_code)),
            code_hits_for_mixed=int(raw.get('code_hits_for_mixed', d.code_hits_for_mixed)),
            educational_score_mixed=float(raw.get('educational_score_mixed', d.educational_score_mixed)),
            prose_ratio_mixed=float(raw.get('prose_ratio_mixed', d.prose_ratio_mixed)),
            ocr_min_text_chars=int(raw.get('ocr_min_text_chars', d.ocr_min_text_chars)),
            ocr_single_char_ratio=float(raw.get('ocr_single_char_ratio', d.ocr_single_char_ratio)),
            ocr_single_char_mult=float(raw.get('ocr_single_char_mult', d.ocr_single_char_mult)),
            ocr_broken_min=int(raw.get('ocr_broken_min', d.ocr_broken_min)),
            ocr_broken_char_div=int(raw.get('ocr_broken_char_div', d.ocr_broken_char_div)),
            ocr_pipe_noise_ratio=float(raw.get('ocr_pipe_noise_ratio', d.ocr_pipe_noise_ratio)),
            ocr_pipe_noise_mult=float(raw.get('ocr_pipe_noise_mult', d.ocr_pipe_noise_mult)),
            invalid_syntax_balance=float(raw.get('invalid_syntax_balance', d.invalid_syntax_balance)),
            generated_score_max=float(raw.get('generated_score_max', d.generated_score_max)),
            secret_pii_score=float(raw.get('secret_pii_score', d.secret_pii_score)),
            truncation_wiki_exempt=float(raw.get('truncation_wiki_exempt', d.truncation_wiki_exempt)),
            truncation_exempt_domains=frozenset(exempt) if exempt else d.truncation_exempt_domains,
            technical_domains=frozenset(technical) if technical else d.technical_domains,
            code_syntax_floor=float(raw.get('code_syntax_floor', d.code_syntax_floor)),
            alpha_syntax_floor=float(raw.get('alpha_syntax_floor', d.alpha_syntax_floor)),
            chars_per_token_floor=float(raw.get('chars_per_token_floor', d.chars_per_token_floor)),
            confidence_base=float(raw.get('confidence_base', d.confidence_base)),
            signal_penalty_floor=float(raw.get('signal_penalty_floor', d.signal_penalty_floor)),
            signal_penalty_weight=float(raw.get('signal_penalty_weight', d.signal_penalty_weight)),
            soft_signal_penalty=float(raw.get('soft_signal_penalty', d.soft_signal_penalty)),
            issue_count_penalty=int(raw.get('issue_count_penalty', d.issue_count_penalty)),
            issue_confidence_mult=float(raw.get('issue_confidence_mult', d.issue_confidence_mult)),
            synthetic_edu_threshold=float(raw.get('synthetic_edu_threshold', d.synthetic_edu_threshold)),
            synthetic_penalty_threshold=float(raw.get('synthetic_penalty_threshold', d.synthetic_penalty_threshold)),
            synthetic_confidence_mult=float(raw.get('synthetic_confidence_mult', d.synthetic_confidence_mult)),
            language_blend_base=float(raw.get('language_blend_base', d.language_blend_base)),
            language_blend_weight=float(raw.get('language_blend_weight', d.language_blend_weight)),
            confidence_min=float(raw.get('confidence_min', d.confidence_min)),
            confidence_max=float(raw.get('confidence_max', d.confidence_max)),
        )

@dataclass
class PipelinePolicy:
    normalization: StageToggle = field(default_factory=StageToggle)
    artifact_cleaning: dict[str, bool] = field(default_factory=lambda: {
        'strip_ui_noise': True,
        'strip_metadata_lines': True,
        'strip_boilerplate': True,
        'strip_social_promo': True,
        'foundation_metadata_clean': True,
    })
    structural_repair: dict[str, Any] = field(default_factory=lambda: {
        'repair_truncation': True,
        'remove_heavy_truncation': True,
        'strip_dominant_code': True,
        'remove_code_dumps': True,
        'min_chars_after_repair': MIN_CHARS_AFTER_REPAIR,
    })
    classification: ClassificationPolicy = field(default_factory=ClassificationPolicy)
    scoring: ScoringPolicy = field(default_factory=ScoringPolicy)
    curator: CuratorPolicy = field(default_factory=CuratorPolicy)
    rewrite: RewritePolicy = field(default_factory=RewritePolicy)
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)
    decision: DecisionHeuristicsPolicy = field(default_factory=DecisionHeuristicsPolicy)
    structural_thresholds: StructuralRepairThresholds = field(default_factory=StructuralRepairThresholds)
    export: dict[str, Any] = field(default_factory=lambda: {
        'text_keys': ['text', 'content', 'body', 'markdown'],
        'attach_scores': True,
        'attach_decision': True,
    })

    @classmethod
    def resolve(cls, spec: Optional[str] = None) -> PipelinePolicy:
        return _resolve_pipeline_cached(spec or DEFAULT_PIPELINE_SPEC)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PipelinePolicy:
        if not raw:
            return cls()
        scoring_raw = raw.get('scoring') or {}
        weights_raw = scoring_raw.get('composite_weights') or {}
        blend_raw = scoring_raw.get('context_blend') or {}
        curator_raw = raw.get('curator') or {}
        classification_raw = raw.get('classification') or {}
        rewrite_raw = raw.get('rewrite') or {}
        validation_raw = raw.get('validation') or {}
        class_th = classification_raw.get('thresholds') or {}

        def _band(key: str, defaults: CuratorBand) -> CuratorBand:
            band = curator_raw.get(key) or {}
            return CuratorBand(
                min_composite=float(band.get('min_composite', defaults.min_composite)),
                min_knowledge=float(band.get('min_knowledge', defaults.min_knowledge)),
                max_artifact_contamination=float(
                    band.get('max_artifact_contamination', defaults.max_artifact_contamination),
                ),
                min_words=int(band.get('min_words', defaults.min_words)),
            )

        keep_defaults = CuratorBand(48.0, 40.0, 42.0, 12)
        rewrite_defaults = CuratorBand(25.0, 25.0, 68.0, 12)
        drop_defaults = CuratorBand()
        drop_raw = curator_raw.get('drop') or {}
        hard_flags = drop_raw.get('hard_reject_flags') or curator_raw.get('hard_reject_flags') or []
        salvage_raw = curator_raw.get('salvage') or {}
        code_rewrite_raw = curator_raw.get('code_rewrite') or {}
        bounds = validation_raw.get('score_bounds') or [0.0, 100.0]
        sr_raw = raw.get('structural_repair') or {}
        decision_raw = raw.get('decision') or {}

        return cls(
            normalization=StageToggle(bool((raw.get('normalization') or {}).get('enabled', True))),
            artifact_cleaning=dict(raw.get('artifact_cleaning') or {
                'strip_ui_noise': True,
                'strip_metadata_lines': True,
                'strip_boilerplate': True,
                'strip_social_promo': True,
                'foundation_metadata_clean': True,
            }),
            structural_repair=dict(raw.get('structural_repair') or {
                'repair_truncation': True,
                'remove_heavy_truncation': True,
                'strip_dominant_code': True,
                'remove_code_dumps': True,
                'min_chars_after_repair': MIN_CHARS_AFTER_REPAIR,
            }),
            classification=ClassificationPolicy(
                enabled=bool(classification_raw.get('enabled', True)),
                thresholds=ClassificationThresholds(
                    content_commercial=float(class_th.get('commercial_dominant', {}).get('content_commercial', class_th.get('content_commercial', 0.55))),
                    signals_commercial=float(class_th.get('commercial_dominant', {}).get('signals_commercial', class_th.get('signals_commercial', 0.45))),
                    max_overall_value=float(class_th.get('commercial_dominant', {}).get('max_overall_value', class_th.get('max_overall_value', class_th.get('low_value_overall', 0.22)))),
                    max_educational=float(class_th.get('commercial_dominant', {}).get('max_educational', class_th.get('max_educational', 0.20))),
                    max_technical=float(class_th.get('commercial_dominant', {}).get('max_technical', class_th.get('max_technical', 0.25))),
                    low_value_overall=float(class_th.get('low_value_overall', 0.22)),
                    government_low_value=float(class_th.get('government_low_value', 0.15)),
                    license_min_hits=int(class_th.get('license_min_hits', 2)),
                    license_max_overall=float(class_th.get('license_max_overall', 0.22)),
                    scaffold_max_words=int(class_th.get('scaffold_max_words', 35)),
                    metadata_only_max_words=int(class_th.get('metadata_only_max_words', 40)),
                    metadata_only_max_facts=int(class_th.get('metadata_only_max_facts', 2)),
                    metadata_only_max_educational=float(class_th.get('metadata_only_max_educational', 0.15)),
                    commercial_edu_floor=float(class_th.get('commercial_edu_floor', 0.25)),
                    structured_code_syntax=float(class_th.get('structured_code_syntax', 0.85)),
                    structured_code_educational=float(class_th.get('structured_code_educational', 0.15)),
                    code_density_text=float(class_th.get('code_density_text', 0.18)),
                    code_density_mixed=float(class_th.get('code_density_mixed', 0.06)),
                ),
            ),
            scoring=ScoringPolicy(
                chars_per_token=float(scoring_raw.get('chars_per_token', 3.8)),
                weights=CompositeWeights(
                    knowledge=float(weights_raw.get('knowledge', 0.22)),
                    educational=float(weights_raw.get('educational', 0.17)),
                    technical=float(weights_raw.get('technical', 0.14)),
                    coherence=float(weights_raw.get('coherence', 0.14)),
                    information_density=float(weights_raw.get('information_density', 0.07)),
                    novelty=float(weights_raw.get('novelty', 0.05)),
                ),
                context_blend=ContextBlendWeights(
                    coherence=float(blend_raw.get('coherence', 0.5)),
                    structural=float(blend_raw.get('structural', 0.3)),
                    artifact_inverse=float(blend_raw.get('artifact_inverse', 0.2)),
                    truncation_penalty=float(blend_raw.get('truncation_penalty', 20.0)),
                ),
                artifact_penalty=float(scoring_raw.get('artifact_penalty', 0.30)),
                noise_penalty=float(scoring_raw.get('noise_penalty', 0.10)),
                duplication_penalty=float(scoring_raw.get('duplication_penalty', 0.08)),
                continuous_weight=float(scoring_raw.get('continuous_weight', 0.70)),
                utility_weight=float(scoring_raw.get('utility_weight', 0.30)),
            ),
            curator=CuratorPolicy(
                keep=_band('keep', keep_defaults),
                rewrite=_band('rewrite', rewrite_defaults),
                drop=_band('drop', drop_defaults),
                rewrite_sample_weight=float(curator_raw.get('rewrite_sample_weight', 0.85)),
                salvage=CuratorSalvagePolicy(
                    technical_value_floor=float(salvage_raw.get('technical_value_floor', 45.0)),
                    educational_value_floor=float(salvage_raw.get('educational_value_floor', 42.0)),
                    salvage_flags=frozenset(
                        str(x) for x in (salvage_raw.get('salvage_flags') or ['synthetic_spam', 'low_value_news'])
                    ),
                ),
                code_rewrite=CodeRewritePolicy(
                    min_structural_integrity=float(code_rewrite_raw.get('min_structural_integrity', 85.0)),
                    min_technical_value=float(code_rewrite_raw.get('min_technical_value', 45.0)),
                ),
                hard_reject_flags=frozenset(str(x) for x in hard_flags) or CuratorPolicy().hard_reject_flags,
            ),
            rewrite=RewritePolicy(
                normalize_qa=bool(rewrite_raw.get('normalize_qa', True)),
                strip_qa_tail=bool(rewrite_raw.get('strip_qa_tail', True)),
                strip_artifact_lines=bool(rewrite_raw.get('strip_artifact_lines', True)),
                max_padding_ratio=float(rewrite_raw.get('max_padding_ratio', 0.20)),
                max_synthetic_score=float(rewrite_raw.get('max_synthetic_score', 0.18)),
                max_seo_score=float(rewrite_raw.get('max_seo_score', 0.10)),
                prefer_compression=bool(rewrite_raw.get('prefer_compression', True)),
            ),
            validation=ValidationPolicy(
                score_bounds=(float(bounds[0]), float(bounds[1])),
                require_doc_id=bool(validation_raw.get('require_doc_id', False)),
                reject_conflicting_action=bool(validation_raw.get('reject_conflicting_action', True)),
            ),
            decision=DecisionHeuristicsPolicy.from_dict(decision_raw),
            structural_thresholds=StructuralRepairThresholds.from_dict(sr_raw.get('thresholds')),
            export=dict(raw.get('export') or {
                'text_keys': ['text', 'content', 'body', 'markdown'],
                'attach_scores': True,
                'attach_decision': True,
            }),
        )

@lru_cache(maxsize=8)
def _resolve_pipeline_cached(spec: str) -> PipelinePolicy:
    try:
        resolved = Resolver.default().resolve(ConfigRef(kind='pipeline', id=spec))
        return PipelinePolicy.from_dict(dict(resolved.raw))
    except (FileNotFoundError, KeyError):
        return PipelinePolicy()
