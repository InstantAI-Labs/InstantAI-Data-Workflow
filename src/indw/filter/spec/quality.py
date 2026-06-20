from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from indw.config.defaults import (
    BALANCE_HIGH_VALUE_DOMAIN_BYPASS,
    BALANCE_MIN_KEPT_BEFORE_CAP,
    BALANCE_QUALITY_CAP_BYPASS,
    BALANCE_SOFT_CAP_OVERFLOW,
    DEDUP_FUZZY_THRESHOLD,
    DEDUP_SEMANTIC_HAMMING,
    DEDUP_SEMANTIC_JACCARD,
    DEDUP_SEMANTIC_RECENT_JACCARD,
    DEFAULT_QUALITY_SPEC,
    MAX_CHARS_GATE,
    MIN_CHARS_GATE,
    MIN_SCORE_LEGACY_METRICS,
    SCORE_SAMPLE_CHARS,
)
from indw.clean.document.config import CleaningConfig
from indw.filter.language.config import LanguagePolicyConfig
from indw.filter.license.config import LicensePolicyConfig
from indw.filter.pii.config import PiiPolicyConfig
from indw.filter.toxicity.config import ToxicityPolicyConfig
from indw.filter.decide.calibrate import AdaptiveCalibrationConfig
from indw.clean.semantic.spec import SemanticSelectionConfig
from indw.dedup.embed.config import EmbeddingDedupConfig

@dataclass
class QualityThresholds:
    min_score: float = MIN_SCORE_LEGACY_METRICS
    max_toxicity: Optional[float] = None
    max_pii_score: Optional[float] = None
    min_entropy: float = 2.5
    max_repetition: float = 0.65
    max_html_score: float = 0.15
    min_alpha_ratio: float = 0.45
    min_chars: int = MIN_CHARS_GATE
    max_chars: int = MAX_CHARS_GATE
    score_sample_chars: int = SCORE_SAMPLE_CHARS
    max_prompt_injection_score: float = 0.4
    max_token_spam_score: float = 0.55
    min_structural_quality: float = 0.25
    min_coherence_score: float = 0.2
    max_reasoning_repetition: float = 0.7
    max_truncation_score: float = 0.55
    max_boilerplate_score: float = 0.40
    max_commercial_score: float = 0.38
    max_seo_spam_score: float = 0.35
    max_low_information_score: float = 0.50
    max_software_piracy_score: float = 0.18
    min_informative_density: float = 0.015
    warn_truncation_score: float = 0.35
    warn_boilerplate_score: float = 0.22
    warn_commercial_score: float = 0.18
    warn_seo_spam_score: float = 0.15
    warn_low_information_score: float = 0.28
    warn_discovery_artifact_score: float = 0.28
    max_discovery_artifact_score: float = 0.55
    high_quality_only: bool = False

@dataclass
class DedupConfig:
    exact: bool = True
    fuzzy: bool = False
    fuzzy_threshold: float = DEDUP_FUZZY_THRESHOLD
    fuzzy_num_perm: int = 128
    fuzzy_quality_margin: float = 0.05
    semantic: bool = True
    semantic_hamming_threshold: int = DEDUP_SEMANTIC_HAMMING
    semantic_jaccard_threshold: float = DEDUP_SEMANTIC_JACCARD
    semantic_recent_jaccard_threshold: float = DEDUP_SEMANTIC_RECENT_JACCARD
    skip_within_document_chunks: bool = True
    embedding: EmbeddingDedupConfig = field(default_factory=EmbeddingDedupConfig)

@dataclass
class BalanceConfig:
    enabled: bool = True
    domain_caps: dict[str, float] = field(default_factory=lambda: {'web': 0.55, 'code': 0.2, 'conversation': 0.15, 'wiki': 0.2, 'reasoning': 0.1, 'docs': 0.1, 'qa': 0.12})
    language_caps: dict[str, float] = field(default_factory=lambda: {'other': 1.0})
    script_targets: dict[str, float] = field(default_factory=dict)
    soft_cap_overflow: float = BALANCE_SOFT_CAP_OVERFLOW
    quality_cap_bypass_score: float = BALANCE_QUALITY_CAP_BYPASS
    quality_high_value_domain_bypass_score: float = BALANCE_HIGH_VALUE_DOMAIN_BYPASS
    min_kept_before_cap: int = BALANCE_MIN_KEPT_BEFORE_CAP

@dataclass
class SyntheticDefenseConfig:
    enabled: bool = True
    max_synthetic_score: float = 0.72
    min_semantic_diversity: float = 0.18
    max_repeated_span_score: float = 0.75

@dataclass
class CurriculumConfig:
    enabled: bool = True
    stage: str = 'core'
    stage_weights: dict[str, float] = field(default_factory=lambda: {'reasoning': 1.2, 'code': 1.1, 'docs': 1.0, 'wiki': 0.95, 'web': 0.85, 'conversation': 0.8, 'qa': 0.85})
    min_stage_score: float = 0.42

@dataclass
class QualityPipelineConfig:
    enabled: bool = True
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    sample_scores: int = 5000
    tokenizer_path: Optional[str] = None
    track_token_efficiency: bool = False
    synthetic_defense: SyntheticDefenseConfig = field(default_factory=SyntheticDefenseConfig)
    semantic_selection: SemanticSelectionConfig = field(default_factory=SemanticSelectionConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    orchestration: Optional[dict[str, Any]] = None
    multilingual: Optional[dict[str, Any]] = None
    toxicity: Optional[dict[str, Any]] = None
    pii: Optional[dict[str, Any]] = None
    language_id: Optional[dict[str, Any]] = None
    corpus_evaluation: Optional[dict[str, Any]] = None
    observability: Optional[dict[str, Any]] = None
    licensing: Optional[dict[str, Any]] = None
    adaptive_calibration: AdaptiveCalibrationConfig = field(default_factory=AdaptiveCalibrationConfig)
    _policy_cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def _cached_policy(self, key: str, builder: Any) -> Any:
        cached = self._policy_cache.get(key)
        if cached is None:
            cached = builder()
            self._policy_cache[key] = cached
        return deepcopy(cached)

    def language_policy(self) -> LanguagePolicyConfig:
        return self._cached_policy('language', self._build_language_policy)

    def _build_language_policy(self) -> LanguagePolicyConfig:
        base = deepcopy(LanguagePolicyConfig.resolve())
        if self.language_id:
            raw_lang = dict(self.language_id)
            fast_only = bool(raw_lang.pop('fast_detector_only', False))
            raw_lang.pop('allow', None)
            raw_lang.pop('supported_languages', None)
            overlay = LanguagePolicyConfig.from_dict(raw_lang)
            base.enabled = overlay.enabled
            base.english_only = overlay.english_only
            base.skip_post_clean_detection = overlay.skip_post_clean_detection
            base.gate = overlay.gate
            base.detector = overlay.detector
            base.mixed = overlay.mixed
            base.hints = overlay.hints
            if fast_only:
                base.detector.max_chars = min(base.detector.max_chars, 8192)
            if base.english_only:
                base.mixed.enabled = overlay.mixed.enabled
        return base

    def pii_policy(self) -> PiiPolicyConfig:
        return self._cached_policy('pii', self._build_pii_policy)

    def _build_pii_policy(self) -> PiiPolicyConfig:
        base = deepcopy(PiiPolicyConfig.resolve())
        if self.pii:
            raw_pii = dict(self.pii)
            structural_only = raw_pii.get('mode') == 'structural_only' or raw_pii.get('ner_enabled') is False
            if structural_only:
                ner = dict(raw_pii.get('ner') or {})
                ner['enabled'] = False
                raw_pii['ner'] = ner
                ctx = dict(raw_pii.get('context') or {})
                ctx['enabled'] = True
                raw_pii['context'] = ctx
                secrets = dict(raw_pii.get('secrets') or {})
                secrets['min_secret_score'] = max(
                    float(secrets.get('min_secret_score', 0.55)),
                    0.72,
                )
                raw_pii['secrets'] = secrets
                weights = dict(base.scoring_weights)
                weights['entities'] = 0.0
                weights['secrets'] = 0.58
                weights['context'] = 0.42
                raw_pii['scoring_weights'] = weights
            overlay = PiiPolicyConfig.from_dict(raw_pii)
            base.enabled = overlay.enabled
            base.ner = overlay.ner
            base.secrets = overlay.secrets
            base.context = overlay.context
            base.scoring_weights = overlay.scoring_weights
            if overlay.max_pii_score is not None:
                base.max_pii_score = overlay.max_pii_score
        if self.thresholds.max_pii_score is not None:
            base.max_pii_score = self.thresholds.max_pii_score
        return base

    def toxicity_policy(self) -> ToxicityPolicyConfig:
        return self._cached_policy('toxicity', self._build_toxicity_policy)

    def _build_toxicity_policy(self) -> ToxicityPolicyConfig:
        base = deepcopy(ToxicityPolicyConfig.resolve())
        if self.toxicity:
            raw_tox = dict(self.toxicity)
            rules_only = raw_tox.get('mode') == 'rules_only' or raw_tox.get('classifier_enabled') is False
            if rules_only:
                clf = dict(raw_tox.get('classifier') or {})
                clf['enabled'] = False
                raw_tox['classifier'] = clf
                ctx = dict(raw_tox.get('context') or {})
                ctx['enabled'] = False
                raw_tox['context'] = ctx
                raw_tox['scoring_weights'] = {
                    'classifier': 0.0,
                    'rule': 0.45,
                    'pattern': 0.45,
                    'context': 0.10,
                }
            overlay = ToxicityPolicyConfig.from_dict(raw_tox)
            base.enabled = overlay.enabled
            base.classifier = overlay.classifier
            base.context = overlay.context
            base.scoring_weights = overlay.scoring_weights
            if overlay.max_toxicity_score is not None:
                base.max_toxicity_score = overlay.max_toxicity_score
        if self.thresholds.max_toxicity is not None:
            base.max_toxicity_score = self.thresholds.max_toxicity
        return base

    def corpus_evaluation_policy(self):
        from indw.store.eval.config import CorpusEvaluationConfig

        base = deepcopy(CorpusEvaluationConfig.resolve())
        if self.corpus_evaluation:
            overlay = CorpusEvaluationConfig.from_dict(self.corpus_evaluation)
            base.enabled = overlay.enabled
            base.lightweight = overlay.lightweight
            base.output_dir = overlay.output_dir
        return base

    def observability_policy(self):
        from indw.tools.metrics.config import ObservabilityPolicyConfig

        base = deepcopy(ObservabilityPolicyConfig.resolve())
        if self.observability:
            overlay = ObservabilityPolicyConfig.from_dict(self.observability)
            base.enabled = overlay.enabled
            if overlay.output_dir:
                base.output_dir = overlay.output_dir
        return base

    def license_policy(self) -> LicensePolicyConfig:
        return self._cached_policy('license', self._build_license_policy)

    def _build_license_policy(self) -> LicensePolicyConfig:
        base = deepcopy(LicensePolicyConfig.resolve())
        if self.licensing:
            overlay = LicensePolicyConfig.from_dict(self.licensing)
            base.enabled = overlay.enabled
            base.reject_proprietary = overlay.reject_proprietary
            base.reject_restricted = overlay.reject_restricted
            base.reject_paywalled = overlay.reject_paywalled
            base.reject_drm = overlay.reject_drm
            base.reject_redistribution_prohibited = overlay.reject_redistribution_prohibited
            base.reject_pirated_books = overlay.reject_pirated_books
            base.reject_incompatible_repos = overlay.reject_incompatible_repos
            base.flag_unknown = overlay.flag_unknown
            base.flag_attribution_required = overlay.flag_attribution_required
            base.allow_cc_by_sa = overlay.allow_cc_by_sa
            base.allow_government = overlay.allow_government
            base.allow_wikipedia_compatible = overlay.allow_wikipedia_compatible
            base.include_provenance_in_jsonl = overlay.include_provenance_in_jsonl
            base.min_confidence_for_reject = overlay.min_confidence_for_reject
            base.incompatible_repo_licenses = overlay.incompatible_repo_licenses
            base.keep_licenses = overlay.keep_licenses
            base.flag_licenses = overlay.flag_licenses
            base.remove_licenses = overlay.remove_licenses
            base.output_dir = overlay.output_dir
        return base

    @classmethod
    def resolve(cls, spec: str = DEFAULT_QUALITY_SPEC) -> QualityPipelineConfig:
        from indw.config.resolve import resolve_quality_config

        return resolve_quality_config(spec)

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> QualityPipelineConfig:
        if not raw:
            return cls()
        th = raw.get('thresholds') or {}
        if 'min_quality_score' in th and 'min_score' not in th:
            th = {**th, 'min_score': th['min_quality_score']}
        if 'max_toxicity' in th:
            th = dict(th)
        dd = dict(raw.get('dedup') or {})
        if 'deduplication' in raw:
            dd['exact'] = bool(raw['deduplication'])
        bal = raw.get('balance') or {}
        syn = raw.get('synthetic_defense') or {}
        sem = raw.get('semantic_selection') or {}
        cur = raw.get('curriculum') or {}
        cal = raw.get('adaptive_calibration') or {}
        track_tok = bool(raw.get('track_token_efficiency', False) or raw.get('tokenizer_validation', False))
        cleaning_raw = raw.get('cleaning') or raw.get('corpus_cleaning')
        return cls(
            enabled=bool(raw.get('enabled', raw.get('quality_scoring', True))),
            cleaning=CleaningConfig.from_dict(cleaning_raw),
            thresholds=QualityThresholds(
                min_score=float(th.get('min_score', th.get('min_quality_score', MIN_SCORE_LEGACY_METRICS))),
                max_toxicity=(
                    float(th['max_toxicity']) if th.get('max_toxicity') is not None else None
                ),
                max_pii_score=(
                    float(th['max_pii_score']) if th.get('max_pii_score') is not None else None
                ),
                min_entropy=float(th.get('min_entropy', 2.5)),
                max_repetition=float(th.get('max_repetition', 0.65)),
                max_html_score=float(th.get('max_html_score', 0.15)),
                min_alpha_ratio=float(th.get('min_alpha_ratio', 0.45)),
                min_chars=int(th.get('min_chars', MIN_CHARS_GATE)),
                max_chars=int(th.get('max_chars', MAX_CHARS_GATE)),
                score_sample_chars=int(th.get('score_sample_chars', SCORE_SAMPLE_CHARS)),
                max_prompt_injection_score=float(th.get('max_prompt_injection_score', 0.4)),
                max_token_spam_score=float(th.get('max_token_spam_score', 0.55)),
                min_structural_quality=float(th.get('min_structural_quality', 0.25)),
                min_coherence_score=float(th.get('min_coherence_score', 0.2)),
                max_reasoning_repetition=float(th.get('max_reasoning_repetition', 0.7)),
                max_truncation_score=float(th.get('max_truncation_score', 0.55)),
                max_boilerplate_score=float(th.get('max_boilerplate_score', 0.40)),
                max_commercial_score=float(th.get('max_commercial_score', 0.38)),
                max_seo_spam_score=float(th.get('max_seo_spam_score', 0.35)),
                max_low_information_score=float(th.get('max_low_information_score', 0.50)),
                max_software_piracy_score=float(th.get('max_software_piracy_score', 0.18)),
                min_informative_density=float(th.get('min_informative_density', 0.015)),
                warn_truncation_score=float(th.get('warn_truncation_score', 0.35)),
                warn_boilerplate_score=float(th.get('warn_boilerplate_score', 0.22)),
                warn_commercial_score=float(th.get('warn_commercial_score', 0.18)),
                warn_seo_spam_score=float(th.get('warn_seo_spam_score', 0.15)),
                warn_low_information_score=float(th.get('warn_low_information_score', 0.28)),
                warn_discovery_artifact_score=float(th.get('warn_discovery_artifact_score', 0.28)),
                max_discovery_artifact_score=float(th.get('max_discovery_artifact_score', 0.55)),
                high_quality_only=bool(th.get('high_quality_only', False)),
            ),
            dedup=DedupConfig(
                exact=bool(dd.get('exact', True)),
                fuzzy=bool(dd.get('fuzzy', False)),
                fuzzy_threshold=float(dd.get('fuzzy_threshold', DEDUP_FUZZY_THRESHOLD)),
                fuzzy_num_perm=int(dd.get('fuzzy_num_perm', 128)),
                fuzzy_quality_margin=float(dd.get('fuzzy_quality_margin', 0.05)),
                semantic=bool(dd.get('semantic', True)),
                semantic_hamming_threshold=int(dd.get('semantic_hamming_threshold', DEDUP_SEMANTIC_HAMMING)),
                semantic_jaccard_threshold=float(dd.get('semantic_jaccard_threshold', DEDUP_SEMANTIC_JACCARD)),
                semantic_recent_jaccard_threshold=float(
                    dd.get('semantic_recent_jaccard_threshold', DEDUP_SEMANTIC_RECENT_JACCARD)
                ),
                skip_within_document_chunks=bool(dd.get('skip_within_document_chunks', True)),
                embedding=EmbeddingDedupConfig.from_dict(
                    dd.get('embedding') or dd.get('embedding_semantic') or dd.get('semantic_embedding')
                ),
            ),
            balance=BalanceConfig(
                enabled=bool(bal.get('enabled', True)),
                domain_caps=dict(bal.get('domain_caps') or BalanceConfig().domain_caps),
                language_caps=dict(bal.get('language_caps') or BalanceConfig().language_caps),
                script_targets=dict(bal.get('script_targets') or {}),
                soft_cap_overflow=float(bal.get('soft_cap_overflow', BALANCE_SOFT_CAP_OVERFLOW)),
                quality_cap_bypass_score=float(bal.get('quality_cap_bypass_score', BALANCE_QUALITY_CAP_BYPASS)),
                quality_high_value_domain_bypass_score=float(
                    bal.get('quality_high_value_domain_bypass_score', BALANCE_HIGH_VALUE_DOMAIN_BYPASS)
                ),
                min_kept_before_cap=int(bal.get('min_kept_before_cap', BALANCE_MIN_KEPT_BEFORE_CAP)),
            ),
            sample_scores=int(raw.get('sample_scores', 5000)),
            tokenizer_path=raw.get('tokenizer_path'),
            track_token_efficiency=track_tok,
            synthetic_defense=SyntheticDefenseConfig(
                enabled=bool(syn.get('enabled', True)),
                max_synthetic_score=float(syn.get('max_synthetic_score', 0.72)),
                min_semantic_diversity=float(syn.get('min_semantic_diversity', 0.18)),
                max_repeated_span_score=float(syn.get('max_repeated_span_score', 0.75)),
            ),
            semantic_selection=SemanticSelectionConfig(
                enabled=bool(sem.get('enabled', True)),
                section_mode=bool(sem.get('section_mode', False)),
            ),
            curriculum=CurriculumConfig(
                enabled=bool(cur.get('enabled', True)),
                stage=str(cur.get('stage', 'core')),
                stage_weights=dict(cur.get('stage_weights') or CurriculumConfig().stage_weights),
                min_stage_score=float(cur.get('min_stage_score', 0.42)),
            ),
            orchestration=raw.get('orchestration'),
            multilingual=raw.get('multilingual'),
            toxicity=raw.get('toxicity'),
            pii=raw.get('pii'),
            language_id=(
                raw.get('language_id')
                if raw.get('language_id') is not None
                else (
                    raw.get('language')
                    if raw.get('language_detection', True)
                    else {'enabled': False}
                )
            ),
            corpus_evaluation=raw.get('corpus_evaluation'),
            observability=raw.get('observability'),
            licensing=raw.get('licensing'),
            adaptive_calibration=AdaptiveCalibrationConfig(
                enabled=bool(cal.get('enabled', True)),
                warmup_samples=int(cal.get('warmup_samples', 200)),
                reservoir_size=int(cal.get('reservoir_size', 10000)),
                downrank_anchor_percentile=float(cal.get('downrank_anchor_percentile', 30.0)),
            ),
        )