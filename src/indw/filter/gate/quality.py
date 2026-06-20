from __future__ import annotations
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional
from indw.filter.language.script_policy import MultilingualPolicyConfig
from indw.filter.decide.calibrate import AdaptiveCalibrator
from indw.filter.spec.quality import QualityPipelineConfig
from typing import TYPE_CHECKING

from indw.filter.decide.engine import DecisionEngine

if TYPE_CHECKING:
    from indw.config.resolve import PipelineConfigContext
from indw.filter.decide.policy import SOFT_ISSUES
from indw.filter.gate.scorer import DocumentScore, score_document
from indw.filter.language.bridge import LiveTokenizerEncoder
from indw.filter.language.telemetry import MultilingualTokenizerTelemetry
from indw.filter.language.detect import LanguageIdentifier
from indw.filter.language.reports import LanguageRunStats
from indw.filter.license.detector import LicenseDetector
from indw.filter.license.reports import LicenseRunStats
from indw.filter.pii.detect import PiiDetector
from indw.filter.pii.reports import PiiRunStats
from indw.filter.toxicity.detect import ToxicityDetector
from indw.filter.toxicity.reports import ToxicityRunStats
logger = logging.getLogger(__name__)

@dataclass
class QualityRunStats:
    kept: int = 0
    downranked: int = 0
    rejected: int = 0
    reject_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    domain_kept: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    domain_rejected_cap: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    language_kept: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    source_scanned: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    source_kept: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    score_samples: list[float] = field(default_factory=list)
    length_samples: list[int] = field(default_factory=list)
    entropy_samples: list[float] = field(default_factory=list)
    token_chars_kept: int = 0
    token_chars_rejected: int = 0
    reasoning_density_sum: float = 0.0
    factual_density_sum: float = 0.0
    educational_value_sum: float = 0.0
    synthetic_score_sum: float = 0.0
    token_spam_sum: float = 0.0
    utility_sum: float = 0.0
    confidence_sum: float = 0.0
    preserve_count: int = 0
    evidence_evaluated: int = 0
    discard_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokenizer_telemetry: MultilingualTokenizerTelemetry = field(
        default_factory=MultilingualTokenizerTelemetry
    )
    evaluated: int = 0
    evaluated_score_samples: list[float] = field(default_factory=list)
    utility_samples: list[float] = field(default_factory=list)

    @property
    def pre_filter_score_mean(self) -> float:
        scores = self.evaluated_score_samples
        return sum(scores) / len(scores) if scores else 0.0

    def record_evaluation(self, score: float) -> None:
        self.evaluated += 1
        if len(self.evaluated_score_samples) < 10000:
            self.evaluated_score_samples.append(score)

    def record_reject(self, reason: str, length: int=0) -> None:
        self.rejected += 1
        self.reject_reasons[reason] += 1
        self.token_chars_rejected += length

    def source_distribution(self) -> dict[str, float]:
        total = sum(self.source_kept.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in self.source_kept.items()}

    def record_source(self, source: str, *, kept: bool) -> None:
        key = source or 'unknown'
        self.source_scanned[key] += 1
        if kept:
            self.source_kept[key] += 1

    def record_keep(self, doc: DocumentScore) -> None:
        if doc.filter_decision == 'KEEP_BUT_DOWNRANK':
            self.downranked += 1
        self.kept += 1
        self.domain_kept[doc.domain] += 1
        self.language_kept[doc.language] += 1
        self.token_chars_kept += doc.signals.length
        self.reasoning_density_sum += doc.signals.reasoning_density
        self.factual_density_sum += doc.signals.factual_density
        self.educational_value_sum += doc.signals.educational_value
        self.synthetic_score_sum += doc.signals.synthetic_score
        self.token_spam_sum += doc.signals.token_spam_score
        if len(self.score_samples) < 10000:
            self.score_samples.append(doc.score)
            self.length_samples.append(doc.signals.length)
            self.entropy_samples.append(doc.signals.char_entropy)
        if doc.content_value is not None and doc.content_value.evidence is not None:
            ev = doc.content_value.evidence
            self.evidence_evaluated += 1
            self.utility_sum += ev.utility
            self.confidence_sum += ev.confidence
            if ev.preserve:
                self.preserve_count += 1
            if not ev.preserve and ev.discard_reason:
                self.discard_reasons[ev.discard_reason] += 1
            if len(self.utility_samples) < 10000:
                self.utility_samples.append(ev.utility)

    def compensate_pre_gate_keep(self, doc: DocumentScore, *, source: str) -> None:
        if self.kept > 0:
            self.kept -= 1
        if self.downranked > 0 and doc.filter_decision == 'KEEP_BUT_DOWNRANK':
            self.downranked -= 1
        if self.domain_kept.get(doc.domain, 0) > 0:
            self.domain_kept[doc.domain] -= 1
        if self.language_kept.get(doc.language, 0) > 0:
            self.language_kept[doc.language] -= 1
        if self.token_chars_kept >= doc.signals.length:
            self.token_chars_kept -= doc.signals.length
        self.reasoning_density_sum -= doc.signals.reasoning_density
        self.factual_density_sum -= doc.signals.factual_density
        self.educational_value_sum -= doc.signals.educational_value
        self.synthetic_score_sum -= doc.signals.synthetic_score
        self.token_spam_sum -= doc.signals.token_spam_score
        key = source or 'unknown'
        if self.source_kept.get(key, 0) > 0:
            self.source_kept[key] -= 1

    def to_dict(self, *, calibration: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        from indw.filter.gate.diagnostics import score_histogram as _score_hist
        scores = self.score_samples
        return {
            'kept': self.kept,
            'downranked': self.downranked,
            'rejected': self.rejected,
            'reject_reasons': dict(self.reject_reasons),
            'domain_kept': dict(self.domain_kept),
            'domain_rejected_cap': dict(self.domain_rejected_cap),
            'language_kept': dict(self.language_kept),
            'score_mean': sum(scores) / len(scores) if scores else 0.0,
            'score_p10': sorted(scores)[len(scores) // 10] if len(scores) >= 10 else 0.0,
            'entropy_mean': sum(self.entropy_samples) / len(self.entropy_samples) if self.entropy_samples else 0.0,
            'token_chars_kept': self.token_chars_kept,
            'token_chars_rejected': self.token_chars_rejected,
            'reasoning_density_mean': self.reasoning_density_sum / max(self.kept, 1),
            'factual_density_mean': self.factual_density_sum / max(self.kept, 1),
            'educational_value_mean': self.educational_value_sum / max(self.kept, 1),
            'synthetic_score_mean': self.synthetic_score_sum / max(self.kept, 1),
            'token_spam_mean': self.token_spam_sum / max(self.kept, 1),
            'utility_mean': self.utility_sum / max(self.evidence_evaluated, 1),
            'confidence_mean': self.confidence_sum / max(self.evidence_evaluated, 1),
            'preserve_rate': self.preserve_count / max(self.evidence_evaluated, 1),
            'evidence_discard_reasons': dict(self.discard_reasons),
            'tokenizer_telemetry': self.tokenizer_telemetry.to_dict(),
            'evaluated': self.evaluated,
            'pre_filter_score_mean': self.pre_filter_score_mean,
            'evaluated_score_histogram': _score_hist(list(self.evaluated_score_samples)),
            'utility_histogram': _score_hist(list(self.utility_samples)),
            'calibration': calibration or {},
        }

class AdaptiveCurriculumGate:
    def __init__(self, stage_weights: dict[str, float], *, enabled: bool = True):
        self.stage_weights = stage_weights
        self.enabled = enabled
        self.kept = 0
        self._rejected = 0

    def allow(self, doc: DocumentScore) -> bool:
        if not self.enabled:
            return True
        weight = float(self.stage_weights.get(doc.domain, 1.0))
        target = min(0.995, max(0.05, 0.5 * weight))
        if self.kept == 0:
            self.kept += 1
            return True
        projected = (self.kept + 1) / (self.kept + self._rejected + 1)
        if projected <= target:
            self.kept += 1
            return True
        if doc.score >= min(0.9, 0.45 + (projected - target) * 0.5):
            self.kept += 1
            return True
        self._rejected += 1
        return False

class DomainBalancer:

    _HIGH_VALUE_DOMAINS = frozenset({'docs', 'reasoning', 'wiki', 'qa', 'code'})

    def __init__(
        self,
        caps: dict[str, float],
        *,
        enabled: bool = True,
        soft_cap_overflow: float = 0.15,
        quality_cap_bypass_score: float = 0.86,
        quality_high_value_domain_bypass_score: float = 0.58,
        min_kept_before_cap: int = 100,
    ):
        self.caps = caps
        self.enabled = enabled
        self.soft_cap_overflow = max(0.0, soft_cap_overflow)
        self.quality_cap_bypass_score = quality_cap_bypass_score
        self.quality_high_value_domain_bypass_score = quality_high_value_domain_bypass_score
        self.min_kept_before_cap = max(0, min_kept_before_cap)
        self._counts: dict[str, int] = defaultdict(int)

    def _kept_total(self) -> int:
        return sum(self._counts.values())

    def allow(self, domain: str, *, quality_score: float = 0.0) -> bool:
        if not self.enabled:
            return True
        cap = self.caps.get(domain, 1.0)
        if cap >= 1.0:
            self._counts[domain] += 1
            return True
        kept_total = self._kept_total()
        if kept_total < self.min_kept_before_cap:
            self._counts[domain] += 1
            return True
        if quality_score >= self.quality_cap_bypass_score:
            self._counts[domain] += 1
            return True
        if (
            domain in self._HIGH_VALUE_DOMAINS
            and quality_score >= self.quality_high_value_domain_bypass_score
        ):
            self._counts[domain] += 1
            return True
        kept_total = self._kept_total()
        if kept_total == 0:
            self._counts[domain] += 1
            return True
        ratio = self._counts[domain] / kept_total
        soft_cap = min(1.0, cap + self.soft_cap_overflow)
        if ratio < cap:
            self._counts[domain] += 1
            return True
        if ratio < soft_cap and quality_score >= cap:
            self._counts[domain] += 1
            return True
        return False

    def distribution(self) -> dict[str, float]:
        kept_total = self._kept_total()
        if kept_total == 0:
            return {}
        return {k: v / kept_total for k, v in self._counts.items()}

    def seed(self, counts: dict[str, int]) -> None:
        self._counts = defaultdict(int, {k: int(v) for k, v in counts.items() if v > 0})

    def counts(self) -> dict[str, int]:
        return dict(self._counts)


class TelemetryScriptBalancer:

    def __init__(
        self,
        targets: dict[str, float],
        *,
        enabled: bool = True,
        floor: float = 0.03,
        soft_cap_overflow: float = 0.15,
        quality_cap_bypass_score: float = 0.86,
        min_kept_before_cap: int = 100,
    ):
        self.targets = targets
        self.enabled = enabled
        self.floor = floor
        self.soft_cap_overflow = max(0.0, soft_cap_overflow)
        self.quality_cap_bypass_score = quality_cap_bypass_score
        self.min_kept_before_cap = max(0, min_kept_before_cap)
        self._counts: dict[str, int] = defaultdict(int)

    def _kept_total(self) -> int:
        return sum(self._counts.values())

    def allow(self, bucket_key: str, *, quality_score: float = 0.0) -> bool:
        if not self.enabled:
            return True
        cap = self.targets.get(bucket_key, self.targets.get('other', 1.0))
        if cap >= 1.0:
            self._counts[bucket_key] += 1
            return True
        kept_total = self._kept_total()
        if kept_total < self.min_kept_before_cap:
            self._counts[bucket_key] += 1
            return True
        if quality_score >= self.quality_cap_bypass_score:
            self._counts[bucket_key] += 1
            return True
        ratio = self._counts[bucket_key] / kept_total
        soft_cap = min(1.0, cap + self.soft_cap_overflow)
        if ratio < cap or ratio < soft_cap:
            self._counts[bucket_key] += 1
            return True
        return False

    def distribution(self) -> dict[str, float]:
        kept_total = self._kept_total()
        if kept_total == 0:
            return {}
        return {k: v / kept_total for k, v in self._counts.items()}

    def seed(self, counts: dict[str, int]) -> None:
        self._counts = defaultdict(int, {k: int(v) for k, v in counts.items() if v > 0})

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

LanguageBalancer = TelemetryScriptBalancer

class QualityGate:

    def __init__(
        self,
        config: Optional[QualityPipelineConfig] = None,
        *,
        ctx: Optional[PipelineConfigContext] = None,
    ):
        from indw.config.resolve import PipelineConfigContext as _Ctx

        if ctx is None:
            ctx = _Ctx.resolve()
        if config is not None:
            ctx = ctx.with_quality(config)
        self.ctx = ctx
        self.config = ctx.quality
        self.stats = QualityRunStats()
        self.domain_balancer = DomainBalancer(
            self.config.balance.domain_caps,
            enabled=self.config.balance.enabled,
            soft_cap_overflow=self.config.balance.soft_cap_overflow,
            quality_cap_bypass_score=self.config.balance.quality_cap_bypass_score,
            quality_high_value_domain_bypass_score=(
                self.config.balance.quality_high_value_domain_bypass_score
            ),
            min_kept_before_cap=self.config.balance.min_kept_before_cap,
        )
        mpol = MultilingualPolicyConfig.from_dict(self.config.multilingual)
        script_targets = dict(mpol.script_targets)
        if not script_targets:
            script_targets = dict(self.config.balance.language_caps)
        if self.config.balance.script_targets:
            script_targets.update(self.config.balance.script_targets)
        self.lang_balancer = TelemetryScriptBalancer(
            script_targets,
            enabled=self.config.balance.enabled,
            soft_cap_overflow=self.config.balance.soft_cap_overflow,
            quality_cap_bypass_score=self.config.balance.quality_cap_bypass_score,
            min_kept_before_cap=self.config.balance.min_kept_before_cap,
        )
        self.multilingual_policy = mpol
        self.curriculum_gate = AdaptiveCurriculumGate(
            self.config.curriculum.stage_weights,
            enabled=self.config.curriculum.enabled,
        )
        self._toxicity_policy = self.config.toxicity_policy()
        self._pii_policy = self.config.pii_policy()
        self._language_policy = self.config.language_policy()
        self.toxicity_stats = ToxicityRunStats()
        self._toxicity_detector: Optional[ToxicityDetector] = None
        if self._toxicity_policy.enabled:
            self._toxicity_detector = ToxicityDetector(self._toxicity_policy)
        self.pii_stats = PiiRunStats()
        self._pii_detector: Optional[PiiDetector] = None
        if self._pii_policy.enabled:
            self._pii_detector = PiiDetector(self._pii_policy)
        self.language_stats = LanguageRunStats()
        self._language_identifier: Optional[LanguageIdentifier] = None
        if self._language_policy.enabled:
            caps = dict(self.config.balance.language_caps) if self.config.balance.enabled else {}
            self._language_identifier = LanguageIdentifier(
                self._language_policy,
                locale_caps=caps,
            )
        if self._language_policy.english_only:
            self.lang_balancer.enabled = False
        self._license_policy = self.config.license_policy()
        self.license_stats = LicenseRunStats()
        self._license_detector: Optional[LicenseDetector] = None
        if self._license_policy.enabled:
            self._license_detector = LicenseDetector(self._license_policy)
        self.calibrator = AdaptiveCalibrator(self.config.adaptive_calibration)
        self._decision_engine = DecisionEngine(ctx=self.ctx, calibrator=self.calibrator)
        self.tokenizer_encoder: Optional[LiveTokenizerEncoder] = None
        if self.config.track_token_efficiency and self.config.tokenizer_path:
            self.tokenizer_encoder = LiveTokenizerEncoder(
                self.config.tokenizer_path,
                target_chars_per_token=self.multilingual_policy.target_chars_per_token,
            )

    def evaluate(
        self,
        text: str,
        *,
        source: str = '',
        exact_duplicate: bool = False,
        near_duplicate: bool = False,
        duplicate_ratio: float = 0.0,
        provenance: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, DocumentScore]:
        doc = score_document(
            text,
            source=source,
            duplicate_ratio=duplicate_ratio,
            thresholds=self.config.thresholds,
            gate=self,
            provenance=provenance,
        )
        return self.finalize_scored_document(
            doc,
            text,
            source=source,
            exact_duplicate=exact_duplicate,
            near_duplicate=near_duplicate,
        )

    def finalize_scored_document(
        self,
        doc: DocumentScore,
        text: str,
        *,
        source: str = '',
        exact_duplicate: bool = False,
        near_duplicate: bool = False,
    ) -> tuple[bool, DocumentScore]:
        q10_preview = doc.score * 10.0
        self.stats.record_evaluation(doc.score)
        decision = self._decision_engine.decide(
            doc,
            text,
            exact_duplicate=exact_duplicate,
            near_duplicate=near_duplicate,
        )
        if doc.reject_reason and doc.reject_reason in SOFT_ISSUES:
            if doc.reject_reason not in decision.issues:
                decision.issues.insert(0, doc.reject_reason)
        if doc.license_assessment is not None and doc.license_assessment.filter_action == 'FLAG':
            flag = doc.license_assessment.filter_reason
            if flag and flag not in decision.issues:
                decision.issues.append(flag)
        if near_duplicate and 'near_duplicate' not in decision.issues:
            decision.issues.append('near_duplicate')
        self._decision_engine.apply_to_score(doc, decision)
        self.calibrator.record(doc.score, decision.quality_score_10 or q10_preview)
        if doc.language_assessment is not None:
            self.language_stats.record(doc.language_assessment)
        if doc.pii_assessment is not None:
            self.pii_stats.record(doc.pii_assessment)
        if doc.toxicity_assessment is not None:
            fin = doc.toxicity_assessment.final
            ml_top = doc.toxicity_assessment.ml.top_category()
            self.toxicity_stats.record(
                final_score=fin.final_toxicity_score,
                band=fin.band,
                reason=fin.toxicity_reason,
                ml_top=ml_top,
            )
        if doc.license_assessment is not None:
            self.license_stats.record(
                doc.license_assessment,
                text_len=doc.signals.length,
                kept=decision.filter_decision != 'REJECT',
            )
        if not self.config.enabled:
            doc.filter_decision = 'KEEP'
            return (True, doc)
        if decision.filter_decision == 'REJECT':
            reason = decision.reason or doc.reject_reason or 'rejected'
            if reason in (
                'language',
                'unknown_language',
                'low_language_confidence',
                'language_fragmentation',
                'language_mixing',
                'script_fragmentation',
            ):
                self.language_stats.record_score_reject(reason)
            self.stats.record_reject(reason, doc.signals.length)
            self.stats.record_source(source, kept=False)
            return (False, doc)
        if not self.domain_balancer.allow(doc.domain, quality_score=doc.score):
            self.stats.record_reject('domain_cap', doc.signals.length)
            self.stats.domain_rejected_cap[doc.domain] += 1
            self.stats.record_source(source, kept=False)
            doc.filter_decision = 'REJECT'
            doc.reject_reason = 'domain_cap'
            return (False, doc)
        if not self.lang_balancer.allow(doc.language, quality_score=doc.score):
            self.stats.record_reject('language_cap', doc.signals.length)
            self.language_stats.record_balancer_reject()
            self.stats.record_source(source, kept=False)
            doc.filter_decision = 'REJECT'
            doc.reject_reason = 'language_cap'
            return (False, doc)
        if not self.curriculum_gate.allow(doc):
            self.stats.record_reject('curriculum_balance', doc.signals.length)
            self.stats.record_source(source, kept=False)
            doc.filter_decision = 'REJECT'
            doc.reject_reason = 'curriculum_balance'
            return (False, doc)
        if doc.tokenizer_ids:
            self.stats.tokenizer_telemetry.record(
                text,
                doc.tokenizer_ids,
                bucket=doc.language,
                profile=doc.script_profile,
                text_delimiter_density=doc.signals.delimiter_density,
                text_reasoning_density=doc.signals.reasoning_density,
                structural_quality=doc.signals.structural_quality,
                target_chars_per_token=self.multilingual_policy.target_chars_per_token,
            )
        self.stats.record_keep(doc)
        self.stats.record_source(source, kept=True)
        return (True, doc)