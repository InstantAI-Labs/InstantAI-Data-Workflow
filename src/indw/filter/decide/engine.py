from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from indw.config.validation import ConfigResolutionError
from indw.filter.score.types import CanonicalDocumentScore
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.decide.curator import CuratorEngine, _band_match, _word_count
from indw.filter.spec.document import ContentClassification, CorpusDocument, CuratorDecision
from indw.filter.decide.calibrate import AdaptiveCalibrator
from indw.filter.spec.quality import CurriculumConfig, QualityThresholds, SyntheticDefenseConfig
from indw.filter.decide.policy import (
    HARD_REJECT,
    build_signals,
    collect_adaptive_quality_issues,
    decision_confidence,
    detect_content_type,
    quality_score_10_from_doc,
    soft_issues,
)
from indw.clean.semantic.spec import SemanticSelectionConfig
from indw.clean.artifact.evidence import AdaptiveBaselineEstimator

if TYPE_CHECKING:
    from indw.config.resolve import PipelineConfigContext

@dataclass
class PipelineDecision:
    action: str
    reason: str
    detail: str
    sample_weight: float = 1.0
    corpus_partition: str = 'main'
    filter_decision: str = ''
    issues: list[str] = field(default_factory=list)
    signals: dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.5
    content_type: str = 'text'
    quality_score_10: float = 0.0

    def to_curator_decision(self) -> CuratorDecision:
        return CuratorDecision(
            action=self.action,
            reason=self.reason,
            detail=self.detail,
            corpus_partition=self.corpus_partition,
            sample_weight=self.sample_weight,
            route_scores={'composite': self.quality_score_10 * 10.0},
        )

class DecisionEngine:
    def __init__(
        self,
        ctx: PipelineConfigContext,
        *,
        calibrator: Optional[AdaptiveCalibrator] = None,
    ) -> None:
        if ctx is None:
            raise ConfigResolutionError('DecisionEngine requires PipelineConfigContext')
        self.ctx = ctx
        self.policy: PipelinePolicy = ctx.pipeline
        self.thresholds = ctx.quality.thresholds
        self.semantic_selection = ctx.quality.semantic_selection
        self.synthetic_defense = ctx.quality.synthetic_defense
        self.curriculum = ctx.quality.curriculum
        self.calibrator = calibrator or AdaptiveCalibrator(ctx.quality.adaptive_calibration)
        self._curator = CuratorEngine(self.policy)

    def decide(
        self,
        score: CanonicalDocumentScore,
        text: str,
        *,
        doc: CorpusDocument | None = None,
        flags: tuple[str, ...] | None = None,
        classification: ContentClassification | None = None,
        text_modified: bool = False,
        exact_duplicate: bool = False,
        near_duplicate: bool = False,
    ) -> PipelineDecision:
        heuristics = self.policy.decision
        content_type = detect_content_type(
            text, domain=score.domain, code=score.code_signals, policy=heuristics,
        )
        q10 = quality_score_10_from_doc(score, text, policy=heuristics)
        issues = soft_issues(
            score,
            text,
            self.thresholds,
            semantic_selection=self.semantic_selection,
            policy=heuristics,
        )
        adaptive = collect_adaptive_quality_issues(
            score,
            text,
            self.thresholds,
            semantic_selection=self.semantic_selection,
            synthetic_defense=self.synthetic_defense,
            curriculum=self.curriculum,
            policy=heuristics,
        )
        for code in adaptive:
            if code not in issues:
                issues.append(code)
        signals = build_signals(
            score,
            duplicate=exact_duplicate,
            near_duplicate=near_duplicate,
            policy=heuristics,
        )

        if exact_duplicate:
            return PipelineDecision(
                action='DROP',
                reason='exact_duplicate',
                detail='duplicate',
                filter_decision='REJECT',
                issues=['exact_duplicate'],
                signals=signals,
                content_type=content_type,
                quality_score_10=q10,
            )

        if score.reject_reason and score.reject_reason in HARD_REJECT:
            return PipelineDecision(
                action='DROP',
                reason=score.reject_reason,
                detail=score.reject_reason,
                filter_decision='REJECT',
                issues=[score.reject_reason, *issues],
                signals=signals,
                content_type=content_type,
                quality_score_10=q10,
            )

        if (
            signals['secret_detected']
            and (
                score.pii_score > 0.58
                or score.pii_reason in ('production_secret', 'credential_leak', 'customer_data')
            )
        ):
            return PipelineDecision(
                action='DROP',
                reason='secret_detected',
                detail='secret_detected',
                filter_decision='REJECT',
                issues=['secret_detected', *issues],
                signals=signals,
                content_type=content_type,
                quality_score_10=q10,
            )

        if signals['toxicity'] and score.toxicity_score > 0.45:
            return PipelineDecision(
                action='DROP',
                reason='toxicity',
                detail='toxicity',
                filter_decision='REJECT',
                issues=['toxicity', *issues],
                signals=signals,
                content_type=content_type,
                quality_score_10=q10,
            )

        if signals['invalid_code'] and score.domain == 'code':
            return PipelineDecision(
                action='DROP',
                reason='invalid_code',
                detail='invalid_code',
                filter_decision='REJECT',
                issues=['invalid_code', *issues],
                signals=signals,
                content_type=content_type,
                quality_score_10=q10,
            )

        if not text.strip():
            return PipelineDecision(
                action='DROP',
                reason='empty',
                detail='no_text',
                filter_decision='REJECT',
                content_type=content_type,
                quality_score_10=q10,
            )

        merged_flags = set(flags or ())
        if doc is not None:
            merged_flags.update(doc.flags)
            if doc.classification is not None:
                merged_flags.update(doc.classification.flags)
            classification = classification or doc.classification
            text_modified = text_modified or doc.text_modified

        curator_doc = CorpusDocument(
            doc_id=doc.doc_id if doc else '',
            raw_text=text,
            text=text,
            classification=classification,
            scores=score,
            flags=tuple(merged_flags),
            text_modified=text_modified,
        )
        curator_decision = self._curator.decide(curator_doc)

        signal_penalty = self.calibrator.signal_confidence(signals, score.signals)
        confidence = decision_confidence(
            score,
            utility=score.training_utility,
            signal_penalty=signal_penalty,
            issue_count=len(issues),
            policy=heuristics,
        )
        utility_score = score.training_utility.utility_score if score.training_utility else score.score
        has_soft = bool(issues) or near_duplicate or signal_penalty > heuristics.soft_signal_penalty

        if curator_decision.action == 'DROP':
            return PipelineDecision(
                action='DROP',
                reason=curator_decision.reason,
                detail=curator_decision.detail,
                filter_decision='REJECT',
                issues=issues,
                signals=signals,
                confidence=confidence,
                content_type=content_type,
                quality_score_10=q10,
            )

        if curator_decision.action == 'REWRITE':
            weight = self.policy.curator.rewrite_sample_weight
            return PipelineDecision(
                action='REWRITE',
                reason=curator_decision.reason,
                detail=curator_decision.detail,
                sample_weight=weight,
                filter_decision='KEEP_BUT_DOWNRANK',
                issues=issues,
                signals=signals,
                confidence=confidence,
                content_type=content_type,
                quality_score_10=q10,
            )

        sample_weight = 1.0
        filter_decision = 'KEEP'
        if has_soft:
            if self.thresholds.high_quality_only and confidence >= 0.42:
                return PipelineDecision(
                    action='DROP',
                    reason=issues[0] if issues else 'borderline_quality',
                    detail='high_quality_only',
                    filter_decision='REJECT',
                    issues=issues or ['borderline_quality'],
                    signals=signals,
                    confidence=confidence,
                    content_type=content_type,
                    quality_score_10=q10,
                )
            weight = self.calibrator.downrank_weight(
                score.score,
                q10,
                issue_count=len(issues),
                signal_penalty=signal_penalty,
                near_duplicate=near_duplicate,
            )
            util_floor = AdaptiveBaselineEstimator().baseline([utility_score, score.score, confidence])
            if utility_score > util_floor:
                weight = max(weight, util_floor)
            if confidence < 0.42:
                weight = max(weight, 0.65)
            pii_cap = AdaptiveBaselineEstimator().baseline([score.pii_score, 0.45])
            if signals['pii'] and score.pii_score < pii_cap:
                weight *= 0.75
            sample_weight = weight
            if sample_weight < 0.99:
                filter_decision = 'KEEP_BUT_DOWNRANK'

        return PipelineDecision(
            action='KEEP',
            reason=curator_decision.reason,
            detail=curator_decision.detail,
            sample_weight=sample_weight,
            filter_decision=filter_decision,
            issues=issues,
            signals=signals,
            confidence=confidence,
            content_type=content_type,
            quality_score_10=q10,
        )

    def apply_to_score(self, score: CanonicalDocumentScore, decision: PipelineDecision) -> CanonicalDocumentScore:
        score.quality_score_10 = decision.quality_score_10
        score.filter_decision = decision.filter_decision
        score.filter_issues = list(decision.issues)
        score.filter_signals = dict(decision.signals)
        score.downrank_weight = decision.sample_weight
        score.filter_confidence = decision.confidence
        score.content_type = decision.content_type
        return score
