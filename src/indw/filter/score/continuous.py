from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from indw.filter.score.artifacts import ArtifactSignalBundle
from indw.filter.content.code import CodeDumpResult
from indw.clean.document.value import ContentValueSignals
from indw.clean.artifact.evidence import SemanticEvidenceBundle
from indw.filter.score.signals import QualitySignals
from indw.filter.refine.truncation import TruncationResult

if TYPE_CHECKING:
    from indw.filter.spec.pipeline import CompositeWeights

@dataclass
class ContinuousScoreVector:
    educational_value: float = 0.0
    knowledge_density: float = 0.0
    factual_density: float = 0.0
    coherence: float = 0.0
    noise: float = 0.0
    evergreen_value: float = 0.0
    technical_depth: float = 0.0
    duplication_risk: float = 0.0
    training_usefulness: float = 0.0
    artifact_severity: float = 0.0
    information_density: float = 0.0
    format_quality: float = 0.0
    token_efficiency: float = 0.0
    document_completeness: float = 0.0
    _raw_composite: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            'knowledge': round(self.knowledge_density, 2),
            'educational': round(self.educational_value, 2),
            'coherence': round(self.coherence, 2),
            'technical': round(self.technical_depth, 2),
            'artifact': round(self.artifact_severity, 2),
            'information': round(self.information_density, 2),
            'evergreen': round(self.evergreen_value, 2),
            'training': round(self.training_usefulness, 2),
        }

    @property
    def composite(self) -> float:
        return self._raw_composite

    def compute_raw_composite(
        self,
        weights: CompositeWeights | None = None,
        *,
        artifact_penalty: float = 0.30,
        noise_penalty: float = 0.10,
        duplication_penalty: float = 0.08,
    ) -> float:
        from indw.filter.spec.pipeline import CompositeWeights as CW

        w = weights or CW()
        factual_w = max(0.0, 1.0 - (
            w.educational + w.knowledge + w.coherence + w.technical
            + w.information_density + w.novelty
        ))
        factual_w = min(0.12, factual_w) if factual_w > 0 else 0.10
        training_w = 0.08
        value = (
            self.educational_value * w.educational
            + self.knowledge_density * w.knowledge
            + self.factual_density * factual_w
            + self.coherence * w.coherence
            + self.technical_depth * w.technical
            + self.training_usefulness * training_w
            + self.evergreen_value * w.novelty
            + self.information_density * w.information_density
        )
        artifact_pen = self.artifact_severity * artifact_penalty
        noise_pen = self.noise * noise_penalty
        dup_pen = self.duplication_risk * duplication_penalty
        format_bonus = max(0.0, (self.format_quality - 50.0) * 0.06)
        token_bonus = max(0.0, (self.token_efficiency - 0.5) * 12.0)
        complete_bonus = max(0.0, (self.document_completeness - 50.0) * 0.05)
        raw = value - artifact_pen - noise_pen - dup_pen + format_bonus + token_bonus + complete_bonus
        self._raw_composite = max(0.0, min(100.0, raw))
        return self._raw_composite

def build_continuous_scores(
    *,
    knowledge_density: float,
    educational_value: float,
    factual_density: float,
    coherence: float,
    language_quality: float,
    code_quality: float,
    evidence: SemanticEvidenceBundle | None,
    cv: ContentValueSignals | None,
    signals: QualitySignals | None,
    trunc: TruncationResult | None,
    code_dump: CodeDumpResult | None,
    artifact_ratio: float = 0.0,
    duplicate_ratio: float = 0.0,
    artifact_signals: ArtifactSignalBundle | None = None,
    text_chars: int = 0,
    chars_per_token: float = 3.8,
    composite_weights: CompositeWeights | None = None,
    artifact_penalty: float = 0.30,
    noise_penalty: float = 0.10,
    duplication_penalty: float = 0.08,
) -> ContinuousScoreVector:
    q = evidence.quality if evidence else None
    intent = evidence.intent if evidence else None

    content_noise = 0.0
    if q is not None:
        content_noise = max(
            q.narrative_filler * 0.35,
            q.entertainment * 0.25,
            q.commercial * 0.30,
        )
    if signals is not None:
        content_noise = max(
            content_noise,
            signals.boilerplate_score * 0.40,
            signals.seo_spam_score * 0.35,
        )
    noise = min(100.0, content_noise * 100.0)

    artifact_sev = min(100.0, artifact_ratio * 100.0)
    if artifact_signals is not None:
        artifact_sev = max(artifact_sev, artifact_signals.severity * 100.0)

    evergreen = 0.0
    if intent is not None:
        evergreen = intent.half_life * 100.0

    technical = max(code_quality, (q.technical if q else 0.0) * 100.0)
    if cv is not None:
        technical = max(technical, cv.technical_score * 100.0)

    info_density = 0.0
    if evidence is not None:
        info_density = min(100.0, evidence.information_density * 100.0)
    if cv is not None:
        info_density = max(info_density, cv.overall_value_score * 100.0)

    format_q = language_quality
    if signals is not None:
        format_q = max(
            format_q,
            signals.structural_quality * 100.0,
            (1.0 - signals.synthetic_score) * 55.0,
        )

    utility = evidence.utility * 100.0 if evidence else (cv.overall_value_score * 100.0 if cv else 0.0)
    trunc_pen = (trunc.probability if trunc else 0.0) * 12.0
    dump_pen = (code_dump.probability if code_dump else 0.0) * 12.0
    training = max(0.0, min(100.0, utility * 0.85 - trunc_pen - dump_pen + info_density * 0.15))

    token_eff = 0.65
    if text_chars > 0:
        est_tokens = text_chars / max(chars_per_token, 1.0)
        density = knowledge_density / 100.0
        token_eff = max(0.35, min(1.0, 0.35 + density * 0.45 + min(1.0, 800.0 / max(est_tokens, 1.0)) * 0.20))

    trunc_prob = trunc.probability if trunc else 0.0
    completeness = max(
        0.0,
        min(
            100.0,
            coherence * 0.35
            + format_q * 0.25
            + (100.0 - artifact_sev) * 0.25
            + (1.0 - trunc_prob) * 100.0 * 0.15,
        ),
    )

    vec = ContinuousScoreVector(
        educational_value=educational_value,
        knowledge_density=knowledge_density,
        factual_density=factual_density,
        coherence=coherence,
        noise=noise,
        evergreen_value=evergreen,
        technical_depth=technical,
        duplication_risk=min(100.0, duplicate_ratio * 100.0),
        training_usefulness=training,
        artifact_severity=artifact_sev,
        information_density=info_density,
        format_quality=format_q,
        token_efficiency=token_eff,
        document_completeness=completeness,
    )
    vec.compute_raw_composite(
        composite_weights,
        artifact_penalty=artifact_penalty,
        noise_penalty=noise_penalty,
        duplication_penalty=duplication_penalty,
    )
    return vec
