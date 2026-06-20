from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from indw.filter.language.detect import LanguageAssessment
from indw.filter.content.code import CodeQualitySignals
from indw.filter.score.signals import QualitySignals
from indw.clean.document.value import ContentValueSignals, TrainingUtilityEstimate
from indw.filter.license.detector import LicenseAssessment
from indw.filter.pii.detect import PiiAssessment
from indw.filter.toxicity.detect import ToxicityAssessment

@dataclass
class CanonicalDocumentScore:
    knowledge: float = 0.0
    educational_value: float = 0.0
    technical_value: float = 0.0
    artifact_contamination: float = 0.0
    coherence: float = 0.0
    information_density: float = 0.0
    novelty: float = 0.0
    structural_integrity: float = 0.0
    context_consistency: float = 0.0
    composite: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    signals: QualitySignals = field(default_factory=QualitySignals)
    domain: str = ''
    language: str = ''
    lang_fragmentation: float = 0.0
    language_confidence: float = 0.0
    mixed_language: bool = False
    language_assessment: Optional[LanguageAssessment] = None
    script_profile: Any = None
    multilingual_quality: float = 0.0
    chars_per_token: float = 0.0
    token_inflation_risk: float = 0.0
    tokenizer_runtime: Any = None
    tokenizer_ids: Optional[list[int]] = None
    code_signals: Optional[CodeQualitySignals] = None
    toxicity_score: float = 0.0
    toxicity_reason: Optional[str] = None
    toxicity_assessment: Optional[ToxicityAssessment] = None
    pii_score: float = 0.0
    pii_entities: int = 0
    pii_secrets: int = 0
    pii_reason: Optional[str] = None
    pii_assessment: Optional[PiiAssessment] = None
    reject_reason: str = ''
    content_type: str = 'text'
    content_value: Optional[ContentValueSignals] = None
    content_category: str = 'blog'
    training_utility: Optional[TrainingUtilityEstimate] = None
    license: str = 'Unknown'
    license_confidence: float = 0.0
    copyright_status: str = 'unknown'
    attribution_required: bool = False
    document_type: str = 'unknown'
    license_assessment: Optional[LicenseAssessment] = None
    artifact_ratio: float = 0.0
    artifact_components: dict[str, float] | None = None
    utility_normalized: float = 0.0

    quality_score_10: float = 0.0
    filter_decision: str = ''
    filter_issues: list[str] | None = None
    filter_signals: dict[str, bool] | None = None
    downrank_weight: float = 1.0
    filter_confidence: float = 0.5

    @property
    def score(self) -> float:
        return self.composite / 100.0

    @property
    def passed(self) -> bool:
        return self.filter_decision in ('KEEP', 'KEEP_BUT_DOWNRANK')

    @property
    def kept(self) -> bool:
        return self.passed

    def in_bounds(self, lo: float = 0.0, hi: float = 100.0) -> bool:
        fields = (
            self.knowledge, self.educational_value, self.technical_value,
            self.artifact_contamination, self.coherence, self.information_density,
            self.novelty, self.structural_integrity, self.context_consistency,
            self.composite,
        )
        return all(lo <= v <= hi for v in fields)

    def to_dict(self) -> dict[str, float]:
        return {
            'knowledge': round(self.knowledge, 2),
            'educational': round(self.educational_value, 2),
            'technical': round(self.technical_value, 2),
            'artifact': round(self.artifact_contamination, 2),
            'coherence': round(self.coherence, 2),
            'information': round(self.information_density, 2),
            'novelty': round(self.novelty, 2),
            'structural': round(self.structural_integrity, 2),
            'context': round(self.context_consistency, 2),
            'composite': round(self.composite, 2),
        }

CanonicalScores = CanonicalDocumentScore
DocumentScore = CanonicalDocumentScore
