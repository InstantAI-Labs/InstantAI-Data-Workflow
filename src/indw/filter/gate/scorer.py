from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from indw.filter.score.canonical import score_document_canonical
from indw.filter.score.types import CanonicalDocumentScore, DocumentScore
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.quality import CurriculumConfig, QualityThresholds, SyntheticDefenseConfig
from indw.filter.score.adaptive import adaptive_document_score
from indw.clean.semantic.spec import SemanticSelectionConfig

if TYPE_CHECKING:
    from indw.filter.gate.quality import QualityGate
    from indw.filter.language.bridge import LiveTokenizerEncoder

__all__ = [
    'DocumentScore',
    'CanonicalDocumentScore',
    'adaptive_document_score',
    'score_document',
]

def score_document(
    text: str,
    *,
    source: str = '',
    duplicate_ratio: float = 0.0,
    thresholds: Optional[QualityThresholds] = None,
    semantic_selection: Optional[SemanticSelectionConfig] = None,
    synthetic_defense: Optional[SyntheticDefenseConfig] = None,
    curriculum: Optional[CurriculumConfig] = None,
    multilingual_policy: Any = None,
    tokenizer_encoder: Optional['LiveTokenizerEncoder'] = None,
    toxicity_policy: Any = None,
    toxicity_detector: Any = None,
    pii_policy: Any = None,
    pii_detector: Any = None,
    language_policy: Any = None,
    language_identifier: Any = None,
    license_policy: Any = None,
    license_detector: Any = None,
    provenance: Optional[dict[str, Any]] = None,
    gate: Optional['QualityGate'] = None,
    policy: PipelinePolicy | None = None,
    analysis_scan: Optional[str] = None,
    analysis_full_len: Optional[int] = None,
    analysis_bundle: Any = None,
    prechecked_language: Any = None,
) -> CanonicalDocumentScore:
    del semantic_selection, synthetic_defense, curriculum
    from indw.filter.spec.quality import QualityPipelineConfig

    cfg = gate.config if gate is not None else QualityPipelineConfig()
    if thresholds is not None:
        cfg.thresholds = thresholds
    return score_document_canonical(
        text,
        source=source,
        duplicate_ratio=duplicate_ratio,
        policy=policy,
        gate=gate,
        quality_config=cfg,
        provenance=provenance,
        tokenizer_encoder=tokenizer_encoder,
        toxicity_policy=toxicity_policy,
        toxicity_detector=toxicity_detector,
        pii_policy=pii_policy,
        pii_detector=pii_detector,
        language_policy=language_policy,
        language_identifier=language_identifier,
        license_policy=license_policy,
        license_detector=license_detector,
        analysis_scan=analysis_scan,
        analysis_full_len=analysis_full_len,
        analysis_bundle=analysis_bundle,
        prechecked_language=prechecked_language,
    )
