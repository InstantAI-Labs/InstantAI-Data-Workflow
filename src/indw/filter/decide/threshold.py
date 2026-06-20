from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from indw.config.defaults import (
    ADAPTIVE_KNOWLEDGE_DENSITY_FLOOR,
    DEDUP_FUZZY_THRESHOLD,
    DEDUP_SEMANTIC_JACCARD,
    DIVERSITY_DENSITY_DISCOUNT_FLOOR,
    EDUCATIONAL_CODE_DISCOUNT_FLOOR,
    INSTRUCTION_UNWRAP_CHAR_MIN,
    INSTRUCTION_UNWRAP_CHAR_DISCOUNT,
    INSTRUCTION_UNWRAP_DENSITY_FLOOR,
    LANGUAGE_MIN_CONFIDENCE,
    MIN_CHARS_AFTER_CLEAN,
    MIN_CHARS_AFTER_REPAIR,
    MIN_CHARS_FINAL,
    MIN_CHARS_GATE,
)
from indw.config.validation import ConfigResolutionError
from indw.filter.spec.pipeline import CuratorBand, PipelinePolicy
from indw.filter.spec.quality import QualityPipelineConfig, QualityThresholds
from indw.filter.content.code import CodeDumpResult
from indw.filter.refine.settings import DIVERSITY_PROTECTED_AXES, RefinerConfig
from indw.clean.artifact.evidence import SemanticEvidenceBundle

DEFAULT_MIN_CHARS_MERGE = MIN_CHARS_AFTER_CLEAN
DEFAULT_MIN_CHARS_REFINE = MIN_CHARS_AFTER_REPAIR
DEFAULT_MIN_CHARS_FINAL = MIN_CHARS_FINAL
DEFAULT_MIN_CHARS_GATE = MIN_CHARS_GATE

@dataclass
class ThresholdProfile:
    min_chars_gate: int = DEFAULT_MIN_CHARS_GATE
    min_chars_refine: int = DEFAULT_MIN_CHARS_REFINE
    min_chars_merge: int = DEFAULT_MIN_CHARS_MERGE
    min_chars_final: int = DEFAULT_MIN_CHARS_FINAL
    adaptive_density_only: bool = True
    diversity_density_discount: float = 8.0
    educational_code_discount: float = 14.0
    instruction_unwrap_discount: float = 8.0
    static_density_floor: float = 0.0
    max_synthetic_score: float = 0.72
    max_seo_score: float = 0.35
    language_confidence_min: float = LANGUAGE_MIN_CONFIDENCE
    dedup_fuzzy_threshold: float = DEDUP_FUZZY_THRESHOLD
    dedup_semantic_jaccard: float = DEDUP_SEMANTIC_JACCARD
    keep: CuratorBand = field(default_factory=lambda: CuratorBand(48.0, 40.0, 42.0, 12))
    rewrite: CuratorBand = field(default_factory=lambda: CuratorBand(25.0, 25.0, 68.0, 12))

    @classmethod
    def from_quality_config(
        cls,
        cfg: QualityPipelineConfig,
        *,
        pipeline: PipelinePolicy,
    ) -> ThresholdProfile:
        if pipeline is None:
            raise ConfigResolutionError('ThresholdProfile requires pipeline from PipelineConfigContext')
        policy = pipeline
        lang = cfg.language_policy()
        return cls(
            min_chars_gate=int(cfg.thresholds.min_chars),
            min_chars_merge=int(getattr(cfg.cleaning, 'min_chars_after_clean', DEFAULT_MIN_CHARS_MERGE)),
            min_chars_refine=int(policy.structural_repair.get('min_chars_after_repair', DEFAULT_MIN_CHARS_REFINE)),
            max_synthetic_score=float(cfg.synthetic_defense.max_synthetic_score),
            max_seo_score=float(cfg.thresholds.max_seo_spam_score),
            language_confidence_min=float(lang.gate.min_confidence),
            dedup_fuzzy_threshold=float(cfg.dedup.fuzzy_threshold),
            dedup_semantic_jaccard=float(cfg.dedup.semantic_jaccard_threshold),
            keep=policy.curator.keep,
            rewrite=policy.curator.rewrite,
        )

    @classmethod
    def from_refiner_config(
        cls,
        cfg: RefinerConfig,
        *,
        pipeline: PipelinePolicy,
    ) -> ThresholdProfile:
        if pipeline is None:
            raise ConfigResolutionError('ThresholdProfile requires pipeline from PipelineConfigContext')
        static_floor = 0.0 if cfg.adaptive_density_only else cfg.min_knowledge_density
        policy = pipeline
        return cls(
            min_chars_refine=cfg.min_chars_after_refine,
            min_chars_merge=int(getattr(cfg, 'min_chars_after_clean', DEFAULT_MIN_CHARS_MERGE)),
            adaptive_density_only=cfg.adaptive_density_only,
            diversity_density_discount=cfg.diversity_density_discount,
            static_density_floor=static_floor,
            keep=policy.curator.keep,
            rewrite=policy.curator.rewrite,
        )

class ThresholdResolver:
    def __init__(self, profile: ThresholdProfile) -> None:
        if profile is None:
            raise ConfigResolutionError('ThresholdResolver requires ThresholdProfile from PipelineConfigContext')
        self.profile = profile

    @classmethod
    def from_quality_config(
        cls,
        cfg: QualityPipelineConfig,
        *,
        pipeline: PipelinePolicy | None = None,
    ) -> ThresholdResolver:
        return cls(ThresholdProfile.from_quality_config(cfg, pipeline=pipeline))

    def min_chars(self, *, mode: str = 'refine', instruction_unwrapped: bool = False) -> int:
        base = {
            'gate': self.profile.min_chars_gate,
            'merge': self.profile.min_chars_merge,
            'refine': self.profile.min_chars_refine,
            'final': self.profile.min_chars_final,
        }.get(mode, self.profile.min_chars_refine)
        if instruction_unwrapped and mode == 'refine':
            return max(INSTRUCTION_UNWRAP_CHAR_MIN, base - INSTRUCTION_UNWRAP_CHAR_DISCOUNT)
        return base

    def knowledge_density_floor(
        self,
        *,
        evidence: SemanticEvidenceBundle | None,
        code_dump: CodeDumpResult | None = None,
        instruction_unwrapped: bool = False,
        category: str = '',
    ) -> float:
        del category
        p = self.profile
        if evidence is None:
            return p.static_density_floor if not p.adaptive_density_only else ADAPTIVE_KNOWLEDGE_DENSITY_FLOOR

        adaptive = evidence.threshold * 100.0
        floor = adaptive if p.adaptive_density_only else max(p.static_density_floor, adaptive)

        if evidence.profile.primary in DIVERSITY_PROTECTED_AXES:
            floor = max(DIVERSITY_DENSITY_DISCOUNT_FLOOR, floor - p.diversity_density_discount)
        if code_dump is not None and code_dump.classification == 'educational_code':
            floor = max(EDUCATIONAL_CODE_DISCOUNT_FLOOR, floor - p.educational_code_discount)
        if instruction_unwrapped:
            floor = max(INSTRUCTION_UNWRAP_DENSITY_FLOOR, floor - p.instruction_unwrap_discount)
        return floor

    def quality_thresholds(self, cfg: Optional[QualityThresholds] = None) -> QualityThresholds:
        th = cfg or QualityThresholds()
        th.min_chars = self.min_chars(mode='gate')
        th.max_seo_spam_score = self.profile.max_seo_score
        return th

    @staticmethod
    def resolve_cleaning_thresholds(cleaning_cfg: Any | None = None) -> dict[str, int | float]:
        min_chars = int(getattr(cleaning_cfg, 'min_chars_after_clean', DEFAULT_MIN_CHARS_MERGE) if cleaning_cfg else DEFAULT_MIN_CHARS_MERGE)
        return {
            'min_chars_after_clean': min_chars,
            'max_ui_noise_ratio': float(getattr(cleaning_cfg, 'max_ui_noise_ratio', 0.45) if cleaning_cfg else 0.45),
            'max_boilerplate_ratio': float(getattr(cleaning_cfg, 'max_boilerplate_ratio', 0.55) if cleaning_cfg else 0.55),
        }
