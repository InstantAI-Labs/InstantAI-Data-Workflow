from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from indw.config.loader import ConfigRef, Resolver, thaw

from indw.config.defaults import DEFAULT_PIPELINE_SPEC, DEFAULT_QUALITY_SPEC
from indw.config.validation import (
    ConfigResolutionError,
    validate_legacy_aliases,
    validate_pipeline_policy,
    validate_quality_config,
)
from indw.filter.language.config import LanguagePolicyConfig
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.quality import QualityPipelineConfig
from indw.filter.decide.threshold import ThresholdResolver
from indw.filter.license.config import LicensePolicyConfig
from indw.filter.pii.config import PiiPolicyConfig
from indw.filter.toxicity.config import ToxicityPolicyConfig

@dataclass(frozen=True)
class PipelineConfigContext:

    quality: QualityPipelineConfig
    pipeline: PipelinePolicy
    thresholds: ThresholdResolver
    language: LanguagePolicyConfig
    pii: PiiPolicyConfig
    toxicity: ToxicityPolicyConfig
    license: LicensePolicyConfig
    quality_spec: str
    pipeline_spec: str
    legacy_aliases: tuple[str, ...] = ()

    @classmethod
    def resolve(
        cls,
        quality_spec: Optional[str] = None,
        *,
        pipeline_spec: Optional[str] = None,
        raw_overrides: Optional[dict[str, Any]] = None,
    ) -> PipelineConfigContext:
        qspec = quality_spec or DEFAULT_QUALITY_SPEC
        pspec = pipeline_spec or DEFAULT_PIPELINE_SPEC
        return _fork_context(_resolve_cached(qspec, pspec, _overrides_key(raw_overrides)))

    @property
    def decision_engine_kwargs(self) -> dict[str, Any]:
        return {
            'ctx': self,
            'thresholds': self.quality.thresholds,
            'semantic_selection': self.quality.semantic_selection,
            'synthetic_defense': self.quality.synthetic_defense,
            'curriculum': self.quality.curriculum,
        }

    def with_quality(self, cfg: QualityPipelineConfig) -> PipelineConfigContext:
        cfg = deepcopy(cfg)
        validate_quality_config(cfg)
        if _quality_policy_overlay_free(cfg):
            return PipelineConfigContext(
                quality=cfg,
                pipeline=self.pipeline,
                thresholds=ThresholdResolver.from_quality_config(cfg, pipeline=self.pipeline),
                language=self.language,
                pii=self.pii,
                toxicity=self.toxicity,
                license=self.license,
                quality_spec=self.quality_spec,
                pipeline_spec=self.pipeline_spec,
                legacy_aliases=self.legacy_aliases,
            )
        return PipelineConfigContext(
            quality=cfg,
            pipeline=self.pipeline,
            thresholds=ThresholdResolver.from_quality_config(cfg, pipeline=self.pipeline),
            language=cfg.language_policy(),
            pii=cfg.pii_policy(),
            toxicity=cfg.toxicity_policy(),
            license=cfg.license_policy(),
            quality_spec=self.quality_spec,
            pipeline_spec=self.pipeline_spec,
            legacy_aliases=self.legacy_aliases,
        )

def _quality_policy_overlay_free(cfg: QualityPipelineConfig) -> bool:
    if cfg.language_id or cfg.pii or cfg.toxicity or cfg.licensing:
        return False
    if cfg.corpus_evaluation or cfg.observability:
        return False
    th = cfg.thresholds
    return th.max_pii_score is None and th.max_toxicity is None

def _fork_context(ctx: PipelineConfigContext) -> PipelineConfigContext:
    quality = deepcopy(ctx.quality)
    pipeline = ctx.pipeline
    return PipelineConfigContext(
        quality=quality,
        pipeline=pipeline,
        thresholds=ThresholdResolver.from_quality_config(quality, pipeline=pipeline),
        language=deepcopy(ctx.language),
        pii=deepcopy(ctx.pii),
        toxicity=deepcopy(ctx.toxicity),
        license=deepcopy(ctx.license),
        quality_spec=ctx.quality_spec,
        pipeline_spec=ctx.pipeline_spec,
        legacy_aliases=ctx.legacy_aliases,
    )

def _overrides_key(raw: Optional[dict[str, Any]]) -> str:
    if not raw:
        return ''
    return json.dumps(raw, sort_keys=True, default=str)

@lru_cache(maxsize=16)
def _resolve_cached(
    quality_spec: str,
    pipeline_spec: str,
    overrides_key: str,
) -> PipelineConfigContext:
    resolver = Resolver.default()
    qref = ConfigRef(kind='quality', id=quality_spec)
    raw = dict(thaw(resolver.resolve(qref).raw))
    if overrides_key:
        raw.update(json.loads(overrides_key))
    legacy = tuple(validate_legacy_aliases(raw))
    quality = QualityPipelineConfig.from_dict(raw)
    validate_quality_config(quality)
    pipeline = PipelinePolicy.resolve(pipeline_spec)
    validate_pipeline_policy(pipeline)
    return PipelineConfigContext(
        quality=quality,
        pipeline=pipeline,
        thresholds=ThresholdResolver.from_quality_config(quality, pipeline=pipeline),
        language=quality.language_policy(),
        pii=quality.pii_policy(),
        toxicity=quality.toxicity_policy(),
        license=quality.license_policy(),
        quality_spec=quality_spec,
        pipeline_spec=pipeline_spec,
        legacy_aliases=legacy,
    )

def resolve_quality_config(
    spec: Optional[str] = None,
    *,
    raw: Optional[dict[str, Any]] = None,
) -> QualityPipelineConfig:
    if raw is not None:
        cfg = QualityPipelineConfig.from_dict(raw)
        validate_quality_config(cfg)
        return cfg
    return PipelineConfigContext.resolve(spec).quality

def ctx_from_gate_or_quality(
    gate: Any | None = None,
    quality_config: QualityPipelineConfig | None = None,
    *,
    quality_spec: str | None = None,
) -> PipelineConfigContext:
    if gate is not None:
        return gate.ctx
    base = PipelineConfigContext.resolve(quality_spec)
    if quality_config is not None:
        return base.with_quality(quality_config)
    return base
