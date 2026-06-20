from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TextIO

from indw.store.corpus.registry import CorpusRegistry
from indw.clean.corpus import CorpusCleaningPipeline
from indw.schedule.mix.config import MixtureOrchestrationConfig
from indw.filter.spec.quality import QualityPipelineConfig
from indw.filter.gate.quality import QualityGate
from indw.schedule.state.checkpoint import MergeCheckpoint, clear_merge_outputs, reconcile_checkpoint_output
from indw.schedule.apply.dedup import (
    build_merge_dedup_stack,
    restore_merge_dedup_from_output,
    restore_merge_gate_balancers,
)
from indw.schedule.apply.lifecycle import maybe_merge_complete_early_return
from indw.schedule.read.sources import (
    corpus_has_multilingual_sources,
    load_mix_weights,
    load_source_registry,
)
from indw.schedule.monitor.invariants import assert_merge_output_synced

logger = logging.getLogger(__name__)


@dataclass
class MergeRunContext:
    raw_dir: Path
    out_path: Path
    merge_work: Path
    cfg: QualityPipelineConfig
    merge_ctx: Any
    cleaning_pipeline: CorpusCleaningPipeline
    gate: QualityGate
    reject_log: Any
    stage_profile: Any
    pipeline_metrics: Any
    orch_cfg: MixtureOrchestrationConfig
    license_policy: Any
    source_registry: dict[str, dict[str, Any]]
    index_path: Path
    sources: list[Path]
    checkpoint: MergeCheckpoint
    index: Any
    exact: Any
    fuzzy: Any
    semantic: Any
    embed_semantic: Any
    resuming: bool
    index_file: Optional[TextIO]
    source_names: list[str]
    mix_weights: dict[str, int]
    mode: str


def bootstrap_merge_run(
    raw_dir: Path,
    out_path: Path,
    *,
    quality_config: Optional[QualityPipelineConfig] = None,
    corpus_registry: Optional[CorpusRegistry] = None,
    work_dir: Optional[Path] = None,
    fresh: bool = False,
    resume: bool = True,
    append: bool = False,
    source_filter: Optional[list[str]] = None,
    parallel_workers: int = 1,
) -> tuple[MergeRunContext | None, dict[str, Any] | None]:
    from indw.config.resolve import PipelineConfigContext
    from indw.clean.document.stage_manifest import log_cleaning_manifest
    from indw.tools.metrics.reject_log import MergeRejectLog
    from indw.tools.metrics.stage_profile import MergeStageProfile
    from indw.schedule.config.pin import bind_checkpoint_config
    from indw.schedule.monitor.obs import (
        obs_mode,
        reject_log_enabled,
        reject_log_flush_every,
    )

    raw_dir = Path(raw_dir)
    out_path = Path(out_path)
    cfg = quality_config or QualityPipelineConfig()
    merge_ctx = PipelineConfigContext.resolve().with_quality(cfg)
    merge_work = Path(work_dir) if work_dir else out_path.parent
    if cfg.cleaning.artifact_discovery:
        cfg.cleaning.artifact_discovery_corpus_dir = str(merge_work)
    cleaning_pipeline = CorpusCleaningPipeline(cfg.cleaning, score_thresholds=cfg.thresholds)
    log_cleaning_manifest(
        cfg.cleaning,
        dedup_semantic=bool(getattr(cfg.dedup, 'semantic', False)),
    )
    obs_cfg = cfg.observability or {}
    reject_log = MergeRejectLog(
        merge_work,
        enabled=reject_log_enabled(obs_cfg),
        flush_every=reject_log_flush_every(),
    )
    stage_profile = MergeStageProfile()
    logger.info('Observability mode: %s', obs_mode())
    gate = QualityGate(ctx=merge_ctx)
    corpus_id = corpus_registry.corpus_id if corpus_registry else 'default'
    pipeline_metrics = None
    try:
        from monitoring.metrics.pipeline_exporter import PipelineMetricsExporter
        pipeline_metrics = PipelineMetricsExporter.begin_merge(
            corpus_id=corpus_id,
            fresh=fresh,
        )
        logger.info('Pipeline metrics enabled (corpus_id=%s)', corpus_id)
    except Exception as exc:
        logger.warning('Pipeline metrics disabled: %s', exc)

    orch_cfg = MixtureOrchestrationConfig.from_dict(cfg.orchestration)
    license_policy = cfg.license_policy()
    source_registry = load_source_registry(raw_dir)
    index_path = out_path.with_suffix('.mixture_index.jsonl')
    if source_filter:
        sources = [raw_dir / n / 'data.jsonl' for n in source_filter]
        sources = [p for p in sources if p.exists()]
    else:
        sources = sorted(raw_dir.glob('*/data.jsonl'))
    if not sources:
        raise FileNotFoundError(f'No raw JSONL under {raw_dir}')

    checkpoint: Optional[MergeCheckpoint] = None
    if fresh:
        if corpus_registry is not None:
            corpus_registry.close()
        removed = clear_merge_outputs(merge_work)
        if removed:
            logger.info('Fresh merge: cleared %s', ', '.join(removed))
        from indw.clean.artifact.discovery_engine import reset_discovery_engines
        reset_discovery_engines()
    elif resume:
        checkpoint = MergeCheckpoint.load(merge_work)

    if checkpoint is None:
        checkpoint = MergeCheckpoint()
    bind_checkpoint_config(checkpoint, work_dir or merge_work, fresh=fresh)

    index = corpus_registry.open_index() if corpus_registry else None
    exact, fuzzy, semantic, embed_semantic = build_merge_dedup_stack(cfg, index)

    pruned = checkpoint.prune_sources({src.parent.name for src in sources})
    if pruned:
        logger.info('Pruned stale checkpoint sources: %s', ', '.join(pruned))
    gate.calibrator.import_state(checkpoint.adaptive_calibrator_state)

    early = maybe_merge_complete_early_return(
        checkpoint=checkpoint,
        source_filter=source_filter,
        merge_work=merge_work,
        out_path=out_path,
        corpus_registry=corpus_registry,
        gate=gate,
        cfg=cfg,
        work_dir=work_dir,
        exact=exact,
        cleaning_pipeline=cleaning_pipeline,
        pipeline_metrics=pipeline_metrics,
        index=index,
        fuzzy=fuzzy,
        semantic=semantic,
        embed_semantic=embed_semantic,
        parallel_workers=parallel_workers,
    )
    if early is not None:
        return None, early

    resuming = any(checkpoint.line_offset(src.parent.name) > 0 for src in sources)
    mode = 'a' if (append or resuming) and out_path.exists() else 'w'
    index_mode = 'a' if (append or resuming) and index_path.exists() else 'w'
    index_file = index_path.open(index_mode, encoding='utf-8') if orch_cfg.enabled else None
    restore_merge_gate_balancers(
        gate=gate,
        checkpoint=checkpoint,
        index_path=index_path,
        resuming=resuming,
        append=append,
    )
    restore_merge_dedup_from_output(
        out_path=out_path,
        checkpoint=checkpoint,
        cfg=cfg,
        exact=exact,
        fuzzy=fuzzy,
        semantic=semantic,
        embed_semantic=embed_semantic,
        index=index,
        resuming=resuming,
        append=append,
    )
    if resuming:
        reconcile_checkpoint_output(checkpoint, out_path, logger=logger)
        assert_merge_output_synced(checkpoint, out_path, exact=exact, context='post-reconcile resume')

    source_names = [src.parent.name for src in sources]
    mix_weights = load_mix_weights(raw_dir, source_names)
    if not corpus_has_multilingual_sources(source_names):
        gate.lang_balancer.enabled = False
        logger.info('Language balancer disabled (monolingual source set)')

    ctx = MergeRunContext(
        raw_dir=raw_dir,
        out_path=out_path,
        merge_work=merge_work,
        cfg=cfg,
        merge_ctx=merge_ctx,
        cleaning_pipeline=cleaning_pipeline,
        gate=gate,
        reject_log=reject_log,
        stage_profile=stage_profile,
        pipeline_metrics=pipeline_metrics,
        orch_cfg=orch_cfg,
        license_policy=license_policy,
        source_registry=source_registry,
        index_path=index_path,
        sources=sources,
        checkpoint=checkpoint,
        index=index,
        exact=exact,
        fuzzy=fuzzy,
        semantic=semantic,
        embed_semantic=embed_semantic,
        resuming=resuming,
        index_file=index_file,
        source_names=source_names,
        mix_weights=mix_weights,
        mode=mode,
    )
    return ctx, None
