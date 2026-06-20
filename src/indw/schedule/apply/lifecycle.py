from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def maybe_merge_complete_early_return(
    *,
    checkpoint: Any,
    source_filter: Optional[list[str]],
    merge_work: Path,
    out_path: Path,
    corpus_registry: Any,
    gate: Any,
    cfg: Any,
    work_dir: Optional[Path],
    exact: Any,
    cleaning_pipeline: Any,
    pipeline_metrics: Any,
    index: Any,
    fuzzy: Any,
    semantic: Any,
    embed_semantic: Any = None,
    parallel_workers: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    from indw.schedule.state.artifacts import merge_quality_report_path, run_merge_finalize

    if source_filter and checkpoint.complete:
        checkpoint.complete = False
        logger.info('Incremental merge: resetting complete flag for filtered sources')
    if not (checkpoint.complete and not source_filter):
        return None
    if merge_quality_report_path(merge_work).exists():
        logger.info('Merge checkpoint already complete; skipping quality merge')
        if corpus_registry is not None:
            corpus_registry.close()
        from indw.store.io.jsonl import checkpoint_kept_lines, count_jsonl_lines
        kept_lines = checkpoint_kept_lines(checkpoint) or (
            count_jsonl_lines(out_path) if out_path.exists() else 0
        )
        return {
            'docs': kept_lines,
            'skipped_parse': 0,
            'bytes': out_path.stat().st_size if out_path.exists() else 0,
            'elapsed_sec': 0.0,
            'resumed': True,
            'complete': True,
        }
    logger.info('Merge checkpoint complete but quality artifacts missing; running finalize only')
    finalize_kwargs: dict[str, Any] = {
        'gate': gate,
        'cfg': cfg,
        'out_path': out_path,
        'work_dir': work_dir,
        'merge_work': merge_work,
        'checkpoint': checkpoint,
        'exact': exact,
        'cleaning_pipeline': cleaning_pipeline,
        'skipped_parse': 0,
        'elapsed': 0.0,
        'pipeline_metrics': pipeline_metrics,
        'corpus_registry': corpus_registry,
        'index': index,
        'fuzzy': fuzzy,
        'semantic': semantic,
        'embed_semantic': embed_semantic,
        'seed_from_progress': True,
    }
    if parallel_workers is not None:
        finalize_kwargs['parallel_workers'] = parallel_workers
        finalize_kwargs['log_message'] = f'Quality merge finalize-only (parallel workers={parallel_workers})'
    else:
        finalize_kwargs['log_message'] = 'Quality merge finalize-only'
    return run_merge_finalize(**finalize_kwargs)


def pause_merge_run(
    *,
    sink: Any,
    index_file: Any,
    index: Any,
    checkpoint: Any,
    merge_work: Path,
    gate: Any,
    exact: Any,
    total_scanned: int,
    elapsed_sec: float,
    message: str,
    extra_progress: Optional[dict[str, Any]] = None,
    record_interrupt: bool = False,
    cleaning_pipeline: Any = None,
    reject_log: Any = None,
    stage_profile: Any = None,
) -> None:
    from indw.schedule.state.checkpoint import MergeCheckpoint, publish_merge_progress

    if sink is not None:
        sink.flush()
    if index_file is not None:
        index_file.flush()
    if index is not None:
        index.flush()
    checkpoint.save(merge_work, interrupted=True, gate=gate)
    if record_interrupt:
        from indw.tools.metrics.recovery import record_recovery_event
        record_recovery_event(merge_work, 'merge_interrupted', status='paused')
    totals = checkpoint.totals()
    publish_merge_progress(
        merge_work,
        gate=gate,
        exact=exact,
        total_scanned=total_scanned,
        elapsed_sec=elapsed_sec,
        status='paused',
        kept=totals['kept'],
        rejected=totals['rejected'],
        extra=extra_progress,
        force=True,
    )
    if reject_log is not None:
        reject_log.flush()
    if cleaning_pipeline is not None or stage_profile is not None:
        from indw.schedule.monitor.obs import stage_metrics_on_finalize
        if stage_metrics_on_finalize():
            from indw.tools.metrics.stage_profile import MergeStageProfile, write_stage_metrics

            profile = stage_profile if stage_profile is not None else MergeStageProfile()
            cleaning_stats = cleaning_pipeline.snapshot() if cleaning_pipeline is not None else None
            write_stage_metrics(
                merge_work,
                profile,
                cleaning_stats=cleaning_stats,
                merge_wall_sec=elapsed_sec,
                docs_scanned=total_scanned,
            )
    logger.info(
        '%s: scanned=%d kept=%d checkpoint=%s',
        message,
        totals['scanned'],
        totals['kept'],
        MergeCheckpoint.path_for(merge_work),
    )
