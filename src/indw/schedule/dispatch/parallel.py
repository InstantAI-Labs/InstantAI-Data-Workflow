from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional, TextIO

from indw.schedule.core import _InterleavedSources
from indw.schedule.state.checkpoint import (
    METRICS_SNAPSHOT_INTERVAL,
    PROGRESS_LOG_INTERVAL,
    load_run_progress,
    publish_live_gate_snapshot,
    log_merge_progress,
    publish_merge_progress,
    publish_resume_gate_snapshot,
    resume_metrics_kwargs,
    make_merge_checkpoint_flusher,
)
from indw.store.io.jsonl import checkpoint_kept_lines, resolve_raw_line_total
from indw.schedule.monitor.invariants import assert_merge_output_synced
from indw.schedule.apply.merge import apply_merge_preprocessed_line
from indw.schedule.row.signals import (
    MergeStopState,
    install_merge_signal_handlers,
    merge_stop_handler,
    restore_merge_signal_handlers,
)
from indw.schedule.apply.lifecycle import pause_merge_run
from indw.schedule.row.provenance import provenance_for_merge_row
from indw.schedule.apply.serialize import merge_cleaning_stats
from indw.schedule.monitor.obs import periodic_stdout_progress_enabled
from indw.schedule.read.preprocess import parse_merge_jsonl_line
from indw.schedule.state.sessions import open_merge_coordinator
from indw.store.corpus.registry import CorpusRegistry
from indw.ingest.sink import BufferedJsonlWriter, DEFAULT_WRITE_BUFFER
from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule.state.artifacts import run_merge_finalize
from monitoring.cpu import collect_cpu_stats

logger = logging.getLogger(__name__)

from indw.schedule.config.policy import MERGE_READ_SENTINEL
from indw.schedule.config.tune import bind_merge_tune, get_merge_tune, merge_drain_sec, resolve_merge_tune


def _quality_config_path(merge_work: Path) -> Path:
    path = merge_work / '_resolved_quality.yaml'
    if path.exists():
        return path
    raise FileNotFoundError(
        f'Parallel merge requires quality config at {path}. '
        'Run prepare via FastDatasetPipeline so _resolved_quality.yaml exists.'
    )


def _reader_thread(
    interleaved: _InterleavedSources,
    read_queue: queue.Queue,
    *,
    chunk_size: int,
    stream_batch: int,
    batch_flush_sec: float,
    stop_event: threading.Event,
    license_policy: Any,
    source_registry: dict[str, dict[str, Any]],
    probe: Any | None = None,
) -> None:
    batch: list[dict[str, Any]] = []
    seq = 0
    batch_target = max(1, stream_batch)
    batches_emitted = 0
    batch_started = time.monotonic()

    def _emit_batch() -> None:
        nonlocal batch, batch_target, batch_started, batches_emitted
        if not batch or stop_event.is_set():
            return
        while not stop_event.is_set():
            try:
                read_queue.put(batch, timeout=0.25)
                break
            except queue.Full:
                if probe is not None:
                    probe.reader_block_events += 1
                continue
        batches_emitted += 1
        if batches_emitted == 1 and batch_target < stream_batch:
            batch_target = max(1, stream_batch)
        elif batch_target < chunk_size:
            batch_target = min(chunk_size, max(batch_target * 2, stream_batch))
        batch = []
        batch_started = time.monotonic()

    try:
        for src_name, _src_path, line_no, line in interleaved:
            if stop_event.is_set():
                break
            _line_kind, row = parse_merge_jsonl_line(line)
            item: dict[str, Any] = {
                'seq': seq,
                'src_name': src_name,
                'line_no': line_no,
            }
            if row is not None:
                item['row'] = row
            else:
                item['line'] = line
            prov = (
                provenance_for_merge_row(
                    row,
                    src_name,
                    license_policy=license_policy,
                    source_registry=source_registry,
                )
                if row is not None
                else None
            )
            if prov is not None:
                item['provenance'] = prov
            from indw.schedule.read.ingest import ingest_line_meta
            item['ingest_meta'] = ingest_line_meta(
                line=line, src_name=src_name, line_no=line_no, seq=seq,
            )
            from indw.filter.stage0.audit import audit_enabled, record_reader_input
            if audit_enabled():
                chars = None
                if row is not None:
                    from indw.clean.corpus import extract_row_text
                    chars = len(extract_row_text(row))
                record_reader_input(seq=seq, source=src_name, chars=chars)
            batch.append(item)
            seq += 1
            now = time.monotonic()
            if len(batch) >= batch_target or (
                batch and (now - batch_started) >= batch_flush_sec
            ):
                _emit_batch()
        if batch and not stop_event.is_set():
            _emit_batch()
    finally:
        interleaved.close()
        read_queue.put(MERGE_READ_SENTINEL)


def merge_with_quality_parallel(
    raw_dir: Path,
    out_path: Path,
    *,
    quality_config: Optional[QualityPipelineConfig] = None,
    corpus_registry: Optional[CorpusRegistry] = None,
    write_buffer_bytes: int = DEFAULT_WRITE_BUFFER,
    source_filter: Optional[list[str]] = None,
    append: bool = False,
    work_dir: Optional[Path] = None,
    resume: bool = True,
    fresh: bool = False,
    workers: int = 2,
    chunk_size: Optional[int] = None,
    checkpoint_interval: int | None = None,
    time_limit_sec: Optional[float] = None,
    validation_collector: Optional[Any] = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)
    cfg = quality_config or QualityPipelineConfig()
    workers = max(1, int(workers))
    merge_work = Path(work_dir) if work_dir else out_path.parent
    coordinator = open_merge_coordinator(
        merge_work,
        workers=workers,
        chunk_size=chunk_size,
        checkpoint_interval=checkpoint_interval,
    )
    policy = coordinator.runtime.policy
    chunk_size = policy.chunk_size
    workers = policy.workers
    checkpoint_interval = policy.checkpoint_interval
    batch_timeout_sec = policy.batch_timeout_sec
    metrics_interval_sec = policy.metrics_snapshot_sec
    from indw.schedule.state.setup import bootstrap_merge_run

    run_ctx, early = bootstrap_merge_run(
        raw_dir,
        out_path,
        quality_config=cfg,
        corpus_registry=corpus_registry,
        work_dir=work_dir,
        fresh=fresh,
        resume=resume,
        append=append,
        source_filter=source_filter,
        parallel_workers=workers,
    )
    if early is not None:
        return early
    assert run_ctx is not None
    cleaning_pipeline = run_ctx.cleaning_pipeline
    gate = run_ctx.gate
    reject_log = run_ctx.reject_log
    stage_profile = run_ctx.stage_profile
    pipeline_metrics = run_ctx.pipeline_metrics
    license_policy = run_ctx.license_policy
    source_registry = run_ctx.source_registry
    sources = run_ctx.sources
    checkpoint = run_ctx.checkpoint
    index = run_ctx.index
    exact = run_ctx.exact
    fuzzy = run_ctx.fuzzy
    semantic = run_ctx.semantic
    embed_semantic = run_ctx.embed_semantic
    resuming = run_ctx.resuming
    index_file = run_ctx.index_file
    source_names = run_ctx.source_names
    mix_weights = run_ctx.mix_weights
    mode = run_ctx.mode
    merge_work = run_ctx.merge_work
    orch_cfg = run_ctx.orch_cfg
    config_path = _quality_config_path(merge_work)
    skipped_parse = 0

    from indw.schedule.config.hardware import probe_system_hardware
    from indw.schedule.dispatch.alloc import plan_graph_alloc

    _hw = probe_system_hardware(merge_work)
    tune = resolve_merge_tune(workers=workers, chunk_size=chunk_size, hw=_hw)
    bind_merge_tune(tune)
    _hw, pipe_alloc = plan_graph_alloc(
        workers=workers,
        chunk_size=chunk_size,
        merge_work=merge_work,
    )
    ingest_batch_size = pipe_alloc.fast_batch
    stream_batch_size = pipe_alloc.stream_batch
    max_buffered_lines = pipe_alloc.apply_queue
    max_completed_results = max(
        max_buffered_lines * tune.result_buffer_factor,
        tune.fast_buffer_floor,
    )
    max_heavy_completed_results = max(
        max_buffered_lines * tune.heavy_result_buffer_factor,
        tune.heavy_buffer_floor,
    )
    from indw.dedup.exact import PersistentHashIndex
    from indw.schedule.dispatch.workers import build_worker_init_bundle

    dedup_index_path = ''
    if index is not None:
        dedup_index_path = str(index.db_path)
    else:
        dedup_index_path = str(PersistentHashIndex.default_path(merge_work))

    dedup_shards = int(getattr(pipe_alloc, 'dedup_shards', 0) or 0)
    worker_init = build_worker_init_bundle(
        str(config_path),
        workers=workers,
        chunk_size=chunk_size,
        checkpoint_interval=checkpoint_interval,
        work_dir=str(merge_work),
        dedup_index_path=dedup_index_path,
        dedup_shards=dedup_shards,
    )
    timed_out = False
    from indw.schedule.admission import TierTracker, TIER3
    from indw.schedule.monitor.cost import StageCostLedger
    tier_tracker = TierTracker()
    cost_ledger = StageCostLedger()

    def _absorb_cost(payload: dict[str, Any]) -> None:
        cost_ledger.absorb_payload(payload)

    def _track_admission_line(line: dict[str, Any]) -> None:
        rt = line.get('_reject_tier')
        if rt is not None:
            tier_tracker.record_reject(int(rt))
            return
        rejects = line.get('cleaning_rejects') or []
        if rejects:
            tier_tracker.record_reject(TIER3)

    for src_name in source_names:
        src_state = checkpoint.source(src_name)
        offset = checkpoint.line_offset(src_name)
        if offset > 0:
            logger.info('  [%s] resume at line %d (kept=%d)', src_name, offset, src_state.kept)
        else:
            logger.info('  [%s] mix_weight=%d', src_name, mix_weights.get(src_name, 1))

    interleaved = _InterleavedSources.open(sources, checkpoint, mix_weights)
    total_scanned = sum(checkpoint.source(n).scanned for n in source_names)
    total_raw_estimate = resolve_raw_line_total(sources)
    resume_progress = load_run_progress(merge_work) if resuming else {}
    resume_metric_base = resume_metrics_kwargs(resume_progress)
    resume_exact_dup_base = int(resume_metric_base.get('exact_duplicates', 0))
    if pipeline_metrics is not None:
        pipeline_metrics.set_scan_baseline(total_scanned)
        publish_resume_gate_snapshot(
            pipeline_metrics,
            gate=gate,
            checkpoint=checkpoint,
            total_scanned=total_scanned,
            resume_metric_base=resume_metric_base,
        )

    stop_state = MergeStopState()
    stop_event = threading.Event()
    sink: Optional[BufferedJsonlWriter] = None
    pci = coordinator.intel
    acim = coordinator.acim
    doc_monitor = coordinator.doc_monitor

    def _pause_merge(*, message: str = 'Merge paused') -> None:
        pause_merge_run(
            sink=sink,
            index_file=index_file,
            index=index,
            checkpoint=checkpoint,
            merge_work=merge_work,
            gate=gate,
            exact=exact,
            total_scanned=total_scanned,
            elapsed_sec=time.perf_counter() - t0,
            message=message,
            extra_progress={'workers': workers},
            record_interrupt=True,
            cleaning_pipeline=cleaning_pipeline,
            reject_log=reject_log,
            stage_profile=stage_profile,
        )

    previous_handler, previous_term = install_merge_signal_handlers(
        merge_stop_handler(
            stop_state,
            lambda: _pause_merge(message='Merge force-stopped'),
            stop_event=stop_event,
        ),
    )

    read_queue: queue.Queue = queue.Queue(maxsize=pipe_alloc.ingest_queue)
    active_workers = 0
    worker_failures = 0
    from indw.schedule.graph.runner import run_graph_merge
    from indw.schedule.read.probe import SchedulerProbe
    from indw.schedule.apply.coordinator import ApplyCoordinator

    sched_probe = SchedulerProbe(t0=t0)
    from indw.filter.stage0.audit import audit_enabled, bind_audit_dir
    if audit_enabled():
        bind_audit_dir(merge_work)
    metrics_stop = threading.Event()
    from indw.schedule.backends.config import pipeline_execution_backend
    exec_backend = pipeline_execution_backend()
    logger.info(
        'Quality merge (graph backend=%s): %d sources -> %s | workers=%d chunk=%d ingest_batch=%d '
        'stream_batch=%d resume=%s batch_timeout=%.0fs max_completed=%d pools=%s metrics_sec=%.0f',
        exec_backend,
        len(sources),
        out_path,
        workers,
        chunk_size,
        ingest_batch_size,
        stream_batch_size,
        resuming,
        batch_timeout_sec,
        max_completed_results,
        pipe_alloc.to_dict(),
        metrics_interval_sec,
    )

    def _publish_metrics_snapshot(*, log_diagnostics: bool = False) -> None:
        from monitoring.cpu import collect_cpu_stats

        cpu = collect_cpu_stats()
        coordinator.refresh_signals(
            cpu_pct=cpu.utilization_pct,
            queue_depth=read_queue.qsize(),
            docs_per_sec=total_scanned / max(time.perf_counter() - t0, 1e-9),
            active_workers=active_workers,
        )
        publish_live_gate_snapshot(
            pipeline_metrics,
            gate=gate,
            source_names=source_names,
            checkpoint=checkpoint,
            total_scanned=total_scanned,
            exact=exact,
            resume_exact_dup_base=resume_exact_dup_base,
            resume_metric_base=resume_metric_base,
            thresholds=cfg.thresholds,
            log_diagnostics=log_diagnostics,
            workers=workers,
            active_workers=active_workers,
            queue_depth=read_queue.qsize(),
            cpu_utilization_pct=cpu.utilization_pct,
        )

    def _metrics_loop() -> None:
        while not metrics_stop.wait(metrics_interval_sec):
            _publish_metrics_snapshot(log_diagnostics=False)
            cp_kept = sum(checkpoint.source(n).kept for n in source_names)
            cp_rejected = sum(checkpoint.source(n).rejected for n in source_names)
            publish_merge_progress(
                merge_work,
                gate=gate,
                exact=exact,
                total_scanned=total_scanned,
                elapsed_sec=time.perf_counter() - t0,
                kept=cp_kept,
                rejected=cp_rejected,
                extra={'filtered_lines': checkpoint_kept_lines(checkpoint)},
            )

    def _prep_apply_line(line: dict[str, Any]) -> dict[str, Any]:
        if line.get('_merge_objects_ready'):
            return line
        chunks_in = line.get('chunks') or []
        if not any(isinstance(chunk.get('clean_result'), dict) for chunk in chunks_in):
            return line
        from indw.schedule.apply.serialize import preprocessed_line_to_objects
        out = preprocessed_line_to_objects(line)
        out['_merge_objects_ready'] = True
        return out

    def _refresh_runtime(**kwargs: Any) -> None:
        nonlocal active_workers
        active_workers = int(kwargs.get('active_workers', active_workers))
        coordinator.refresh_signals(**kwargs)

    scan_counter_ref = {'n': total_scanned}

    def _merge_stop_requested() -> bool:
        return stop_state.requested

    def _on_merge_time_limit() -> None:
        nonlocal timed_out
        if timed_out:
            return
        timed_out = True
        stop_event.set()
        logger.info('Merge time limit reached (%.0fs) — draining', time_limit_sec)

    def _apply_preprocessed_line(line: dict[str, Any]) -> bool:
        nonlocal total_scanned, skipped_parse
        scan_counters = {
            'total_scanned': total_scanned,
            'skipped_parse': skipped_parse,
        }
        assert sink is not None
        apply_t0 = time.perf_counter()
        kept_any = apply_merge_preprocessed_line(
            line,
            cleaning_pipeline=cleaning_pipeline,
            gate=gate,
            cfg=cfg,
            checkpoint=checkpoint,
            exact=exact,
            fuzzy=fuzzy,
            semantic=semantic,
            embed_semantic=embed_semantic,
            sink=sink,
            index_file=index_file,
            scan_counters=scan_counters,
            license_policy=license_policy,
            validation_collector=validation_collector,
            pipeline_metrics=pipeline_metrics,
            stop_requested=stop_state.requested,
            reject_log=reject_log,
            pci=pci,
        )
        from indw.filter.stage0.audit import audit_enabled, record_apply
        if audit_enabled():
            record_apply(line, kept=kept_any, wall_ms=(time.perf_counter() - apply_t0) * 1000.0)
        if kept_any:
            tier_tracker.record_accept()
        elif line.get('chunks'):
            tier_tracker.record_reject(TIER3)
        total_scanned = scan_counters['total_scanned']
        skipped_parse = scan_counters['skipped_parse']
        scan_counter_ref['n'] = total_scanned
        return kept_any

    def _log_parallel_progress(
        *,
        total_scanned: int,
        src_name: str,
        line_no: int,
        read_qsize: int,
        active_workers: int,
        total_raw_estimate: int,
    ) -> None:
        cp_kept = sum(checkpoint.source(n).kept for n in source_names)
        cp_rejected = sum(checkpoint.source(n).rejected for n in source_names)
        cpu = collect_cpu_stats()
        elapsed = max(time.perf_counter() - t0, 1e-6)
        dps = total_scanned / elapsed
        remaining = max(total_raw_estimate - total_scanned, 0)
        eta_sec = remaining / dps if dps > 0 else 0.0
        extra = {
            'workers': workers,
            'active_workers': active_workers,
            'docs_per_sec': round(dps, 1),
            'read_queue_size': read_qsize,
            'cpu_utilization_pct': cpu.utilization_pct,
            'eta_sec': round(eta_sec, 1),
            'filtered_lines': checkpoint_kept_lines(checkpoint),
        }
        publish_merge_progress(
            merge_work,
            gate=gate,
            exact=exact,
            total_scanned=total_scanned,
            elapsed_sec=time.perf_counter() - t0,
            kept=cp_kept,
            rejected=cp_rejected,
            extra=extra,
            force=True,
        )
        if not periodic_stdout_progress_enabled():
            return
        log_merge_progress(
            logger,
            total_scanned=total_scanned,
            gate=gate,
            t0=t0,
            src_name=src_name,
            line_no=line_no,
            exact_dup=exact.duplicates,
            session_kept=cp_kept,
            session_rejected=cp_rejected,
            workers=workers,
            active_workers=active_workers,
            read_queue_size=read_qsize,
            cpu_utilization=cpu.utilization_pct,
            eta_sec=eta_sec,
        )

    apply_coordinator = ApplyCoordinator(
        apply_fn=_apply_preprocessed_line,
        prep_fn=_prep_apply_line,
        tune=tune,
        sched_probe=sched_probe,
        checkpoint_interval=checkpoint_interval,
        pipeline_metrics=pipeline_metrics,
        publish_metrics=lambda: _publish_metrics_snapshot(log_diagnostics=False),
        sink=sink,
        total_scanned_ref=scan_counter_ref,
        on_progress=lambda ctx: _log_parallel_progress(
            total_scanned=total_scanned,
            src_name=ctx['line']['src_name'],
            line_no=ctx['line']['line_no'] + 1,
            read_qsize=read_queue.qsize(),
            active_workers=active_workers,
            total_raw_estimate=total_raw_estimate,
        ),
    )

    def _ingest_lines(lines: list[dict[str, Any]]) -> None:
        for line in lines:
            _track_admission_line(line)
        apply_coordinator.ingest_lines(lines)

    def _ingest_batch_payload(batch_payload: dict[str, Any]) -> None:
        _absorb_cost(batch_payload)
        for line in batch_payload.get('items') or []:
            _track_admission_line(line)
        apply_coordinator.ingest_batch(batch_payload)
        if batch_payload.get('cleaning_stats') is not None:
            merge_cleaning_stats(cleaning_pipeline.stats, batch_payload['cleaning_stats'])

    def _can_submit_heavy() -> bool:
        return apply_coordinator.can_accept(max_completed=max_heavy_completed_results)

    def _apply_next_write_seq() -> int:
        return apply_coordinator.next_write_seq

    def _apply_buffer_depth() -> tuple[int, int]:
        return apply_coordinator.buffer_depth()

    def _notify_apply() -> None:
        apply_coordinator.notify()

    reader = threading.Thread(
        target=_reader_thread,
        args=(interleaved, read_queue),
        kwargs={
            'chunk_size': ingest_batch_size,
            'stream_batch': stream_batch_size,
            'batch_flush_sec': pipe_alloc.batch_flush_sec,
            'stop_event': stop_event,
            'license_policy': license_policy,
            'source_registry': source_registry,
            'probe': sched_probe,
        },
        daemon=True,
        name='merge-reader',
    )

    pci_stats_final: dict[str, Any] = {}
    doc_monitor_stats_final: dict[str, Any] = {}
    try:
        metrics_thread = threading.Thread(
            target=_metrics_loop,
            daemon=True,
            name='merge-metrics',
        )
        metrics_thread.start()
        apply_thread = threading.Thread(
            target=apply_coordinator.run_loop, daemon=True, name='merge-apply',
        )
        apply_thread.start()
        with BufferedJsonlWriter(
            out_path,
            buffer_bytes=write_buffer_bytes,
            mode=mode,
            on_flush=make_merge_checkpoint_flusher(
                checkpoint=checkpoint,
                merge_work=merge_work,
                gate=gate,
                index_file=index_file,
                index=index,
                out_path=out_path,
            ),
        ) as sink_writer:
            sink = sink_writer
            reader.start()
            merge_kwargs = dict(
                config_path=str(config_path),
                workers=workers,
                chunk_size=chunk_size,
                merge_work=merge_work,
                read_queue=read_queue,
                apply_next_write_seq=_apply_next_write_seq,
                ingest_batch=_ingest_batch_payload,
                ingest_line_results=_ingest_lines,
                merge_cleaning_stats=lambda stats: merge_cleaning_stats(
                    cleaning_pipeline.stats, stats,
                ),
                notify_apply=_notify_apply,
                apply_buffer_depth=_apply_buffer_depth,
                stop_requested=_merge_stop_requested,
                refresh_runtime=_refresh_runtime,
                t0=t0,
                total_scanned_ref=scan_counter_ref,
                worker_init=worker_init,
                alloc=pipe_alloc,
                probe=sched_probe,
                batch_timeout_sec=batch_timeout_sec,
                time_limit_sec=time_limit_sec,
                on_time_limit=_on_merge_time_limit,
                drain_sec=merge_drain_sec(time_limit_sec=time_limit_sec),
            )
            worker_failures = run_graph_merge(
                can_submit=_can_submit_heavy,
                on_cost_payload=_absorb_cost,
                **merge_kwargs,
            )
            apply_coordinator.stop()
            apply_coordinator.notify()
            join_timeout = merge_drain_sec(time_limit_sec=time_limit_sec) if timed_out else 120.0
            apply_thread.join(timeout=join_timeout)

            if timed_out:
                _pause_merge(message='Merge time limit reached')
            elif stop_state.requested:
                _pause_merge(message='Merge force-stopped')
    except KeyboardInterrupt:
        stop_state.requested = True
        stop_event.set()
        logger.warning('Merge interrupted — saving checkpoint …')
        if sink is not None:
            _pause_merge(message='Merge interrupted')
    finally:
        metrics_stop.set()
        stop_event.set()
        if reader.is_alive():
            reader.join(timeout=2.0)
        restore_merge_signal_handlers(previous_handler, previous_term)
        reject_log.close()
        sched_probe.finalize_phases()
        sched_probe.publish(merge_work)
        if audit_enabled():
            from indw.filter.stage0.audit import (
                build_report,
                human_summary,
                publish_report,
            )
            sched_path = merge_work / 'pipeline_scheduler_report.json'
            sched_data = {}
            if sched_path.is_file():
                import json as _json
                sched_data = _json.loads(sched_path.read_text(encoding='utf-8'))
            progress_path = merge_work / 'pipeline_progress.json'
            progress_data = {}
            if progress_path.is_file():
                import json as _json
                progress_data = _json.loads(progress_path.read_text(encoding='utf-8'))
            metrics_path = merge_work / 'stage_metrics.json'
            metrics_data = {}
            if metrics_path.is_file():
                import json as _json
                metrics_data = _json.loads(metrics_path.read_text(encoding='utf-8'))
            audit_report = build_report(
                merge_work,
                scheduler=sched_data,
                progress=progress_data,
                stage_metrics=metrics_data,
            )
            publish_report(merge_work, audit_report)
            logger.info('%s', human_summary(audit_report))
        pci_stats_final, doc_monitor_stats_final = coordinator.close()

        def _write_pipeline_audit_report() -> None:
            from indw.tools.reports.pipeline_audit import build_pipeline_audit_report
            from indw.store.io.json_codec import dumps_pretty
            audit_path = merge_work / 'pipeline_audit_report.json'
            audit_path.write_text(
                dumps_pretty(build_pipeline_audit_report(merge_work, workers=workers)),
                encoding='utf-8',
            )

        for _publish in (
            lambda: tier_tracker.publish(merge_work),
            lambda: cost_ledger.publish(merge_work),
            _write_pipeline_audit_report,
        ):
            try:
                _publish()
            except Exception:
                logger.exception('merge report publish failed')

    if index_file is not None:
        index_file.flush()
        index_file.close()
    if index is not None:
        index.flush()
    if timed_out:
        result = run_merge_finalize(
            gate=gate,
            cfg=cfg,
            out_path=out_path,
            work_dir=work_dir,
            merge_work=merge_work,
            checkpoint=checkpoint,
            exact=exact,
            cleaning_pipeline=cleaning_pipeline,
            skipped_parse=skipped_parse,
            elapsed=time.perf_counter() - t0,
            pipeline_metrics=pipeline_metrics,
            corpus_registry=corpus_registry,
            index=index,
            fuzzy=fuzzy,
            semantic=semantic,
            embed_semantic=embed_semantic,
            parallel_workers=workers,
            worker_failures=worker_failures,
            pci_stats=pci_stats_final,
            doc_monitor_stats=doc_monitor_stats_final,
            log_message='Quality merge timed out (parallel workers=%d)' % workers,
        )
        if index is not None:
            index.close()
        return result
    if stop_state.requested:
        if corpus_registry is not None:
            corpus_registry.close()
        if sink is None:
            _pause_merge(message='Merge paused')
        raise SystemExit(130)

    checkpoint.complete = True
    assert_merge_output_synced(checkpoint, out_path, exact=exact, context='parallel merge complete')
    checkpoint.save(merge_work, gate=gate)
    elapsed = time.perf_counter() - t0
    result = run_merge_finalize(
        gate=gate,
        cfg=cfg,
        out_path=out_path,
        work_dir=work_dir,
        merge_work=merge_work,
        checkpoint=checkpoint,
        exact=exact,
        cleaning_pipeline=cleaning_pipeline,
        skipped_parse=skipped_parse,
        elapsed=elapsed,
        pipeline_metrics=pipeline_metrics,
        corpus_registry=corpus_registry,
        index=index,
        fuzzy=fuzzy,
        semantic=semantic,
        embed_semantic=embed_semantic,
        parallel_workers=workers,
        worker_failures=worker_failures,
        pci_stats=pci_stats_final,
        doc_monitor_stats=doc_monitor_stats_final,
        log_message='Quality merge done (parallel workers=%d pipelined)' % workers,
    )
    if index is not None:
        index.close()
    return result
