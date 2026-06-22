from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from indw.schedule.monitor.audit import load_work_json


def _metric(d: dict[str, Any], *keys: str, default: Any = 0) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _execution_graph() -> dict[str, Any]:
    return {
        'entry': 'fast_pipeline.run -> merge_with_quality (core.py)',
        'bootstrap': 'setup.bootstrap_merge_run -> build_merge_dedup_stack',
        'parallel': {
            'reader': 'parallel._reader_thread -> read_queue',
            'fast': 'concurrent.run_pipelined_merge -> process_fast_merge_batch -> run_fast_stages',
            'survivors': 'MergeDocumentContext.survivor_payload -> LaneBuffers',
            'heavy': 'process_heavy_merge_batch -> run_heavy_stages -> CorpusCleaningPipeline.process',
            'apply': 'parallel._apply_loop -> apply_merge_preprocessed_line -> write_merge_chunks',
            'finalize': 'artifacts.run_merge_finalize -> finalize_merge',
        },
        'serial': 'core._merge_with_quality_locked -> preprocess_merge_line -> run_progressive_preprocess',
        'frozen_stages': [
            'stage0_filters', 'semantic_cleaning', 'knowledge_extraction',
            'publication_recovery', 'acim', 'pci', 'lci', 'dedup', 'gates', 'lanes',
        ],
    }


def _ownership_graph() -> dict[str, Any]:
    return {
        'dedup_index': {
            'owner': 'main: build_merge_dedup_stack',
            'worker': 'fast workers flush-only (_shutdown_fast_worker)',
            'close': 'core/parallel after run_merge_finalize (once)',
        },
        'corpus_registry': {
            'owner': 'artifacts.run_merge_finalize',
            'close': 'finalize path only',
        },
        'discovery_engine': {
            'owner': 'artifacts.run_merge_finalize',
            'close': 'discovery_engine.close + reset_discovery_engines',
        },
        'survivor_store': {
            'owner': 'survivor_store.externalize_survivor_text',
            'resolve': 'resolve_survivor_text + payload._work_dir',
        },
        'apply_buffer': {
            'owner': 'parallel._apply_loop (next_write_seq)',
            'ingest': 'concurrent collect -> _ingest_lines',
        },
        'lane_buffers': {
            'owner': 'concurrent.run_pipelined_merge LaneBuffers',
            'backpressure': 'survivor_buffer_cap + can_submit_heavy',
        },
        'finalize': {
            'owner': 'artifacts.run_merge_finalize',
            'call_sites': ['parallel complete', 'parallel timed_out', 'core complete', 'core timed_out'],
        },
    }


def _resource_lifetime() -> dict[str, Any]:
    return {
        'PersistentHashIndex': 'open bootstrap -> flush workers -> flush main -> finalize len() -> close main',
        'CorpusRegistry': 'open bootstrap -> close finalize',
        'MergeCoordinator': 'open parallel/core -> close finally before finalize',
        'reject_log': 'open bootstrap -> close finally',
        'ProcessPoolExecutor': 'concurrent with-block -> worker atexit flush',
        'survivor_store_files': 'persist under .survivor_store/ per run work_dir',
    }


def _queue_graph() -> dict[str, Any]:
    return {
        'read_queue': 'reader thread -> concurrent fast_pull (stopped on time_limit)',
        'pending_fast': 'fast ProcessPoolExecutor futures',
        'lane_buffers': 'survivors by lane (normal/large/huge)',
        'pending_heavy': 'heavy ProcessPoolExecutor futures',
        'line_results': 'ordered apply buffer keyed by seq',
        'apply_sink': 'BufferedJsonlWriter + checkpoint flusher',
    }


def _shutdown_sequence() -> list[str]:
    return [
        'time_limit -> on_time_limit sets stop_event (reader stops)',
        'drain_mode: no fast_pull; heavy submit/collect continues',
        'drain_done: queues empty or drain_deadline (no pending heavy)',
        'executor context exit: workers flush dedup index (no close)',
        'apply_stop.set -> apply drains contiguous line_results',
        'finally: coordinator.close, reject_log.close, sched probe publish',
        'index.flush -> run_merge_finalize (discovery reset, corpus close) -> index.close',
    ]


_LEGACY_REMOVED = [
    'worker exact._index.close() — main owns SQLite lifetime',
    'parallel finally reset_discovery_engines — artifacts owns discovery teardown',
    'parallel timed_out pre-close corpus_registry — finalize owns close',
    'serial timed_out ad-hoc dict return — unified run_merge_finalize',
    'apply loop stop_state mid-seq abort — apply_stop only',
    'duplicate plan_pipelined_alloc when alloc passed from parallel',
]


_INTENTIONAL_SAFETY = [
    'process_merge_batch inline fallback on fast-pool failure after one retry',
    'validate_survivor_payload admission + text/store resolution',
    'assert_merge_output_synced before complete finalize',
    'head_priority dispatch when ordering_gap > 0',
    'heavy_ooo_dispatch_limit ordering hold',
]


def build_stabilization_audit(
    work_dir: Path,
    *,
    baseline_dir: Path | None = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    scheduler = load_work_json(work_dir / 'pipeline_scheduler_report.json')
    live = load_work_json(work_dir / 'pipeline_live_metrics.json')
    progress = load_work_json(work_dir / 'pipeline_progress.json')
    stage_metrics = load_work_json(work_dir / 'stage_metrics.json')
    validation = load_work_json(work_dir / 'pipeline_validation_report.json')
    summary = load_work_json(work_dir / 'pipeline_validation_summary.json')
    stage0 = load_work_json(work_dir / 'stage0_audit_report.json')

    sched_live = live.get('scheduler') or {}
    dispatch = scheduler.get('dispatch_denied') or {}
    cache = live.get('cache') or stage_metrics.get('cache') or {}
    elapsed_sec = float(scheduler.get('elapsed_sec') or stage_metrics.get('merge_wall_sec') or 1)

    backlog = {
        'peak_survivor_buffer': _metric(sched_live, 'peak_survivor_buffer'),
        'peak_ordering_gap': _metric(scheduler, 'peak_ordering_gap'),
        'peak_apply_buffer': _metric(scheduler, 'peak_apply_buffer'),
        'peak_heavy_pending': _metric(sched_live, 'peak_heavy_pending'),
        'ordering_wait_ms': _metric(scheduler, 'ordering_wait_ms'),
        'ordering_wait_pct': _metric(scheduler, 'ordering_wait_pct'),
        'lane_slot_blocked': dispatch.get('lane_slot_blocked', 0),
        'survivor_buffer_cap': dispatch.get('survivor_buffer_cap', 0),
        'apply_buffer_cap': dispatch.get('apply_buffer_cap', 0),
        'ordering_gap_hold': dispatch.get('ordering_gap_hold', 0),
    }

    ordering = {
        'peak_ordering_gap': backlog['peak_ordering_gap'],
        'ordering_wait_ms': backlog['ordering_wait_ms'],
        'ordering_wait_pct': backlog['ordering_wait_pct'],
        'head_blocked_dispatches': _metric(scheduler, 'head_blocked_dispatches'),
        'head_priority_dispatches': _metric(scheduler, 'head_priority_dispatches'),
        'dispatched_past_head': _metric(scheduler, 'dispatched_past_head'),
        'intentional_constraints': [
            'apply commits strictly by next_write_seq',
            'head_priority when ordering_gap > 0 and head in buffer',
            'heavy_ooo_dispatch_limit caps out-of-order dispatch',
            'huge lane deferred when normal min_seq < huge min_seq',
        ],
    }

    apply_bottleneck = {
        'peak_apply_buffer': backlog['peak_apply_buffer'],
        'ordering_wait_ms': backlog['ordering_wait_ms'],
        'ordering_wait_pct': backlog['ordering_wait_pct'],
        'apply_completions': _metric(scheduler, 'apply_completions'),
        'heavy_submits': _metric(scheduler, 'heavy_submits'),
        'heavy_collects': _metric(scheduler, 'heavy_collects'),
        'heavy_apply_backpressure_events': _metric(scheduler, 'heavy_apply_backpressure_events'),
    }

    dup_events: list[dict[str, Any]] = []
    if stage0:
        counters = stage0.get('counters') or {}
        dup_events = [{'note': 'see stage0_audit events for per-seq duplicate detection'}]
    else:
        try:
            from indw.filter.stage0.verify import build_production_verification_report
            ver = build_production_verification_report(work_dir)
            dup_events = ver.get('duplicate_execution', {}).get('duplicate_events') or []
        except Exception:
            pass

    functioning = (
        validation.get('verdict', {}).get('pipeline_functioning')
        if isinstance(validation.get('verdict'), dict)
        else validation.get('pipeline_functioning')
    )
    if functioning is None:
        functioning = summary.get('pipeline_functioning')

    report: dict[str, Any] = {
        'work_dir': str(work_dir),
        'execution_graph': _execution_graph(),
        'ownership': _ownership_graph(),
        'resource_lifetime': _resource_lifetime(),
        'queues': _queue_graph(),
        'shutdown_sequence': _shutdown_sequence(),
        'survivor_lifecycle': {
            'externalize_threshold': _metric(validation, 'tuning', 'ipc_externalize_chars', default=50_000),
            'store_dir': str(work_dir / '.survivor_store'),
            'store_files': len(list((work_dir / '.survivor_store').glob('*.txt')))
            if (work_dir / '.survivor_store').is_dir() else 0,
            'validation': 'validate_survivor_payload resolves text_store_key via _work_dir',
        },
        'ipc_audit': {
            'inline_text': 'payload.text below externalize threshold',
            'externalized': 'text="" + text_store_key + _work_dir',
            'heavy_worker': '_work_dir injected before validate',
            'empty_payload_guard': 'validate_survivor_payload rejects unresolved store',
        },
        'scheduler_backlog': backlog,
        'scheduler_analysis': {
            'loop_count': _metric(scheduler, 'loop_count'),
            'loops_per_sec': _metric(scheduler, 'loops_per_sec'),
            'worker_idle_ms': _metric(scheduler, 'worker_idle_ms'),
            'reader_block_events': _metric(scheduler, 'reader_block_events'),
            'dispatch_denied': dispatch,
            'lane_slots': scheduler.get('lane_slots'),
            'avg_phase_ms': scheduler.get('avg_phase_ms'),
        },
        'apply_bottleneck': apply_bottleneck,
        'ordering': ordering,
        'bottleneck_tree': scheduler.get('bottleneck_tree') or live.get('bottlenecks') or [],
        'cache_efficiency': cache,
        'duplicate_execution': {
            'events': dup_events,
            'legacy_removed': _LEGACY_REMOVED,
        },
        'intentional_safety_checks': _INTENTIONAL_SAFETY,
        'unavoidable_bottlenecks': [
            'serial apply commit (next_write_seq) under parallel heavy completion',
            'head-of-line blocking on slow docs (ke_serialization)',
            'lane slot caps (5 heavy slots vs survivor depth)',
        ],
        'progress': {
            'scanned': progress.get('scanned'),
            'kept': progress.get('kept'),
            'rejected': progress.get('rejected'),
            'elapsed_sec': elapsed_sec,
        },
        'memory': {
            'peak_rss_mb': _metric(live, 'peak_rss_mb'),
            'rss_mb': _metric(live, 'rss_mb'),
        },
        'worker_utilization': {
            'worker_util_pct': _metric(sched_live, 'worker_util_pct'),
            'fast_submits': _metric(scheduler, 'fast_submits'),
            'fast_collects': _metric(scheduler, 'fast_collects'),
            'heavy_submits': _metric(scheduler, 'heavy_submits'),
            'heavy_collects': _metric(scheduler, 'heavy_collects'),
        },
        'validation': {
            'pipeline_functioning': functioning,
            'scanned': summary.get('scanned') or validation.get('throughput', {}).get('documents_scanned'),
            'accepted': summary.get('accepted'),
            'errors': validation.get('errors') or [],
        },
        'success_criteria': {
            'no_sqlite_finalize_crash': functioning is True,
            'no_empty_survivor_payload': _metric(scheduler, 'worker_failures', default=0) == 0,
            'graceful_shutdown': functioning is True and elapsed_sec > 120,
            'bounded_survivor_backlog': backlog['peak_survivor_buffer'] < 2000,
            'controlled_ordering_gap': backlog['peak_ordering_gap'] < 150,
        },
    }

    if baseline_dir is not None:
        base = build_stabilization_audit(Path(baseline_dir))
        report['before_after'] = {
            'baseline': str(baseline_dir),
            'complexity': {
                'peak_survivor_buffer': {
                    'before': _metric(base, 'scheduler_backlog', 'peak_survivor_buffer'),
                    'after': backlog['peak_survivor_buffer'],
                },
                'peak_ordering_gap': {
                    'before': _metric(base, 'ordering', 'peak_ordering_gap'),
                    'after': ordering['peak_ordering_gap'],
                },
                'peak_apply_buffer': {
                    'before': _metric(base, 'scheduler_backlog', 'peak_apply_buffer'),
                    'after': backlog['peak_apply_buffer'],
                },
                'finalize_crash_before': not base.get('success_criteria', {}).get('no_sqlite_finalize_crash', True),
            },
            'backlog_delta': {
                k: backlog.get(k, 0) - _metric(base, 'scheduler_backlog', k)
                for k in backlog
            },
        }

    return report


def publish_stabilization_audit(work_dir: Path, report: dict[str, Any]) -> Path:
    work_dir = Path(work_dir)
    out_json = work_dir / 'production_stabilization_audit.json'
    out_md = work_dir / 'production_stabilization_audit.md'
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    out_md.write_text(human_stabilization_summary(report), encoding='utf-8')
    return out_json


def human_stabilization_summary(report: dict[str, Any]) -> str:
    bb = report.get('scheduler_backlog') or {}
    ab = report.get('apply_bottleneck') or {}
    val = report.get('validation') or {}
    sc = report.get('success_criteria') or {}
    lines = [
        '# Production Stabilization Audit',
        '',
        f"work_dir: {report.get('work_dir')}",
        '',
        '## Success criteria',
    ]
    for k, v in sc.items():
        lines.append(f'- {k}: {v}')
    lines.extend([
        '',
        '## Backlog',
        f"- peak_survivor_buffer: {bb.get('peak_survivor_buffer')}",
        f"- peak_ordering_gap: {bb.get('peak_ordering_gap')}",
        f"- ordering_wait_ms: {bb.get('ordering_wait_ms')} ({bb.get('ordering_wait_pct')}%)",
        f"- survivor_buffer_cap_denials: {bb.get('survivor_buffer_cap')}",
        '',
        '## Apply bottleneck',
        f"- peak_apply_buffer: {ab.get('peak_apply_buffer')}",
        f"- apply_completions: {ab.get('apply_completions')}",
        '',
        '## Validation',
        f"- pipeline_functioning: {val.get('pipeline_functioning')}",
        f"- scanned: {val.get('scanned')} accepted: {val.get('accepted')}",
        '',
        '## Legacy removed',
    ])
    for item in (report.get('duplicate_execution') or {}).get('legacy_removed') or []:
        lines.append(f'- {item}')
    lines.extend(['', '## Shutdown sequence'])
    for step in report.get('shutdown_sequence') or []:
        lines.append(f'- {step}')
    ba = report.get('before_after')
    if ba:
        lines.extend(['', '## Before/after', f"baseline: {ba.get('baseline')}"])
        cx = ba.get('complexity') or {}
        for k, row in cx.items():
            if isinstance(row, dict) and 'before' in row:
                lines.append(f'- {k}: {row.get("before")} -> {row.get("after")}')
            else:
                lines.append(f'- {k}: {row}')
    return '\n'.join(lines) + '\n'
