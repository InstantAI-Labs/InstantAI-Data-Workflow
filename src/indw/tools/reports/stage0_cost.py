from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from indw.schedule.monitor.audit import load_work_json
from indw.filter.stage0.audit import build_report, load_events
from indw.filter.stage0.verify import _duplicate_execution


def _stage0_stage_rows(stage_metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    stages = stage_metrics.get('stages') or {}
    rows: dict[str, dict[str, float]] = {}
    for name, row in stages.items():
        if not isinstance(row, dict):
            continue
        if not str(name).startswith('s1_') and not str(name).startswith('s2_') and name != 's3_admission':
            continue
        in_docs = int(row.get('in_docs') or 0)
        wall = float(row.get('wall_sec') or 0.0)
        rows[str(name)] = {
            'wall_sec': round(wall, 4),
            'cpu_sec': round(float(row.get('cpu_sec') or wall), 4),
            'in_docs': in_docs,
            'ms_per_doc': round(1000.0 * wall / max(in_docs, 1), 3),
            'reject_rate': float(row.get('reject_rate') or 0.0),
        }
    return rows


def _cold_start_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    stage0 = [ev for ev in events if ev.get('event') == 'stage0_fast']
    if not stage0:
        return {'cold_doc_count': 0, 'steady_doc_count': 0}
    walls = [float(ev.get('wall_ms') or 0.0) for ev in stage0]
    cold = walls[0] if walls else 0.0
    steady = walls[1:] if len(walls) > 1 else []
    return {
        'cold_doc_count': 1,
        'steady_doc_count': len(steady),
        'cold_wall_ms': round(cold, 2),
        'steady_avg_ms': round(statistics.mean(steady), 2) if steady else 0.0,
        'steady_median_ms': round(statistics.median(steady), 2) if steady else 0.0,
        'steady_p95_ms': round(sorted(steady)[int(0.95 * (len(steady) - 1))], 2) if len(steady) > 1 else 0.0,
        'cold_overhead_ms': round(max(cold - (statistics.mean(steady) if steady else 0.0), 0.0), 2),
    }


def _duplicate_computation_audit() -> list[dict[str, str]]:
    return [
        {
            'item': 'raw_feature_extract',
            'before': '2x per doc (structural + metadata structural)',
            'after': '1x per survivor (content_filters pass)',
            'status': 'eliminated',
        },
        {
            'item': 'resolve_gate_policy',
            'before': 'per filter sub-call',
            'after': '1x via worker_gate_policy / run_stage0_content_filters',
            'status': 'cached',
        },
        {
            'item': 'resolve_doc_max_chars',
            'before': 'per doc size gate',
            'after': 'worker init bind',
            'status': 'cached',
        },
        {
            'item': 'language_detector_init',
            'before': 'first doc cold-start in fast pool',
            'after': 'init_fast_merge_worker warmup',
            'status': 'moved_to_init',
        },
        {
            'item': 'line_nav_ratio',
            'before': 're-extract via document_gate_raw',
            'after': 'raw.nav_line_ratio reuse',
            'status': 'eliminated',
        },
        {
            'item': 'raw_features IPC',
            'before': 'pickled in survivor_payload',
            'after': 'restored in survivor_payload for PCI reuse in heavy worker',
            'status': 'restored',
        },
        {
            'item': 'stage_order',
            'before': 'size → lang → dedup → structural',
            'after': 'size → lang → dedup → stage0 (dedup insert parity locked)',
            'status': 'parity_locked',
        },
        {
            'item': 'evaluate_document_gate',
            'before': 'heavy cleaning re-extracts document_gate_raw',
            'after': 'raw=ctx.raw_features via gate_raw document context',
            'status': 'deduplicated',
        },
    ]


def _ipc_profile(work_dir: Path) -> dict[str, Any]:
    events = load_events(work_dir)
    survivors = sum(1 for ev in events if ev.get('event') == 'heavy_enter')
    dup = _duplicate_execution(events)
    return {
        'survivor_dispatches': survivors,
        'duplicate_execution_docs': dup.get('duplicate_doc_count', 0),
        'raw_features_in_payload': True,
        'payload_fields': [
            'seq', 'src_name', 'line_no', 'text', 'meaningful_chars',
            'language_assessment', 'stage_trace', 'doc_tier', 'admission',
            'ingest_meta', 'doc_content_hash', 'row',
        ],
    }


def build_stage0_cost_report(work_dir: Path, *, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    work_dir = Path(work_dir)
    events = load_events(work_dir)
    stage0 = build_report(
        work_dir,
        scheduler=load_work_json(work_dir / 'pipeline_scheduler_report.json'),
        progress=load_work_json(work_dir / 'pipeline_progress.json'),
        stage_metrics=load_work_json(work_dir / 'stage_metrics.json'),
    )
    stage_metrics = load_work_json(work_dir / 'stage_metrics.json')
    live = load_work_json(work_dir / 'pipeline_live_metrics.json')
    cache_stats = live.get('cache') or stage_metrics.get('cache') or {}
    wall = stage0.get('wall_time_ms') or {}
    stage_rows = _stage0_stage_rows(stage_metrics)

    cost = {
        'cost_breakdown': {
            'stage0_total_ms': wall.get('stage0') or {},
            'heavy_total_ms': wall.get('heavy') or {},
            'apply_total_ms': wall.get('apply') or {},
            'per_stage': stage_rows,
        },
        'cache_efficiency': cache_stats,
        'duplicate_computation_audit': _duplicate_computation_audit(),
        'cold_start': _cold_start_from_events(events),
        'memory_allocation_profile': {
            'raw_features_ipc': 'restored_in_survivor_payload',
            'raw_extract_site': 'single_pass_at_s2_structural_filter',
            'dedup_hash': 'computed_once_at_s2_doc_dedup',
            'gate_raw_heavy': 'reused_via_document_execution_context',
        },
        'ipc_profile': _ipc_profile(work_dir),
        'scheduler_interaction': stage0.get('scheduler') or {},
    }
    if baseline:
        b_wall = (baseline.get('wall_time_ms') or {}).get('stage0') or {}
        a_wall = wall.get('stage0') or {}
        b_avg = float(b_wall.get('avg_ms') or 0.0)
        a_avg = float(a_wall.get('avg_ms') or 0.0)
        cost['before_after'] = {
            'baseline_stage0_avg_ms': b_avg,
            'optimized_stage0_avg_ms': a_avg,
            'delta_avg_ms': round(a_avg - b_avg, 2),
            'delta_pct': round(100.0 * (a_avg - b_avg) / max(b_avg, 1e-9), 2),
        }
    return cost
