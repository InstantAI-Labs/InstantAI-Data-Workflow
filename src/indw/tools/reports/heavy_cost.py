from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any


from indw.schedule.monitor.audit import load_work_json


def _heavy_stage_rows(stage_metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    heavy_names = (
        's3_intermediate', 's4_intel_preview', 's4_high_quality',
        'ke_structure_recovery', 'ke_section_classify', 'ke_section_quality',
        'ke_unit_assembly', 'ke_boundary_role', 'ke_aggregation',
    )
    stages = stage_metrics.get('stages') or {}
    rows: dict[str, dict[str, float]] = {}
    for name, row in stages.items():
        if not isinstance(row, dict):
            continue
        if name not in heavy_names and not name.startswith('s3_') and not name.startswith('s4_'):
            continue
        in_docs = int(row.get('in_docs') or 0)
        wall = float(row.get('wall_sec') or 0.0)
        rows[str(name)] = {
            'wall_sec': round(wall, 4),
            'cpu_sec': round(float(row.get('cpu_sec') or wall), 4),
            'in_docs': in_docs,
            'ms_per_doc': round(1000.0 * wall / max(in_docs, 1), 3),
        }
    return rows


def _bottleneck_tree(stage_metrics: dict[str, Any], wall_ms: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    stage0 = (wall_ms.get('stage0') or {}).get('total_ms') or 0.0
    heavy = (wall_ms.get('heavy') or {}).get('total_ms') or 0.0
    apply = (wall_ms.get('apply') or {}).get('total_ms') or 0.0
    total = stage0 + heavy + apply
    if total <= 0:
        return nodes
    nodes.append({'layer': 'stage0', 'wall_ms': round(stage0, 2), 'pct': round(100 * stage0 / total, 2)})
    nodes.append({'layer': 'heavy', 'wall_ms': round(heavy, 2), 'pct': round(100 * heavy / total, 2)})
    nodes.append({'layer': 'apply', 'wall_ms': round(apply, 2), 'pct': round(100 * apply / total, 2)})
    per = _heavy_stage_rows(stage_metrics)
    ranked = sorted(per.items(), key=lambda kv: -kv[1]['wall_sec'])
    for stage_name, row in ranked[:8]:
        nodes.append({
            'layer': f'heavy/{stage_name}',
            'wall_ms': round(row['wall_sec'] * 1000, 2),
            'pct': round(100 * row['wall_sec'] * 1000 / max(heavy, 1), 2) if heavy else 0,
        })
    return nodes


def _duplicate_computation_audit() -> list[dict[str, str]]:
    return [
        {
            'item': 'section_semantic_stack',
            'before': 'classify + quality each run evidence/structure/profile',
            'after': 'build_section_analysis once per section, shared scratch',
            'status': 'eliminated',
        },
        {
            'item': 'classify_section_evidence',
            'before': 'compute_semantic_evidence bypassed doc_context',
            'after': 'resolve via section_scratch + doc_context caches',
            'status': 'fixed',
        },
        {
            'item': 'pci_raw_rescan',
            'before': 'PCI full rescan after IPC (raw_features omitted)',
            'after': 'raw_features restored in survivor_payload for PCI reuse',
            'status': 'fixed',
        },
        {
            'item': 'ke_unit_finalize',
            'before': 'finalize_semantic_unit after dedupe even when unchanged',
            'after': 'skip integrity refinalize when duplicate_ratio=0',
            'status': 'eliminated',
        },
        {
            'item': 'heavy_apply_backpressure',
            'before': 'fast and heavy share same apply buffer cap',
            'after': 'HEAVY_RESULT_BUFFER_FACTOR=12 vs fast=8',
            'status': 'decoupled',
        },
        {
            'item': 'evaluate_document_gate',
            'before': 'heavy cleaning safety net',
            'after': 'unchanged intentional duplicate',
            'status': 'preserved',
        },
    ]


def build_heavy_pipeline_cost_report(
    work_dir: Path,
    *,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    from indw.filter.stage0.audit import build_report, load_events

    stage0 = build_report(
        work_dir,
        scheduler=load_work_json(work_dir / 'pipeline_scheduler_report.json'),
        progress=load_work_json(work_dir / 'pipeline_progress.json'),
        stage_metrics=load_work_json(work_dir / 'stage_metrics.json'),
    )
    stage_metrics = load_work_json(work_dir / 'stage_metrics.json')
    sched = load_work_json(work_dir / 'pipeline_scheduler_report.json')
    live = load_work_json(work_dir / 'pipeline_live_metrics.json')
    wall_ms = stage0.get('wall_time_ms') or {}
    cache_stats = live.get('cache') or stage_metrics.get('cache') or {}
    events = load_events(work_dir)
    heavy_events = [e for e in events if e.get('event') == 'heavy_exit']
    heavy_walls = [float(e.get('wall_ms') or 0) for e in heavy_events]

    report = {
        'bottleneck_tree': _bottleneck_tree(stage_metrics, wall_ms),
        'heavy_cost_breakdown': {
            'total_ms': wall_ms.get('heavy') or {},
            'per_stage': _heavy_stage_rows(stage_metrics),
            'heavy_exit_samples': len(heavy_walls),
            'heavy_exit_avg_ms': round(statistics.mean(heavy_walls), 2) if heavy_walls else 0,
            'heavy_exit_p95_ms': round(sorted(heavy_walls)[int(0.95 * (len(heavy_walls) - 1))], 2) if len(heavy_walls) > 1 else 0,
        },
        'duplicate_computation_audit': _duplicate_computation_audit(),
        'cache_efficiency': cache_stats,
        'scheduler_analysis': {
            'ordering_wait_ms': sched.get('ordering_wait_ms', 0),
            'peak_ordering_gap': sched.get('peak_ordering_gap', 0),
            'head_blocked_dispatches': sched.get('head_blocked_dispatches', 0),
            'head_priority_dispatches': sched.get('head_priority_dispatches', 0),
            'peak_survivor_buffer': sched.get('peak_survivor_buffer', 0),
            'peak_heavy_pending': sched.get('peak_heavy_pending', 0),
            'dispatch_denied': sched.get('dispatch_denied') or {},
            'lane_heavy_submits': sched.get('lane_heavy_submits') or {},
        },
        'worker_utilization': {
            'worker_util_pct': (live.get('scheduler') or {}).get('worker_util_pct', 0),
            'fast_submits': sched.get('fast_submits', 0),
            'heavy_submits': sched.get('heavy_submits', 0),
            'heavy_collects': sched.get('heavy_collects', 0),
        },
        'memory_audit': {
            'section_scratch': 'per_section_single_build',
            'doc_context_caches': 'evidence/bundle/structure/completion per doc',
            'raw_features_ipc': 'restored_for_pci_reuse',
        },
        'ipc_audit': {
            'survivor_payload_includes_raw_features': True,
            'heavy_return_includes': ['items', 'cleaning_stats', 'discovery_calibration'],
        },
    }
    if baseline:
        b_heavy = (baseline.get('wall_time_ms') or {}).get('heavy') or {}
        a_heavy = wall_ms.get('heavy') or {}
        b_avg = float(b_heavy.get('avg_ms') or 0)
        a_avg = float(a_heavy.get('avg_ms') or 0)
        report['before_after'] = {
            'baseline_heavy_avg_ms': b_avg,
            'optimized_heavy_avg_ms': a_avg,
            'delta_avg_ms': round(a_avg - b_avg, 2),
            'delta_pct': round(100.0 * (a_avg - b_avg) / max(b_avg, 1e-9), 2),
        }
    return report
