from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from indw.schedule.monitor.audit import load_work_json
from indw.tools.reports.stabilization_audit import build_stabilization_audit


def _batch_lifecycle() -> dict[str, Any]:
    return {
        'reader': {
            'owner': 'parallel._reader_thread',
            'batch_target': 'stream_batch ramp -> fast_batch',
            'flush': 'batch_flush_sec time cap',
        },
        'fast_pool': {
            'owner': 'concurrent fast_pull -> process_fast_merge_batch',
            'unit': 'reader batch (one submit per reader batch)',
            'collect': 'batched ingest_line_results(terminal)',
        },
        'survivor_buffers': {
            'owner': 'LaneBuffers.route_many',
            'order': 'seq-sorted insert per lane',
            'pick': 'pick_lane_batch (normal multi, large/huge single)',
        },
        'heavy_pool': {
            'owner': 'process_heavy_merge_batch',
            'unit': 'pick_lane_batch chunk capped by heavy_batch',
            'ipc': 'externalize >= ipc_externalize_chars',
        },
        'apply': {
            'owner': 'parallel._apply_loop',
            'prep': '_prep_apply_line once (_merge_objects_ready)',
            'commit': 'strict next_write_seq',
        },
        'checkpoint': {
            'owner': 'BufferedJsonlWriter on_flush + checkpoint_interval',
        },
    }


_BATCH_REMOVED = [
    'per-terminal ingest_line_results([line]) loop -> single batched call',
    'pick_lane_batch full-buffer sort each pick -> sorted insert at route',
    'duplicate preprocessed_line_to_objects on apply ingress + apply',
    'duplicate resolve_survivor_text in validate + from_survivor_payload',
    'unused heavy_batch_cap binding -> wired into pick_lane_batch',
    'lane_min_seq O(n) scan -> O(1) head read on sorted buffer',
]


def build_batch_efficiency_audit(
    work_dir: Path,
    *,
    baseline_dir: Path | None = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    stab = build_stabilization_audit(work_dir, baseline_dir=baseline_dir)
    scheduler = load_work_json(work_dir / 'pipeline_scheduler_report.json')
    live = load_work_json(work_dir / 'pipeline_live_metrics.json')
    stage_metrics = load_work_json(work_dir / 'stage_metrics.json')
    alloc = load_work_json(work_dir / 'pipeline_live_metrics.json').get('allocation') or {}

    fast_sub = int(scheduler.get('fast_submits') or 0)
    fast_col = int(scheduler.get('fast_collects') or 0)
    heavy_sub = int(scheduler.get('heavy_submits') or 0)
    heavy_col = int(scheduler.get('heavy_collects') or 0)
    apply_done = int(scheduler.get('apply_completions') or 0)

    stages = stage_metrics.get('stages') or {}
    wall_rank = sorted(
        (
            {
                'stage': name,
                'wall_sec': float(row.get('wall_sec') or 0),
                'in_docs': int(row.get('in_docs') or 0),
                'docs_per_sec': round(
                    int(row.get('in_docs') or 0) / max(float(row.get('wall_sec') or 0), 1e-9), 3,
                ),
            }
            for name, row in stages.items()
            if isinstance(row, dict)
        ),
        key=lambda r: -r['wall_sec'],
    )

    avg_heavy_batch = round(heavy_col and (heavy_col / max(heavy_sub, 1)), 2)
    avg_fast_batch = round(fast_col and (fast_sub and fast_col / fast_sub or 0), 2)

    report: dict[str, Any] = {
        'work_dir': str(work_dir),
        'batch_lifecycle': _batch_lifecycle(),
        'batch_efficiency': {
            'fast_submits': fast_sub,
            'fast_collects': fast_col,
            'avg_fast_batches_per_collect': avg_fast_batch,
            'heavy_submits': heavy_sub,
            'heavy_collects': heavy_col,
            'avg_docs_per_heavy_submit': avg_heavy_batch,
            'apply_completions': apply_done,
            'heavy_to_apply_ratio': round(heavy_col / max(apply_done, 1), 3),
            'allocation': alloc,
        },
        'queue_health': stab.get('scheduler_backlog'),
        'scheduler_efficiency': stab.get('scheduler_analysis'),
        'worker_utilization': stab.get('worker_utilization'),
        'ipc_overhead': stab.get('ipc_audit'),
        'survivor_backlog': {
            'peak': stab.get('scheduler_backlog', {}).get('peak_survivor_buffer'),
            'cap_denials': stab.get('scheduler_backlog', {}).get('survivor_buffer_cap'),
        },
        'ordering_gaps': stab.get('ordering'),
        'bottleneck_ranking': wall_rank[:12],
        'duplicate_work_removed': _BATCH_REMOVED,
        'intentional_safety': stab.get('intentional_safety_checks'),
        'unavoidable_bottlenecks': stab.get('unavoidable_bottlenecks'),
        'validation': stab.get('validation'),
        'success_criteria': stab.get('success_criteria'),
    }

    if baseline_dir is not None:
        base = build_batch_efficiency_audit(Path(baseline_dir))
        bb = report['batch_efficiency']
        bbb = base.get('batch_efficiency') or {}
        report['before_after'] = {
            'baseline': str(baseline_dir),
            'peak_survivor_buffer': {
                'before': base.get('survivor_backlog', {}).get('peak'),
                'after': report['survivor_backlog']['peak'],
            },
            'avg_docs_per_heavy_submit': {
                'before': bbb.get('avg_docs_per_heavy_submit'),
                'after': bb.get('avg_docs_per_heavy_submit'),
            },
            'ordering_wait_ms_delta': (
                stab.get('scheduler_backlog', {}).get('ordering_wait_ms', 0)
                - base.get('queue_health', {}).get('ordering_wait_ms', 0)
            ),
        }

    return report


def publish_batch_efficiency_audit(work_dir: Path, report: dict[str, Any]) -> Path:
    work_dir = Path(work_dir)
    out_json = work_dir / 'batch_efficiency_audit.json'
    out_md = work_dir / 'batch_efficiency_audit.md'
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = [
        '# Batch Efficiency Audit',
        '',
        f"work_dir: {report.get('work_dir')}",
        '',
        '## Batch metrics',
    ]
    be = report.get('batch_efficiency') or {}
    for k in (
        'fast_submits', 'heavy_submits', 'avg_docs_per_heavy_submit',
        'apply_completions', 'heavy_to_apply_ratio',
    ):
        lines.append(f'- {k}: {be.get(k)}')
    lines.extend(['', '## Bottleneck ranking (wall)'])
    for row in report.get('bottleneck_ranking') or []:
        lines.append(
            f"- {row.get('stage')}: {row.get('wall_sec')}s "
            f"dps={row.get('docs_per_sec')}",
        )
    lines.extend(['', '## Removed duplicate batch paths'])
    for item in report.get('duplicate_work_removed') or []:
        lines.append(f'- {item}')
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out_json
