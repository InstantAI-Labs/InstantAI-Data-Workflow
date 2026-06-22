from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from indw.schedule.config.tune import MergeTuneProfile, resolve_merge_tune
from indw.tools.reports.foundation_cost import build_foundation_pipeline_report
from indw.filter.stage0.verify import build_production_verification_report


from indw.schedule.monitor.audit import load_work_json


def _baseline_profile() -> MergeTuneProfile:
    from indw.schedule.config.tune import _DEFAULT
    return _DEFAULT


def _worker_metrics(sched: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    sched_live = live.get('scheduler') or {}
    fast_sub = int(sched.get('fast_submits') or 0)
    heavy_sub = int(sched.get('heavy_submits') or 0)
    elapsed = float(sched.get('elapsed_sec') or 1)
    return {
        'fast_submits': fast_sub,
        'heavy_submits': heavy_sub,
        'heavy_to_fast_ratio': round(heavy_sub / max(fast_sub, 1), 4),
        'worker_util_pct': sched_live.get('worker_util_pct', 0),
        'fast_submits_per_sec': round(fast_sub / max(elapsed, 1e-6), 3),
        'heavy_submits_per_sec': round(heavy_sub / max(elapsed, 1e-6), 3),
    }


def _queue_efficiency(sched: dict[str, Any]) -> dict[str, Any]:
    denied = sched.get('dispatch_denied') or {}
    denied_total = sum(int(v) for v in denied.values())
    return {
        'peak_survivor_buffer': int(sched.get('peak_survivor_buffer') or 0),
        'peak_heavy_pending': int(sched.get('peak_heavy_pending') or 0),
        'peak_ordering_gap': int(sched.get('peak_ordering_gap') or 0),
        'dispatch_denied_total': denied_total,
        'dispatch_denied': denied,
        'head_blocked_dispatches': int(sched.get('head_blocked_dispatches') or 0),
        'head_priority_dispatches': int(sched.get('head_priority_dispatches') or 0),
    }


def build_pipeline_tuning_report(
    work_dir: Path,
    *,
    workers: int,
    chunk_size: int,
    tune: MergeTuneProfile,
    parity: dict[str, Any] | None = None,
    wall_sec: float = 0.0,
    baseline_tune: MergeTuneProfile | None = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    sched = load_work_json(work_dir / 'pipeline_scheduler_report.json')
    live = load_work_json(work_dir / 'pipeline_live_metrics.json')
    stage0_audit = load_work_json(work_dir / 'stage0_audit_report.json')
    foundation = build_foundation_pipeline_report(work_dir)
    verify = build_production_verification_report(work_dir, parity=parity or {})

    wall = stage0_audit.get('wall_time_ms') or {}
    stage0_ms = float((wall.get('stage0') or {}).get('total_ms') or 0)
    heavy_ms = float((wall.get('heavy') or {}).get('total_ms') or 0)
    apply_ms = float((wall.get('apply') or {}).get('total_ms') or 0)
    total_ms = stage0_ms + heavy_ms + apply_ms

    baseline = baseline_tune or _baseline_profile()
    ordering_wait = float(sched.get('ordering_wait_ms') or 0)

    ranked = sorted(
        [
            {'layer': 'heavy', 'ms': heavy_ms, 'pct': round(100 * heavy_ms / max(total_ms, 1), 2)},
            {'layer': 'stage0', 'ms': stage0_ms, 'pct': round(100 * stage0_ms / max(total_ms, 1), 2)},
            {'layer': 'apply', 'ms': apply_ms, 'pct': round(100 * apply_ms / max(total_ms, 1), 2)},
            {'layer': 'ordering_wait', 'ms': ordering_wait, 'pct': round(100 * ordering_wait / max(wall_sec * 1000, 1), 2)},
        ],
        key=lambda r: -r['ms'],
    )

    docs = int((verify.get('waterfall') or {}).get('input') or 0)
    dps = docs / max(wall_sec, 1e-6)

    expected = {
        'throughput_docs_per_sec_delta_pct': round(
            100.0 * (tune.result_buffer_factor - baseline.result_buffer_factor)
            / max(baseline.result_buffer_factor, 1),
            1,
        ),
        'apply_stall_reduction': 'adaptive wait + ooo_limit=3 balances gap vs dispatch',
        'ipc_savings_large_docs': f"externalize >= {tune.ipc_externalize_chars} chars",
        'memory_pressure': 'cache trim unchanged; larger apply buffer reduces backpressure churn',
    }

    return {
        'tuning_pass': 'production_operational',
        'work_dir': str(work_dir),
        'workload': {'workers': workers, 'chunk_size': chunk_size, 'docs': docs, 'wall_sec': round(wall_sec, 3)},
        'parity': parity or {},
        'current_settings': baseline.to_dict(),
        'recommended_settings': tune.to_dict(),
        'settings_delta': {
            k: {'before': baseline.to_dict()[k], 'after': tune.to_dict()[k]}
            for k in baseline.to_dict()
            if baseline.to_dict()[k] != tune.to_dict()[k]
        },
        'bottleneck_ranking': ranked,
        'operational_metrics': {
            'docs_per_sec': round(dps, 3),
            'stage0_ms_per_doc': round(stage0_ms / max(docs, 1), 2),
            'heavy_ms_per_doc': round(heavy_ms / max(docs, 1), 2),
            'apply_ms_per_doc': round(apply_ms / max(docs, 1), 2),
            'ordering_wait_ms': ordering_wait,
            'ordering_wait_pct_of_wall': round(100 * ordering_wait / max(wall_sec * 1000, 1), 2),
        },
        'worker_utilization': _worker_metrics(sched, live),
        'queue_efficiency': _queue_efficiency(sched),
        'scheduler_efficiency': {
            'ordering_wait_ms': ordering_wait,
            'peak_ordering_gap': int(sched.get('peak_ordering_gap') or 0),
            'lane_heavy_submits': sched.get('lane_heavy_submits') or {},
            'lane_slots': sched.get('lane_slots') or {},
        },
        'cache_efficiency': foundation.get('cache_efficiency') or live.get('cache') or {},
        'memory_audit': foundation.get('memory_audit') or {},
        'ipc_audit': foundation.get('ipc_audit') or {},
        'ordering_gap_analysis': {
            'peak_gap': int(sched.get('peak_ordering_gap') or 0),
            'ordering_wait_ms': ordering_wait,
            'ooo_dispatch_limit': tune.heavy_ooo_dispatch_limit,
            'head_priority_dispatches': int(sched.get('head_priority_dispatches') or 0),
        },
        'expected_improvements': expected,
        'verification': {
            'hash_match': (parity or {}).get('hash_match'),
            'stage0_reject_rate': (verify.get('stage0_efficiency') or {}).get('reject_rate_pct'),
        },
    }
