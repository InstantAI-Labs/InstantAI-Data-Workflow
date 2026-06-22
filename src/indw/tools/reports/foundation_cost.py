from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from indw.extract.structure.document import STRUCTURE_OWNERS
from indw.tools.reports.heavy_cost import build_heavy_pipeline_cost_report
from indw.tools.reports.stage0_cost import build_stage0_cost_report


from indw.schedule.monitor.audit import load_work_json


def _ownership_graph() -> list[dict[str, str]]:
    return [
        {'operation': 'structure_recovery', 'owner': 'document_structure.recover_document_structure', 'scope': 'document'},
        {'operation': 'aggregation_refine', 'owner': 'document_structure.recover_document_structure', 'scope': 'document'},
        {'operation': 'topic_expand', 'owner': 'document_structure.recover_document_structure', 'scope': 'document'},
        {'operation': 'section_semantic_stack', 'owner': 'section_scratch.build_section_analysis', 'scope': 'section'},
        {'operation': 'publication_scaffold_strip', 'owner': 'publication_role.cached_scaffold', 'scope': 'text_slice'},
        {'operation': 'completion_analysis', 'owner': 'doc_context.completion + semantic_boundary', 'scope': 'text'},
        {'operation': 'pci_fingerprint', 'owner': 'pci.build_fingerprint_bundle_detail', 'scope': 'document'},
        {'operation': 'survivor_text_ipc', 'owner': 'survivor_store.externalize_survivor_text', 'scope': 'large_doc'},
        {'operation': 'apply_commit', 'owner': 'parallel._apply_loop', 'scope': 'ordered_seq'},
        {'operation': 'heavy_dispatch', 'owner': 'concurrent._submit_heavy_lanes', 'scope': 'scheduler'},
    ]


def _apply_analysis(sched: dict[str, Any]) -> dict[str, Any]:
    return {
        'ordering_wait_ms': sched.get('ordering_wait_ms', 0),
        'peak_ordering_gap': sched.get('peak_ordering_gap', 0),
        'apply_stall_ms': sched.get('ordering_wait_ms', 0),
        'head_blocked_dispatches': sched.get('head_blocked_dispatches', 0),
        'head_priority_dispatches': sched.get('head_priority_dispatches', 0),
        'dispatch_denied': sched.get('dispatch_denied') or {},
        'apply_buffer_decoupled': True,
        'heavy_ooo_dispatch_limit': 3,
    }


def _large_document_audit(events: list[dict[str, Any]]) -> dict[str, Any]:
    heavy = [e for e in events if e.get('event') == 'heavy_enter']
    chars = [int(e.get('chars') or 0) for e in heavy]
    large = sum(1 for c in chars if c >= 30_000)
    huge = sum(1 for c in chars if c >= 80_000)
    return {
        'heavy_enter_count': len(heavy),
        'large_lane_docs': large,
        'huge_lane_docs': huge,
        'avg_heavy_chars': round(statistics.mean(chars), 1) if chars else 0,
        'text_externalization': 'survivor_store for docs >= LARGE_SURVIVOR_CHARS',
    }


def build_foundation_pipeline_report(
    work_dir: Path,
    *,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    from indw.filter.stage0.audit import load_events

    events = load_events(work_dir)
    sched = load_work_json(work_dir / 'pipeline_scheduler_report.json')
    stage0_cost = build_stage0_cost_report(work_dir, baseline=baseline)
    heavy_cost = build_heavy_pipeline_cost_report(work_dir, baseline=baseline)

    duplicate = list(stage0_cost.get('duplicate_computation_audit') or [])
    duplicate.extend(heavy_cost.get('duplicate_computation_audit') or [])
    duplicate.extend([
        {
            'item': 'document_structure_passes',
            'before': 'recover_structure + refine + expand per KE entry',
            'after': 'single recover_document_structure with doc_context cache',
            'status': 'consolidated',
        },
        {
            'item': 'scaffold_probe_slices',
            'before': 'repeated pub/evidence/nav per probe index',
            'after': 'left/right slice caches in trailing scaffold probe',
            'status': 'cached',
        },
        {
            'item': 'large_text_ipc',
            'before': 'full text in every survivor pickle',
            'after': 'hash-keyed survivor_store for large docs',
            'status': 'externalized',
        },
        {
            'item': 'fast_pool_fallback',
            'before': 'immediate inline process_merge_batch',
            'after': 'one fast-pool retry before inline fallback',
            'status': 'hardened',
        },
        {
            'item': 'apply_head_of_line',
            'before': 'Event spin + unbounded far-ahead heavy dispatch',
            'after': 'Condition wait + HEAVY_OOO_DISPATCH_LIMIT gating',
            'status': 'redesigned',
        },
    ])

    wall = load_work_json(work_dir / 'stage0_audit_report.json').get('wall_time_ms') or {}
    if not wall:
        wall = heavy_cost.get('cost_breakdown', {}).get('stage0_total_ms') or {}

    ranked = []
    for layer, key in (('stage0', 'stage0'), ('heavy', 'heavy'), ('apply', 'apply')):
        row = wall.get(key) if isinstance(wall, dict) else {}
        if isinstance(row, dict) and row.get('total_ms'):
            ranked.append({'layer': key, 'wall_ms': row.get('total_ms'), 'avg_ms': row.get('avg_ms')})
    ranked.sort(key=lambda r: -float(r.get('wall_ms') or 0))

    report = {
        'bottleneck_tree_ranked': ranked,
        'ownership_graph': _ownership_graph(),
        'structure_owners': list(STRUCTURE_OWNERS),
        'duplicate_computation_audit': duplicate,
        'stage0_cost': stage0_cost,
        'heavy_cost': heavy_cost,
        'apply_analysis': _apply_analysis(sched),
        'scheduler_analysis': heavy_cost.get('scheduler_analysis') or {},
        'large_document_audit': _large_document_audit(events),
        'ipc_audit': {
            **(stage0_cost.get('ipc_profile') or {}),
            'survivor_text_store': '.survivor_store/{doc_content_hash}.txt',
            'large_doc_threshold_chars': 30_000,
        },
        'memory_audit': {
            **(heavy_cost.get('memory_audit') or {}),
            'document_structure_cache': 'doc_context._document_structure',
        },
        'cache_efficiency': heavy_cost.get('cache_efficiency') or stage0_cost.get('cache_efficiency') or {},
        'worker_utilization': heavy_cost.get('worker_utilization') or {},
    }
    if baseline:
        ba = {}
        for key in ('stage0', 'heavy', 'apply'):
            b = (baseline.get('wall_time_ms') or {}).get(key) or {}
            a = (load_work_json(work_dir / 'stage0_audit_report.json').get('wall_time_ms') or {}).get(key) or {}
            if b.get('avg_ms') or a.get('avg_ms'):
                ba[key] = {
                    'baseline_avg_ms': b.get('avg_ms'),
                    'optimized_avg_ms': a.get('avg_ms'),
                    'delta_pct': round(
                        100.0 * (float(a.get('avg_ms') or 0) - float(b.get('avg_ms') or 0))
                        / max(float(b.get('avg_ms') or 1), 1e-9),
                        2,
                    ),
                }
        report['before_after'] = ba
    return report
