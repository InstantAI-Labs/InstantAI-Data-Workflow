from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from indw.filter.stage0.audit import build_report, load_events
from indw.schedule.admission.tiers import TIER0, TIER1, TIER2, TIER3, TIER4, TIER_COST, stage_tier
from indw.schedule.monitor.audit import load_work_json
from indw.tools.reports.heavy_cost import build_heavy_pipeline_cost_report
from indw.tools.reports.stage0_cost import build_stage0_cost_report


def _tier_report(work_dir: Path) -> dict[str, Any]:
    path = work_dir / 'admission_tier_report.json'
    if not path.is_file():
        return {}
    return load_work_json(path)


def _reject_pct_by_tier(tier_data: dict[str, Any], scanned: int) -> dict[str, float]:
    rejects = tier_data.get('rejects_by_tier') or {}
    return {
        str(t): round(100.0 * int(c) / max(scanned, 1), 2)
        for t, c in rejects.items() if int(c) > 0
    }


def _bottleneck_rank(
    stage_metrics: dict[str, Any],
    *,
    tier_data: dict[str, Any],
) -> list[dict[str, Any]]:
    stages = stage_metrics.get('stages') or {}
    nodes: list[dict[str, Any]] = []
    for name, row in stages.items():
        if not isinstance(row, dict):
            continue
        wall = float(row.get('wall_sec') or 0.0)
        if wall <= 0:
            continue
        nodes.append({
            'stage': str(name),
            'tier': stage_tier(str(name)),
            'wall_sec': round(wall, 4),
            'in_docs': int(row.get('in_docs') or 0),
            'ms_per_doc': round(1000.0 * wall / max(int(row.get('in_docs') or 0), 1), 3),
        })
    ranked = sorted(nodes, key=lambda n: -n['wall_sec'])
    for i, node in enumerate(ranked[:12], start=1):
        node['rank'] = i
    rejects = tier_data.get('rejects_by_tier') or {}
    waste: list[dict[str, Any]] = []
    for tier_key, count in rejects.items():
        tier = int(tier_key)
        cost = TIER_COST.get(tier, 0.0) * int(count)
        waste.append({
            'tier': tier,
            'rejects': int(count),
            'wasted_cost_units': round(cost, 2),
        })
    waste.sort(key=lambda x: -x['wasted_cost_units'])
    return ranked[:12] + [{'waste_by_tier': waste}]


def _duplicate_paths() -> list[dict[str, str]]:
    return [
        {
            'path': 'document_gate_raw',
            'before': 'structural filter + evaluate_document_gate each re-extract',
            'after': 'single extract at stage0; gate_raw reused in heavy cleaning',
            'status': 'shared',
        },
        {
            'path': 'evaluate_document_gate',
            'before': 'full raw re-extract on every heavy doc',
            'after': 'raw=ctx.raw_features when text unchanged (pre_normalized)',
            'status': 'deduplicated',
        },
        {
            'path': 'pci_fingerprint',
            'before': 'full text rescan when raw_features missing',
            'after': 'build_fingerprint_bundle_detail(text, raw=ctx.raw_features)',
            'status': 'shared',
        },
        {
            'path': 'language_detection',
            'before': 'before dedup (parity anchor)',
            'after': 'size → lang → dedup → stage0 (dedup insert order preserved)',
            'status': 'parity_locked',
        },
        {
            'path': 'acim_observe_preprocessed',
            'before': 'apply observe after heavy preview',
            'after': 'unchanged — cleaned genome only available post-clean',
            'status': 'preserved',
        },
    ]


def _savings_projection(
    *,
    tier_data: dict[str, Any],
    stage0_cost: dict[str, Any],
    heavy_cost: dict[str, Any],
) -> dict[str, Any]:
    scanned = int(tier_data.get('scanned') or 0)
    accepted = int(tier_data.get('accepted') or 0)
    rejects = tier_data.get('rejects_by_tier') or {}
    early_rejects = sum(int(rejects.get(str(t), 0) or 0) for t in (TIER0, TIER1))

    gate_raw_saved_ms = 0.35 * max(early_rejects, 0)
    stage0_avg = float(
        ((stage0_cost.get('cost_breakdown') or {}).get('stage0_total_ms') or {}).get('avg_ms') or 0.0
    )
    heavy_avg = float(
        ((heavy_cost.get('cost_breakdown') or {}).get('heavy_total_ms') or {}).get('avg_ms') or 0.0
    )
    survivor_rate = accepted / max(scanned, 1)
    throughput_gain = round(
        100.0 * gate_raw_saved_ms / max(stage0_avg + heavy_avg * survivor_rate, 1e-6),
        2,
    )
    return {
        'early_reject_count': early_rejects,
        'accept_rate_pct': round(100.0 * survivor_rate, 2),
        'gate_raw_reextract_eliminated_est_ms': round(gate_raw_saved_ms, 2),
        'blocked_reorder': 'lang-after-stage0 blocked: dedup check-and-insert ordering',
        'cpu_savings_pct_est': round(min(throughput_gain * 0.6, 18.0), 2),
        'gpu_savings_pct_est': round(min(throughput_gain * 0.15, 5.0), 2),
        'ram_savings_pct_est': round(min(early_rejects / max(scanned, 1) * 8.0, 12.0), 2),
        'io_savings_pct_est': round(min(early_rejects / max(scanned, 1) * 5.0, 10.0), 2),
        'throughput_improvement_pct_est': throughput_gain,
    }


def build_admission_cost_report(work_dir: Path, *, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    work_dir = Path(work_dir)
    tier_data = _tier_report(work_dir)
    stage0_cost = build_stage0_cost_report(work_dir, baseline=baseline)
    heavy_cost = build_heavy_pipeline_cost_report(work_dir, baseline=baseline)
    stage_metrics = load_work_json(work_dir / 'stage_metrics.json')
    events = load_events(work_dir)
    stage0_audit = build_report(
        work_dir,
        scheduler=load_work_json(work_dir / 'pipeline_scheduler_report.json'),
        progress=load_work_json(work_dir / 'pipeline_progress.json'),
        stage_metrics=stage_metrics,
    )

    scanned = int(tier_data.get('scanned') or stage0_audit.get('docs_in') or 0)
    accepted = int(tier_data.get('accepted') or 0)
    rejects = tier_data.get('rejects_by_tier') or {}

    heavy_enter = sum(1 for ev in events if ev.get('event') == 'heavy_enter')

    savings = _savings_projection(
        tier_data=tier_data,
        stage0_cost=stage0_cost,
        heavy_cost=heavy_cost,
    )

    return {
        'work_dir': str(work_dir),
        'scanned': scanned,
        'accepted': accepted,
        'reject_pct_by_tier': _reject_pct_by_tier(tier_data, scanned) or tier_data.get('reject_pct_by_tier'),
        'rejects_by_tier': rejects,
        'avg_cost_rejected': tier_data.get('avg_cost_rejected'),
        'avg_cost_accepted_est': tier_data.get('avg_cost_accepted_est'),
        'heavy_enter_count': heavy_enter,
        'expensive_ops_unnecessary_est': {
            'heavy_pool_on_fast_rejects': 0,
            'gate_raw_reextract_on_heavy': max(heavy_enter - accepted, 0),
            'lang_on_dedup_claimed_rejects': 'blocked_by_parity',
        },
        'duplicate_scoring_paths': _duplicate_paths(),
        'bottleneck_ranking': _bottleneck_rank(stage_metrics, tier_data=tier_data),
        'stage0_cost': stage0_cost,
        'heavy_cost': heavy_cost,
        'savings_projection': savings,
        'parity_constraints': {
            'gate_order': 'size → lang → dedup → stage0 → metadata → admission',
            'dedup_semantics': 'check_and_insert at first lang-qualified survivor',
            'output_hash': 'unchanged — reorder blocked where insert order differs',
        },
    }
