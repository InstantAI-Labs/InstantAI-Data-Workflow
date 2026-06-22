from __future__ import annotations

import platform
import statistics
from pathlib import Path
from typing import Any

from indw.schedule.config.hardware import SystemHardwareProfile
from indw.schedule.config.tune import (
    MergeTuneProfile,
    ProductionRunProfile,
    merge_tune_env_exports,
    recommend_production_profiles,
    resolve_merge_tune,
)
from indw.tools.reports.pipeline_tune_report import build_pipeline_tuning_report


def _scaling_efficiency(scaling: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not scaling:
        return []
    baseline = float(scaling[0].get('docs_per_sec') or 0)
    out: list[dict[str, Any]] = []
    for row in scaling:
        dps = float(row.get('docs_per_sec') or 0)
        workers = int(row.get('workers') or 1)
        eff = dps / max(baseline, 1e-9)
        out.append({
            **row,
            'efficiency_vs_1_worker': round(eff, 3),
            'parallel_efficiency_pct': round(100.0 * eff / max(workers, 1), 1),
        })
    return out


def _rank_bottlenecks(tuning: dict[str, Any], scaling: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = list(tuning.get('bottleneck_ranking') or [])
    if scaling:
        best = max(scaling, key=lambda r: float(r.get('docs_per_sec') or 0))
        ranked.append({
            'layer': 'scheduler_ordering',
            'ms': float(best.get('ordering_wait_ms') or 0),
            'pct': float(best.get('ordering_wait_pct_of_wall') or 0),
            'kind': 'deterministic',
        })
    impl = [r for r in ranked if r.get('layer') not in ('ordering_wait', 'scheduler_ordering')]
    det = [r for r in ranked if r.get('layer') in ('ordering_wait', 'scheduler_ordering')]
    impl.sort(key=lambda r: -float(r.get('ms') or 0))
    det.sort(key=lambda r: -float(r.get('ms') or 0))
    return [
        {**r, 'category': 'implementation' if r not in det else 'deterministic_ordering'}
        for r in impl + det
    ]


def build_hardware_recommendations(
    hw: SystemHardwareProfile,
    scaling: list[dict[str, Any]],
) -> dict[str, Any]:
    profiles = recommend_production_profiles(hw)
    prod = profiles['production']
    tune = resolve_merge_tune(workers=prod.workers, chunk_size=prod.chunk_size, hw=hw)
    lane = tune.to_dict()
    recs: list[str] = []
    if hw.storage.storage_class in ('hdd', 'slow'):
        recs.append('Use NVMe or fast SSD for work_dir; HDD limits fast-worker IPC throughput.')
    if hw.mem_budget_mb < 8192:
        recs.append('Keep merge workers <= 4 on <8GB mem budget to avoid RSS pressure trim churn.')
    if hw.cpu_logical >= 8 and scaling:
        peak_eff = max(float(r.get('parallel_efficiency_pct') or 0) for r in scaling)
        if peak_eff < 45:
            recs.append(
                'Heavy semantic lane dominates; scale workers until ordering_wait_pct plateaus, not CPU count.',
            )
    if not recs:
        recs.append('Current hardware supports tuned defaults; no override required beyond env exports.')
    return {
        'hardware': hw.to_dict(),
        'recommended_workers': prod.workers,
        'recommended_chunk_size': prod.chunk_size,
        'lane_routing': {
            'large_chars': lane.get('lane_routing_large_chars'),
            'huge_chars': lane.get('lane_routing_huge_chars'),
        },
        'profiles': {k: v.to_dict() for k, v in profiles.items()},
        'recommendations': recs,
    }


def build_inherent_limitations(
    tuning: dict[str, Any],
    scaling: list[dict[str, Any]],
) -> dict[str, Any]:
    ops = tuning.get('operational_metrics') or {}
    sched = tuning.get('scheduler_efficiency') or {}
    gap = tuning.get('ordering_gap_analysis') or {}
    heavy_pct = 0.0
    for row in tuning.get('bottleneck_ranking') or []:
        if row.get('layer') == 'heavy':
            heavy_pct = float(row.get('pct') or 0)
            break
    ordering_pct = float(ops.get('ordering_wait_pct_of_wall') or 0)
    peak_gap = int(gap.get('peak_gap') or sched.get('peak_ordering_gap') or 0)
    ooo = int(gap.get('ooo_dispatch_limit') or 0)
    scaling_plateau = False
    if len(scaling) >= 2:
        effs = [float(r.get('efficiency_vs_1_worker') or 0) for r in scaling]
        if len(effs) >= 2 and effs[-1] - effs[-2] < 0.08:
            scaling_plateau = True
    return {
        'frozen_quality_components': [
            'stage0_acceptance',
            'semantic_evidence',
            'knowledge_extraction',
            'publication_recovery',
            'acim_pci_lci',
            'deduplication',
            'deterministic_apply_order',
        ],
        'unavoidable_costs': [
            {
                'name': 'heavy_semantic_processing',
                'pct_of_stage_wall': heavy_pct,
                'note': 'Dominates per-doc wall; cannot reduce without weakening semantic gates.',
            },
            {
                'name': 'deterministic_ordering_wait',
                'pct_of_wall': ordering_pct,
                'peak_gap': peak_gap,
                'ooo_limit': ooo,
                'note': 'Apply loop waits for head sequence; gap up to ooo_limit is expected under parallel heavy.',
            },
            {
                'name': 'serial_apply_commit',
                'note': 'Accepted rows commit in sequence; apply ms/doc stays low but cannot parallelize ordering.',
            },
        ],
        'implementation_bottlenecks_addressed': [
            'stage0_reordered_filters_and_dedup',
            'shared_section_analysis_for_heavy',
            'survivor_ipc_externalization',
            'adaptive_apply_wait_and_scheduler_buffers',
            'hardware_adaptive_merge_tune',
        ],
        'scaling_plateau_at_high_workers': scaling_plateau,
    }


def _shell_exports(env: dict[str, str]) -> str:
    lines = []
    for key, val in sorted(env.items()):
        lines.append(f'export {key}={val}')
    return '\n'.join(lines)


def build_production_ready_commands(
    *,
    repo_root: Path,
    hw: SystemHardwareProfile,
    production_profile: ProductionRunProfile,
    tune: MergeTuneProfile,
) -> dict[str, Any]:
    root = Path(repo_root)
    mr = root.as_posix()
    env = merge_tune_env_exports(tune)
    env_block = _shell_exports({
        'INSTANT_QUALITY_CONFIG': 'configs/filtering/quality_fast_first.yaml',
        'INSTANT_SKIP_METRICS_PROBE': '1',
        **env,
    })
    common = f'cd {mr}'
    val_stage0 = 'indw audit --kind stage0 --workers 2'
    val_tune = 'python scripts/pipeline_tuning_audit.py --workers 2 --out reports/pipeline_tuning_report.json'
    val_scale = 'indw audit --kind production'
    val_parity = 'indw validate'
    prepare = (
        'indw merge {raw} {out} '
        '--work-dir {work_dir} '
        '--workers {workers} '
        '--chunk-size {chunk} '
        '{extra}'
    )
    profiles = recommend_production_profiles(hw)
    sections: dict[str, Any] = {}
    for name, prof in profiles.items():
        prof_tune = resolve_merge_tune(workers=prof.workers, chunk_size=prof.chunk_size, hw=hw)
        prof_env = merge_tune_env_exports(prof_tune)
        extra = '--skip-download' if name != 'production' else ''
        if name == 'production':
            extra = '--fresh-merge'
        cmd = prepare.format(
            work_dir=f'artifacts/data_{name}',
            workers=prof.workers,
            chunk=prof.chunk_size,
            extra=extra,
        )
        sections[name] = {
            'workers': prof.workers,
            'chunk_size': prof.chunk_size,
            'purpose': prof.purpose,
            'env': prof_env,
            'merge_command': cmd,
        }
    script = '\n\n'.join([
        '# INDW production validation',
        '',
        common,
        env_block,
        'mkdir -p reports',
        '',
        '# --- Validation ---',
        val_stage0,
        val_tune,
        val_scale,
        val_parity,
        '',
        '# --- Merge profiles ---',
        sections['small']['merge_command'],
        '',
        sections['medium']['merge_command'],
        '',
        sections['production']['merge_command'],
    ])
    return {
        'platform': platform.system(),
        'repo_root': mr,
        'tuned_env': env,
        'profiles': sections,
        'validation_commands': {
            'stage0_verify': val_stage0,
            'pipeline_tuning_audit': val_tune,
            'production_scale_audit': val_scale,
            'parity_tests': val_parity,
        },
        'copy_paste_script': script,
    }


def build_production_scale_report(
    work_dir: Path,
    *,
    hw: SystemHardwareProfile,
    scaling_runs: list[dict[str, Any]],
    reference_hash: str,
    production_workers: int,
    production_chunk_size: int,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    tune = resolve_merge_tune(
        workers=production_workers, chunk_size=production_chunk_size, hw=hw,
    )
    profiles = recommend_production_profiles(hw)
    prod_profile = profiles['production']
    parity = {
        'reference_hash': reference_hash,
        'all_scaling_hash_match': all(bool(r.get('hash_match')) for r in scaling_runs),
        'scaling': [
            {
                'workers': r.get('workers'),
                'hash_match': r.get('hash_match'),
                'output_hash': r.get('output_hash'),
            }
            for r in scaling_runs
        ],
    }
    best_run = scaling_runs[-1] if scaling_runs else {}
    tuning = build_pipeline_tuning_report(
        work_dir,
        workers=production_workers,
        chunk_size=production_chunk_size,
        tune=tune,
        parity=parity,
        wall_sec=float(best_run.get('wall_sec') or 0),
    )
    scaling_curve = _scaling_efficiency(scaling_runs)
    hw_rec = build_hardware_recommendations(hw, scaling_curve)
    limitations = build_inherent_limitations(tuning, scaling_curve)
    repo_root = Path(__file__).resolve().parents[4]
    commands = build_production_ready_commands(
        repo_root=repo_root,
        hw=hw,
        production_profile=prod_profile,
        tune=tune,
    )
    dps_vals = [float(r.get('docs_per_sec') or 0) for r in scaling_curve if r.get('docs_per_sec')]
    return {
        'audit': 'production_scale_final',
        'production_ready': parity['all_scaling_hash_match'] and bool(reference_hash),
        'hardware': hw.to_dict(),
        'recommended_tune': tune.to_dict(),
        'scaling_curve': scaling_curve,
        'scaling_summary': {
            'workers_tested': [int(r.get('workers') or 0) for r in scaling_curve],
            'peak_docs_per_sec': round(max(dps_vals), 3) if dps_vals else 0,
            'median_docs_per_sec': round(statistics.median(dps_vals), 3) if dps_vals else 0,
        },
        'throughput_scaling': {
            'curve': scaling_curve,
            'plateau_hint': limitations.get('scaling_plateau_at_high_workers'),
        },
        'scheduler_efficiency': tuning.get('scheduler_efficiency'),
        'queue_efficiency': tuning.get('queue_efficiency'),
        'worker_utilization': tuning.get('worker_utilization'),
        'cache_efficiency': tuning.get('cache_efficiency'),
        'ipc_audit': tuning.get('ipc_audit'),
        'memory_audit': tuning.get('memory_audit'),
        'ordering_gap_analysis': tuning.get('ordering_gap_analysis'),
        'large_document_behavior': {
            'ipc_externalize_chars': tune.ipc_externalize_chars,
            'lane_routing_large_chars': tune.to_dict().get('lane_routing_large_chars'),
            'lane_routing_huge_chars': tune.to_dict().get('lane_routing_huge_chars'),
            'survivor_store': (tuning.get('ipc_audit') or {}).get('externalized_count'),
        },
        'bottleneck_ranking': _rank_bottlenecks(tuning, scaling_curve),
        'hardware_recommendations': hw_rec,
        'inherent_limitations': limitations,
        'operational_metrics': tuning.get('operational_metrics'),
        'parity': parity,
        'production_ready_commands': commands,
    }


def human_production_summary(report: dict[str, Any]) -> str:
    lines = [
        'Production Scale Audit',
        f"  production_ready={report.get('production_ready')}",
    ]
    parity = report.get('parity') or {}
    lines.append(f"  scaling_hash_match={parity.get('all_scaling_hash_match')}")
    summary = report.get('scaling_summary') or {}
    lines.append(f"  peak_docs_per_sec={summary.get('peak_docs_per_sec')}")
    curve = report.get('scaling_curve') or []
    if curve:
        lines.append('  scaling_curve:')
        for row in curve:
            lines.append(
                f"    w={row.get('workers')} dps={row.get('docs_per_sec')} "
                f"eff={row.get('efficiency_vs_1_worker')} "
                f"hash_match={row.get('hash_match')}"
            )
    ranked = report.get('bottleneck_ranking') or []
    if ranked:
        lines.append('  bottlenecks:')
        for row in ranked[:5]:
            cat = row.get('category', '')
            lines.append(
                f"    [{cat}] {row.get('layer')} {row.get('ms')}ms ({row.get('pct')}%)"
            )
    hw = report.get('hardware_recommendations') or {}
    prof = (hw.get('profiles') or {}).get('production') or {}
    lines.append(
        f"  recommended_production: workers={prof.get('workers')} chunk={prof.get('chunk_size')}"
    )
    return '\n'.join(lines)
