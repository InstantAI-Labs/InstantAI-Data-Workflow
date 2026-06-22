from __future__ import annotations

from pathlib import Path
from typing import Any

from indw.schedule.architecture.classify import classification_summary, commodity_stages, intelligence_stages
from indw.schedule.architecture.graph import horizontal_graph_spec
from indw.schedule.architecture.ownership import ownership_graph
from indw.schedule.architecture.resources import resource_allocation_spec
from indw.schedule.monitor.audit import load_work_json
from indw.store.io.json_codec import backend_name
from indw.tools.reports.admission_cost import build_admission_cost_report
from indw.tools.reports.execution_consolidation import build_execution_consolidation_report
from indw.tools.reports.heavy_cost import build_heavy_pipeline_cost_report
from indw.tools.reports.library_migration import build_library_migration_report
from indw.tools.reports.stage0_cost import build_stage0_cost_report


def _duplicate_computation_audit() -> list[dict[str, str]]:
    return [
        {
            'item': 'fast_gate_execution',
            'before': 'engine _stage* + workers inline + graph pools',
            'after': 'tier01 owner + chain pools single path',
            'status': 'consolidated',
        },
        {
            'item': 'heavy_execution',
            'before': 'run_heavy_stages inline in workers',
            'after': 'process_heavy_chain_batch (pci→acim→clean)',
            'status': 'consolidated',
        },
        {
            'item': 'jsonl_parse',
            'before': 'parse_merge_jsonl_line duplicate logic',
            'after': 'data.store.io.jsonl.parse_jsonl_line',
            'status': 'consolidated',
        },
        {
            'item': 'document_gate_raw',
            'before': 're-extract per heavy doc',
            'after': 'gate_raw shared from stage0 survivor payload',
            'status': 'consolidated',
        },
        {
            'item': 'evaluate_document_gate',
            'before': 'full re-extract',
            'after': 'raw=ctx.raw_features when pre_normalized',
            'status': 'deduplicated',
        },
    ]


def _duplicate_infrastructure_audit() -> list[dict[str, str]]:
    return [
        {
            'item': 'checkpoint_json',
            'before': 'stdlib json',
            'after': f'orjson ({backend_name()})',
            'status': 'consolidated',
        },
        {
            'item': 'merge_batch_workers',
            'before': '3 independent fast/heavy/merge implementations',
            'after': 'all delegate to chain.py',
            'status': 'consolidated',
        },
        {
            'item': 'stage_metrics_files',
            'before': 'stage_metrics.json + pipeline_stage_metrics.json',
            'after': 'pipeline_cost_accounting.json + stage_metrics.json',
            'status': 'partial',
        },
        {
            'item': 'legacy_pipelined_v1',
            'before': 'dual graph + pipelined paths',
            'after': 'single canonical graph (concurrent.py removed)',
            'status': 'consolidated',
        },
    ]


def _dead_code_removed() -> list[str]:
    return [
        'engine._stage2_fast_filter',
        'engine._stage2_structural_filter',
        'engine._stage_doc_dedup',
        'engine._stage_metadata_validate',
        'workers inline run_fast_stages/run_heavy_stages loops',
    ]


def _engineering_debt() -> list[dict[str, str]]:
    return [
        {
            'area': 'dedup_order',
            'debt': 'lang must precede exact dedup (check-and-insert)',
            'mitigation': 'lookup-only fast dedup + apply insert (future, parity-gated)',
        },
        {
            'area': 'fuzzy_dedup',
            'debt': 'custom MinHash vs datasketch',
            'mitigation': 'datasketch optional extra; frozen-corpus parity required',
        },
        {
            'area': 'embed_ann',
            'debt': 'bucket index vs faiss',
            'mitigation': 'faiss optional extra for Tier4 scale',
        },
        {
            'area': 'pipelined_v1',
            'debt': 'removed',
            'mitigation': 'concurrent.py deleted; canonical graph only',
        },
    ]


def _migration_plan() -> list[dict[str, str]]:
    return [
        {'phase': '1', 'action': 'chain single execution path', 'status': 'done'},
        {'phase': '2', 'action': 'tier01 progressive admission', 'status': 'done'},
        {'phase': '3', 'action': 'orjson checkpoint/jsonl', 'status': 'done'},
        {'phase': '4', 'action': 'pipeline cost accounting', 'status': 'done'},
        {'phase': '5', 'action': 'datasketch fuzzy backend', 'status': 'parity_pending'},
        {'phase': '6', 'action': 'faiss embed ANN', 'status': 'parity_pending'},
        {'phase': '7', 'action': 'zstandard jsonl compression', 'status': 'planned'},
        {'phase': '8', 'action': 'deprecate pipelined v1 / unify execution graph', 'status': 'done'},
    ]


def _cpu_gpu_reports(work_dir: Path | None) -> dict[str, Any]:
    cpu: dict[str, Any] = {'pools': resource_allocation_spec()['cpu_pools']}
    gpu: dict[str, Any] = {'pools': resource_allocation_spec()['gpu_pools']}
    if work_dir and (work_dir / 'pipeline_scheduler_report.json').is_file():
        sched = load_work_json(work_dir / 'pipeline_scheduler_report.json')
        cpu['scheduler'] = sched.get('cpu') or {}
        gpu['scheduler'] = sched.get('gpu') or {}
    if work_dir and (work_dir / 'pipeline_live_metrics.json').is_file():
        live = load_work_json(work_dir / 'pipeline_live_metrics.json')
        cpu['utilization_pct'] = live.get('cpu_pct')
        cpu['rss_mb'] = live.get('rss_mb')
    return {'cpu': cpu, 'gpu': gpu}


def build_pipeline_audit_report(
    work_dir: Path | str | None = None,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    work = Path(work_dir) if work_dir else None
    cost = load_work_json(work / 'pipeline_cost_accounting.json') if work and (work / 'pipeline_cost_accounting.json').is_file() else {}
    live = load_work_json(work / 'pipeline_live_metrics.json') if work and (work / 'pipeline_live_metrics.json').is_file() else {}
    sched = load_work_json(work / 'pipeline_scheduler_report.json') if work and (work / 'pipeline_scheduler_report.json').is_file() else {}
    tier = load_work_json(work / 'admission_tier_report.json') if work and (work / 'admission_tier_report.json').is_file() else {}

    lib = build_library_migration_report(work, workers=workers)
    admission = build_admission_cost_report(work) if work and work.is_dir() else {}
    stage0 = build_stage0_cost_report(work) if work and work.is_dir() else {}
    heavy = build_heavy_pipeline_cost_report(work) if work and work.is_dir() else {}

    throughput = cost.get('throughput_docs_per_sec') or 0.0
    savings = admission.get('savings_projection') or {}

    return {
        'architecture_graph': horizontal_graph_spec(),
        'ownership_graph': ownership_graph(),
        'commodity_vs_intelligence': classification_summary(),
        'commodity_count': len(commodity_stages()),
        'intelligence_count': len(intelligence_stages()),
        'duplicate_computation_audit': _duplicate_computation_audit(),
        'duplicate_infrastructure_audit': _duplicate_infrastructure_audit(),
        'dead_code_removed': _dead_code_removed(),
        'cpu_utilization': _cpu_gpu_reports(work)['cpu'],
        'gpu_utilization': _cpu_gpu_reports(work)['gpu'],
        'scheduler_report': sched,
        'queue_analysis': {
            'live': live.get('scheduler') or {},
            'bottlenecks': live.get('bottlenecks') or [],
            'peak_read_queue': (live.get('scheduler') or {}).get('peak_read_queue'),
            'peak_heavy_pending': (live.get('scheduler') or {}).get('peak_heavy_pending'),
        },
        'cache_analysis': {
            'hit_rate': live.get('cache_hit_rate'),
            'stage0_cache': (stage0.get('cache_efficiency') if stage0 else {}),
        },
        'ipc_analysis': {
            'raw_features_in_survivor': True,
            'gate_raw_document_context': True,
            'orjson_backend': backend_name(),
        },
        'memory_analysis': {
            'rss_mb': live.get('rss_mb'),
            'survivor_mmap': 'data.schedule.state.survivor',
        },
        'pipeline_cost_accounting': cost,
        'bottleneck_tree': cost.get('bottleneck_tree') or (heavy.get('bottleneck_tree') if heavy else []),
        'gate_recommendations': cost.get('gate_recommendations') or [],
        'throughput_estimate': {
            'docs_per_sec': throughput,
            'improvement_pct_est': savings.get('throughput_improvement_pct_est'),
        },
        'library_adoption_report': lib.get('commodity_replaced'),
        'engineering_debt_report': _engineering_debt(),
        'migration_plan': _migration_plan(),
        'remaining_optimization_opportunities': lib.get('library_adoption_opportunities'),
        'competitive_advantage': lib.get('competitive_advantage_components'),
        'tier_reject_pct': tier.get('reject_pct_by_tier'),
        'parity_contract': lib.get('parity_contract'),
        'execution_consolidation': build_execution_consolidation_report(workers=workers, work_dir=work),
        'dask_integration': __import__(
            'data.tools.reports.dask_integration',
            fromlist=['build_dask_integration_report'],
        ).build_dask_integration_report(workers=workers),
        'stage0_cost': stage0,
        'heavy_cost': heavy,
    }
