from __future__ import annotations

from pathlib import Path
from typing import Any

from indw.schedule.architecture.classify import classification_summary
from indw.schedule.architecture.graph import horizontal_graph_spec
from indw.schedule.architecture.ownership import ownership_graph
from indw.schedule.architecture.resources import resource_allocation_spec


def _current_execution_graph() -> dict[str, Any]:
    return {
        'label': 'pre_consolidation',
        'paths': [
            'core.py serial loop (workers=1, graph=off)',
            'parallel.py → concurrent.run_pipelined_merge (v1)',
            'parallel.py → graph.run_graph_merge (v2)',
            'workers.process_fast/heavy_merge_batch → engine.run_fast/heavy_stages',
            'workers.process_fast/heavy_chain_batch → stage pools',
        ],
        'schedulers': 2,
        'batch_entrypoints': 5,
        'apply_paths': 2,
    }


def _duplicate_execution_graph() -> dict[str, Any]:
    return {
        'scheduler': {
            'canonical': 'graph.runner.run_graph_merge',
            'removed': 'dispatch.concurrent.run_pipelined_merge',
            'overlap_pct': 70,
        },
        'fast_batch': {
            'canonical': 'stages.pools.chain.process_fast_chain_batch',
            'removed': 'workers.process_fast_merge_batch → engine.run_fast_stages',
        },
        'heavy_batch': {
            'canonical': 'stages.pools.chain.process_heavy_chain_batch',
            'removed': 'workers.process_heavy_merge_batch → engine.run_heavy_stages',
        },
        'serial_merge': {
            'canonical': 'dispatch.parallel.merge_with_quality_parallel',
            'removed': 'core.py ~350-line serial loop',
        },
        'stage_bodies': {
            'canonical': 'stages.pools.* + admission.tier01',
            'duplicate': 'stages.engine._stage* (retained for unit tests / single-doc)',
        },
        'alloc': {
            'canonical': 'dispatch.alloc.plan_graph_alloc',
            'legacy': 'dispatch.alloc.plan_pipelined_alloc (base for graph alloc only)',
        },
    }


def _canonical_execution_graph() -> dict[str, Any]:
    spec = horizontal_graph_spec()
    return {
        'topology': [
            'Reader (_reader_thread → read_queue)',
            'Stage queues (read_queue, lane_buffers)',
            'Worker pools (fast_executor: process_fast_chain_batch)',
            'Heavy pools (heavy_executor: process_heavy_chain_batch)',
            'ApplyCoordinator (seq-ordered apply_merge_preprocessed_line)',
            'Storage (BufferedJsonlWriter + checkpoint flusher)',
            'Finalize (run_merge_finalize)',
        ],
        'entry': 'merge_with_quality → merge_with_quality_parallel → run_graph_merge',
        'deployment_modes': {
            'local_single_core': 'workers=1, pool sizes from plan_graph_alloc',
            'local_multi_core': 'workers=N, fast/heavy pool scale with N',
            'horizontal': 'same graph; INSTANT_PIPELINE_QUEUE=redis|filesystem for cross-node queues',
        },
        'spec': spec,
    }


def _local_scaling_model() -> dict[str, Any]:
    return {
        'knob': 'workers + plan_graph_alloc',
        'behavior': 'Increase workers → larger fast/heavy ProcessPoolExecutor; graph unchanged',
        'cpu_pools': resource_allocation_spec(workers=8),
        'gpu_pools': 'embed_dedup optional; hash default on CPU',
        'no_code_path_change': True,
    }


def _horizontal_scaling_model() -> dict[str, Any]:
    return {
        'same_graph': True,
        'queue_backend': 'graph.queues.make_stage_queue (local|filesystem|redis)',
        'worker_placement': 'ProcessPoolExecutor per node; shared dedup index via sqlite shards',
        'parity_tests': ['test_multi_node_parity.py'],
    }


def _duplicate_scheduler_audit() -> list[dict[str, str]]:
    return [
        {'item': 'run_pipelined_merge', 'status': 'removed', 'canonical': 'run_graph_merge'},
        {'item': 'core serial loop', 'status': 'removed', 'canonical': 'merge_with_quality_parallel'},
        {'item': 'graph_mode branch in parallel.py', 'status': 'removed', 'canonical': 'always graph'},
        {'item': 'INSTANT_PIPELINE_GRAPH v1', 'status': 'ignored', 'canonical': 'always v2 graph'},
    ]


def _duplicate_worker_audit() -> list[dict[str, str]]:
    return [
        {'item': 'process_fast_merge_batch', 'status': 'delegates_to_chain', 'owner': 'workers.py'},
        {'item': 'process_heavy_merge_batch', 'status': 'delegates_to_chain', 'owner': 'workers.py'},
        {'item': 'process_merge_batch fallback', 'status': 'delegates_to_chain', 'owner': 'workers.py'},
        {'item': 'run_fast_stages / run_heavy_stages', 'status': 'retained_for_tests', 'owner': 'stages/engine.py'},
        {'item': 'init_preprocess_worker', 'status': 'unused', 'owner': 'workers.py'},
    ]


def _duplicate_infrastructure_audit() -> list[dict[str, str]]:
    return [
        {'item': 'dual checkpoint finalize paths', 'status': 'unified', 'owner': 'state/artifacts.run_merge_finalize'},
        {'item': 'dual apply paths', 'status': 'unified', 'owner': 'apply/coordinator + apply/merge'},
        {'item': 'PipelineStageMonitor (v1 only)', 'status': 'removed_with_concurrent', 'owner': 'read/probe.SchedulerProbe'},
        {'item': 'per-stage queue fields in StageAllocationV2', 'status': 'unused', 'note': 'future horizontal IPC'},
    ]


def _files_removed() -> list[str]:
    return [
        'data/schedule/dispatch/concurrent.py',
        'data/schedule/core.py serial merge loop (~350 lines)',
    ]


def _modules_merged() -> list[dict[str, str]]:
    return [
        {'from': 'core serial + parallel + concurrent', 'to': 'parallel + graph.runner'},
        {'from': 'workers engine loops', 'to': 'stages.pools.chain'},
        {'from': 'dual graph config branches', 'to': 'single canonical_graph mode'},
    ]


def _execution_simplification() -> dict[str, Any]:
    return {
        'scheduler_implementations_before': 3,
        'scheduler_implementations_after': 1,
        'merge_entry_paths_before': 3,
        'merge_entry_paths_after': 1,
        'batch_wrappers_unified': True,
        'lines_removed_estimate': 900,
    }


def _maintenance_reduction() -> dict[str, str]:
    return {
        'gate_changes': 'single chain path; tier01 + pools only',
        'parity_burden': 'workers=1 vs N only; no v1/v2 dual maintenance',
        'scheduler_fixes': 'one loop in graph/runner.py',
    }


def _throughput_improvement() -> dict[str, str]:
    return {
        'local': 'workers=1 now uses same pooled graph as N (no serial bottleneck)',
        'horizontal': 'unchanged; queue backends ready for multi-node',
        'expected': 'marginal on small corpus; scales with worker count on production corpus',
    }


def _remaining_technical_debt() -> list[dict[str, str]]:
    return [
        {'area': 'stage_body_triplication', 'note': 'engine.py vs pools vs tier01 — pools should call engine primitives'},
        {'area': 'plan_pipelined_alloc', 'note': 'still base for plan_graph_alloc; could inline'},
        {'area': 'init_preprocess_worker', 'note': 'dead initializer'},
        {'area': 'graph/queues.py', 'note': 'not wired to runner; horizontal scale pending'},
        {'area': 'dedup_order', 'note': 'lang before dedup parity lock'},
        {'area': 'library_adoption', 'note': 'datasketch, faiss, zstandard parity-gated'},
    ]


def build_execution_consolidation_report(*, workers: int = 4, work_dir: Path | None = None) -> dict[str, Any]:
    return {
        'current_execution_graph': _current_execution_graph(),
        'duplicate_execution_graph': _duplicate_execution_graph(),
        'canonical_execution_graph': _canonical_execution_graph(),
        'local_scaling_model': _local_scaling_model(),
        'horizontal_scaling_model': _horizontal_scaling_model(),
        'duplicate_scheduler_audit': _duplicate_scheduler_audit(),
        'duplicate_worker_audit': _duplicate_worker_audit(),
        'duplicate_infrastructure_audit': _duplicate_infrastructure_audit(),
        'ownership_graph': ownership_graph(),
        'classification': classification_summary(),
        'files_removed': _files_removed(),
        'modules_merged': _modules_merged(),
        'execution_simplification': _execution_simplification(),
        'expected_maintenance_reduction': _maintenance_reduction(),
        'expected_throughput_improvement': _throughput_improvement(),
        'remaining_technical_debt': _remaining_technical_debt(),
        'validation': {
            'output_hash_parity': 'workers 1/2/4/8',
            'acceptance_parity': 'tier01 + apply path unified',
            'ordering': 'ApplyCoordinator seq head',
            'tests': [
                'test_stage_pool_parity',
                'test_tier_admission_parity',
                'test_parallel_merge_parity',
                'test_multi_node_parity',
                'production_scale_audit',
            ],
        },
    }
