from __future__ import annotations

from typing import Any

from indw.schedule.architecture.graph import horizontal_graph_spec
from indw.schedule.architecture.ownership import ownership_graph
from indw.schedule.backends.config import dask_scheduler_address, pipeline_execution_backend
from indw.schedule.backends.factory import backend_topology, resolve_execution_backend


def _current_execution_graph() -> dict[str, Any]:
    spec = horizontal_graph_spec()
    return {
        'topology': [
            'Reader → read_queue',
            'Fast chain (preprocess → filter → stage0)',
            'LaneBuffers → Heavy chain (pci → acim → clean)',
            'ApplyCoordinator (deterministic seq apply)',
            'BufferedJsonlWriter + checkpoint',
            'run_merge_finalize',
        ],
        'entry': 'merge_with_quality → merge_with_quality_parallel → run_graph_merge',
        'spec': spec,
    }


def _dask_execution_graph() -> dict[str, Any]:
    return {
        'unchanged_pipeline_graph': True,
        'dask_role': 'batch task executor only',
        'orchestration_owner': 'graph.runner.run_graph_merge',
        'dask_schedules': ['process_fast_chain_batch', 'process_heavy_chain_batch'],
        'dask_does_not_own': [
            'ApplyCoordinator ordering',
            'exact/fuzzy/semantic dedup at apply',
            'checkpoint/finalize',
            'Stage0/PCI/ACIM/clean intelligence',
            'acceptance decisions',
        ],
        'worker_bootstrap': 'client.run(_dask_worker_bootstrap, worker_init)',
        'scheduler_address': dask_scheduler_address() or 'LocalCluster (ephemeral)',
        'fault_tolerance': 'dask task retry + in-process process_merge_batch fallback',
    }


def _backend_abstraction() -> dict[str, Any]:
    return {
        'contract': 'data.schedule.backends.contract.ExecutionBackend',
        'session': 'ExecutionSession.submit_fast/submit_heavy/run_fallback_merge',
        'implementations': backend_topology(),
        'default': pipeline_execution_backend(),
        'env': 'INSTANT_PIPELINE_BACKEND=local|thread|multiprocess|dask',
        'dask_env': 'INSTANT_DASK_SCHEDULER or DASK_SCHEDULER_ADDRESS',
    }


def _worker_topology() -> dict[str, Any]:
    backend = pipeline_execution_backend()
    return {
        'local': {'processes': 0, 'threads': 0, 'description': 'sync in driver process'},
        'thread': {'processes': 0, 'threads': 'plan_graph_alloc fast/heavy', 'description': 'shared-memory thread pool'},
        'multiprocess': {'processes': 'spawn pools', 'description': 'default production local/multi-core'},
        'dask': {'processes': 'cluster workers', 'description': 'horizontal multi-node'},
        'active': backend,
        'gpu': 'embed dedup optional; backend-independent',
    }


def _scheduler_ownership() -> dict[str, str]:
    return {
        'graph_orchestration': 'graph.runner.run_graph_merge',
        'batch_dispatch': 'backends.*.ExecutionSession',
        'ordered_apply': 'apply.coordinator.ApplyCoordinator',
        'lane_routing': 'dispatch.lanes.LaneBuffers',
        'ingest_queue': 'dispatch.parallel._reader_thread',
        'finalize': 'state.artifacts.run_merge_finalize',
    }


def _queue_ownership() -> dict[str, str]:
    return {
        'ingest': 'stdlib queue.Queue (reader → graph loop)',
        'survivor_lanes': 'dispatch.lanes.LaneBuffers',
        'apply_buffer': 'apply.coordinator seq-ordered dict',
        'cross_node_optional': 'graph.queues make_stage_queue (fs/redis)',
    }


def _local_vs_distributed_mapping() -> list[dict[str, str]]:
    return [
        {'deployment': 'single PC debug', 'backend': 'local', 'config': 'INSTANT_PIPELINE_BACKEND=local workers=1'},
        {'deployment': 'multi-core PC', 'backend': 'multiprocess', 'config': 'workers=N (default)'},
        {'deployment': 'I/O threaded local', 'backend': 'thread', 'config': 'INSTANT_PIPELINE_BACKEND=thread'},
        {'deployment': 'Dask cluster', 'backend': 'dask', 'config': 'INSTANT_PIPELINE_BACKEND=dask INSTANT_DASK_SCHEDULER=tcp://...'},
    ]


def _scaling_characteristics() -> dict[str, Any]:
    return {
        'local': {'scale_axis': 'workers alloc only', 'bottleneck': 'driver CPU'},
        'multiprocess': {'scale_axis': 'CPU cores on host', 'bottleneck': 'apply ordering + sqlite dedup'},
        'dask': {'scale_axis': 'cluster worker count', 'bottleneck': 'shared merge_work FS for dedup shards'},
        'deterministic_apply': 'always single-threaded ApplyCoordinator in driver',
    }


def _bottleneck_analysis() -> list[dict[str, str]]:
    return [
        {'stage': 'apply', 'note': 'seq-ordered; not parallelized across backends'},
        {'stage': 'exact_dedup', 'note': 'sqlite check-and-insert; requires shared work_dir on dask cluster'},
        {'stage': 'heavy_chain', 'note': 'primary dask scale target'},
        {'stage': 'fast_chain', 'note': 'second dask scale target'},
    ]


def _migration_plan() -> list[dict[str, str]]:
    return [
        {'phase': '1', 'action': 'ExecutionBackend abstraction', 'status': 'done'},
        {'phase': '2', 'action': 'Refactor run_graph_merge to backend session', 'status': 'done'},
        {'phase': '3', 'action': 'DaskBackend with worker bootstrap', 'status': 'done'},
        {'phase': '4', 'action': 'Parity tests local/multiprocess/dask', 'status': 'done'},
        {'phase': '5', 'action': 'Wire per-stage dask queues for multi-node ingest', 'status': 'planned'},
        {'phase': '6', 'action': 'Shared dedup store for multi-node (NFS/object)', 'status': 'planned'},
    ]


def build_dask_integration_report(*, workers: int = 4) -> dict[str, Any]:
    return {
        'current_execution_graph': _current_execution_graph(),
        'dask_execution_graph': _dask_execution_graph(),
        'backend_abstraction': _backend_abstraction(),
        'worker_topology': _worker_topology(),
        'scheduler_ownership': _scheduler_ownership(),
        'queue_ownership': _queue_ownership(),
        'duplicate_paths_removed': [
            'ProcessPoolExecutor hardcoded in runner (now MultiprocessBackend)',
        ],
        'local_vs_distributed_mapping': _local_vs_distributed_mapping(),
        'expected_scaling_characteristics': _scaling_characteristics(),
        'bottleneck_analysis': _bottleneck_analysis(),
        'migration_plan': _migration_plan(),
        'ownership_graph': ownership_graph(),
        'resolved_backend': resolve_execution_backend().name,
        'validation': {
            'hash_parity': 'multiprocess vs local vs dask (LocalCluster)',
            'ordering': 'ApplyCoordinator unchanged',
            'intelligence': 'stages/pools/chain only; no dask imports in stages',
        },
    }
