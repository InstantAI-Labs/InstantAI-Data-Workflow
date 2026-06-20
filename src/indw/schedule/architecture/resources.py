from __future__ import annotations

from typing import Any


def resource_allocation_spec(*, workers: int = 8) -> dict[str, Any]:
    from indw.schedule.dispatch.alloc import plan_graph_alloc
    from indw.schedule.config.hardware import probe_system_hardware

    hw = probe_system_hardware()
    _, alloc = plan_graph_alloc(workers=workers, chunk_size=32)
    v2 = alloc
    return {
        'hardware': {
            'cpu_logical': hw.cpu_logical,
            'mem_budget_mb': hw.mem_budget_mb,
            'gpu_count': getattr(hw, 'gpu_count', 0),
        },
        'cpu_pools': {
            'preprocess': getattr(v2, 'preprocess_workers', max(1, workers // 4)),
            'filter': getattr(v2, 'filter_workers', max(1, workers // 8)),
            'stage0': getattr(v2, 'stage0_workers', max(1, workers // 4)),
            'pci': getattr(v2, 'pci_workers', max(1, workers // 4)),
            'acim': getattr(v2, 'acim_workers', max(1, workers // 4)),
            'clean': getattr(v2, 'clean_workers', max(1, workers // 2)),
            'apply': 1,
            'ingest_reader': 1,
        },
        'gpu_pools': {
            'embed_encode': {
                'default': 'cpu_hash',
                'optional': 'sentence-transformers E5',
                'isolated_from': ['preprocess', 'filter', 'stage0', 'pci', 'acim', 'clean'],
            },
        },
        'dedup_shards': getattr(v2, 'dedup_shards', 0),
        'allocation': alloc.to_dict() if hasattr(alloc, 'to_dict') else {},
        'policy': {
            'lightweight_filter_cpu_only': True,
            'custom_intelligence_cpu': True,
            'neural_inference_gpu_optional': True,
            'tier_gated_execution': True,
        },
    }
