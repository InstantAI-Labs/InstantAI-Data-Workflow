from __future__ import annotations

from typing import Any

from indw.schedule.graph.config import pipeline_queue_backend
from indw.schedule.backends.config import pipeline_execution_backend


def horizontal_graph_spec() -> dict[str, Any]:
    queue = pipeline_queue_backend()
    return {
        'mode': 'canonical_graph',
        'execution_backend': pipeline_execution_backend(),
        'queue_backend': queue,
        'ingest': {
            'owner': 'data.schedule.dispatch.parallel',
            'workers': 1,
            'resource': 'cpu',
            'output': 'read_queue',
        },
        'fast_chain': {
            'owner': 'data.schedule.stages.pools.chain.process_fast_chain_batch',
            'resource': 'cpu',
            'stages': [
                {'pool': 'preprocess', 'stage': 's1_fast_preprocess', 'tier': 0},
                {'pool': 'filter', 'stage': 's2_fast_filter', 'tier': 0},
                {'pool': 'stage0', 'stage': 's2_doc_dedup+s2_structural+s3_admission', 'tier': 1},
            ],
            'terminal_sink': 'apply_coordinator',
            'survivor_sink': 'lane_buffers',
        },
        'heavy_chain': {
            'owner': 'data.schedule.stages.pools.chain.process_heavy_chain_batch',
            'resource': 'cpu',
            'stages': [
                {'pool': 'pci', 'stage': 's3_intermediate', 'tier': 2},
                {'pool': 'acim', 'stage': 's4_intel_preview', 'tier': 3},
                {'pool': 'clean', 'stage': 's4_high_quality', 'tier': 3},
            ],
            'output': 'apply_coordinator',
        },
        'apply': {
            'owner': 'data.schedule.apply.coordinator.ApplyCoordinator',
            'workers': 1,
            'resource': 'cpu',
            'ordered': True,
            'stages': ['s5_final_validation', 's6_output', 'fuzzy_dedup', 'semantic_dedup', 'embed_dedup'],
        },
        'gpu_pools': {
            'embed_dedup': {
                'owner': 'data.dedup.embed.pools.gpu_worker',
                'resource': 'cpu_default',
                'gpu_when': 'embedding extra + sentence-transformers',
                'note': 'hash embed default; E5/GPU optional',
            },
        },
        'bounded_buffers': {
            'fast_result_buffer': 'merge_tune.fast_result_buffer_factor',
            'heavy_result_buffer': 'merge_tune.heavy_result_buffer_factor',
            'apply_coordinator': 'seq-ordered line_results dict',
        },
        'progressive_admission': {
            'owner': 'data.schedule.admission.tier01',
            'order': 'size → lang → dedup → stage0 → metadata → admission',
            'parity_lock': 'dedup check-and-insert after language',
        },
    }
