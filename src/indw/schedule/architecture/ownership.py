from __future__ import annotations

from typing import Any

OWNERSHIP: dict[str, dict[str, str]] = {
    'json_parse': {
        'owner': 'data.store.io.jsonl',
        'canonical': 'parse_jsonl_line',
        'cleanup': 'data.store.io.json_codec',
        'resource': 'cpu',
    },
    'json_serialize': {
        'owner': 'data.store.io.json_codec',
        'canonical': 'dumps_line',
        'cleanup': 'data.ingest.sink.BufferedJsonlWriter',
        'resource': 'cpu',
    },
    'checkpoint': {
        'owner': 'data.schedule.state.checkpoint',
        'canonical': 'MergeCheckpoint',
        'cleanup': 'data.store.io.atomic',
        'resource': 'cpu',
    },
    'html_extract': {
        'owner': 'data.clean.document.html',
        'canonical': 'extract_html_body',
        'library': 'trafilatura',
        'resource': 'cpu',
    },
    'exact_dedup': {
        'owner': 'data.dedup.service.exact_shard',
        'canonical': 'ShardedExactDedup',
        'library': 'sqlite3+tenacity',
        'resource': 'cpu',
    },
    'fuzzy_dedup': {
        'owner': 'data.dedup.fuzzy',
        'canonical': 'StreamingFuzzyDedup',
        'resource': 'cpu',
    },
    'embed_dedup': {
        'owner': 'data.dedup.embed.pipeline',
        'canonical': 'EmbedDedupPipeline',
        'gpu_pool': 'data.dedup.embed.pools.gpu_worker',
        'resource': 'cpu_default',
    },
    'fast_gates': {
        'owner': 'data.schedule.admission.tier01',
        'canonical': 'run_tier01_gates',
        'resource': 'cpu',
    },
    'fast_chain': {
        'owner': 'data.schedule.stages.pools.chain',
        'canonical': 'process_fast_chain_batch',
        'resource': 'cpu',
    },
    'heavy_chain': {
        'owner': 'data.schedule.stages.pools.chain',
        'canonical': 'process_heavy_chain_batch',
        'resource': 'cpu',
    },
    'stage0': {
        'owner': 'data.filter.stage0.engine',
        'canonical': 'run_stage0_content_filters',
        'resource': 'cpu',
        'kind': 'intelligence',
    },
    'pci': {
        'owner': 'data.schedule.intel.pci',
        'canonical': 'build_fingerprint_bundle_detail',
        'resource': 'cpu',
        'kind': 'intelligence',
    },
    'acim': {
        'owner': 'data.schedule.intel.session',
        'canonical': 'ACIMSession',
        'resource': 'cpu',
        'kind': 'intelligence',
    },
    'lci': {
        'owner': 'data.schedule.intel.lci_graph',
        'canonical': 'LCIGraph',
        'resource': 'cpu',
        'kind': 'intelligence',
    },
    'semantic_clean': {
        'owner': 'data.clean.corpus',
        'canonical': 'CorpusCleaningPipeline',
        'resource': 'cpu',
        'kind': 'intelligence',
    },
    'knowledge_extraction': {
        'owner': 'data.extract.core.units',
        'canonical': 'extract_knowledge_units',
        'resource': 'cpu',
        'kind': 'intelligence',
    },
    'apply': {
        'owner': 'data.schedule.apply.coordinator',
        'canonical': 'ApplyCoordinator',
        'resource': 'cpu',
    },
    'scheduler': {
        'owner': 'data.schedule.graph.runner',
        'canonical': 'run_graph_merge',
        'resource': 'cpu',
    },
    'execution_backend': {
        'owner': 'data.schedule.backends.factory',
        'canonical': 'resolve_execution_backend',
        'resource': 'cpu',
        'backends': 'local,thread,multiprocess,dask',
    },
    'curriculum': {
        'owner': 'data.schedule.mix.curriculum',
        'canonical': 'CurriculumPlanner',
        'kind': 'intelligence',
        'resource': 'cpu',
    },
}


def ownership_graph() -> dict[str, Any]:
    commodity = [k for k, v in OWNERSHIP.items() if v.get('kind') != 'intelligence']
    intelligence = [k for k, v in OWNERSHIP.items() if v.get('kind') == 'intelligence']
    return {
        'capabilities': OWNERSHIP,
        'commodity_owners': commodity,
        'intelligence_owners': intelligence,
        'single_owner_enforced': True,
    }
