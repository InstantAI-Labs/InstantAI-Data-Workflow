from __future__ import annotations

from pathlib import Path
from typing import Any

from indw.schedule.architecture.classify import (
    classification_summary,
    commodity_stages,
    intelligence_stages,
)
from indw.schedule.architecture.graph import horizontal_graph_spec
from indw.schedule.architecture.resources import resource_allocation_spec
from indw.schedule.monitor.audit import load_work_json
from indw.store.io.json_codec import backend_name


def _library_adoption() -> list[dict[str, Any]]:
    return [
        {
            'capability': 'json_jsonl',
            'library': 'orjson',
            'status': 'adopted',
            'owner': 'data.store.io.json_codec',
            'coverage': 'jsonl read/write, checkpoint, progress',
        },
        {
            'capability': 'html_extraction',
            'library': 'trafilatura',
            'status': 'adopted',
            'owner': 'data.clean.document.html',
            'coverage': 'primary path; regex fallback on miss',
        },
        {
            'capability': 'caching',
            'library': 'cachetools',
            'status': 'adopted',
            'owner': 'data.store.io.cache',
            'coverage': 'BoundedLRU for dedup/evidence',
        },
        {
            'capability': 'retries',
            'library': 'tenacity',
            'status': 'adopted',
            'owner': 'data.store.io.retry',
            'coverage': 'sqlite IO, distributed ops',
        },
        {
            'capability': 'language_detection',
            'library': 'langid',
            'status': 'optional',
            'owner': 'data.filter.language.fast_detector',
            'coverage': 'english_only fast path when extra installed',
        },
        {
            'capability': 'embeddings',
            'library': 'sentence-transformers',
            'status': 'optional',
            'owner': 'data.dedup.embed.e5',
            'coverage': 'E5 GPU path; hash embed default CPU',
        },
        {
            'capability': 'fuzzy_dedup',
            'library': 'datasketch',
            'status': 'opportunity',
            'owner': 'data.dedup.fuzzy',
            'coverage': 'custom MinHash LSH today; datasketch optional extra',
        },
        {
            'capability': 'vector_ann',
            'library': 'faiss',
            'status': 'opportunity',
            'owner': 'data.dedup.embed.ann',
            'coverage': 'bucket blocking today',
        },
        {
            'capability': 'compression',
            'library': 'zstandard',
            'status': 'opportunity',
            'owner': 'data.ingest.sink',
            'coverage': 'pyarrow zstd for parquet export only',
        },
        {
            'capability': 'distributed_queue',
            'library': 'redis',
            'status': 'optional',
            'owner': 'data.schedule.graph.queues',
            'coverage': 'INSTANT_PIPELINE_QUEUE=redis',
        },
    ]


def _duplicates_removed() -> list[dict[str, str]]:
    return [
        {
            'item': 'jsonl_parse',
            'before': 'parse_merge_jsonl_line + parse_jsonl_batch duplicate logic',
            'after': 'single owner data.store.io.jsonl.parse_jsonl_line',
            'status': 'consolidated',
        },
        {
            'item': 'checkpoint_json',
            'before': 'stdlib json in checkpoint/progress',
            'after': f'orjson via data.store.io.json_codec ({backend_name()})',
            'status': 'consolidated',
        },
        {
            'item': 'document_gate_raw',
            'before': 're-extract in structural + evaluate_document_gate',
            'after': 'single extract; gate_raw shared via survivor payload',
            'status': 'consolidated',
        },
        {
            'item': 'pci_raw_rescan',
            'before': 'PCI full rescan without raw_features',
            'after': 'build_fingerprint_bundle_detail(text, raw=ctx.raw_features)',
            'status': 'consolidated',
        },
        {
            'item': 'fast_gate_paths',
            'before': 'engine _stage* + tier01 + graph pools',
            'after': 'tier01 single owner; pools delegate',
            'status': 'unified',
        },
        {
            'item': 'heavy_gate_paths',
            'before': 'run_heavy_stages + pci/acim/clean pools',
            'after': 'graph pools primary; engine retained for workers=1 parity',
            'status': 'dual_path_parity_locked',
        },
    ]


def _intelligence_preserved() -> list[dict[str, str]]:
    return [
        {'system': 'Stage0', 'owner': 'data.filter.stage0.engine', 'tier': '0-1'},
        {'system': 'PCI', 'owner': 'data.schedule.intel.pci', 'tier': '2'},
        {'system': 'ACIM', 'owner': 'data.schedule.intel.session', 'tier': '3'},
        {'system': 'LCI', 'owner': 'data.schedule.intel.lci_graph', 'tier': '3'},
        {'system': 'Semantic clean', 'owner': 'data.clean.semantic.pipeline', 'tier': '3'},
        {'system': 'Educational scoring', 'owner': 'data.extract.roles.education', 'tier': '3-4'},
        {'system': 'Publication recovery', 'owner': 'data.extract.roles.publication', 'tier': '3-4'},
        {'system': 'Knowledge extraction', 'owner': 'data.extract.core.units', 'tier': '4'},
        {'system': 'Curriculum balancing', 'owner': 'data.schedule.mix.curriculum', 'tier': 'cross'},
        {'system': 'Quality gate / calibrator', 'owner': 'data.filter.gate.quality', 'tier': '3'},
        {'system': 'Artifact discovery', 'owner': 'data.clean.artifact.discovery_engine', 'tier': '3'},
    ]


def _bottleneck_rank(work_dir: Path | None) -> list[dict[str, Any]]:
    if work_dir is None or not Path(work_dir).is_dir():
        return []
    metrics = load_work_json(Path(work_dir) / 'stage_metrics.json')
    stages = metrics.get('stages') or {}
    nodes = []
    for name, row in stages.items():
        if not isinstance(row, dict):
            continue
        wall = float(row.get('wall_sec') or 0.0)
        if wall <= 0:
            continue
        nodes.append({
            'stage': name,
            'wall_sec': round(wall, 4),
            'in_docs': int(row.get('in_docs') or 0),
        })
    return sorted(nodes, key=lambda n: -n['wall_sec'])[:12]


def build_library_migration_report(
    work_dir: Path | str | None = None,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    work = Path(work_dir) if work_dir else None
    admission = load_work_json(work / 'admission_cost_report.json') if work and (work / 'admission_cost_report.json').is_file() else {}
    tier = load_work_json(work / 'admission_tier_report.json') if work and (work / 'admission_tier_report.json').is_file() else {}

    savings = admission.get('savings_projection') or {}
    return {
        'architecture': 'horizontal_library_first',
        'json_backend': backend_name(),
        'commodity_replaced': _library_adoption(),
        'custom_intelligence_preserved': _intelligence_preserved(),
        'duplicate_engineering_removed': _duplicates_removed(),
        'horizontal_execution_graph': horizontal_graph_spec(),
        'cpu_gpu_allocation': resource_allocation_spec(workers=workers),
        'stage_classification': classification_summary(),
        'commodity_stage_count': len(commodity_stages()),
        'intelligence_stage_count': len(intelligence_stages()),
        'library_adoption_opportunities': [
            row for row in _library_adoption() if row['status'] == 'opportunity'
        ],
        'throughput_improvement_pct_est': savings.get('throughput_improvement_pct_est'),
        'compute_savings': {
            'cpu_pct_est': savings.get('cpu_savings_pct_est'),
            'gpu_pct_est': savings.get('gpu_savings_pct_est'),
            'ram_pct_est': savings.get('ram_savings_pct_est'),
            'io_pct_est': savings.get('io_savings_pct_est'),
        },
        'tier_reject_pct': tier.get('reject_pct_by_tier'),
        'bottleneck_ranking': _bottleneck_rank(work),
        'competitive_advantage_components': [
            s['system'] for s in _intelligence_preserved()
        ],
        'parity_contract': {
            'sorted_output_hash': 'unchanged',
            'gate_order': 'size → lang → dedup → stage0',
            'workers_parity': 'workers=1 ≡ workers=N',
            'tests': [
                'test_tier_admission_parity',
                'test_stage_pool_parity',
                'test_stage0_gate_parity',
                'test_multi_node_parity',
            ],
        },
    }
