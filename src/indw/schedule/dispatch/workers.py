from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any

import yaml

from indw.config.resolve import PipelineConfigContext
from indw.clean.corpus import CorpusCleaningPipeline
from indw.filter.spec.quality import QualityPipelineConfig
from indw.filter.gate.quality import QualityGate
from indw.schedule.stages.pools.chain import process_fast_chain_batch, process_heavy_chain_batch
from indw.schedule.read.gates import worker_quality_config
from indw.schedule.intel.merge_session import (
    close_worker_intelligence_session,
    open_worker_intelligence_session,
)
from indw.clean.artifact.evidence_cache import reset_evidence_session

_FAST_CTX: dict[str, Any] | None = None
_HEAVY_CTX: dict[str, Any] | None = None
_PREPROCESS_CTX: dict[str, Any] | None = None
_SHUTDOWN_REGISTERED = False
WorkerInitBundle = tuple[str, dict[str, Any]]


def fast_doc_dedup_check(*, digest: str, source: str) -> bool:
    if _FAST_CTX is None:
        return False
    exact = _FAST_CTX.get('exact_dedup')
    if exact is None:
        return False
    return exact.is_duplicate('', source=source, digest=digest)


def build_worker_init_bundle(
    config_path: str,
    *,
    workers: int,
    chunk_size: int,
    checkpoint_interval: int,
    work_dir: str,
    dedup_index_path: str = '',
    dedup_shards: int = 0,
) -> WorkerInitBundle:
    return (
        config_path,
        {
            'workers': int(workers),
            'chunk_size': int(chunk_size),
            'checkpoint_interval': int(checkpoint_interval),
            'work_dir': work_dir,
            'dedup_index_path': dedup_index_path,
            'dedup_shards': int(dedup_shards),
        },
    )


def _bind_worker_runtime(pol: dict[str, Any]) -> None:
    from indw.schedule.config.policy import MergeRuntime, bind_merge_runtime
    bind_merge_runtime(MergeRuntime.bootstrap(
        workers=int(pol['workers']),
        chunk_size=int(pol['chunk_size']),
        checkpoint_interval=int(pol['checkpoint_interval']),
        work_dir=pol.get('work_dir') or None,
    ))


def _shutdown_fast_worker() -> None:
    global _FAST_CTX
    if _FAST_CTX is None:
        return
    exact = _FAST_CTX.get('exact_dedup')
    if exact is not None:
        flush = getattr(exact, 'flush', None)
        if flush is not None:
            try:
                flush()
            except Exception:
                pass
        index = getattr(exact, '_index', None)
        if index is not None:
            try:
                index.flush()
            except Exception:
                pass
    from indw.schedule.config.policy import bind_merge_runtime
    bind_merge_runtime(None)
    _FAST_CTX = None


def _shutdown_heavy_worker() -> None:
    global _HEAVY_CTX
    if _HEAVY_CTX is None:
        return
    cleaning_pipeline: CorpusCleaningPipeline | None = _HEAVY_CTX.get('cleaning_pipeline')
    if cleaning_pipeline is not None and cleaning_pipeline.discovery_engine is not None:
        cleaning_pipeline.discovery_engine.close()
    from indw.clean.artifact.discovery_engine import reset_discovery_engines
    reset_discovery_engines()
    from indw.schedule.monitor.doc import bind_doc_monitor
    doc_mon = _HEAVY_CTX.get('doc_monitor')
    if doc_mon is not None:
        doc_mon.close()
    bind_doc_monitor(None)
    close_worker_intelligence_session()
    from indw.schedule.config.policy import bind_merge_runtime
    bind_merge_runtime(None)
    _HEAVY_CTX = None


def _shutdown_preprocess_worker() -> None:
    global _PREPROCESS_CTX
    from indw.schedule.config.policy import bind_merge_runtime
    bind_merge_runtime(None)
    _PREPROCESS_CTX = None


def _init_fast_dedup(pol: dict[str, Any]) -> Any | None:
    from indw.dedup.exact import PersistentHashIndex
    from indw.dedup.service.exact_shard import ShardedExactDedup
    from indw.ingest.hash import ExactHashDedup

    shards = int(pol.get('dedup_shards') or 0)
    work_dir = str(pol.get('work_dir') or '').strip()
    raw_path = str(pol.get('dedup_index_path') or '').strip()
    if shards > 1 and work_dir:
        return ShardedExactDedup(work_dir, shards=shards)
    if not raw_path:
        if work_dir:
            raw_path = str(PersistentHashIndex.default_path(work_dir))
    if not raw_path:
        return None
    index = PersistentHashIndex(raw_path)
    return ExactHashDedup(index)


def init_preprocess_worker(init_bundle: WorkerInitBundle) -> None:
    global _PREPROCESS_CTX, _SHUTDOWN_REGISTERED
    if _PREPROCESS_CTX is not None:
        _shutdown_preprocess_worker()
    config_path, pol = init_bundle
    _bind_worker_runtime(pol)
    _PREPROCESS_CTX = {'work_dir': str(pol.get('work_dir') or '').strip()}
    if not _SHUTDOWN_REGISTERED:
        atexit.register(_shutdown_heavy_worker)
        atexit.register(_shutdown_fast_worker)
        atexit.register(_shutdown_preprocess_worker)
        _SHUTDOWN_REGISTERED = True


def init_fast_merge_worker(init_bundle: WorkerInitBundle) -> None:
    global _FAST_CTX, _SHUTDOWN_REGISTERED
    if _FAST_CTX is not None:
        _shutdown_fast_worker()
    config_path, pol = init_bundle
    _bind_worker_runtime(pol)
    config_path_obj = Path(config_path)
    raw = yaml.safe_load(config_path_obj.read_text(encoding='utf-8')) or {}
    cfg = worker_quality_config(
        QualityPipelineConfig.from_dict(raw),
        merge_work=str(config_path_obj.parent),
    )
    ctx = PipelineConfigContext.resolve().with_quality(cfg)
    gate = QualityGate(ctx=ctx)
    exact = _init_fast_dedup(pol) if cfg.dedup.exact else None
    from indw.filter.stage0.engine import bind_fast_stage0_worker
    bind_fast_stage0_worker(gate)
    work_dir = str(pol.get('work_dir') or '').strip()
    _FAST_CTX = {'gate': gate, 'exact_dedup': exact, 'work_dir': work_dir}
    if not _SHUTDOWN_REGISTERED:
        atexit.register(_shutdown_heavy_worker)
        atexit.register(_shutdown_fast_worker)
        atexit.register(_shutdown_preprocess_worker)
        _SHUTDOWN_REGISTERED = True


def init_merge_worker(init_bundle: WorkerInitBundle) -> None:
    global _HEAVY_CTX, _SHUTDOWN_REGISTERED
    if _HEAVY_CTX is not None:
        _shutdown_heavy_worker()
    config_path, pol = init_bundle
    _bind_worker_runtime(pol)
    reset_evidence_session()
    config_path_obj = Path(config_path)
    merge_work = config_path_obj.parent
    open_worker_intelligence_session(merge_work)
    raw = yaml.safe_load(config_path_obj.read_text(encoding='utf-8')) or {}
    cfg = worker_quality_config(QualityPipelineConfig.from_dict(raw), merge_work=str(merge_work))
    ctx = PipelineConfigContext.resolve().with_quality(cfg)
    from indw.schedule.monitor.doc import DocMonitorSession, bind_doc_monitor
    doc_mon = DocMonitorSession(merge_work)
    bind_doc_monitor(doc_mon)
    _HEAVY_CTX = {
        'cfg': cfg,
        'cleaning_pipeline': CorpusCleaningPipeline(cfg.cleaning, score_thresholds=cfg.thresholds),
        'gate': QualityGate(ctx=ctx),
        'doc_monitor': doc_mon,
        'work_dir': str(pol.get('work_dir') or merge_work).strip(),
    }
    if not _SHUTDOWN_REGISTERED:
        atexit.register(_shutdown_heavy_worker)
        atexit.register(_shutdown_fast_worker)
        atexit.register(_shutdown_preprocess_worker)
        _SHUTDOWN_REGISTERED = True


def process_merge_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if _HEAVY_CTX is None:
        raise RuntimeError('merge worker not initialized')
    work_dir = (_HEAVY_CTX or {}).get('work_dir') or (_FAST_CTX or {}).get('work_dir')
    for row in batch:
        if work_dir and not row.get('_work_dir'):
            row['_work_dir'] = work_dir
    fast = process_fast_chain_batch(batch)
    terminal = list(fast.get('terminal') or [])
    survivors = list(fast.get('survivors') or [])
    cost_rows = list(fast.get('_cost_rows') or [])
    if not survivors:
        return {'items': terminal, 'cleaning_stats': None, '_cost_rows': cost_rows}
    heavy = process_heavy_chain_batch(survivors)
    cost_rows.extend(heavy.get('_cost_rows') or [])
    items = terminal + list(heavy.get('items') or [])
    payload: dict[str, Any] = {
        'items': items,
        'cleaning_stats': heavy.get('cleaning_stats'),
        '_cost_rows': cost_rows,
    }
    if heavy.get('discovery_calibration') is not None:
        payload['discovery_calibration'] = heavy['discovery_calibration']
    return payload


def process_fast_merge_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if _FAST_CTX is None:
        raise RuntimeError('fast merge worker not initialized')
    work_dir = (_FAST_CTX or {}).get('work_dir')
    for row in batch:
        if work_dir and not row.get('_work_dir'):
            row['_work_dir'] = work_dir
    return process_fast_chain_batch(batch)


def process_heavy_merge_batch(survivors: list[dict[str, Any]]) -> dict[str, Any]:
    if _HEAVY_CTX is None:
        raise RuntimeError('merge worker not initialized')
    work_dir = (_HEAVY_CTX or {}).get('work_dir')
    for row in survivors:
        if work_dir and not row.get('_work_dir'):
            row['_work_dir'] = work_dir
    return process_heavy_chain_batch(survivors)
