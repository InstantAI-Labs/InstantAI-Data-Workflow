from __future__ import annotations

import pytest

from indw.config.resolve import PipelineConfigContext
from indw.store.corpus.registry import CorpusRegistry
from indw.schedule.monitor.audit import sorted_output_hash
from indw.schedule.core import merge_with_quality
from tests.fixtures.pipeline_corpus import MERGE_PASSAGE_A, write_raw_sources, write_resolved_quality

pytestmark = [pytest.mark.critical, pytest.mark.integration, pytest.mark.property]

def _merge_hash(tmp_path, *, workers: int, cfg) -> str:
    from copy import deepcopy

    from indw.clean.artifact.discovery_engine import reset_discovery_engines

    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a', 'source_b'), texts=(MERGE_PASSAGE_A, MERGE_PASSAGE_A))
    work = tmp_path / f'w{workers}'
    out = work / 'filtered.jsonl'
    run_cfg = deepcopy(cfg)
    write_resolved_quality(work, run_cfg)
    corpus = CorpusRegistry(work)
    try:
        reset_discovery_engines()
        merge_with_quality(
            raw_dir, out, quality_config=run_cfg, work_dir=work,
            corpus_registry=corpus, fresh=True, resume=False,
            workers=workers, chunk_size=2,
        )
        return sorted_output_hash(out)
    finally:
        corpus.close()
        reset_discovery_engines()

def test_parallel_merge_hash_parity_lenient(tmp_path):
    from tests.fixtures.pipeline_corpus import lenient_merge_config

    cfg = lenient_merge_config()
    assert _merge_hash(tmp_path, workers=1, cfg=cfg) == _merge_hash(tmp_path, workers=2, cfg=cfg)

def test_parallel_merge_hash_parity_discovery_trim(tmp_path):
    ctx = PipelineConfigContext.resolve()
    cfg = ctx.quality
    cfg.cleaning.enabled = True
    cfg.cleaning.artifact_discovery = True
    cfg.cleaning.artifact_discovery_shadow = False
    cfg.cleaning.artifact_discovery_trim = True
    cfg.dedup.exact = False
    cfg.balance.enabled = False
    cfg.thresholds.min_score = 0.0

    assert _merge_hash(tmp_path, workers=1, cfg=cfg) == _merge_hash(tmp_path, workers=2, cfg=cfg)

def test_parallel_merge_hash_parity_fuzzy(tmp_path):
    from copy import deepcopy
    from tests.fixtures.pipeline_corpus import lenient_merge_config

    cfg = deepcopy(lenient_merge_config())
    cfg.dedup.fuzzy = True
    cfg.dedup.exact = True
    assert _merge_hash(tmp_path, workers=1, cfg=cfg) == _merge_hash(tmp_path, workers=2, cfg=cfg)

def test_parallel_merge_worker_failures_recover(tmp_path, monkeypatch):
    from concurrent.futures import Future

    from tests.fixtures.pipeline_corpus import lenient_merge_config

    import indw.schedule.backends.multiprocess as mp_backend
    import indw.schedule.dispatch.workers as workers_mod

    calls = {'n': 0}
    original_submit = mp_backend.ProcessPoolExecutor.submit

    def flaky_submit(self, fn, /, *args, **kwargs):
        if fn not in (
            workers_mod.process_fast_chain_batch,
            workers_mod.process_heavy_chain_batch,
            workers_mod.process_merge_batch,
        ):
            return original_submit(self, fn, *args, **kwargs)
        calls['n'] += 1
        if calls['n'] == 1:
            fut: Future = Future()
            fut.set_exception(RuntimeError('simulated worker crash'))
            return fut
        return original_submit(self, fn, *args, **kwargs)

    monkeypatch.setattr(mp_backend.ProcessPoolExecutor, 'submit', flaky_submit)
    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a', 'source_b'))
    cfg = lenient_merge_config()
    work = tmp_path / 'work'
    out = work / 'filtered.jsonl'
    write_resolved_quality(work, cfg)
    stats = merge_with_quality(
        raw_dir, out, quality_config=cfg, work_dir=work,
        fresh=True, resume=False, workers=2, chunk_size=2,
    )
    assert stats['kept'] >= 1
    assert int(stats.get('worker_failures', 0)) >= 1
