from __future__ import annotations

import os
import sys

import pytest

from indw.schedule.backends.config import pipeline_execution_backend
from indw.schedule.backends.factory import resolve_execution_backend
from indw.schedule.monitor.audit import sorted_output_hash
from indw.schedule.core import merge_with_quality
from tests.fixtures.pipeline_corpus import MERGE_PASSAGE_A, lenient_merge_config, write_raw_sources, write_resolved_quality

pytestmark = [pytest.mark.critical, pytest.mark.integration]


def _run_merge(tmp_path, *, workers: int, backend: str) -> str:
    from indw.clean.artifact.discovery_engine import reset_discovery_engines

    os.environ['INSTANT_MERGE_HW_PROBE'] = '0'
    os.environ['INSTANT_PIPELINE_BACKEND'] = backend
    cfg = lenient_merge_config()
    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a',), texts=(MERGE_PASSAGE_A,))
    work = tmp_path / f'{backend}_w{workers}'
    out = work / 'filtered.jsonl'
    write_resolved_quality(work, cfg)
    reset_discovery_engines()
    merge_with_quality(
        raw_dir,
        out,
        quality_config=cfg,
        work_dir=work,
        fresh=True,
        resume=False,
        workers=workers,
        chunk_size=2,
    )
    return sorted_output_hash(out)


def test_backend_factory_defaults():
    os.environ.pop('INSTANT_PIPELINE_BACKEND', None)
    assert resolve_execution_backend().name == 'multiprocess'
    assert pipeline_execution_backend() == 'multiprocess'


def test_backend_aliases():
    from indw.schedule.backends.config import normalize_backend_name

    assert normalize_backend_name('sync') == 'local'
    assert normalize_backend_name('cluster') == 'dask'
    assert resolve_execution_backend('local').name == 'local'


def test_multiprocess_matches_local_backend_parity(tmp_path):
    h_mp = _run_merge(tmp_path / 'a', workers=1, backend='multiprocess')
    h_local = _run_merge(tmp_path / 'b', workers=1, backend='local')
    assert h_mp == h_local


def test_multiprocess_workers_parity(tmp_path):
    h1 = _run_merge(tmp_path / 'a', workers=1, backend='multiprocess')
    h2 = _run_merge(tmp_path / 'b', workers=2, backend='multiprocess')
    assert h1 == h2


@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('dask'),
    reason='dask not installed',
)
@pytest.mark.skipif(
    sys.platform == 'win32',
    reason='Dask LocalCluster full-merge parity is slow/flaky on Windows CI',
)
def test_dask_local_cluster_parity(tmp_path):
    h_mp = _run_merge(tmp_path / 'a', workers=1, backend='multiprocess')
    h_dask = _run_merge(tmp_path / 'b', workers=1, backend='dask')
    assert h_mp == h_dask


def test_dask_backend_requires_package(monkeypatch):
    import indw.schedule.backends.dask as dask_mod

    class _Ctx:
        def __enter__(self):
            raise RuntimeError('dask[distributed] required')

        def __exit__(self, *a):
            return None

    monkeypatch.setattr(dask_mod, '_DaskSessionContext', lambda *a, **k: _Ctx())
    monkeypatch.setitem(sys.modules, 'dask.distributed', None)
    backend = resolve_execution_backend('dask')
    with pytest.raises(RuntimeError, match='dask'):
        with backend.open(('x', {}), fast_workers=1, heavy_workers=1):
            pass


def test_dask_integration_report():
    from indw.tools.reports.dask_integration import build_dask_integration_report

    report = build_dask_integration_report(workers=2)
    assert report['dask_execution_graph']['unchanged_pipeline_graph'] is True
    assert 'local' in report['backend_abstraction']['implementations']
