from __future__ import annotations

import os

import pytest

from indw.schedule.monitor.audit import sorted_output_hash
from indw.schedule.core import merge_with_quality
from tests.fixtures.pipeline_corpus import MERGE_PASSAGE_A, lenient_merge_config, write_raw_sources, write_resolved_quality

pytestmark = [pytest.mark.critical, pytest.mark.property, pytest.mark.integration]


def _run_merge(tmp_path, *, workers: int) -> str:
    from indw.clean.artifact.discovery_engine import reset_discovery_engines

    os.environ['INSTANT_MERGE_HW_PROBE'] = '0'
    cfg = lenient_merge_config()
    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a',), texts=(MERGE_PASSAGE_A,))
    work = tmp_path / f'w{workers}'
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


def test_workers_1_matches_2(tmp_path):
    assert _run_merge(tmp_path / 'a', workers=1) == _run_merge(tmp_path / 'b', workers=2)


def test_workers_1_matches_4(tmp_path):
    assert _run_merge(tmp_path / 'a', workers=1) == _run_merge(tmp_path / 'b', workers=4)
