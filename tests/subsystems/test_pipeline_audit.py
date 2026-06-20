from __future__ import annotations

import os
import pytest

from indw.schedule.monitor.audit import sorted_output_hash
from indw.schedule.core import merge_with_quality
from indw.tools.reports.pipeline_audit import build_pipeline_audit_report
from tests.fixtures.pipeline_corpus import MERGE_PASSAGE_A, lenient_merge_config, write_raw_sources, write_resolved_quality

pytestmark = [pytest.mark.critical, pytest.mark.integration]


def _run_merge(tmp_path, *, workers: int, graph: str = 'v2') -> str:
    from indw.clean.artifact.discovery_engine import reset_discovery_engines
    import os

    os.environ['INSTANT_MERGE_HW_PROBE'] = '0'
    os.environ['INSTANT_PIPELINE_GRAPH'] = graph
    cfg = lenient_merge_config()
    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a',), texts=(MERGE_PASSAGE_A,))
    work = tmp_path / f'w{workers}_{graph}'
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


def test_chain_consolidation_workers_parity(tmp_path):
    h1 = _run_merge(tmp_path / 'a', workers=1)
    h2 = _run_merge(tmp_path / 'b', workers=2)
    assert h1 == h2


def test_pipeline_audit_after_merge(tmp_path):
    from indw.clean.artifact.discovery_engine import reset_discovery_engines
    import os

    os.environ['INSTANT_MERGE_HW_PROBE'] = '0'
    os.environ['INSTANT_PIPELINE_GRAPH'] = 'v2'
    cfg = lenient_merge_config()
    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a',), texts=(MERGE_PASSAGE_A,))
    work = tmp_path / 'audit'
    out = work / 'filtered.jsonl'
    write_resolved_quality(work, cfg)
    reset_discovery_engines()
    merge_with_quality(
        raw_dir, out, quality_config=cfg, work_dir=work,
        fresh=True, resume=False, workers=1, chunk_size=2,
    )
    report = build_pipeline_audit_report(work, workers=1)
    assert report['commodity_count'] >= 8
    assert report['intelligence_count'] >= 6
    assert 'architecture_graph' in report
    assert 'ownership_graph' in report
    cost_path = work / 'pipeline_cost_accounting.json'
    assert cost_path.is_file()
