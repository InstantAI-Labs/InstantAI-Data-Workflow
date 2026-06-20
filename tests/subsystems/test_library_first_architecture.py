from __future__ import annotations

import pytest

from indw.store.io.json_codec import dumps_pretty, loads
from indw.store.io.jsonl import parse_jsonl_line

pytestmark = pytest.mark.critical


def test_parse_jsonl_line_owner():
    kind, row = parse_jsonl_line('{"text": "hello"}')
    assert kind == 'ok'
    assert row['text'] == 'hello'


def test_parse_merge_delegates_to_jsonl_owner():
    from indw.schedule.read.gates import parse_merge_jsonl_line
    kind, row = parse_merge_jsonl_line('{"a": 1}')
    assert kind == 'ok'
    assert row == {'a': 1}


def test_checkpoint_orjson_roundtrip(tmp_path):
    from indw.schedule.state.checkpoint import MergeCheckpoint, SourceCheckpoint

    cp = MergeCheckpoint(
        sources={'src': SourceCheckpoint(line_offset=3, scanned=10, kept=2, rejected=8)},
        domain_counts={'web': 2},
    )
    cp.save(tmp_path)
    loaded = MergeCheckpoint.load(tmp_path)
    assert loaded is not None
    assert loaded.sources['src'].line_offset == 3
    assert loaded.domain_counts == {'web': 2}


def test_library_classification():
    from indw.schedule.architecture.classify import commodity_stages, intelligence_stages

    commodity = {r.stage for r in commodity_stages()}
    intel = {r.stage for r in intelligence_stages()}
    assert 's2_doc_dedup' in commodity
    assert 's4_intel_preview' in intel
    assert 'knowledge_extraction' in intel


def test_migration_report_builds():
    from indw.tools.reports.library_migration import build_library_migration_report

    report = build_library_migration_report(workers=4)
    assert report['json_backend'] == 'orjson'
    assert len(report['custom_intelligence_preserved']) >= 8
    assert report['horizontal_execution_graph']['mode'] == 'canonical_graph'
