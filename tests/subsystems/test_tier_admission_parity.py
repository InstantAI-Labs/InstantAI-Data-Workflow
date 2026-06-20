from __future__ import annotations

import os

import pytest

from indw.clean.gate.evaluate import document_gate_raw, evaluate_document_gate
from indw.schedule.monitor.audit import sorted_output_hash
from indw.schedule.core import merge_with_quality
from tests.fixtures.pipeline_corpus import MERGE_PASSAGE_A, lenient_merge_config, write_raw_sources, write_resolved_quality

pytestmark = [pytest.mark.critical, pytest.mark.property, pytest.mark.integration]


def _run_merge(tmp_path, *, workers: int, graph: str = 'v2') -> str:
    from indw.clean.artifact.discovery_engine import reset_discovery_engines

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


def test_tier_admission_workers_parity(tmp_path):
    h1 = _run_merge(tmp_path / 'a', workers=1)
    h2 = _run_merge(tmp_path / 'b', workers=2)
    h4 = _run_merge(tmp_path / 'c', workers=4)
    assert h1 == h2 == h4


def test_gate_raw_reuse_matches_full_extract():
    text = (
        'Gradient descent optimizes parameters by following the negative gradient of a loss function. '
        'Stochastic variants sample mini-batches to reduce per-step cost while preserving convergence '
        'properties under appropriate learning-rate schedules.'
    ) * 6
    raw = document_gate_raw(text)
    full = evaluate_document_gate(text)
    reused = evaluate_document_gate(text, raw=raw)
    assert full.keep == reused.keep
    assert full.reason == reused.reason


def test_tier01_rejects_short_text():
    from indw.schedule.admission.tier01 import run_tier01_gates
    from indw.schedule.state.context import MergeDocumentContext
    from indw.filter.gate.quality import QualityGate

    gate = QualityGate()
    ctx = MergeDocumentContext(
        seq=0,
        src_name='test.jsonl',
        line_no=0,
        text='x' * 20,
        meaningful_chars=20,
    )
    reason = run_tier01_gates(ctx, gate=gate, src_name='test.jsonl')
    assert reason is not None
