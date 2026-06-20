from __future__ import annotations

import hashlib
import json

import pytest

from indw.schedule.state.checkpoint import MergeCheckpoint
from indw.schedule.core import merge_with_quality
from tests.fixtures.pipeline_corpus import (
    MERGE_CORPUS,
    lenient_merge_config,
    write_raw_sources,
    write_resolved_quality,
)

pytestmark = pytest.mark.critical

def test_sequential_merge_keeps_documents(tmp_path):
    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a', 'source_b'))
    work = tmp_path / 'work'
    out = work / 'filtered.jsonl'
    cfg = lenient_merge_config()
    write_resolved_quality(work, cfg)
    stats = merge_with_quality(
        raw_dir,
        out,
        quality_config=cfg,
        work_dir=work,
        fresh=True,
        resume=False,
        workers=1,
        checkpoint_interval=1,
    )
    assert stats['kept'] >= 2
    lines = out.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == stats['kept']
    for line in lines:
        row = json.loads(line)
        assert row.get('text')

def test_merge_exact_dedup_across_sources(tmp_path):
    raw_dir = tmp_path / 'raw'
    dup = MERGE_CORPUS[0]
    write_raw_sources(raw_dir, ('source_a',), texts=(dup,))
    write_raw_sources(raw_dir, ('source_b',), texts=(dup,))
    work = tmp_path / 'work'
    out = work / 'filtered.jsonl'
    cfg = lenient_merge_config()
    write_resolved_quality(work, cfg)
    stats = merge_with_quality(
        raw_dir,
        out,
        quality_config=cfg,
        work_dir=work,
        fresh=True,
        resume=False,
        workers=1,
    )
    assert stats['kept'] == 1

def test_merge_empty_batch_source_skipped(tmp_path):
    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a',))
    empty = raw_dir / 'empty_source'
    empty.mkdir()
    (empty / 'data.jsonl').write_text('', encoding='utf-8')
    work = tmp_path / 'work'
    out = work / 'filtered.jsonl'
    cfg = lenient_merge_config()
    write_resolved_quality(work, cfg)
    stats = merge_with_quality(
        raw_dir,
        out,
        quality_config=cfg,
        work_dir=work,
        fresh=True,
        resume=False,
        workers=1,
    )
    assert stats['kept'] >= 1

def test_merge_corrupted_jsonl_skipped(tmp_path):
    raw_dir = tmp_path / 'raw' / 'source_a'
    raw_dir.mkdir(parents=True)
    path = raw_dir / 'data.jsonl'
    path.write_text(
        'not-json\n'
        + json.dumps({'text': MERGE_CORPUS[0]}) + '\n',
        encoding='utf-8',
    )
    work = tmp_path / 'work'
    out = work / 'filtered.jsonl'
    cfg = lenient_merge_config()
    write_resolved_quality(work, cfg)
    stats = merge_with_quality(
        raw_dir.parent,
        out,
        quality_config=cfg,
        work_dir=work,
        fresh=True,
        resume=False,
        workers=1,
    )
    assert stats['kept'] >= 1

@pytest.mark.integration
def test_parallel_merge_matches_sequential_count(tmp_path):
    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a', 'source_b'))
    cfg = lenient_merge_config()

    work_seq = tmp_path / 'seq'
    out_seq = work_seq / 'filtered.jsonl'
    write_resolved_quality(work_seq, cfg)
    seq_stats = merge_with_quality(
        raw_dir,
        out_seq,
        quality_config=cfg,
        work_dir=work_seq,
        fresh=True,
        resume=False,
        workers=1,
    )

    work_par = tmp_path / 'par'
    out_par = work_par / 'filtered.jsonl'
    write_resolved_quality(work_par, cfg)
    par_stats = merge_with_quality(
        raw_dir,
        out_par,
        quality_config=cfg,
        work_dir=work_par,
        fresh=True,
        resume=False,
        workers=2,
        chunk_size=2,
    )
    assert par_stats['kept'] == seq_stats['kept']
    assert par_stats['kept'] >= 2

@pytest.mark.integration
def test_parallel_merge_output_hash_matches_sequential(tmp_path):
    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a', 'source_b'))

    def _hash(workers: int) -> str:
        cfg = lenient_merge_config()
        work = tmp_path / f'w{workers}'
        out = work / 'filtered.jsonl'
        write_resolved_quality(work, cfg)
        merge_with_quality(
            raw_dir, out, quality_config=cfg, work_dir=work,
            fresh=True, resume=False, workers=workers, chunk_size=2,
        )
        texts = [
            json.loads(line)['text']
            for line in out.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        return hashlib.sha256('\n'.join(sorted(texts)).encode()).hexdigest()

    assert _hash(1) == _hash(2)

@pytest.mark.integration
def test_parallel_merge_with_sqlite_dedup_matches_sequential(tmp_path):
    from indw.store.corpus.registry import CorpusRegistry

    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a', 'source_b'))
    cfg = lenient_merge_config()
    cfg.dedup.exact = True

    def _run(workers: int) -> int:
        work = tmp_path / f'dedup_w{workers}'
        out = work / 'filtered.jsonl'
        write_resolved_quality(work, cfg)
        corpus = CorpusRegistry(work)
        try:
            stats = merge_with_quality(
                raw_dir, out, quality_config=cfg, work_dir=work,
                corpus_registry=corpus, fresh=True, resume=False,
                workers=workers, chunk_size=2,
            )
            return int(stats['kept'])
        finally:
            corpus.close()

    assert _run(1) == _run(2)

def test_merge_checkpoint_save_and_resume(tmp_path):
    raw_dir = tmp_path / 'raw'
    texts = tuple(MERGE_CORPUS) * 3
    write_raw_sources(raw_dir, ('source_a',), texts=texts)
    work = tmp_path / 'work'
    out = work / 'filtered.jsonl'
    cfg = lenient_merge_config()
    write_resolved_quality(work, cfg)

    merge_with_quality(
        raw_dir,
        out,
        quality_config=cfg,
        work_dir=work,
        fresh=True,
        resume=False,
        workers=1,
        time_limit_sec=0.001,
        checkpoint_interval=1,
    )
    checkpoint = MergeCheckpoint.load(work)
    assert checkpoint is not None
    partial_kept = checkpoint.totals()['kept']

    stats = merge_with_quality(
        raw_dir,
        out,
        quality_config=cfg,
        work_dir=work,
        fresh=False,
        resume=True,
        workers=1,
        checkpoint_interval=1,
    )
    assert stats['kept'] >= partial_kept
    assert checkpoint.path_for(work).exists()

def test_merge_kept_matches_filtered_lines(tmp_path):
    raw_dir = tmp_path / 'raw'
    write_raw_sources(raw_dir, ('source_a', 'source_b'))
    work = tmp_path / 'work'
    out = work / 'filtered.jsonl'
    cfg = lenient_merge_config()
    write_resolved_quality(work, cfg)
    stats = merge_with_quality(
        raw_dir,
        out,
        quality_config=cfg,
        work_dir=work,
        fresh=True,
        resume=False,
        workers=1,
    )
    from indw.schedule.state.checkpoint import count_jsonl_lines

    assert count_jsonl_lines(out) == stats['kept']

def test_reconcile_checkpoint_output(tmp_path):
    from indw.schedule.state.checkpoint import MergeCheckpoint, reconcile_checkpoint_output

    out = tmp_path / 'filtered.jsonl'
    out.write_text('{"text":"a"}\n{"text":"b"}\n', encoding='utf-8')
    cp = MergeCheckpoint()
    cp.source('src').kept = 10
    result = reconcile_checkpoint_output(cp, out)
    assert result['adjusted'] == 8
    assert cp.totals()['kept'] == 2

def test_reconcile_checkpoint_output_file_above_kept(tmp_path):
    from indw.schedule.state.checkpoint import MergeCheckpoint, reconcile_checkpoint_output

    out = tmp_path / 'filtered.jsonl'
    out.write_text('{"text":"a"}\n{"text":"b"}\n{"text":"c"}\n', encoding='utf-8')
    cp = MergeCheckpoint()
    row = cp.source('src')
    row.kept = 1
    row.scanned = 2
    row.rejected = 1
    result = reconcile_checkpoint_output(cp, out)
    assert result['adjusted'] == 2
    assert cp.totals()['kept'] == 3
    assert cp.totals()['scanned'] == 4
