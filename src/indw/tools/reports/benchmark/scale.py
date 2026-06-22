from __future__ import annotations

import json
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from indw.config.resolve import PipelineConfigContext
from indw.tools.metrics.pipeline_health import append_benchmark_history, load_benchmark_history, record_pipeline_health
from indw.filter.spec.quality import QualityPipelineConfig
from indw.filter.gate.quality import QualityGate
from indw.schedule.core import merge_with_quality

GB = 1024 ** 3

_rss_reader: Any = 'pending'


def peak_rss_mb() -> float:
    global _rss_reader
    if _rss_reader == 'pending':
        try:
            import resource

            def _read() -> float:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform == 'darwin':
                    return rss / (1024 * 1024)
                return rss / 1024

            _rss_reader = _read
        except Exception:
            _rss_reader = lambda: 0.0
    return float(_rss_reader())

def _bench_ctx() -> PipelineConfigContext:
    ctx = PipelineConfigContext.resolve()
    cfg = ctx.quality
    cfg.cleaning.enabled = False
    cfg.synthetic_defense.enabled = False
    cfg.curriculum.enabled = False
    cfg.dedup.exact = False
    cfg.dedup.fuzzy = False
    cfg.dedup.semantic = False
    cfg.balance.enabled = False
    cfg.thresholds.min_score = 0.0
    return ctx.with_quality(cfg)

def _write_bench_quality(work_dir: Path, cfg: QualityPipelineConfig) -> Path:
    from dataclasses import asdict

    import yaml

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / '_resolved_quality.yaml'
    payload = {
        'enabled': cfg.enabled,
        'thresholds': asdict(cfg.thresholds),
        'dedup': asdict(cfg.dedup),
        'balance': asdict(cfg.balance),
        'cleaning': asdict(cfg.cleaning),
        'synthetic_defense': asdict(cfg.synthetic_defense),
        'curriculum': asdict(cfg.curriculum),
        'orchestration': cfg.orchestration or {'enabled': False},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding='utf-8')
    return path

@dataclass
class ScaleBenchmarkResult:
    tier: str
    target_bytes: int
    actual_bytes: int
    doc_count: int
    elapsed_sec: float
    docs_per_sec: float
    bytes_per_sec: float
    peak_rss_mb: float
    kept: int
    rejected: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'tier': self.tier,
            'target_bytes': self.target_bytes,
            'actual_bytes': self.actual_bytes,
            'corpus_gb': round(self.actual_bytes / GB, 4),
            'doc_count': self.doc_count,
            'elapsed_sec': round(self.elapsed_sec, 3),
            'docs_per_sec': round(self.docs_per_sec, 3),
            'bytes_per_sec': round(self.bytes_per_sec, 1),
            'peak_rss_mb': round(self.peak_rss_mb, 2),
            'kept': self.kept,
            'rejected': self.rejected,
        }

@dataclass
class PipelineScaleBenchmarkResult:
    tier: str
    target_bytes: int
    actual_bytes: int
    gate: ScaleBenchmarkResult
    merge_elapsed_sec: float
    merge_docs_per_sec: float
    merge_kept: int
    merge_rejected: int
    merge_scanned: int
    export_elapsed_sec: float = 0.0
    export_tokens_per_sec: float = 0.0
    shards_written: int = 0
    tokens_exported: int = 0
    checkpoint_recoverable: bool = False
    peak_rss_mb: float = 0.0
    workers: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'tier': self.tier,
            'target_bytes': self.target_bytes,
            'actual_bytes': self.actual_bytes,
            'corpus_gb': round(self.actual_bytes / GB, 4),
            'docs_per_sec': round(self.merge_docs_per_sec, 3),
            'workers': self.workers,
            'peak_rss_mb': round(self.peak_rss_mb, 2),
            'gate': self.gate.to_dict(),
            'merge_elapsed_sec': round(self.merge_elapsed_sec, 3),
            'merge_docs_per_sec': round(self.merge_docs_per_sec, 3),
            'merge_kept': self.merge_kept,
            'merge_rejected': self.merge_rejected,
            'merge_scanned': self.merge_scanned,
            'export_elapsed_sec': round(self.export_elapsed_sec, 3),
            'export_tokens_per_sec': round(self.export_tokens_per_sec, 1),
            'shards_written': self.shards_written,
            'tokens_exported': self.tokens_exported,
            'checkpoint_recoverable': self.checkpoint_recoverable,
            **self.extra,
        }

def write_streaming_corpus(
    path: Path,
    target_bytes: int,
    *,
    template: str,
    source: str = 'bench',
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    count = 0
    with path.open('w', encoding='utf-8') as fh:
        while written < target_bytes:
            row = {'text': f'{template} doc={count}', 'source': source}
            line = json.dumps(row, ensure_ascii=False) + '\n'
            fh.write(line)
            written += len(line.encode('utf-8'))
            count += 1
    return count

def iter_corpus_rows(path: Path, *, max_docs: int | None = None) -> Iterator[dict[str, Any]]:
    count = 0
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if max_docs is not None and count >= max_docs:
                break
            yield json.loads(line)
            count += 1

def run_gate_scale_benchmark(
    work_dir: Path,
    *,
    target_gb: float,
    template: str,
    tier: str | None = None,
    history_dir: Path | None = None,
    max_docs: int | None = None,
) -> ScaleBenchmarkResult:
    target_bytes = int(target_gb * GB)
    tier_name = tier or f'{int(target_gb)}gb'
    corpus_path = work_dir / f'{tier_name}.jsonl'
    corpus_docs = write_streaming_corpus(corpus_path, target_bytes, template=template)
    actual_bytes = corpus_path.stat().st_size

    ctx = _bench_ctx()
    gate = QualityGate(ctx=ctx)

    tracemalloc.start()
    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    kept = rejected = 0
    processed = 0
    for row in iter_corpus_rows(corpus_path, max_docs=max_docs):
        ok, _ = gate.evaluate(row['text'], source=row.get('source', 'bench'))
        processed += 1
        if ok:
            kept += 1
        else:
            rejected += 1
    elapsed = time.perf_counter() - t0
    _, trace_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_peak = max(peak_rss_mb(), rss_before, trace_peak / (1024 * 1024))

    result = ScaleBenchmarkResult(
        tier=tier_name,
        target_bytes=target_bytes,
        actual_bytes=actual_bytes,
        doc_count=processed,
        elapsed_sec=elapsed,
        docs_per_sec=processed / max(elapsed, 1e-6),
        bytes_per_sec=actual_bytes / max(elapsed, 1e-6),
        peak_rss_mb=rss_peak,
        kept=kept,
        rejected=rejected,
    )
    if history_dir is not None:
        append_benchmark_history(history_dir, result.to_dict())
        record_pipeline_health(
            history_dir,
            gate_stats={'kept': kept, 'rejected': rejected},
            benchmark_stats=result.to_dict(),
        )
    return result

def run_pipeline_scale_benchmark(
    work_dir: Path,
    *,
    target_gb: float,
    template: str,
    tier: str | None = None,
    history_dir: Path | None = None,
    max_docs: int | None = None,
    workers: int = 1,
    run_export: bool = False,
    tokenizer_path: Path | None = None,
) -> PipelineScaleBenchmarkResult:
    target_bytes = int(target_gb * GB)
    tier_name = tier or f'{int(target_gb)}gb'
    bench_root = work_dir / tier_name
    raw_dir = bench_root / 'raw' / 'bench'
    raw_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = raw_dir / 'data.jsonl'
    doc_count = write_streaming_corpus(corpus_path, target_bytes, template=template)
    actual_bytes = corpus_path.stat().st_size

    ctx = _bench_ctx()
    cfg = ctx.quality
    merge_work = bench_root / 'merge'
    merge_work.mkdir(parents=True, exist_ok=True)
    _write_bench_quality(merge_work, cfg)
    out_path = merge_work / 'filtered.jsonl'

    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    merge_stats = merge_with_quality(
        raw_dir.parent,
        out_path,
        quality_config=cfg,
        work_dir=merge_work,
        fresh=True,
        resume=False,
        workers=workers,
    )
    merge_elapsed = time.perf_counter() - t0
    merge_scanned = int(merge_stats.get('scanned', doc_count))
    merge_kept = int(merge_stats.get('kept', 0))
    merge_rejected = int(merge_stats.get('rejected', 0))

    from indw.tools.metrics.pipeline_health import checkpoint_stats_from_path

    cp_stats = checkpoint_stats_from_path(merge_work)
    checkpoint_recoverable = bool(cp_stats.get('recoverable'))

    export_elapsed = 0.0
    tokens_exported = 0
    shards_written = 0
    if run_export and tokenizer_path is not None and out_path.exists():
        from indw.store.export.fast_export import export_token_bins_fast

        export_dir = bench_root / 'shards'
        t1 = time.perf_counter()
        export_out = export_token_bins_fast(
            out_path,
            tokenizer_path,
            export_dir,
            shard_tokens=8192,
            val_ratio=0.1,
            flush_tokens=2048,
        )
        export_elapsed = time.perf_counter() - t1
        stats = export_out.get('_export_stats', {})
        tokens_exported = int(stats.get('tokens_exported', 0))
        shards_written = int(stats.get('shards_written', 0))

    rss_peak = max(peak_rss_mb(), rss_before)
    gate_result = ScaleBenchmarkResult(
        tier=f'{tier_name}_gate',
        target_bytes=target_bytes,
        actual_bytes=actual_bytes,
        doc_count=merge_scanned,
        elapsed_sec=merge_elapsed,
        docs_per_sec=merge_scanned / max(merge_elapsed, 1e-6),
        bytes_per_sec=actual_bytes / max(merge_elapsed, 1e-6),
        peak_rss_mb=rss_peak,
        kept=merge_kept,
        rejected=merge_rejected,
    )

    result = PipelineScaleBenchmarkResult(
        tier=tier_name,
        target_bytes=target_bytes,
        actual_bytes=actual_bytes,
        gate=gate_result,
        merge_elapsed_sec=merge_elapsed,
        merge_docs_per_sec=merge_scanned / max(merge_elapsed, 1e-6),
        merge_kept=merge_kept,
        merge_rejected=merge_rejected,
        merge_scanned=merge_scanned,
        export_elapsed_sec=export_elapsed,
        export_tokens_per_sec=tokens_exported / max(export_elapsed, 1e-6),
        shards_written=shards_written,
        tokens_exported=tokens_exported,
        checkpoint_recoverable=checkpoint_recoverable,
        peak_rss_mb=rss_peak,
        workers=workers,
        extra={'worker_failures': int(merge_stats.get('worker_failures', 0))},
    )
    if history_dir is not None:
        append_benchmark_history(history_dir, result.to_dict())
        record_pipeline_health(
            history_dir,
            gate_stats={'kept': gate_result.kept, 'rejected': gate_result.rejected},
            merge_stats={
                'kept': merge_kept,
                'rejected': merge_rejected,
                'scanned': merge_scanned,
                'worker_failures': int(merge_stats.get('worker_failures', 0)),
                'resumed': bool(merge_stats.get('resumed', False)),
            },
            export_stats={
                'shards_written': shards_written,
                'tokens_exported': tokens_exported,
            },
            checkpoint_stats=cp_stats,
            benchmark_stats={
                'tier': tier_name,
                'docs_per_sec': result.merge_docs_per_sec,
                'peak_rss_mb': rss_peak,
                'corpus_gb': actual_bytes / GB,
            },
        )
    return result

def detect_benchmark_regression(
    history_dir: Path,
    tier: str,
    *,
    metric: str = 'docs_per_sec',
    min_rate: float = 0.1,
    max_drop_ratio: float = 0.5,
) -> tuple[bool, str]:
    rows = [r for r in load_benchmark_history(history_dir) if r.get('tier') == tier]
    if not rows:
        return False, 'no_history'
    latest = rows[-1]
    rate = float(_metric_value(latest, metric))
    if rate < min_rate:
        return True, f'below_floor:{rate:.3f}<{min_rate}'
    if len(rows) >= 2:
        prev = float(_metric_value(rows[-2], metric))
        if prev > 0 and rate < prev * (1.0 - max_drop_ratio):
            return True, f'regression:{rate:.3f}<{prev * (1.0 - max_drop_ratio):.3f}'
    return False, 'ok'

def _metric_value(row: dict[str, Any], metric: str) -> float:
    if metric in row:
        return float(row[metric])
    gate = row.get('gate')
    if isinstance(gate, dict) and metric in gate:
        return float(gate[metric])
    nested = {
        'merge_docs_per_sec': 'merge_docs_per_sec',
        'export_tokens_per_sec': 'export_tokens_per_sec',
    }
    key = nested.get(metric, metric)
    if key in row:
        return float(row[key])
    return 0.0

def certify_benchmark_tier(
    history_dir: Path,
    tier: str,
    *,
    metrics: tuple[tuple[str, float], ...] = (
        ('docs_per_sec', 0.05),
        ('merge_docs_per_sec', 0.02),
    ),
    max_drop_ratio: float = 0.5,
) -> tuple[bool, list[str]]:
    rows = [r for r in load_benchmark_history(history_dir) if r.get('tier') == tier]
    if not rows:
        return False, [f'{tier}:no_history']
    latest = rows[-1]
    failures: list[str] = []
    for metric, floor in metrics:
        rate = _metric_value(latest, metric)
        if rate <= 0:
            continue
        regressed, reason = detect_benchmark_regression(
            history_dir, tier, metric=metric, min_rate=floor, max_drop_ratio=max_drop_ratio,
        )
        if regressed:
            failures.append(f'{tier}/{metric}:{reason}')
    return (not failures), failures


def profile_merge_io_hotspots(
    raw_dir: Path,
    *,
    workers: int = 1,
    max_docs: int = 200,
) -> dict[str, float]:
    import cProfile
    import pstats
    from io import StringIO

    from indw.tools.reports.benchmark.scale import _bench_ctx, _write_bench_quality

    from indw.schedule.core import merge_with_quality

    raw_dir = Path(raw_dir)
    cfg = _bench_ctx().quality
    merge_work = raw_dir.parent / '_profile_merge'
    merge_work.mkdir(parents=True, exist_ok=True)
    _write_bench_quality(merge_work, cfg)
    out_path = merge_work / 'filtered.jsonl'
    profiler = cProfile.Profile()
    profiler.enable()
    merge_with_quality(
        raw_dir,
        out_path,
        quality_config=cfg,
        work_dir=merge_work,
        fresh=True,
        resume=False,
        workers=workers,
    )
    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(25)
    return {
        'workers': float(workers),
        'max_docs': float(max_docs),
        'profile_top': stream.getvalue()[:4000],
    }
