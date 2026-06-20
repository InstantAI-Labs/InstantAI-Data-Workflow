from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

@dataclass
class PipelineHealthSnapshot:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    gate: dict[str, Any] = field(default_factory=dict)
    merge: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)
    adaptive: dict[str, Any] = field(default_factory=dict)
    recovery: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'gate': self.gate,
            'merge': self.merge,
            'export': self.export,
            'checkpoint': self.checkpoint,
            'benchmark': self.benchmark,
            'adaptive': self.adaptive,
            'recovery': self.recovery,
        }

    @property
    def healthy(self) -> bool:
        if self.checkpoint.get('corrupt'):
            return False
        if self.export.get('checksum_failures', 0) > 0:
            return False
        if self.merge.get('worker_failures', 0) > 0 and not self.checkpoint.get('recoverable'):
            return False
        return True

def build_pipeline_health(
    *,
    gate_stats: Optional[dict[str, Any]] = None,
    merge_stats: Optional[dict[str, Any]] = None,
    export_stats: Optional[dict[str, Any]] = None,
    checkpoint_stats: Optional[dict[str, Any]] = None,
    benchmark_stats: Optional[dict[str, Any]] = None,
    adaptive_stats: Optional[dict[str, Any]] = None,
    recovery_stats: Optional[dict[str, Any]] = None,
) -> PipelineHealthSnapshot:
    gate = gate_stats or {}
    merge = merge_stats or {}
    export = export_stats or {}
    checkpoint = checkpoint_stats or {}
    benchmark = benchmark_stats or {}
    adaptive = adaptive_stats or {}
    recovery = recovery_stats or {}
    return PipelineHealthSnapshot(
        gate={
            'kept': int(gate.get('kept', 0)),
            'rejected': int(gate.get('rejected', 0)),
            'reject_reasons': dict(gate.get('reject_reasons') or {}),
            'score_mean': float(gate.get('score_mean', 0.0)),
        },
        merge={
            'kept': int(merge.get('kept', 0)),
            'rejected': int(merge.get('rejected', 0)),
            'scanned': int(merge.get('scanned', 0)),
            'worker_failures': int(merge.get('worker_failures', 0)),
            'resumed': bool(merge.get('resumed', False)),
        },
        export={
            'shards_written': int(export.get('shards_written', 0)),
            'tokens_exported': int(export.get('tokens_exported', 0)),
            'checksum_failures': int(export.get('checksum_failures', 0)),
            'partial': bool(export.get('partial', False)),
        },
        checkpoint={
            'exists': bool(checkpoint.get('exists', False)),
            'recoverable': bool(checkpoint.get('recoverable', False)),
            'corrupt': bool(checkpoint.get('corrupt', False)),
            'last_batch': int(checkpoint.get('last_batch', 0)),
        },
        benchmark={
            'docs_per_sec': float(benchmark.get('docs_per_sec', 0.0)),
            'peak_rss_mb': float(benchmark.get('peak_rss_mb', 0.0)),
            'corpus_gb': float(benchmark.get('corpus_gb', 0.0)),
            'tier': str(benchmark.get('tier', '')),
        },
        adaptive={
            'calibration_samples': int(adaptive.get('calibration_samples', 0)),
            'drift_score': float(adaptive.get('drift_score', 0.0)),
            'threshold_shift': float(adaptive.get('threshold_shift', 0.0)),
        },
        recovery={
            'total_events': int(recovery.get('total_events', 0)),
            'worker_crashes': int(recovery.get('worker_crashes', 0)),
            'sqlite_retries': int(recovery.get('sqlite_retries', 0)),
            'disk_failures': int(recovery.get('disk_failures', 0)),
            'checkpoint_corrupt': int(recovery.get('checkpoint_corrupt', 0)),
            'checkpoint_recoveries': int(recovery.get('checkpoint_recoveries', 0)),
            'export_failures': int(recovery.get('export_failures', 0)),
            'stream_interruptions': int(recovery.get('stream_interruptions', 0)),
            'by_type': dict(recovery.get('by_type') or {}),
        },
    )

def append_benchmark_history(output_dir: Path, entry: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'benchmark_history.jsonl'
    row = {'recorded_at': datetime.now(timezone.utc).isoformat(), **entry}
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, default=str) + '\n')
    return path

def write_health_dashboard(
    output_dir: Path,
    health: PipelineHealthSnapshot,
    *,
    recovery_events: list[dict[str, Any]] | None = None,
) -> Path:
    from indw.store.io.atomic import atomic_write_text
    from indw.tools.metrics.recovery import load_recovery_events

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'pipeline_health.json'
    if recovery_events is None:
        recovery_events = load_recovery_events(output_dir, limit=20)
    else:
        recovery_events = recovery_events[-20:]
    payload = {
        'status': 'healthy' if health.healthy else 'degraded',
        'snapshot': health.to_dict(),
        'benchmark_history': load_benchmark_history(output_dir, limit=20),
        'recovery_events': recovery_events,
    }
    atomic_write_text(path, json.dumps(payload, indent=2))
    return path

def load_benchmark_history(output_dir: Path, *, limit: int = 100) -> list[dict[str, Any]]:
    from indw.tools.metrics.recovery import load_jsonl_tail

    return load_jsonl_tail(Path(output_dir) / 'benchmark_history.jsonl', limit=limit)

def gate_stats_from_gate(gate: Any) -> dict[str, Any]:
    calibration = (
        gate.calibrator.distribution_stats()
        if hasattr(gate, 'calibrator') and gate.calibrator is not None
        else {}
    )
    return gate.stats.to_dict(calibration=calibration)

def checkpoint_stats_from_path(merge_work: Path) -> dict[str, Any]:
    from indw.schedule.state.checkpoint import CHECKPOINT_NAME, MergeCheckpoint

    path = Path(merge_work) / CHECKPOINT_NAME
    if not path.exists():
        return {'exists': False, 'recoverable': False, 'corrupt': False, 'last_batch': 0}
    cp = MergeCheckpoint.load(merge_work)
    if cp is None:
        return {'exists': True, 'recoverable': False, 'corrupt': True, 'last_batch': 0}
    totals = cp.totals()
    return {
        'exists': True,
        'recoverable': bool(cp.complete or cp.interrupted),
        'corrupt': False,
        'last_batch': int(totals.get('scanned', 0)),
    }

def record_pipeline_health(
    output_dir: Path,
    *,
    gate_stats: Optional[dict[str, Any]] = None,
    merge_stats: Optional[dict[str, Any]] = None,
    export_stats: Optional[dict[str, Any]] = None,
    checkpoint_stats: Optional[dict[str, Any]] = None,
    benchmark_stats: Optional[dict[str, Any]] = None,
    adaptive_stats: Optional[dict[str, Any]] = None,
    recovery_stats: Optional[dict[str, Any]] = None,
) -> Path:
    from indw.tools.metrics.recovery import load_recovery_events, recovery_stats_from_events

    events = load_recovery_events(output_dir, limit=200)
    if recovery_stats is None:
        recovery_stats = recovery_stats_from_events(events)
    health = build_pipeline_health(
        gate_stats=gate_stats,
        merge_stats=merge_stats,
        export_stats=export_stats,
        checkpoint_stats=checkpoint_stats,
        benchmark_stats=benchmark_stats,
        adaptive_stats=adaptive_stats,
        recovery_stats=recovery_stats,
    )
    return write_health_dashboard(output_dir, health, recovery_events=events)
