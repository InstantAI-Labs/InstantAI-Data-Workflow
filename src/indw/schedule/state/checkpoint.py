from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

from indw.store.io.json_codec import dumps_canonical, dumps_pretty, loads

if TYPE_CHECKING:
    from indw.filter.gate.quality import QualityGate

from indw.config.defaults import (
    MERGE_CHECKPOINT_FORMAT_VERSION as CHECKPOINT_VERSION,
    MERGE_METRICS_SNAPSHOT_INTERVAL as METRICS_SNAPSHOT_INTERVAL,
    MERGE_PROGRESS_LOG_INTERVAL as PROGRESS_LOG_INTERVAL,
)

CHECKPOINT_NAME = 'merge_checkpoint.json'
RUN_PROGRESS_NAME = 'pipeline_progress.json'


def resolve_metrics_snapshot_interval_sec() -> float:
    from indw.schedule.config.policy import active_or_built_policy
    return active_or_built_policy().metrics_snapshot_sec


@dataclass
class SourceCheckpoint:
    line_offset: int = 0
    scanned: int = 0
    kept: int = 0
    rejected: int = 0


@dataclass
class MergeCheckpoint:
    version: int = CHECKPOINT_VERSION
    complete: bool = False
    interrupted: bool = False
    sources: dict[str, SourceCheckpoint] = field(default_factory=dict)
    domain_counts: dict[str, int] = field(default_factory=dict)
    language_counts: dict[str, int] = field(default_factory=dict)
    updated_at: str = ''
    quality_config_fingerprint: str = ''
    sources_config_fingerprint: str = ''
    filtered_line_count: int = 0
    adaptive_calibrator_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def path_for(cls, work_dir: Path) -> Path:
        return Path(work_dir) / CHECKPOINT_NAME

    @classmethod
    def load(cls, work_dir: Path) -> Optional['MergeCheckpoint']:
        path = cls.path_for(work_dir)
        if not path.exists():
            return None
        try:
            raw = loads(path.read_text(encoding='utf-8'))
            return cls._from_raw(raw)
        except (ValueError, OSError, TypeError):
            from indw.tools.metrics.recovery import record_recovery_event
            record_recovery_event(work_dir, 'checkpoint_corrupt', path=str(path))
            bak = path.with_name(f'{path.name}.bak')
            if bak.exists():
                try:
                    loaded = cls._from_raw(loads(bak.read_text(encoding='utf-8')))
                    record_recovery_event(work_dir, 'checkpoint_recovered', source='backup')
                    return loaded
                except (ValueError, OSError, TypeError):
                    pass
            return None

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> 'MergeCheckpoint':
        sources: dict[str, SourceCheckpoint] = {}
        for name, row in (raw.get('sources') or {}).items():
            if not isinstance(row, dict):
                continue
            sources[name] = SourceCheckpoint(
                line_offset=int(row.get('line_offset', 0)),
                scanned=int(row.get('scanned', 0)),
                kept=int(row.get('kept', 0)),
                rejected=int(row.get('rejected', 0)),
            )
        return cls(
            version=int(raw.get('version', CHECKPOINT_VERSION)),
            complete=bool(raw.get('complete', False)),
            interrupted=bool(raw.get('interrupted', False)),
            sources=sources,
            domain_counts={
                str(k): int(v)
                for k, v in (raw.get('domain_counts') or {}).items()
                if int(v) > 0
            },
            language_counts={
                str(k): int(v)
                for k, v in (raw.get('language_counts') or {}).items()
                if int(v) > 0
            },
            updated_at=str(raw.get('updated_at', '')),
            quality_config_fingerprint=str(raw.get('quality_config_fingerprint', '')),
            sources_config_fingerprint=str(raw.get('sources_config_fingerprint', '')),
            filtered_line_count=int(raw.get('filtered_line_count', 0)),
            adaptive_calibrator_state=dict(raw.get('adaptive_calibrator_state') or {}),
        )

    def save(
        self,
        work_dir: Path,
        *,
        interrupted: bool = False,
        gate: Optional['QualityGate'] = None,
    ) -> Path:
        self.interrupted = interrupted
        if gate is not None:
            self.domain_counts = gate.domain_balancer.counts()
            self.language_counts = gate.lang_balancer.counts()
        self.updated_at = datetime.now(timezone.utc).isoformat()
        from indw.store.io.atomic import DiskFullError, atomic_write_text
        from indw.tools.metrics.recovery import record_recovery_event

        path = self.path_for(work_dir)
        try:
            atomic_write_text(path, dumps_pretty(asdict(self)))
        except DiskFullError as exc:
            record_recovery_event(work_dir, 'disk_full', path=str(path), phase='checkpoint_save')
            raise
        return path

    def source(self, name: str) -> SourceCheckpoint:
        if name not in self.sources:
            self.sources[name] = SourceCheckpoint()
        return self.sources[name]

    def line_offset(self, name: str) -> int:
        return self.source(name).line_offset

    def totals(self) -> dict[str, int]:
        scanned = kept = rejected = 0
        for row in self.sources.values():
            scanned += row.scanned
            kept += row.kept
            rejected += row.rejected
        return {'scanned': scanned, 'kept': kept, 'rejected': rejected}

    def prune_sources(self, valid_names: set[str]) -> list[str]:
        removed = [name for name in self.sources if name not in valid_names]
        for name in removed:
            del self.sources[name]
        return removed


def clear_merge_outputs(work_dir: Path) -> list[str]:
    import logging
    import os
    import time

    from indw.clean.artifact.discovery_engine import reset_discovery_engines
    from indw.schedule.state.lock import read_merge_run_lock, _pid_alive

    log = logging.getLogger(__name__)
    reset_discovery_engines()
    removed: list[str] = []

    def _holder_hint() -> str:
        lock = read_merge_run_lock(work_dir)
        if not lock:
            return ''
        pid = int(lock.get('pid', 0) or 0)
        if pid and pid != os.getpid() and _pid_alive(pid):
            return (
                f' Active merge pid={pid} owner={lock.get("owner")}.'
                ' Stop that process or use a new validation --run-id.'
            )
        return ''

    def _remove_file(path: Path, *, label: str) -> None:
        if not path.exists():
            return

        def _unlink() -> None:
            path.unlink()

        try:
            from indw.store.io.retry import retry_permission_denied
            retry_permission_denied(_unlink, attempts=6, backoff_sec=0.15)
            removed.append(label)
            return
        except PermissionError as last_exc:
            hint = _holder_hint()
            log.error(
                'Cannot delete %s — file locked by another process.%s',
                path,
                hint,
            )
            raise PermissionError(
                f'Cannot delete {path}: file locked by another process.{hint}'
            ) from last_exc

    work_dir = Path(work_dir)
    for rel in (
        CHECKPOINT_NAME,
        'filtered.jsonl',
        'filtered.mixture_index.jsonl',
        RUN_PROGRESS_NAME,
    ):
        _remove_file(work_dir / rel, label=rel)
    from indw.dedup.storage import unlink_sqlite_files

    for rel_path in (
        Path('corpus') / 'dedup_index.sqlite',
        Path('artifact_discovery.sqlite'),
        Path('clean_pass_dedup.sqlite'),
        Path('refine_dedup.sqlite'),
    ):
        db_path = Path(work_dir) / rel_path
        if not db_path.exists():
            continue
        try:
            for name in unlink_sqlite_files(db_path):
                tag = rel_path.as_posix() if name == db_path.name else f'{rel_path.name}:{name}'
                if tag not in removed:
                    removed.append(tag)
        except PermissionError:
            log.error(
                'Cannot delete %s — file locked by another process. '
                'Stop other prepare/metrics processes and retry --fresh-merge.',
                db_path,
            )
            raise
    return removed


def restore_gate_balancers(gate: 'QualityGate', index_path: Path) -> int:
    if not index_path.exists():
        return 0
    domain_counts: dict[str, int] = defaultdict(int)
    lang_counts: dict[str, int] = defaultdict(int)
    rows = 0
    with index_path.open(encoding='utf-8') as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                row = loads(line)
            except ValueError:
                continue
            domain_counts[str(row.get('domain') or 'web')] += 1
            lang_counts[str(row.get('language') or 'en')] += 1
            rows += 1
    if rows <= 0:
        return 0
    gate.domain_balancer.seed(dict(domain_counts))
    gate.lang_balancer.seed(dict(lang_counts))
    return rows


def restore_balancers_from_checkpoint(gate: 'QualityGate', checkpoint: MergeCheckpoint) -> bool:
    restored = False
    if checkpoint.domain_counts:
        gate.domain_balancer.seed(checkpoint.domain_counts)
        restored = True
    if checkpoint.language_counts:
        gate.lang_balancer.seed(checkpoint.language_counts)
        restored = True
    return restored


def count_jsonl_lines(path: Path) -> int:
    from indw.store.io.jsonl import count_jsonl_lines as _count_jsonl_lines

    return _count_jsonl_lines(path)


def resolve_merge_checkpoint_interval(explicit: int | None = None) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    from indw.schedule.config.policy import active_or_built_policy
    return active_or_built_policy().checkpoint_interval


def resolve_merge_checkpoint_save_sec(explicit: float | None = None) -> float:
    if explicit is not None:
        return max(1.0, float(explicit))
    from indw.schedule.config.policy import active_or_built_policy
    return active_or_built_policy().checkpoint_save_sec


def make_merge_checkpoint_flusher(
    *,
    checkpoint: MergeCheckpoint,
    merge_work: Path,
    gate: Optional['QualityGate'],
    index_file: Any = None,
    index: Any = None,
    out_path: Path | None = None,
    min_save_interval_sec: float | None = None,
) -> Callable[[], None]:
    save_interval = resolve_merge_checkpoint_save_sec(min_save_interval_sec)
    state = {'last_save': 0.0}

    def _flush_checkpoint() -> None:
        if index_file is not None:
            index_file.flush()
        if index is not None:
            index.flush()
        now = time.monotonic()
        if now - state['last_save'] < save_interval:
            return
        state['last_save'] = now
        checkpoint.filtered_line_count = int(checkpoint.totals().get('kept', 0))
        if gate is not None and hasattr(gate, 'calibrator'):
            checkpoint.adaptive_calibrator_state = gate.calibrator.export_state()
        checkpoint.save(merge_work, gate=gate)

    return _flush_checkpoint


def reconcile_checkpoint_output(
    checkpoint: MergeCheckpoint,
    out_path: Path,
    *,
    logger: Any = None,
) -> dict[str, int]:
    file_lines = count_jsonl_lines(out_path)
    totals = checkpoint.totals()
    cp_kept = int(totals.get('kept', 0))
    if file_lines == cp_kept:
        return {'file_lines': file_lines, 'checkpoint_kept': cp_kept, 'adjusted': 0}
    log = logger or __import__('logging').getLogger(__name__)
    if file_lines == 0 and cp_kept > 0:
        from indw.schedule.monitor.invariants import MergeAccountingError
        raise MergeAccountingError(
            f'filtered.jsonl is empty but checkpoint kept={cp_kept}; '
            'output was lost. Delete merge_checkpoint.json and rerun without --resume, '
            'or restore filtered.jsonl from backup.'
        )
    if 0 < file_lines < cp_kept:
        log.warning(
            'filtered.jsonl lines (%d) < checkpoint kept (%d); reconciling counters',
            file_lines,
            cp_kept,
        )
        names = list(checkpoint.sources.keys())
        running = 0
        for i, name in enumerate(names):
            row = checkpoint.source(name)
            if i == len(names) - 1:
                row.kept = max(0, file_lines - running)
            else:
                share = int(row.kept * file_lines / cp_kept) if cp_kept else 0
                row.kept = share
                running += share
        return {'file_lines': file_lines, 'checkpoint_kept': cp_kept, 'adjusted': cp_kept - file_lines}
    if file_lines > cp_kept:
        log.warning(
            'filtered.jsonl lines (%d) exceed checkpoint kept (%d); reconciling counters',
            file_lines,
            cp_kept,
        )
        names = list(checkpoint.sources.keys())
        if not names:
            return {'file_lines': file_lines, 'checkpoint_kept': cp_kept, 'adjusted': file_lines - cp_kept}
        running = 0
        for i, name in enumerate(names):
            row = checkpoint.source(name)
            old_kept = row.kept
            if i == len(names) - 1:
                new_kept = max(0, file_lines - running)
            else:
                new_kept = int(row.kept * file_lines / cp_kept) if cp_kept else 0
                running += new_kept
            row.scanned += new_kept - old_kept
            row.kept = new_kept
        return {
            'file_lines': file_lines,
            'checkpoint_kept': cp_kept,
            'adjusted': file_lines - cp_kept,
        }
    return {'file_lines': file_lines, 'checkpoint_kept': cp_kept, 'adjusted': 0}


def _top_reject_reasons(reject_reasons: dict[str, int], *, limit: int = 5) -> list[tuple[str, int]]:
    return sorted(reject_reasons.items(), key=lambda item: item[1], reverse=True)[:limit]


def log_merge_progress(
    logger,
    *,
    total_scanned: int,
    gate: 'QualityGate',
    t0: float,
    src_name: str,
    line_no: int,
    exact_dup: int = 0,
    session_kept: int | None = None,
    session_rejected: int | None = None,
    workers: int | None = None,
    active_workers: int | None = None,
    read_queue_size: int | None = None,
    pending_batches: int | None = None,
    cpu_utilization: float | None = None,
    eta_sec: float | None = None,
) -> None:
    import time

    stats = gate.stats
    kept = session_kept if session_kept is not None else stats.kept
    rejected = session_rejected if session_rejected is not None else stats.rejected
    elapsed = max(time.perf_counter() - t0, 1e-6)
    keep_pct = 100.0 * kept / max(total_scanned, 1)
    dps = total_scanned / elapsed
    qs = stats.to_dict()
    top = _top_reject_reasons(dict(qs.get('reject_reasons') or {}))
    top_str = ', '.join(f'{k}={v}' for k, v in top) if top else '-'
    domain = gate.domain_balancer.distribution()
    dom_str = ', '.join(
        f'{k}={v:.0%}' for k, v in sorted(domain.items(), key=lambda x: -x[1])[:4]
    ) if domain else '-'
    parallel_bits: list[str] = []
    if workers is not None:
        parallel_bits.append(f'workers={workers}')
    if active_workers is not None:
        parallel_bits.append(f'active={active_workers}')
    if read_queue_size is not None:
        parallel_bits.append(f'read_q={read_queue_size}')
    if pending_batches is not None:
        parallel_bits.append(f'pending={pending_batches}')
    if cpu_utilization is not None:
        parallel_bits.append(f'cpu={cpu_utilization:.0f}%')
    if eta_sec is not None and eta_sec > 0:
        parallel_bits.append(f'eta={eta_sec / 60:.1f}m')
    parallel_str = (' | ' + ' '.join(parallel_bits)) if parallel_bits else ''
    logger.info(
        '[merge] scanned=%d kept=%d (%.1f%%) rejected=%d dup=%d '
        'score_mean=%.3f docs/s=%.1f last=%s:%d%s',
        total_scanned,
        kept,
        keep_pct,
        rejected,
        exact_dup,
        float(qs.get('score_mean', 0.0)),
        dps,
        src_name,
        line_no,
        parallel_str,
    )
    logger.info('[merge] rejects: %s | domains: %s', top_str, dom_str)


def write_run_progress(
    work_dir: Path,
    payload: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    from indw.store.io.atomic import atomic_write_text
    import hashlib

    path = Path(work_dir) / RUN_PROGRESS_NAME
    payload_body = {
        **payload,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }
    key = str(path.parent.resolve())
    sig = hashlib.blake2b(
        dumps_canonical(payload),
        digest_size=8,
    ).hexdigest()
    now = time.monotonic()
    last = _last_progress_write.get(key)
    if not force and last is not None:
        elapsed, last_sig = last
        if elapsed >= 0 and now - elapsed < _PROGRESS_WRITE_MIN_SEC and last_sig == sig:
            return
    atomic_write_text(path, dumps_pretty(payload_body))
    _last_progress_write[key] = (now, sig)
    if len(_last_progress_write) > _PROGRESS_WRITE_CACHE_MAX:
        oldest = sorted(_last_progress_write.items(), key=lambda item: item[1][0])[
            : len(_last_progress_write) - _PROGRESS_WRITE_CACHE_MAX
        ]
        for stale_key, _ in oldest:
            _last_progress_write.pop(stale_key, None)


_PROGRESS_WRITE_MIN_SEC = 5.0
_PROGRESS_WRITE_CACHE_MAX = 64
_last_progress_write: dict[str, tuple[float, str]] = {}


def publish_merge_progress(
    work_dir: Path,
    *,
    gate: Any,
    exact: Any,
    total_scanned: int,
    elapsed_sec: float,
    kept: Optional[int] = None,
    rejected: Optional[int] = None,
    status: str = 'running',
    extra: Optional[dict[str, Any]] = None,
    force: bool = False,
) -> None:
    from indw.schedule.monitor.obs import progress_reject_reasons

    payload = merge_run_progress_payload(
        gate=gate,
        exact=exact,
        total_scanned=total_scanned,
        elapsed_sec=elapsed_sec,
        status=status,
        kept=kept,
        rejected=rejected,
        extra=extra,
    )
    payload['reject_reasons'] = progress_reject_reasons(work_dir, gate)
    write_run_progress(work_dir, payload, force=force)


def load_run_progress(work_dir: Path) -> dict[str, Any]:
    path = Path(work_dir) / RUN_PROGRESS_NAME
    if not path.exists():
        return {}
    try:
        raw = loads(path.read_text(encoding='utf-8'))
        return raw if isinstance(raw, dict) else {}
    except (ValueError, OSError, TypeError):
        bak = path.with_name(f'{path.name}.bak')
        if bak.exists():
            try:
                raw = loads(bak.read_text(encoding='utf-8'))
                return raw if isinstance(raw, dict) else {}
            except (ValueError, OSError, TypeError):
                pass
        return {}


def merge_run_progress_payload(
    *,
    gate: Any,
    exact: Any,
    total_scanned: int,
    elapsed_sec: float,
    status: str = 'running',
    kept: Optional[int] = None,
    rejected: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'phase': 'merge',
        'status': status,
        'total_scanned': total_scanned,
        'kept': kept if kept is not None else int(gate.stats.kept),
        'rejected': rejected if rejected is not None else int(gate.stats.rejected),
        'exact_duplicates': int(dict(gate.stats.reject_reasons).get('exact_dup', exact.duplicates)),
        'score_mean': gate.stats.to_dict().get('score_mean', 0.0),
        'reject_reasons': dict(gate.stats.reject_reasons),
        'domain_distribution': gate.domain_balancer.distribution(),
        'elapsed_sec': round(elapsed_sec, 1),
    }
    chunk_outcomes = int(payload['kept']) + int(payload['rejected'])
    payload['chunk_outcomes'] = chunk_outcomes
    payload['accounting_gap'] = int(payload['total_scanned']) - chunk_outcomes
    if extra:
        payload.update(extra)
    return payload


def publish_resume_gate_snapshot(
    pipeline_metrics: Any,
    *,
    gate: Any,
    checkpoint: Any,
    total_scanned: int,
    resume_metric_base: dict[str, Any],
) -> None:
    if pipeline_metrics is None or total_scanned <= 0:
        return
    totals = checkpoint.totals()
    pipeline_metrics.publish_gate_snapshot(
        gate,
        merge_kept=totals['kept'],
        merge_rejected=totals['rejected'],
        total_scanned=total_scanned,
        **resume_metric_base,
    )


def publish_live_gate_snapshot(
    pipeline_metrics: Any,
    *,
    gate: Any,
    source_names: list[str],
    checkpoint: Any,
    total_scanned: int,
    exact: Any,
    resume_exact_dup_base: int,
    resume_metric_base: dict[str, Any],
    thresholds: Any = None,
    log_diagnostics: bool = False,
    workers: int | None = None,
    active_workers: int | None = None,
    queue_depth: int | None = None,
    cpu_utilization_pct: float | None = None,
) -> dict[str, Any] | None:
    if pipeline_metrics is None:
        return None
    snap = live_gate_snapshot_kwargs(
        gate=gate,
        source_names=source_names,
        checkpoint=checkpoint,
        total_scanned=total_scanned,
        exact=exact,
        resume_exact_dup_base=resume_exact_dup_base,
        resume_metric_base=resume_metric_base,
    )
    runtime = {}
    if workers is not None:
        runtime['workers'] = workers
    if active_workers is not None:
        runtime['active_workers'] = active_workers
    if queue_depth is not None:
        runtime['queue_depth'] = queue_depth
    if cpu_utilization_pct is not None:
        runtime['cpu_utilization_pct'] = cpu_utilization_pct
    pipeline_metrics.publish_gate_snapshot(gate, **snap, **runtime)
    if log_diagnostics and thresholds is not None:
        from indw.schedule.monitor.obs import gate_diagnostics_enabled
        if gate_diagnostics_enabled():
            from indw.filter.gate.diagnostics import log_gate_diagnostics

            log_gate_diagnostics(
                gate,
                total_scanned=total_scanned,
                merge_kept=snap['merge_kept'],
                merge_rejected=snap['merge_rejected'],
                thresholds=thresholds,
            )
    return snap


def live_gate_snapshot_kwargs(
    *,
    gate: Any,
    source_names: list[str],
    checkpoint: Any,
    total_scanned: int,
    exact: Any,
    resume_exact_dup_base: int,
    resume_metric_base: dict[str, Any],
) -> dict[str, Any]:
    cp_kept = sum(checkpoint.source(n).kept for n in source_names)
    cp_rejected = sum(checkpoint.source(n).rejected for n in source_names)
    qs = gate.stats.to_dict()
    live_score = float(qs.get('score_mean', 0.0))
    return {
        'merge_kept': cp_kept,
        'merge_rejected': cp_rejected,
        'total_scanned': total_scanned,
        'exact_duplicates': resume_exact_dup_base + exact.duplicates,
        'score_mean': live_score if live_score > 0 else resume_metric_base.get('score_mean'),
        'reject_reasons': resume_metric_base.get('reject_reasons'),
    }


def resume_metrics_kwargs(progress: dict[str, Any]) -> dict[str, Any]:
    if not progress:
        return {}
    rejects = progress.get('reject_reasons')
    if not isinstance(rejects, dict):
        rejects = {}
    exact_dup = rejects.get('exact_dup', progress.get('exact_duplicates', 0))
    out: dict[str, Any] = {}
    if rejects:
        out['reject_reasons'] = dict(rejects)
    try:
        out['exact_duplicates'] = int(exact_dup)
    except (TypeError, ValueError):
        pass
    try:
        score = float(progress.get('score_mean', 0.0))
        if score > 0:
            out['score_mean'] = score
    except (TypeError, ValueError):
        pass
    return out


def triage_merge(work_dir: Path, *, raw_dir: Optional[Path] = None) -> dict[str, Any]:
    work = Path(work_dir)
    raw = Path(raw_dir) if raw_dir else work / 'raw'
    checkpoint = MergeCheckpoint.load(work)
    filtered = work / 'filtered.jsonl'
    filtered_lines = count_jsonl_lines(filtered)
    filtered_bytes = filtered.stat().st_size if filtered.exists() else 0

    sources: list[dict[str, Any]] = []
    total_raw = 0
    for src in sorted(raw.glob('*/data.jsonl')):
        name = src.parent.name
        raw_lines = count_jsonl_lines(src)
        total_raw += raw_lines
        cp = checkpoint.source(name) if checkpoint else SourceCheckpoint()
        pct = round(100.0 * cp.line_offset / raw_lines, 1) if raw_lines else 0.0
        sources.append({
            'name': name,
            'raw_lines': raw_lines,
            'line_offset': cp.line_offset,
            'scanned': cp.scanned,
            'kept': cp.kept,
            'rejected': cp.rejected,
            'progress_pct': pct,
            'done': cp.line_offset >= raw_lines and raw_lines > 0,
        })

    totals = checkpoint.totals() if checkpoint else {'scanned': 0, 'kept': 0, 'rejected': 0}
    can_resume = bool(
        checkpoint
        and not checkpoint.complete
        and any(row['line_offset'] > 0 for row in sources)
    )
    action = 'resume' if can_resume else 'fresh'
    if checkpoint and checkpoint.complete:
        action = 'complete'

    return {
        'work_dir': str(work.resolve()),
        'checkpoint_path': str(MergeCheckpoint.path_for(work)),
        'checkpoint_exists': checkpoint is not None,
        'complete': bool(checkpoint and checkpoint.complete),
        'interrupted': bool(checkpoint and checkpoint.interrupted),
        'updated_at': checkpoint.updated_at if checkpoint else None,
        'filtered_lines': filtered_lines,
        'filtered_bytes': filtered_bytes,
        'raw_lines_total': total_raw,
        'totals': totals,
        'sources': sources,
        'recommended_action': action,
        'hints': {
            'resume': 'python -m cli data prepare --work-dir <dir> --skip-download ...',
            'fresh': 'python -m cli data prepare --work-dir <dir> --fresh-merge --skip-download ...',
            'triage': 'python -m cli data prepare --work-dir <dir> --triage',
        },
    }


def print_merge_triage(work_dir: Path, *, raw_dir: Optional[Path] = None) -> dict[str, Any]:
    status = triage_merge(work_dir, raw_dir=raw_dir)
    print(dumps_pretty(status))
    print('')
    print('Prepare triage:')
    print(f"  action:      {status['recommended_action']}")
    print(f"  raw lines:   {status['raw_lines_total']}")
    print(f"  scanned:     {status['totals']['scanned']}")
    print(f"  kept:        {status['totals']['kept']} (filtered.jsonl lines: {status['filtered_lines']})")
    for row in status['sources']:
        print(
            f"  [{row['name']}] {row['line_offset']}/{row['raw_lines']} lines "
            f"({row['progress_pct']}%) kept={row['kept']}"
        )
    if status['recommended_action'] == 'resume':
        print('  Stop with Ctrl+C anytime; rerun the same command to resume.')
        print('  Press Ctrl+C twice to force quit immediately (checkpoint saved).')
    elif status['recommended_action'] == 'fresh':
        print('  No checkpoint to resume. Use --fresh-merge only to restart from scratch.')
    return status
