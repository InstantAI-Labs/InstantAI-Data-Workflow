from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from indw.config.defaults import MERGE_DOC_STALL_WARN_SEC
from indw.schedule.config.resolve import env_float

logger = logging.getLogger(__name__)

_tls = threading.local()
_MONITOR: DocMonitorSession | None = None
_warned: set[int] = set()
_warn_lock = threading.Lock()
_MAX_WARNED_SEQ = 10_000


def stall_warn_sec() -> float:
    return env_float('INSTANT_MERGE_DOC_STALL_WARN_SEC', MERGE_DOC_STALL_WARN_SEC, minimum=1.0)


def _attach_runtime_stats(snap: dict[str, Any]) -> None:
    try:
        from indw.schedule.monitor.obs import cache_stats_enabled
        if cache_stats_enabled():
            from indw.clean.artifact.evidence_cache import session_cache_stats
            snap['cache'] = session_cache_stats()
    except Exception:
        pass
    try:
        from indw.tools.reports.benchmark import peak_rss_mb
        snap['rss_mb'] = round(peak_rss_mb(), 1)
    except Exception:
        pass


@dataclass
class ActiveDocState:
    seq: int = -1
    src_name: str = ''
    line_no: int = -1
    doc_id: str = ''
    chars: int = 0
    words: int = 0
    fingerprint: str = ''
    stage: str = ''
    stage_started: float = 0.0
    doc_started: float = 0.0
    stall_logged: bool = False

    def stage_elapsed(self) -> float:
        if not self.stage or self.stage_started <= 0:
            return 0.0
        return max(0.0, time.perf_counter() - self.stage_started)

    def doc_elapsed(self) -> float:
        if self.doc_started <= 0:
            return 0.0
        return max(0.0, time.perf_counter() - self.doc_started)

    def to_dict(self) -> dict[str, Any]:
        return {
            'seq': self.seq,
            'source': self.src_name,
            'line_no': self.line_no,
            'doc_id': self.doc_id,
            'chars': self.chars,
            'words': self.words,
            'fingerprint': self.fingerprint,
            'stage': self.stage,
            'stage_elapsed_sec': round(self.stage_elapsed(), 3),
            'doc_elapsed_sec': round(self.doc_elapsed(), 3),
        }


@dataclass
class DocMonitorSession:
    merge_work: Path
    _stall_fp: Any = field(default=None, repr=False)
    _stalls: int = 0

    def __post_init__(self) -> None:
        self.merge_work = Path(self.merge_work)
        from indw.schedule.monitor.obs import doc_stalls_enabled
        if not doc_stalls_enabled():
            self._stall_fp = None
            return
        log_dir = self.merge_work / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self._stall_fp = (log_dir / 'doc_stalls.jsonl').open('a', encoding='utf-8')

    def begin_doc(
        self,
        *,
        seq: int,
        src_name: str,
        line_no: int,
        chars: int,
        fingerprint: str = '',
        doc_id: str = '',
        words: int = 0,
    ) -> None:
        now = time.perf_counter()
        _tls.active = ActiveDocState(
            seq=seq,
            src_name=src_name,
            line_no=line_no,
            doc_id=doc_id,
            chars=chars,
            words=words or max(1, chars // 5),
            fingerprint=fingerprint,
            stage='preprocess',
            stage_started=now,
            doc_started=now,
        )

    def set_stage(self, stage: str) -> None:
        active = getattr(_tls, 'active', None)
        if active is None:
            return
        now = time.perf_counter()
        active.stage = stage
        active.stage_started = now
        self._maybe_warn_stall(active)

    def end_doc(self, *, outcome: str = 'ok') -> dict[str, Any] | None:
        active = getattr(_tls, 'active', None)
        if active is None:
            return None
        summary = {**active.to_dict(), 'outcome': outcome}
        elapsed = active.doc_elapsed()
        if elapsed >= stall_warn_sec():
            self._record_stall(active, reason='slow_doc', outcome=outcome)
        _tls.active = None
        with _warn_lock:
            _warned.discard(active.seq)
        return summary

    def _maybe_warn_stall(self, active: ActiveDocState) -> None:
        elapsed = active.doc_elapsed()
        warn = stall_warn_sec()
        if elapsed < warn:
            return
        with _warn_lock:
            if active.seq in _warned:
                return
            if len(_warned) >= _MAX_WARNED_SEQ:
                _warned.clear()
            _warned.add(active.seq)
        self._record_stall(active, reason='stall_warn')
        logger.warning(
            '[doc_monitor] slow doc seq=%d source=%s line=%d stage=%s '
            'doc_elapsed=%.1fs stage_elapsed=%.1fs chars=%d fp=%s',
            active.seq,
            active.src_name,
            active.line_no,
            active.stage,
            elapsed,
            active.stage_elapsed(),
            active.chars,
            active.fingerprint[:16],
        )

    def _record_stall(self, active: ActiveDocState, *, reason: str, outcome: str = '') -> None:
        if self._stall_fp is None:
            return
        self._stalls += 1
        payload = {
            'reason': reason,
            'outcome': outcome,
            **active.to_dict(),
        }
        _attach_runtime_stats(payload)
        self._stall_fp.write(json.dumps(payload, ensure_ascii=False) + '\n')
        self._stall_fp.flush()

    def close(self) -> None:
        if self._stall_fp is not None:
            self._stall_fp.flush()
            self._stall_fp.close()
            self._stall_fp = None

    def stats(self) -> dict[str, Any]:
        return {'stall_events': self._stalls}


def active_doc_state() -> ActiveDocState | None:
    return getattr(_tls, 'active', None)


def bind_doc_monitor(session: DocMonitorSession | None) -> None:
    global _MONITOR
    _MONITOR = session


def monitor_begin_doc(
    *,
    seq: int,
    src_name: str,
    line_no: int,
    chars: int,
    fingerprint: str = '',
    doc_id: str = '',
    words: int = 0,
) -> None:
    if _MONITOR is not None:
        _MONITOR.begin_doc(
            seq=seq,
            src_name=src_name,
            line_no=line_no,
            chars=chars,
            fingerprint=fingerprint,
            doc_id=doc_id,
            words=words,
        )


def monitor_end_doc(*, outcome: str = 'ok') -> dict[str, Any] | None:
    if _MONITOR is None:
        return None
    return _MONITOR.end_doc(outcome=outcome)


def monitor_snapshot() -> dict[str, Any] | None:
    return active_snapshot()


def active_snapshot() -> dict[str, Any] | None:
    active = getattr(_tls, 'active', None)
    if active is None:
        return None
    snap = active.to_dict()
    _attach_runtime_stats(snap)
    return snap


def set_doc_stage(stage: str) -> None:
    if _MONITOR is not None:
        _MONITOR.set_stage(stage)
        return
    active = getattr(_tls, 'active', None)
    if active is None:
        return
    now = time.perf_counter()
    active.stage = stage
    active.stage_started = now


@contextmanager
def monitored_stage(stage: str) -> Iterator[None]:
    set_doc_stage(stage)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        active = getattr(_tls, 'active', None)
        if active is not None and active.stage == stage:
            active.stage_started = t0
