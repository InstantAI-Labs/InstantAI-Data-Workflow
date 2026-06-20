from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from indw.schedule.state.checkpoint import METRICS_SNAPSHOT_INTERVAL, PROGRESS_LOG_INTERVAL

logger = logging.getLogger(__name__)


class ApplyCoordinator:
    def __init__(
        self,
        *,
        apply_fn: Callable[[dict[str, Any]], bool],
        prep_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        on_apply_complete: Callable[[], None] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        tune: Any,
        sched_probe: Any | None = None,
        checkpoint_interval: int,
        pipeline_metrics: Any | None = None,
        publish_metrics: Callable[[], None] | None = None,
        sink: Any | None = None,
        total_scanned_ref: dict[str, int] | None = None,
    ):
        self._apply_fn = apply_fn
        self._prep_fn = prep_fn or (lambda line: line)
        self._on_apply_complete = on_apply_complete
        self._on_progress = on_progress
        self._tune = tune
        self._sched_probe = sched_probe
        self._checkpoint_interval = checkpoint_interval
        self._pipeline_metrics = pipeline_metrics
        self._publish_metrics = publish_metrics
        self._sink = sink
        self._total_scanned_ref = total_scanned_ref or {'n': 0}

        self._line_results: dict[int, dict[str, Any]] = {}
        self._next_write_seq = 0
        self._apply_buffer_size = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()

        self.total_scanned = 0
        self.skipped_parse = 0
        self.last_progress_at = time.monotonic()

    @property
    def next_write_seq(self) -> int:
        with self._lock:
            return self._next_write_seq

    @property
    def apply_buffer_size(self) -> int:
        return self._apply_buffer_size

    def pending_state(self) -> tuple[int, int, int]:
        with self._lock:
            total = len(self._line_results)
            gap = 0
            if self._next_write_seq not in self._line_results and self._line_results:
                gap = min(self._line_results) - self._next_write_seq
            ready = 0
            seq = self._next_write_seq
            while seq in self._line_results:
                ready += 1
                seq += 1
            out_of_order = total - ready
            return total, gap, out_of_order

    def buffer_depth(self) -> tuple[int, int]:
        total, gap, _ = self.pending_state()
        return total, gap

    def can_accept(self, *, max_completed: int) -> bool:
        total, _, _ = self.pending_state()
        if total >= max_completed:
            if self._sched_probe is not None:
                self._sched_probe.note_heavy_apply_backpressure()
            return False
        return True

    def ingest_lines(self, lines: list[dict[str, Any]]) -> None:
        prepared = [self._prep_fn(line) for line in lines]
        with self._lock:
            for line in prepared:
                self._line_results[int(line['seq'])] = line
            self._apply_buffer_size = len(self._line_results)
        with self._cond:
            self._cond.notify_all()

    def ingest_batch(self, batch_payload: dict[str, Any]) -> None:
        items = [self._prep_fn(line) for line in batch_payload.get('items') or []]
        with self._lock:
            for line in items:
                self._line_results[int(line['seq'])] = line
            self._apply_buffer_size = len(self._line_results)
        with self._cond:
            self._cond.notify_all()

    def notify(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def stop(self) -> None:
        self._stop.set()
        self.notify()

    def run_loop(self) -> None:
        while True:
            with self._cond:
                while self._next_write_seq not in self._line_results:
                    if self._stop.is_set():
                        return
                    gap = 0
                    if self._line_results:
                        gap = min(self._line_results) - self._next_write_seq
                    wait_sec = (
                        self._tune.apply_wait_blocked_sec if gap > 0 else self._tune.apply_wait_idle_sec
                    )
                    wait_t0 = time.perf_counter()
                    notified = self._cond.wait(timeout=wait_sec)
                    if (
                        not notified
                        and gap > 0
                        and self._next_write_seq not in self._line_results
                    ):
                        elapsed_ms = (time.perf_counter() - wait_t0) * 1000.0
                        if (
                            self._sched_probe is not None
                            and elapsed_ms >= self._tune.apply_wait_record_min_ms
                        ):
                            self._sched_probe.note_ordering_wait(gap, elapsed_ms)
                line = self._line_results.pop(self._next_write_seq)
                self._apply_buffer_size = len(self._line_results)

            kept_any = self._apply_fn(line)
            if self._sched_probe is not None:
                self._sched_probe.note_apply_complete()
            if (
                self._pipeline_metrics is not None
                and self.total_scanned % METRICS_SNAPSHOT_INTERVAL == 0
                and self._publish_metrics is not None
            ):
                self._publish_metrics()
            if self.total_scanned > 0 and self.total_scanned % self._checkpoint_interval == 0:
                if self._sink is not None:
                    self._sink.flush()
            if self.total_scanned % PROGRESS_LOG_INTERVAL == 0 and self._on_progress is not None:
                self._on_progress({
                    'line': line,
                    'kept_any': kept_any,
                    'total_scanned': self.total_scanned,
                })
            with self._cond:
                self._next_write_seq += 1
                self._cond.notify_all()
            self.last_progress_at = time.monotonic()
            self._total_scanned_ref['n'] = self.total_scanned
            if self._on_apply_complete is not None:
                self._on_apply_complete()
