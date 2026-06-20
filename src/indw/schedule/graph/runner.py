from __future__ import annotations

import logging
import queue
import time
from concurrent.futures import FIRST_COMPLETED, wait
from typing import Any, Callable

from indw.schedule.backends.factory import resolve_execution_backend
from indw.schedule.config.policy import MERGE_READ_SENTINEL
from indw.schedule.config.tune import get_merge_tune, merge_drain_sec, survivor_buffer_cap
from indw.schedule.dispatch.alloc import StageAllocationV2, plan_graph_alloc
from indw.schedule.dispatch.lanes import (
    ALL_LANES,
    LaneBuffers,
    LaneWorkerSlots,
    has_buffered_seq,
    pick_lane_batch,
    pop_target_seq,
)
from indw.schedule.dispatch.workers import WorkerInitBundle
from indw.schedule.read.probe import SchedulerProbe

logger = logging.getLogger(__name__)

_COLLECT_TIMEOUT = 0.25
_SIGNAL_INTERVAL_SEC = 0.15


def run_graph_merge(
    *,
    config_path: str,
    workers: int,
    chunk_size: int,
    merge_work: Any,
    read_queue: queue.Queue,
    can_submit: Callable[[], bool],
    apply_next_write_seq: Callable[[], int],
    ingest_line_results: Callable[[list[dict[str, Any]]], None],
    ingest_batch: Callable[[dict[str, Any]], None],
    merge_cleaning_stats: Callable[[Any], None],
    notify_apply: Callable[[], None],
    apply_buffer_depth: Callable[[], tuple[int, int]],
    stop_requested: Callable[[], bool],
    refresh_runtime: Callable[..., None],
    t0: float,
    total_scanned_ref: dict[str, int],
    worker_init: WorkerInitBundle,
    alloc: StageAllocationV2 | None = None,
    probe: SchedulerProbe | None = None,
    batch_timeout_sec: float = 120.0,
    time_limit_sec: float | None = None,
    on_time_limit: Callable[[], None] | None = None,
    drain_sec: float | None = None,
    on_cost_payload: Callable[[dict[str, Any]], None] | None = None,
    execution_backend: str | None = None,
) -> int:
    if alloc is None:
        _, alloc = plan_graph_alloc(
            workers=workers, chunk_size=chunk_size, merge_work=merge_work,
        )
    probe = probe or SchedulerProbe(t0=t0)
    fast_workers = max(alloc.preprocess_workers, alloc.filter_workers, alloc.stage0_workers, alloc.fast_workers)
    heavy_workers = max(alloc.clean_workers, alloc.pci_workers, alloc.acim_workers, alloc.heavy_workers)
    lane_slots = LaneWorkerSlots.from_heavy_workers(heavy_workers)
    probe.lane_slots = lane_slots.to_dict()
    backend = resolve_execution_backend(execution_backend)
    probe.execution_backend = backend.name
    worker_failures = 0
    collect_timeout = min(float(batch_timeout_sec), 120.0)
    collect_timeout = max(collect_timeout, 30.0)
    drain_budget = float(drain_sec) if drain_sec is not None else merge_drain_sec(
        time_limit_sec=time_limit_sec,
    )

    pending_fast: dict[int, tuple[Any, list[dict[str, Any]], int]] = {}
    pending_heavy: dict[int, tuple[Any, list[dict[str, Any]], int, str]] = {}
    lane_buffers = LaneBuffers()
    next_fast_id = 0
    next_heavy_id = 0
    reader_done = False
    drain_mode = False
    drain_deadline = 0.0
    last_signal_at = 0.0

    def _timed_out() -> bool:
        return time_limit_sec is not None and (time.perf_counter() - t0) >= time_limit_sec

    def _enter_drain() -> bool:
        nonlocal drain_mode, drain_deadline
        if not _timed_out():
            return False
        if not drain_mode:
            drain_mode = True
            drain_deadline = time.perf_counter() + drain_budget
            if on_time_limit is not None:
                on_time_limit()
        return True

    def _drain_done() -> bool:
        if not drain_mode:
            return False
        if reader_done and not pending_fast and not pending_heavy and lane_buffers.empty():
            return True
        if time.perf_counter() >= drain_deadline and not pending_heavy:
            return True
        return False

    def _survivor_cap() -> int:
        return survivor_buffer_cap(
            apply_queue=alloc.apply_queue,
            heavy_queue=alloc.clean_queue,
        )

    def _pending_lane(lane: str) -> int:
        return sum(1 for _, (_, _, _, ln) in pending_heavy.items() if ln == lane)

    def _collect_fast(
        bid: int,
        task: Any,
        batch: list[dict[str, Any]],
        sess: Any,
    ) -> bool:
        nonlocal worker_failures
        try:
            payload = task.result(timeout=collect_timeout)
        except Exception as exc:
            worker_failures += 1
            logger.warning('Graph fast chain %d failed (%s); fallback', bid, exc)
            fb = sess.run_fallback_merge(batch)
            if on_cost_payload is not None:
                on_cost_payload(fb)
            ingest_batch(fb)
            notify_apply()
            return True
        if on_cost_payload is not None:
            on_cost_payload(payload)
        terminal = payload.get('terminal') or []
        survivors = payload.get('survivors') or []
        if terminal:
            ingest_line_results(terminal)
        if survivors:
            lane_buffers.route_many(survivors)
            probe.note_lane_backlog(lane_buffers.depths())
        if terminal or survivors:
            notify_apply()
        return bool(terminal or survivors)

    def _collect_heavy(hid: int, task: Any, lane: str) -> bool:
        nonlocal worker_failures
        try:
            payload = task.result(timeout=collect_timeout * 2)
        except Exception as exc:
            worker_failures += 1
            logger.warning('Graph heavy chain %d lane=%s failed (%s)', hid, lane, exc)
            return False
        if on_cost_payload is not None:
            on_cost_payload(payload)
        items = payload.get('items') or []
        if items:
            ingest_line_results(items)
            notify_apply()
        stats = payload.get('cleaning_stats')
        if stats is not None:
            merge_cleaning_stats(stats)
        return bool(items)

    def _dispatch_heavy_chunk(
        sess: Any,
        chunk: list[dict[str, Any]],
        lane: str,
        *,
        head_priority: bool = False,
    ) -> bool:
        nonlocal next_heavy_id
        if not chunk or len(pending_heavy) >= alloc.clean_queue or not can_submit():
            return False
        task = sess.submit_heavy(chunk)
        pending_heavy[next_heavy_id] = (task, chunk, int(chunk[0]['seq']), lane)
        next_heavy_id += 1
        if head_priority:
            probe.head_priority_dispatches += 1
        return True

    def _submit_heavy_lanes(sess: Any) -> None:
        _, ordering_gap = apply_buffer_depth()
        head = apply_next_write_seq()
        if len(pending_heavy) >= alloc.clean_queue or not can_submit() or lane_buffers.empty():
            return
        if lane_buffers.total() > _survivor_cap() and ordering_gap > 0:
            return
        if ordering_gap > 0 and has_buffered_seq(lane_buffers, head):
            picked = pop_target_seq(lane_buffers, head)
            if picked is not None:
                item, lane = picked
                _dispatch_heavy_chunk(sess, [item], lane, head_priority=True)
                return
        for lane in ALL_LANES:
            buf = lane_buffers.lane_buffer(lane)
            if not buf:
                continue
            while buf and _pending_lane(lane) < max(1, lane_slots.slot_limit(lane)):
                if not can_submit() or len(pending_heavy) >= alloc.clean_queue:
                    break
                chunk = pick_lane_batch(
                    buf, lane, alloc.heavy_batch, ordering_gap=ordering_gap, batch_cap=alloc.heavy_batch,
                )
                if not chunk:
                    break
                if not _dispatch_heavy_chunk(sess, chunk, lane):
                    break

    def _maybe_refresh(sess: Any) -> None:
        nonlocal last_signal_at
        now = time.perf_counter()
        if now - last_signal_at < _SIGNAL_INTERVAL_SEC:
            return
        last_signal_at = now
        try:
            from indw.tools.reports.benchmark import peak_rss_mb
            rss = peak_rss_mb()
        except Exception:
            rss = 0.0
        try:
            from monitoring.cpu import collect_cpu_stats
            cpu_pct = collect_cpu_stats().utilization_pct or 0.0
        except Exception:
            cpu_pct = 0.0
        refresh_runtime(
            cpu_pct=cpu_pct,
            rss_mb=rss,
            queue_depth=read_queue.qsize(),
            docs_per_sec=total_scanned_ref.get('n', 0) / max(now - t0, 1e-9),
            active_workers=sess.active_workers() or (len(pending_fast) + len(pending_heavy)),
        )

    with backend.open(worker_init, fast_workers=fast_workers, heavy_workers=heavy_workers) as sess:
        while not (reader_done and not pending_fast and not pending_heavy and lane_buffers.empty()):
            if stop_requested() or _drain_done():
                break
            _enter_drain()

            fast_backpressure = not can_submit() or lane_buffers.total() >= _survivor_cap()
            if not drain_mode and not fast_backpressure:
                while not reader_done and len(pending_fast) < alloc.fast_queue:
                    try:
                        item = read_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is MERGE_READ_SENTINEL:
                        reader_done = True
                        break
                    if not item:
                        continue
                    for row in item:
                        row.setdefault('_work_dir', str(merge_work))
                    task = sess.submit_fast(item)
                    pending_fast[next_fast_id] = (task, item, int(item[0]['seq']))
                    next_fast_id += 1

            for bid in sorted(bid for bid, (t, _, _) in pending_fast.items() if t.done()):
                task, batch, _ = pending_fast.pop(bid)
                _collect_fast(bid, task, batch, sess)

            _submit_heavy_lanes(sess)
            for hid in sorted(hid for hid, (t, _, _, _) in pending_heavy.items() if t.done()):
                task, _, _, lane = pending_heavy.pop(hid)
                _collect_heavy(hid, task, lane)

            if reader_done and not pending_fast and lane_buffers.total():
                _submit_heavy_lanes(sess)

            _maybe_refresh(sess)
            progressed = pending_fast or pending_heavy or not reader_done
            if not progressed:
                tasks = [t for t, _, _ in pending_fast.values()] + [
                    t for t, _, _, _ in pending_heavy.values()
                ]
                poll = get_merge_tune().scheduler_idle_poll_sec
                futs = [getattr(t, '_fut', None) for t in tasks]
                futs = [f for f in futs if f is not None]
                if futs:
                    wait(futs, timeout=poll, return_when=FIRST_COMPLETED)
                else:
                    time.sleep(poll)

        notify_apply()
        probe.publish(merge_work)

    logger.info(
        'Graph merge backend=%s fast=%d heavy=%d dedup_shards=%d stage_pools=chain',
        backend.name,
        fast_workers,
        heavy_workers,
        alloc.dedup_shards,
    )
    return worker_failures
