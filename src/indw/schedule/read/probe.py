from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DENIAL_TO_GATE = {
    'heavy_queue_full': 'buffer_gate',
    'apply_buffer_cap': 'apply_gate',
    'no_survivors': 'buffer_gate',
    'huge_lane_skipped': 'lane_gate',
    'lane_slot_blocked': 'lane_gate',
}


@dataclass
class SchedulerProbe:
    t0: float = field(default_factory=time.perf_counter)
    loop_count: int = 0
    scheduler_sleep_ms: float = 0.0
    ordering_wait_ms: float = 0.0
    backpressure_events: int = 0
    reader_block_events: int = 0
    fast_submits: int = 0
    heavy_submits: int = 0
    fast_collects: int = 0
    heavy_collects: int = 0
    apply_completions: int = 0
    peak_apply_buffer: int = 0
    peak_ordering_gap: int = 0
    peak_fast_pending: int = 0
    peak_heavy_pending: int = 0
    peak_survivor_buffer: int = 0
    peak_read_queue: int = 0
    large_doc_blocking_events: int = 0
    worker_idle_ms: float = 0.0
    heavy_apply_backpressure_events: int = 0
    seq_priority_dispatches: int = 0
    head_blocked_dispatches: int = 0
    dispatched_past_head: int = 0
    dispatch_denied: dict[str, int] = field(default_factory=dict)
    head_priority_dispatches: int = 0
    gate_blocked: dict[str, int] = field(default_factory=dict)
    last_dispatch_audit: dict[str, Any] = field(default_factory=dict)
    dispatch_audit_samples: list[dict[str, Any]] = field(default_factory=list)
    lane_slots: dict[str, int] = field(default_factory=dict)
    peak_lane_backlog: dict[str, int] = field(default_factory=dict)
    lane_heavy_submits: dict[str, int] = field(default_factory=dict)
    heavy_worker_dispatches: list[int] = field(default_factory=list)
    phase_ms: dict[str, float] = field(default_factory=dict)
    _phase_acc: dict[str, float] = field(default_factory=dict)
    execution_backend: str = 'multiprocess'

    def record_loop(
        self,
        *,
        phases: dict[str, float],
        depths: dict[str, int],
        ordering_gap: int,
        apply_buffer: int,
        backpressure: bool,
        slept_ms: float,
    ) -> None:
        self.loop_count += 1
        self.scheduler_sleep_ms += slept_ms
        if backpressure:
            self.backpressure_events += 1
        self.peak_apply_buffer = max(self.peak_apply_buffer, apply_buffer)
        self.peak_ordering_gap = max(self.peak_ordering_gap, ordering_gap)
        self.peak_fast_pending = max(self.peak_fast_pending, depths.get('fast_pending', 0))
        self.peak_heavy_pending = max(self.peak_heavy_pending, depths.get('heavy_pending', 0))
        self.peak_survivor_buffer = max(self.peak_survivor_buffer, depths.get('survivor_buffer', 0))
        self.peak_read_queue = max(self.peak_read_queue, depths.get('read_queue', 0))
        for lane in ('lane_normal', 'lane_large', 'lane_huge'):
            self.peak_lane_backlog[lane] = max(
                self.peak_lane_backlog.get(lane, 0),
                depths.get(lane, 0),
            )
        for name, ms in phases.items():
            self._phase_acc[name] = self._phase_acc.get(name, 0.0) + ms

    def note_lane_backlog(self, depths: dict[str, int]) -> None:
        for lane in ('lane_normal', 'lane_large', 'lane_huge'):
            self.peak_lane_backlog[lane] = max(
                self.peak_lane_backlog.get(lane, 0),
                depths.get(lane, 0),
            )

    def note_heavy_dispatch(self, pending_heavy: int, *, lane: str = 'normal') -> None:
        self.heavy_submits += 1
        self.heavy_worker_dispatches.append(pending_heavy)
        self.lane_heavy_submits[lane] = self.lane_heavy_submits.get(lane, 0) + 1

    def note_apply_complete(self) -> None:
        self.apply_completions += 1

    def note_ordering_wait(self, gap: int, wait_ms: float) -> None:
        if gap > 0:
            self.ordering_wait_ms += wait_ms
            self.peak_ordering_gap = max(self.peak_ordering_gap, gap)

    def note_dispatch_denied(self, reason: str) -> None:
        self.dispatch_denied[reason] = self.dispatch_denied.get(reason, 0) + 1
        gate = _DENIAL_TO_GATE.get(reason)
        if gate:
            self.gate_blocked[gate] = self.gate_blocked.get(gate, 0) + 1

    def record_dispatch_audit(self, audit: dict[str, Any]) -> None:
        self.last_dispatch_audit = audit
        if len(self.dispatch_audit_samples) < 200:
            self.dispatch_audit_samples.append(audit)

    def note_heavy_apply_backpressure(self) -> None:
        self.heavy_apply_backpressure_events += 1

    def finalize_phases(self) -> None:
        if self.loop_count > 0:
            self.phase_ms = {
                k: round(v / self.loop_count, 4)
                for k, v in self._phase_acc.items()
            }

    def bottleneck_tree(self) -> list[dict[str, Any]]:
        self.finalize_phases()
        elapsed_ms = max((time.perf_counter() - self.t0) * 1000.0, 1.0)
        phase_totals = {k: v * self.loop_count for k, v in self._phase_acc.items()}
        nodes: list[dict[str, Any]] = [
            {
                'id': 'apply_stall',
                'wall_ms': round(self.ordering_wait_ms, 1),
                'pct': round(100.0 * self.ordering_wait_ms / elapsed_ms, 2),
                'detail': f"peak_gap={self.peak_ordering_gap}",
            },
            {
                'id': 'heavy_survivor_backlog',
                'wall_ms': round(phase_totals.get('heavy_submit', 0) + phase_totals.get('heavy_collect', 0), 1),
                'pct': round(100.0 * (phase_totals.get('heavy_submit', 0) + phase_totals.get('heavy_collect', 0)) / elapsed_ms, 2),
                'detail': f"peak={self.peak_survivor_buffer} lanes={self.peak_lane_backlog}",
            },
            {
                'id': 'large_doc_blocking',
                'wall_ms': float(self.large_doc_blocking_events),
                'pct': round(100.0 * self.large_doc_blocking_events / max(self.heavy_submits, 1), 2),
                'detail': f"dispatches={self.lane_heavy_submits}",
            },
            {
                'id': 'fast_filter',
                'wall_ms': round(phase_totals.get('fast_pull', 0) + phase_totals.get('fast_collect', 0), 1),
                'pct': round(100.0 * (phase_totals.get('fast_pull', 0) + phase_totals.get('fast_collect', 0)) / elapsed_ms, 2),
                'detail': f"submits={self.fast_submits}",
            },
            {
                'id': 'worker_idle',
                'wall_ms': round(self.worker_idle_ms, 1),
                'pct': round(100.0 * self.worker_idle_ms / elapsed_ms, 2),
                'detail': '',
            },
            {
                'id': 'reader_block',
                'wall_ms': float(self.reader_block_events),
                'pct': round(100.0 * self.backpressure_events / max(self.loop_count, 1), 2),
                'detail': f"events={self.reader_block_events}",
            },
            {
                'id': 'heavy_dispatch_denied',
                'wall_ms': float(sum(self.dispatch_denied.values())),
                'pct': 0.0,
                'detail': str(dict(sorted(self.dispatch_denied.items(), key=lambda kv: -kv[1])[:6])),
            },
            {
                'id': 'heavy_apply_backpressure',
                'wall_ms': float(self.heavy_apply_backpressure_events),
                'pct': 0.0,
                'detail': '',
            },
        ]
        nodes.sort(key=lambda n: -float(n['wall_ms']))
        return nodes

    def to_dict(self) -> dict[str, Any]:
        self.finalize_phases()
        elapsed = max(time.perf_counter() - self.t0, 1e-9)
        return {
            'elapsed_sec': round(elapsed, 2),
            'execution_backend': self.execution_backend,
            'loop_count': self.loop_count,
            'loops_per_sec': round(self.loop_count / elapsed, 1),
            'scheduler_sleep_ms': round(self.scheduler_sleep_ms, 1),
            'ordering_wait_ms': round(self.ordering_wait_ms, 1),
            'ordering_wait_pct': round(100.0 * self.ordering_wait_ms / max(elapsed * 1000.0, 1.0), 2),
            'apply_stall_ms': round(self.ordering_wait_ms, 1),
            'worker_idle_ms': round(self.worker_idle_ms, 1),
            'heavy_apply_backpressure_events': self.heavy_apply_backpressure_events,
            'seq_priority_dispatches': self.seq_priority_dispatches,
            'head_blocked_dispatches': self.head_blocked_dispatches,
            'dispatched_past_head': self.dispatched_past_head,
            'head_priority_dispatches': self.head_priority_dispatches,
            'dispatch_denied': dict(self.dispatch_denied),
            'dispatch_denied_total': sum(self.dispatch_denied.values()),
            'gate_blocked': dict(self.gate_blocked),
            'last_dispatch_audit': self.last_dispatch_audit,
            'backpressure_events': self.backpressure_events,
            'reader_block_events': self.reader_block_events,
            'fast_submits': self.fast_submits,
            'heavy_submits': self.heavy_submits,
            'fast_collects': self.fast_collects,
            'heavy_collects': self.heavy_collects,
            'apply_completions': self.apply_completions,
            'peak_apply_buffer': self.peak_apply_buffer,
            'peak_ordering_gap': self.peak_ordering_gap,
            'peak_fast_pending': self.peak_fast_pending,
            'peak_heavy_pending': self.peak_heavy_pending,
            'peak_survivor_buffer': self.peak_survivor_buffer,
            'peak_lane_backlog': self.peak_lane_backlog,
            'peak_read_queue': self.peak_read_queue,
            'large_doc_blocking_events': self.large_doc_blocking_events,
            'lane_slots': self.lane_slots,
            'lane_heavy_submits': self.lane_heavy_submits,
            'avg_phase_ms': self.phase_ms,
            'bottleneck_tree': self.bottleneck_tree(),
            'heavy_dispatch_depths': self.heavy_worker_dispatches[-50:],
        }

    def publish(self, work_dir: Path) -> None:
        from indw.store.io.atomic import atomic_write_text
        path = Path(work_dir) / 'pipeline_scheduler_report.json'
        atomic_write_text(path, json.dumps(self.to_dict(), indent=2))
