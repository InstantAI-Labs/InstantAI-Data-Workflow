from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indw.schedule.dispatch.alloc import PIPELINE_STAGES


@dataclass
class StageLiveStats:
    calls: int = 0
    in_docs: int = 0
    out_docs: int = 0
    dropped: int = 0
    wall_sec: float = 0.0
    queue_depth: int = 0
    latency_ms_sum: float = 0.0

    def record(
        self,
        *,
        wall_sec: float,
        in_docs: int = 1,
        out_docs: int = 1,
        dropped: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        self.calls += 1
        self.in_docs += in_docs
        self.out_docs += out_docs
        self.dropped += dropped
        self.wall_sec += max(0.0, wall_sec)
        self.latency_ms_sum += max(0.0, latency_ms)

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_sum / max(self.calls, 1)

    @property
    def docs_per_sec(self) -> float:
        return self.in_docs / max(self.wall_sec, 1e-9)

    def to_dict(self) -> dict[str, Any]:
        return {
            'calls': self.calls,
            'in_docs': self.in_docs,
            'out_docs': self.out_docs,
            'dropped': self.dropped,
            'wall_sec': round(self.wall_sec, 4),
            'queue_depth': self.queue_depth,
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'docs_per_sec': round(self.docs_per_sec, 3),
        }


@dataclass
class SchedulerLiveStats:
    fast_submits: int = 0
    heavy_submits: int = 0
    fast_collects: int = 0
    heavy_collects: int = 0
    idle_loops: int = 0
    first_result_ms: float = 0.0
    first_apply_ms: float = 0.0
    peak_fast_pending: int = 0
    peak_heavy_pending: int = 0
    peak_survivor_buffer: int = 0
    peak_read_queue: int = 0
    worker_util_pct: float = 0.0

    def record_depths(
        self,
        *,
        fast_pending: int,
        heavy_pending: int,
        survivor_buffer: int,
        read_queue: int,
        fast_workers: int,
        heavy_workers: int,
    ) -> None:
        self.peak_fast_pending = max(self.peak_fast_pending, fast_pending)
        self.peak_heavy_pending = max(self.peak_heavy_pending, heavy_pending)
        self.peak_survivor_buffer = max(self.peak_survivor_buffer, survivor_buffer)
        self.peak_read_queue = max(self.peak_read_queue, read_queue)
        cap = max(fast_workers + heavy_workers, 1)
        active = fast_pending + heavy_pending
        self.worker_util_pct = min(100.0, (active / cap) * 100.0)

    def note_first_result(self, *, elapsed_ms: float) -> None:
        if self.first_result_ms <= 0:
            self.first_result_ms = elapsed_ms

    def note_first_apply(self, *, elapsed_ms: float) -> None:
        if self.first_apply_ms <= 0:
            self.first_apply_ms = elapsed_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            'fast_submits': self.fast_submits,
            'heavy_submits': self.heavy_submits,
            'fast_collects': self.fast_collects,
            'heavy_collects': self.heavy_collects,
            'idle_loops': self.idle_loops,
            'first_result_ms': round(self.first_result_ms, 1),
            'first_apply_ms': round(self.first_apply_ms, 1),
            'peak_fast_pending': self.peak_fast_pending,
            'peak_heavy_pending': self.peak_heavy_pending,
            'peak_survivor_buffer': self.peak_survivor_buffer,
            'peak_read_queue': self.peak_read_queue,
            'worker_util_pct': round(self.worker_util_pct, 1),
        }


@dataclass
class PipelineStageMonitor:
    stages: dict[str, StageLiveStats] = field(default_factory=dict)
    scheduler: SchedulerLiveStats = field(default_factory=SchedulerLiveStats)
    cpu_pct: float = 0.0
    rss_mb: float = 0.0
    cache_hit_rate: float = 0.0
    tokens_per_sec: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)
    bottlenecks: list[str] = field(default_factory=list)

    def stage(self, name: str) -> StageLiveStats:
        if name not in self.stages:
            self.stages[name] = StageLiveStats()
        return self.stages[name]

    def update_runtime(
        self,
        *,
        cpu_pct: float = 0.0,
        rss_mb: float = 0.0,
        cache_hit_rate: float = 0.0,
        tokens_per_sec: float = 0.0,
        queue_depths: dict[str, int] | None = None,
        scheduler_depths: dict[str, int] | None = None,
        fast_workers: int = 0,
        heavy_workers: int = 0,
    ) -> None:
        self.cpu_pct = cpu_pct
        self.rss_mb = rss_mb
        self.cache_hit_rate = cache_hit_rate
        self.tokens_per_sec = tokens_per_sec
        if queue_depths:
            for name, depth in queue_depths.items():
                self.stage(name).queue_depth = depth
        if scheduler_depths is not None:
            self.scheduler.record_depths(
                fast_pending=int(scheduler_depths.get('fast_pending', 0)),
                heavy_pending=int(scheduler_depths.get('heavy_pending', 0)),
                survivor_buffer=int(scheduler_depths.get('survivor_buffer', 0)),
                read_queue=int(scheduler_depths.get('read_queue', 0)),
                fast_workers=fast_workers,
                heavy_workers=heavy_workers,
            )
        self._detect_bottlenecks(scheduler_depths or {})

    def _detect_bottlenecks(self, scheduler_depths: dict[str, int]) -> None:
        issues: list[str] = []
        fast_p = int(scheduler_depths.get('fast_pending', 0))
        heavy_p = int(scheduler_depths.get('heavy_pending', 0))
        survivor = int(scheduler_depths.get('survivor_buffer', 0))
        read_q = int(scheduler_depths.get('read_queue', 0))
        backlog = read_q + fast_p + heavy_p + survivor

        if self.cpu_pct < 35.0 and backlog > 4:
            issues.append('cpu_idle_with_backlog')
        if self.cpu_pct > 95.0:
            issues.append('cpu_saturated')
        if survivor > 8 and heavy_p < 2:
            issues.append('heavy_starvation')
        if read_q > 4 and fast_p == 0:
            issues.append('fast_pool_starvation')
        if self.scheduler.worker_util_pct < 40.0 and backlog > 2:
            issues.append('worker_underutilized')
        ranked = sorted(
            self.stages.items(),
            key=lambda item: item[1].queue_depth,
            reverse=True,
        )
        for name, st in ranked[:3]:
            if st.queue_depth > 4 and st.docs_per_sec < 0.5:
                issues.append(f'queue_backlog:{name}')
        idle = [n for n, s in self.stages.items() if s.calls > 0 and s.docs_per_sec < 0.05]
        if idle:
            issues.append(f'low_throughput:{",".join(idle[:3])}')
        self.bottlenecks = issues

    def to_dict(self) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        stage_rows = {name: self.stage(name).to_dict() for name in PIPELINE_STAGES if name in self.stages}
        return {
            'elapsed_sec': round(elapsed, 2),
            'cpu_pct': round(self.cpu_pct, 1),
            'rss_mb': round(self.rss_mb, 1),
            'cache_hit_rate': round(self.cache_hit_rate, 4),
            'tokens_per_sec': round(self.tokens_per_sec, 2),
            'bottlenecks': self.bottlenecks,
            'scheduler': self.scheduler.to_dict(),
            'stages': stage_rows,
        }

    def publish(self, work_dir: Path, *, force: bool = False) -> None:
        from indw.store.io.atomic import atomic_write_text
        path = Path(work_dir) / 'pipeline_live_metrics.json'
        atomic_write_text(path, json.dumps(self.to_dict(), indent=2))
