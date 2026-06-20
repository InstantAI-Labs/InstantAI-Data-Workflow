from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indw.schedule.config.hardware import SystemHardwareProfile, probe_system_hardware
from indw.schedule.config.policy import PipelineSignals

STAGE_INGEST = 'ingest'
STAGE_FAST_PREPROCESS = 's1_fast_preprocess'
STAGE_STRUCTURAL_FILTER = 's2_structural_filter'
STAGE_FAST_FILTER = 's2_fast_filter'
STAGE_DOC_DEDUP = 's2_doc_dedup'
STAGE_METADATA = 's2_metadata'
STAGE_INTERMEDIATE = 's3_intermediate'
STAGE_ADMISSION = 's3_admission'
STAGE_HIGH_QUALITY = 's4_high_quality'
STAGE_FINAL_VALIDATION = 's5_final_validation'
STAGE_OUTPUT = 's6_output'
STAGE_INTEL_PREVIEW = 's4_intel_preview'

PIPELINE_STAGES = (
    STAGE_INGEST,
    STAGE_FAST_PREPROCESS,
    STAGE_FAST_FILTER,
    STAGE_DOC_DEDUP,
    STAGE_STRUCTURAL_FILTER,
    STAGE_METADATA,
    STAGE_ADMISSION,
    STAGE_INTERMEDIATE,
    STAGE_INTEL_PREVIEW,
    STAGE_HIGH_QUALITY,
    STAGE_FINAL_VALIDATION,
    STAGE_OUTPUT,
)

PREPROCESS_STAGES = (
    STAGE_FAST_PREPROCESS,
    STAGE_FAST_FILTER,
    STAGE_DOC_DEDUP,
    STAGE_STRUCTURAL_FILTER,
    STAGE_METADATA,
    STAGE_ADMISSION,
    STAGE_INTERMEDIATE,
    STAGE_INTEL_PREVIEW,
    STAGE_HIGH_QUALITY,
)

APPLY_STAGES = (STAGE_FINAL_VALIDATION, STAGE_OUTPUT)
FAST_STAGES = (
    STAGE_FAST_PREPROCESS,
    STAGE_FAST_FILTER,
    STAGE_DOC_DEDUP,
    STAGE_STRUCTURAL_FILTER,
    STAGE_METADATA,
)
HEAVY_STAGES = (STAGE_INTERMEDIATE, STAGE_INTEL_PREVIEW, STAGE_HIGH_QUALITY)


@dataclass(frozen=True)
class StageAllocation:
    fast_workers: int
    heavy_workers: int
    ingest_queue: int
    fast_queue: int
    heavy_queue: int
    apply_queue: int
    fast_batch: int
    heavy_batch: int
    stream_batch: int
    batch_flush_sec: float

    def total_workers(self) -> int:
        return self.fast_workers + self.heavy_workers

    def to_dict(self) -> dict[str, Any]:
        return {
            'fast_workers': self.fast_workers,
            'heavy_workers': self.heavy_workers,
            'ingest_queue': self.ingest_queue,
            'fast_queue': self.fast_queue,
            'heavy_queue': self.heavy_queue,
            'apply_queue': self.apply_queue,
            'fast_batch': self.fast_batch,
            'heavy_batch': self.heavy_batch,
            'stream_batch': self.stream_batch,
            'batch_flush_sec': self.batch_flush_sec,
        }


@dataclass
class StageScheduler:
    hw: SystemHardwareProfile
    total_workers: int
    chunk_size: int
    fast_q_depth: int = 0
    heavy_q_depth: int = 0
    signals: PipelineSignals = field(default_factory=PipelineSignals)

    def allocate(self, *, force: bool = False) -> StageAllocation:
        cores = max(1, self.hw.cpu_logical)
        cap = max(1, min(cores, self.total_workers))
        rss = self.signals.rss_mb
        pressure = self.hw.mem_budget_mb

        from indw.schedule.config.tune import get_merge_tune
        tune = get_merge_tune()
        fast_share = tune.fast_worker_share
        if self.heavy_q_depth > self.fast_q_depth * 2:
            fast_share = min(fast_share, 0.28)
        elif self.fast_q_depth > self.heavy_q_depth * 2 and self.signals.cpu_pct < 70:
            fast_share = min(0.48, fast_share + 0.06)
        if rss >= pressure * 0.88:
            fast_share = min(fast_share, 0.30)

        fast_w = max(1, int(round(cap * fast_share)))
        heavy_w = max(1, cap - fast_w)

        fast_batch = max(1, min(64 if cap <= 8 else 128, int(self.chunk_size * 0.12 * tune.fast_batch_scale)))
        heavy_batch = max(1, min(32, int(self.chunk_size * 0.08 * tune.heavy_batch_scale)))
        stream_batch = max(4 if cap <= 8 else 1, min(8, fast_batch // 4 or 1))
        if self.hw.storage.storage_class == 'nvme':
            fast_batch = min(64 if cap <= 8 else 128, int(fast_batch * 1.05))
        elif self.hw.storage.storage_class in ('hdd', 'slow'):
            fast_batch = max(1, int(fast_batch * 0.9))
            heavy_batch = max(1, int(heavy_batch * 0.9))

        return StageAllocation(
            fast_workers=fast_w,
            heavy_workers=heavy_w,
            ingest_queue=max(4, cap * 2),
            fast_queue=max(4, fast_w * 2),
            heavy_queue=max(4, heavy_w * tune.heavy_queue_multiplier),
            apply_queue=max(12, cap * tune.apply_queue_multiplier),
            fast_batch=fast_batch,
            heavy_batch=heavy_batch,
            stream_batch=stream_batch,
            batch_flush_sec=tune.batch_flush_sec,
        )

    def heavy_submit_threshold(
        self,
        alloc: StageAllocation,
        *,
        pending_heavy: int,
        survivor_count: int,
        ordering_gap: int = 0,
    ) -> int:
        if ordering_gap > 0:
            return 1
        idle = alloc.heavy_workers - pending_heavy
        if idle > 0 and survivor_count > 0:
            return 1
        if pending_heavy >= alloc.heavy_workers:
            return alloc.heavy_batch
        return max(1, min(alloc.heavy_batch, alloc.heavy_workers))

    def refresh_depths(
        self,
        *,
        fast_pending: int,
        heavy_pending: int,
        survivor_buffer: int,
        read_queue: int,
    ) -> None:
        self.fast_q_depth = fast_pending + read_queue
        self.heavy_q_depth = heavy_pending + survivor_buffer


def plan_pipelined_alloc(
    *,
    workers: int,
    chunk_size: int,
    merge_work: Path | str | None = None,
) -> tuple[SystemHardwareProfile, StageAllocation]:
    hw = probe_system_hardware(Path(merge_work) if merge_work is not None else None)
    alloc = StageScheduler(hw=hw, total_workers=workers, chunk_size=chunk_size).allocate()
    return hw, alloc


@dataclass(frozen=True)
class StageAllocationV2(StageAllocation):
    preprocess_workers: int = 1
    filter_workers: int = 1
    stage0_workers: int = 1
    dedup_shards: int = 2
    clean_workers: int = 1
    pci_workers: int = 1
    acim_workers: int = 1
    embed_workers: int = 0
    preprocess_queue: int = 4
    filter_queue: int = 4
    stage0_queue: int = 4
    pci_queue: int = 4
    acim_queue: int = 4
    clean_queue: int = 4

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            'preprocess_workers': self.preprocess_workers,
            'filter_workers': self.filter_workers,
            'stage0_workers': self.stage0_workers,
            'dedup_shards': self.dedup_shards,
            'clean_workers': self.clean_workers,
            'pci_workers': self.pci_workers,
            'acim_workers': self.acim_workers,
            'embed_workers': self.embed_workers,
            'preprocess_queue': self.preprocess_queue,
            'filter_queue': self.filter_queue,
            'stage0_queue': self.stage0_queue,
            'pci_queue': self.pci_queue,
            'acim_queue': self.acim_queue,
            'clean_queue': self.clean_queue,
        })
        return base


def plan_graph_alloc(
    *,
    workers: int,
    chunk_size: int,
    merge_work: Path | str | None = None,
    dedup_shards: int = 0,
) -> tuple[SystemHardwareProfile, StageAllocationV2]:
    import os
    hw, base = plan_pipelined_alloc(
        workers=workers, chunk_size=chunk_size, merge_work=merge_work,
    )
    cap = max(1, base.fast_workers + base.heavy_workers)
    from indw.schedule.graph.config import pipeline_dedup_shards
    shards = dedup_shards or pipeline_dedup_shards() or max(2, min(4, cap // 4 or 1))
    fast_third = max(1, base.fast_workers // 3)
    preprocess_w = max(1, fast_third)
    filter_w = max(1, fast_third)
    stage0_w = max(1, base.fast_workers - preprocess_w - filter_w)
    clean_w = max(1, base.heavy_workers - 2) if base.heavy_workers > 2 else max(1, base.heavy_workers)
    pci_w = 1
    acim_w = 1
    if base.heavy_workers <= 2:
        clean_w = max(1, base.heavy_workers)

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name, '').strip()
        if not raw:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    preprocess_w = _env_int('INSTANT_PREPROCESS_WORKERS', preprocess_w)
    filter_w = _env_int('INSTANT_FILTER_WORKERS', filter_w)
    stage0_w = _env_int('INSTANT_STAGE0_WORKERS', stage0_w)
    clean_w = _env_int('INSTANT_CLEAN_WORKERS', clean_w)
    shard_raw = os.environ.get('INSTANT_DEDUP_SHARDS', '').strip()
    if shard_raw:
        try:
            shards = max(1, int(shard_raw))
        except ValueError:
            pass

    return hw, StageAllocationV2(
        fast_workers=base.fast_workers,
        heavy_workers=base.heavy_workers,
        ingest_queue=base.ingest_queue,
        fast_queue=base.fast_queue,
        heavy_queue=base.heavy_queue,
        apply_queue=base.apply_queue,
        fast_batch=base.fast_batch,
        heavy_batch=base.heavy_batch,
        stream_batch=base.stream_batch,
        batch_flush_sec=base.batch_flush_sec,
        preprocess_workers=preprocess_w,
        filter_workers=filter_w,
        stage0_workers=stage0_w,
        dedup_shards=shards,
        clean_workers=clean_w,
        pci_workers=pci_w,
        acim_workers=acim_w,
        embed_workers=0,
        preprocess_queue=max(4, preprocess_w * 2),
        filter_queue=max(4, filter_w * 2),
        stage0_queue=max(4, stage0_w * 2),
        pci_queue=max(4, pci_w * 2),
        acim_queue=max(4, acim_w * 2),
        clean_queue=max(4, clean_w * 2),
    )
