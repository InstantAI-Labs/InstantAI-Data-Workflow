from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from indw.schedule.config.resolve import env_float, env_int
from indw.schedule.config.policy import HUGE_SURVIVOR_CHARS, LARGE_SURVIVOR_CHARS
from indw.schedule.config.hardware import SystemHardwareProfile, probe_system_hardware

_tls = threading.local()
_BOUND: MergeTuneProfile | None = None


@dataclass(frozen=True)
class MergeTuneProfile:
    result_buffer_factor: int
    heavy_result_buffer_factor: int
    heavy_ooo_dispatch_limit: int
    apply_wait_idle_sec: float
    apply_wait_blocked_sec: float
    apply_wait_record_min_ms: float
    scheduler_idle_poll_sec: float
    batch_flush_sec: float
    fast_worker_share: float
    heavy_queue_multiplier: int
    apply_queue_multiplier: int
    fast_batch_scale: float
    heavy_batch_scale: float
    ipc_externalize_chars: int
    fast_buffer_floor: int
    heavy_buffer_floor: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'result_buffer_factor': self.result_buffer_factor,
            'heavy_result_buffer_factor': self.heavy_result_buffer_factor,
            'heavy_ooo_dispatch_limit': self.heavy_ooo_dispatch_limit,
            'apply_wait_idle_sec': self.apply_wait_idle_sec,
            'apply_wait_blocked_sec': self.apply_wait_blocked_sec,
            'apply_wait_record_min_ms': self.apply_wait_record_min_ms,
            'scheduler_idle_poll_sec': self.scheduler_idle_poll_sec,
            'batch_flush_sec': self.batch_flush_sec,
            'fast_worker_share': self.fast_worker_share,
            'heavy_queue_multiplier': self.heavy_queue_multiplier,
            'apply_queue_multiplier': self.apply_queue_multiplier,
            'fast_batch_scale': self.fast_batch_scale,
            'heavy_batch_scale': self.heavy_batch_scale,
            'ipc_externalize_chars': self.ipc_externalize_chars,
            'fast_buffer_floor': self.fast_buffer_floor,
            'heavy_buffer_floor': self.heavy_buffer_floor,
            'lane_routing_large_chars': LARGE_SURVIVOR_CHARS,
            'lane_routing_huge_chars': HUGE_SURVIVOR_CHARS,
        }


_DEFAULT = MergeTuneProfile(
    result_buffer_factor=10,
    heavy_result_buffer_factor=14,
    heavy_ooo_dispatch_limit=3,
    apply_wait_idle_sec=0.05,
    apply_wait_blocked_sec=0.015,
    apply_wait_record_min_ms=2.0,
    scheduler_idle_poll_sec=0.004,
    batch_flush_sec=0.05,
    fast_worker_share=0.42,
    heavy_queue_multiplier=4,
    apply_queue_multiplier=4,
    fast_batch_scale=1.0,
    heavy_batch_scale=1.0,
    ipc_externalize_chars=50_000,
    fast_buffer_floor=48,
    heavy_buffer_floor=64,
)


def resolve_merge_tune(
    *,
    workers: int,
    chunk_size: int,
    hw: SystemHardwareProfile | None = None,
) -> MergeTuneProfile:
    hw = hw or probe_system_hardware()
    cap = max(1, min(int(workers), max(1, hw.cpu_logical)))
    fast_share = 0.38
    if cap <= 2:
        fast_share = 0.42
    elif cap <= 4:
        fast_share = 0.40
    if hw.storage.storage_class in ('hdd', 'slow'):
        fast_share = min(fast_share, 0.32)

    fast_batch_scale = 1.1 if hw.storage.storage_class == 'nvme' else 1.0
    heavy_batch_scale = 1.05 if hw.storage.storage_class == 'nvme' else 1.0
    if hw.storage.storage_class in ('hdd', 'slow'):
        fast_batch_scale = 0.9
        heavy_batch_scale = 0.9

    chunk = max(1, int(chunk_size))
    buffer_boost = 1 if chunk >= 1000 else 0

    return MergeTuneProfile(
        result_buffer_factor=env_int('INSTANT_MERGE_RESULT_BUFFER_FACTOR', 10 + buffer_boost, minimum=4),
        heavy_result_buffer_factor=env_int(
            'INSTANT_MERGE_HEAVY_RESULT_BUFFER_FACTOR', 14 + buffer_boost, minimum=6,
        ),
        heavy_ooo_dispatch_limit=env_int(
            'INSTANT_MERGE_OOO_DISPATCH_LIMIT', 3 if cap >= 2 else 4, minimum=1,
        ),
        apply_wait_idle_sec=env_float('INSTANT_MERGE_APPLY_WAIT_IDLE_SEC', 0.05, minimum=0.005),
        apply_wait_blocked_sec=env_float('INSTANT_MERGE_APPLY_WAIT_BLOCKED_SEC', 0.015, minimum=0.005),
        apply_wait_record_min_ms=env_float('INSTANT_MERGE_APPLY_WAIT_RECORD_MIN_MS', 2.0, minimum=0.5),
        scheduler_idle_poll_sec=env_float('INSTANT_MERGE_SCHEDULER_IDLE_POLL_SEC', 0.004, minimum=0.001),
        batch_flush_sec=env_float('INSTANT_MERGE_BATCH_FLUSH_SEC', 0.05, minimum=0.02),
        fast_worker_share=env_float('INSTANT_MERGE_FAST_WORKER_SHARE', fast_share, minimum=0.2),
        heavy_queue_multiplier=env_int('INSTANT_MERGE_HEAVY_QUEUE_MULT', 4, minimum=2),
        apply_queue_multiplier=env_int('INSTANT_MERGE_APPLY_QUEUE_MULT', 4, minimum=2),
        fast_batch_scale=env_float('INSTANT_MERGE_FAST_BATCH_SCALE', fast_batch_scale, minimum=0.5),
        heavy_batch_scale=env_float('INSTANT_MERGE_HEAVY_BATCH_SCALE', heavy_batch_scale, minimum=0.5),
        ipc_externalize_chars=env_int(
            'INSTANT_MERGE_IPC_EXTERNALIZE_CHARS', 50_000, minimum=LARGE_SURVIVOR_CHARS,
        ),
        fast_buffer_floor=env_int('INSTANT_MERGE_FAST_BUFFER_FLOOR', max(48, cap * 24), minimum=16),
        heavy_buffer_floor=env_int('INSTANT_MERGE_HEAVY_BUFFER_FLOOR', max(64, cap * 32), minimum=24),
    )


def bind_merge_tune(profile: MergeTuneProfile | None) -> None:
    global _BOUND
    _BOUND = profile
    _tls.profile = profile


def get_merge_tune() -> MergeTuneProfile:
    prof = getattr(_tls, 'profile', None)
    if prof is not None:
        return prof
    if _BOUND is not None:
        return _BOUND
    return _DEFAULT


def lane_worker_slots(heavy_workers: int) -> tuple[int, int, int]:
    n = max(1, int(heavy_workers))
    tune = get_merge_tune()
    if n == 1:
        return 1, 1, 1
    if n == 2:
        return 2, 1, 0
    if n <= 4:
        normal = max(2, n - 1)
        return normal, 1, 0 if n < 4 else 1
    huge = max(1, n // 5)
    large = max(1, n // 4)
    normal = max(2, n - huge - large)
    if tune.fast_worker_share >= 0.4 and n <= 8:
        normal = max(normal, n - 2)
        huge = min(huge, 1)
    over = normal + large + huge - n
    if over > 0:
        normal = max(1, normal - over)
    return normal, large, huge


def ipc_externalize_threshold() -> int:
    return get_merge_tune().ipc_externalize_chars


def survivor_buffer_cap(*, apply_queue: int, heavy_queue: int) -> int:
    tune = get_merge_tune()
    return max(int(apply_queue) * 3, int(heavy_queue), tune.heavy_buffer_floor * 4)


def merge_drain_sec(*, time_limit_sec: float | None = None) -> float:
    base = env_float('INSTANT_MERGE_DRAIN_SEC', 90.0, minimum=15.0)
    if time_limit_sec is not None and time_limit_sec > 0:
        return min(base, max(15.0, float(time_limit_sec) * 0.75))
    return base


@dataclass(frozen=True)
class ProductionRunProfile:
    scale: str
    workers: int
    chunk_size: int
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'scale': self.scale,
            'workers': self.workers,
            'chunk_size': self.chunk_size,
            'purpose': self.purpose,
        }


def recommended_workers_for_hardware(hw: SystemHardwareProfile) -> int:
    from indw.config import defaults as D

    logical = max(1, hw.cpu_logical)
    physical = max(1, hw.cpu_count)
    cap = max(2, min(logical - 1, D.DEFAULT_MERGE_WORKERS_CAP))
    if hw.mem_budget_mb < 8192:
        cap = min(cap, 4)
    elif hw.mem_budget_mb < 16384:
        cap = min(cap, 6)
    if hw.storage.storage_class in ('hdd', 'slow'):
        cap = min(cap, max(2, physical))
    return max(2, cap)


def recommend_production_profiles(
    hw: SystemHardwareProfile | None = None,
) -> dict[str, ProductionRunProfile]:
    hw = hw or probe_system_hardware()
    prod_workers = recommended_workers_for_hardware(hw)
    medium_workers = min(4, prod_workers)
    return {
        'small': ProductionRunProfile(
            scale='small',
            workers=2,
            chunk_size=500,
            purpose='CI validation, gate parity, dev corpus slices',
        ),
        'medium': ProductionRunProfile(
            scale='medium',
            workers=medium_workers,
            chunk_size=750,
            purpose='partial corpus staging and throughput calibration',
        ),
        'production': ProductionRunProfile(
            scale='production',
            workers=prod_workers,
            chunk_size=1000,
            purpose='full corpus merge at hardware-adaptive worker cap',
        ),
    }


def merge_tune_env_exports(tune: MergeTuneProfile) -> dict[str, str]:
    return {
        'INSTANT_MERGE_RESULT_BUFFER_FACTOR': str(tune.result_buffer_factor),
        'INSTANT_MERGE_HEAVY_RESULT_BUFFER_FACTOR': str(tune.heavy_result_buffer_factor),
        'INSTANT_MERGE_OOO_DISPATCH_LIMIT': str(tune.heavy_ooo_dispatch_limit),
        'INSTANT_MERGE_APPLY_WAIT_IDLE_SEC': str(tune.apply_wait_idle_sec),
        'INSTANT_MERGE_APPLY_WAIT_BLOCKED_SEC': str(tune.apply_wait_blocked_sec),
        'INSTANT_MERGE_APPLY_WAIT_RECORD_MIN_MS': str(tune.apply_wait_record_min_ms),
        'INSTANT_MERGE_SCHEDULER_IDLE_POLL_SEC': str(tune.scheduler_idle_poll_sec),
        'INSTANT_MERGE_BATCH_FLUSH_SEC': str(tune.batch_flush_sec),
        'INSTANT_MERGE_FAST_WORKER_SHARE': str(tune.fast_worker_share),
        'INSTANT_MERGE_HEAVY_QUEUE_MULT': str(tune.heavy_queue_multiplier),
        'INSTANT_MERGE_APPLY_QUEUE_MULT': str(tune.apply_queue_multiplier),
        'INSTANT_MERGE_FAST_BATCH_SCALE': str(tune.fast_batch_scale),
        'INSTANT_MERGE_HEAVY_BATCH_SCALE': str(tune.heavy_batch_scale),
        'INSTANT_MERGE_IPC_EXTERNALIZE_CHARS': str(tune.ipc_externalize_chars),
        'INSTANT_MERGE_FAST_BUFFER_FLOOR': str(tune.fast_buffer_floor),
        'INSTANT_MERGE_HEAVY_BUFFER_FLOOR': str(tune.heavy_buffer_floor),
    }
