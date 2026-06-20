from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StorageProfile:
    write_mbps: float
    read_mbps: float
    storage_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'write_mbps': round(self.write_mbps, 1),
            'read_mbps': round(self.read_mbps, 1),
            'storage_class': self.storage_class,
        }


@dataclass(frozen=True)
class SystemHardwareProfile:
    cpu_count: int
    cpu_logical: int
    mem_total_mb: float
    mem_available_mb: float
    mem_budget_mb: float
    storage: StorageProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            'cpu_count': self.cpu_count,
            'cpu_logical': self.cpu_logical,
            'mem_total_mb': round(self.mem_total_mb, 1),
            'mem_available_mb': round(self.mem_available_mb, 1),
            'mem_budget_mb': round(self.mem_budget_mb, 1),
            'storage': self.storage.to_dict(),
        }


def _mem_stats() -> tuple[float, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return float(vm.total) / (1024 * 1024), float(vm.available) / (1024 * 1024)
    except Exception:
        return 4096.0, 2048.0


def _probe_disk_mbps(work_dir: Path | None, *, size_mb: int = 32) -> tuple[float, float]:
    root = work_dir if work_dir is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'.instant_io_probe_{os.getpid()}.bin'
    payload = os.urandom(size_mb * 1024 * 1024)
    try:
        t0 = time.perf_counter()
        path.write_bytes(payload)
        write_sec = max(time.perf_counter() - t0, 1e-6)
        t1 = time.perf_counter()
        _ = path.read_bytes()
        read_sec = max(time.perf_counter() - t1, 1e-6)
        write_mbps = (len(payload) / (1024 * 1024)) / write_sec
        read_mbps = (len(payload) / (1024 * 1024)) / read_sec
        return write_mbps, read_mbps
    except OSError:
        return 120.0, 200.0
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _storage_class(write_mbps: float) -> str:
    if write_mbps >= 800:
        return 'nvme'
    if write_mbps >= 250:
        return 'ssd'
    if write_mbps >= 80:
        return 'hdd'
    return 'slow'


_HW_CACHE: SystemHardwareProfile | None = None


def probe_system_hardware(work_dir: Path | None = None, *, force: bool = False) -> SystemHardwareProfile:
    global _HW_CACHE
    if _HW_CACHE is not None and not force:
        return _HW_CACHE
    from indw.schedule.config.resolve import env_flag
    if not env_flag('INSTANT_MERGE_HW_PROBE', True):
        logical = max(1, int(os.cpu_count() or 1))
        mem_total, mem_avail = _mem_stats()
        profile = SystemHardwareProfile(
            cpu_count=logical,
            cpu_logical=logical,
            mem_total_mb=mem_total,
            mem_available_mb=mem_avail,
            mem_budget_mb=min(mem_total * 0.65, max(mem_avail * 0.85, 1024.0)),
            storage=StorageProfile(write_mbps=200.0, read_mbps=300.0, storage_class='ssd'),
        )
        _HW_CACHE = profile
        return profile
    logical = max(1, int(os.cpu_count() or 1))
    physical = logical
    try:
        import psutil
        physical = max(1, int(psutil.cpu_count(logical=False) or logical))
    except Exception:
        pass
    mem_total, mem_avail = _mem_stats()
    env_mem = os.environ.get('INSTANT_MERGE_MEM_BUDGET_MB')
    if env_mem:
        try:
            budget = float(env_mem)
        except ValueError:
            budget = min(mem_total * 0.65, mem_avail * 0.85)
    else:
        budget = min(mem_total * 0.65, max(mem_avail * 0.85, 1024.0))
    write_mbps, read_mbps = _probe_disk_mbps(work_dir)
    storage = StorageProfile(
        write_mbps=write_mbps,
        read_mbps=read_mbps,
        storage_class=_storage_class(write_mbps),
    )
    profile = SystemHardwareProfile(
        cpu_count=physical,
        cpu_logical=logical,
        mem_total_mb=mem_total,
        mem_available_mb=mem_avail,
        mem_budget_mb=budget,
        storage=storage,
    )
    _HW_CACHE = profile
    return profile
