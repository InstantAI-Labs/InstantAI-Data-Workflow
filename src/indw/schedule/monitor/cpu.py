from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CpuStats:
    utilization_pct: Optional[float] = None


def collect_cpu_stats() -> CpuStats:
    try:
        import psutil
        return CpuStats(utilization_pct=float(psutil.cpu_percent(interval=None)))
    except Exception:
        return CpuStats(utilization_pct=None)
