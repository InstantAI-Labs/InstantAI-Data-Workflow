from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from indw.schedule.monitor.cpu import collect_cpu_stats as _collect_cpu_stats
except Exception:
    def _collect_cpu_stats() -> Any:
        return type('CpuStats', (), {'utilization_pct': 0.0})()

try:
    from indw.tools.reports.benchmark import peak_rss_mb as _peak_rss_mb
except Exception:
    def _peak_rss_mb() -> float:
        return 0.0

try:
    from indw.clean.artifact.evidence_cache import session_cache_stats as _session_cache_stats
except Exception:
    def _session_cache_stats() -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class HardwareSnapshot:
    cpu_pct: float = 0.0
    rss_mb: float = 0.0
    cache_hit_rate: float = 0.0
    queue_depth: int = 0
    docs_per_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'cpu_pct': round(self.cpu_pct, 1),
            'rss_mb': round(self.rss_mb, 1),
            'cache_hit_rate': round(self.cache_hit_rate, 4),
            'queue_depth': self.queue_depth,
            'docs_per_sec': round(self.docs_per_sec, 3),
        }


def collect_hardware_snapshot(
    *,
    queue_depth: int = 0,
    docs_per_sec: float = 0.0,
    cpu_pct: float | None = None,
    rss_mb: float | None = None,
    cache_hit_rate: float | None = None,
) -> HardwareSnapshot:
    if cpu_pct is None:
        try:
            cpu_pct = float(_collect_cpu_stats().utilization_pct or 0.0)
        except Exception:
            cpu_pct = 0.0
    if rss_mb is None:
        try:
            rss_mb = float(_peak_rss_mb())
        except Exception:
            rss_mb = 0.0
    if cache_hit_rate is None:
        cache_hit = 0.0
        try:
            stats = _session_cache_stats()
            hits = sum(int(v.get('hits', 0)) for v in stats.values() if isinstance(v, dict))
            misses = sum(int(v.get('misses', 0)) for v in stats.values() if isinstance(v, dict))
            cache_hit = hits / max(hits + misses, 1)
        except Exception:
            pass
    else:
        cache_hit = cache_hit_rate
    return HardwareSnapshot(
        cpu_pct=float(cpu_pct),
        rss_mb=float(rss_mb),
        cache_hit_rate=cache_hit,
        queue_depth=queue_depth,
        docs_per_sec=docs_per_sec,
    )


def adapt_routing_params(
    hw: HardwareSnapshot,
    *,
    verify_threshold: float,
    cache_boost: int,
) -> tuple[float, int]:
    if not _hardware_adapt_enabled():
        return verify_threshold, cache_boost
    thr = verify_threshold
    boost = cache_boost
    if hw.cpu_pct >= 92.0 or hw.rss_mb >= _rss_pressure_mb():
        thr = min(0.99, thr + 0.03)
        boost = max(1, boost - 1)
    elif hw.cpu_pct <= 55.0 and hw.cache_hit_rate >= 0.65:
        thr = max(0.85, thr - 0.01)
        boost = min(4, boost + 1)
    if hw.queue_depth > 32:
        thr = min(0.99, thr + 0.01)
    return thr, boost


def _hardware_adapt_enabled() -> bool:
    from indw.schedule.config.policy import active_or_built_policy
    return active_or_built_policy().hardware_adapt_enabled


def _rss_pressure_mb() -> float:
    from indw.schedule.config.policy import active_or_built_policy
    return active_or_built_policy().rss_pressure_mb
