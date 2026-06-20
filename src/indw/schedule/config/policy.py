from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, replace
from typing import Any

from indw.config import defaults as D
from indw.schedule.config.resolve import (
    env_bool,
    env_float,
    env_int,
    env_optional_int,
    resolve_explicit_or_env,
)

MERGE_READ_SENTINEL = object()
LARGE_SURVIVOR_CHARS = 30_000
HUGE_SURVIVOR_CHARS = 80_000

_tls = threading.local()
_BOUND: MergeRuntime | None = None


@dataclass(frozen=True)
class HardwareProfile:
    cpu_count: int
    mem_budget_mb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            'cpu_count': self.cpu_count,
            'mem_budget_mb': round(self.mem_budget_mb, 1),
        }


@dataclass
class PipelineSignals:
    cpu_pct: float = 0.0
    rss_mb: float = 0.0
    cache_hit_rate: float = 0.0
    queue_depth: int = 0
    docs_per_sec: float = 0.0
    active_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'cpu_pct': round(self.cpu_pct, 1),
            'rss_mb': round(self.rss_mb, 1),
            'cache_hit_rate': round(self.cache_hit_rate, 4),
            'queue_depth': self.queue_depth,
            'docs_per_sec': round(self.docs_per_sec, 3),
            'active_workers': self.active_workers,
        }


@dataclass(frozen=True)
class RuntimePolicy:
    workers: int
    chunk_size: int
    checkpoint_interval: int
    checkpoint_save_sec: float
    metrics_snapshot_sec: float
    doc_wall_budget_sec: float
    doc_max_chars: int
    batch_timeout_sec: float
    evidence_cache_size: int
    raw_feature_cache_size: int
    nav_feature_cache_size: int
    publication_role_cache_size: int
    structure_cache_size: int
    layout_cache_size: int
    scaffold_cache_size: int
    rss_pressure_mb: float
    hardware_adapt_enabled: bool
    lci_num_shards: int
    source: str = 'adaptive'

    def to_dict(self) -> dict[str, Any]:
        return {
            'workers': self.workers,
            'chunk_size': self.chunk_size,
            'checkpoint_interval': self.checkpoint_interval,
            'checkpoint_save_sec': self.checkpoint_save_sec,
            'metrics_snapshot_sec': self.metrics_snapshot_sec,
            'doc_wall_budget_sec': self.doc_wall_budget_sec,
            'doc_max_chars': self.doc_max_chars,
            'batch_timeout_sec': self.batch_timeout_sec,
            'evidence_cache_size': self.evidence_cache_size,
            'raw_feature_cache_size': self.raw_feature_cache_size,
            'nav_feature_cache_size': self.nav_feature_cache_size,
            'publication_role_cache_size': self.publication_role_cache_size,
            'structure_cache_size': self.structure_cache_size,
            'layout_cache_size': self.layout_cache_size,
            'scaffold_cache_size': self.scaffold_cache_size,
            'rss_pressure_mb': self.rss_pressure_mb,
            'hardware_adapt_enabled': self.hardware_adapt_enabled,
            'lci_num_shards': self.lci_num_shards,
            'source': self.source,
        }

    def cache_sizes(self) -> dict[str, int]:
        return {
            'evidence': self.evidence_cache_size,
            'raw': self.raw_feature_cache_size,
            'nav': self.nav_feature_cache_size,
            'publication': self.publication_role_cache_size,
            'structure': self.structure_cache_size,
            'layout': self.layout_cache_size,
            'scaffold': self.scaffold_cache_size,
        }


def detect_hardware_profile(work_dir: Any = None) -> HardwareProfile:
    from indw.schedule.config.resolve import env_flag
    if env_flag('INSTANT_MERGE_HW_PROBE', True):
        from pathlib import Path
        from indw.schedule.config.hardware import probe_system_hardware
        path = Path(work_dir) if work_dir is not None else None
        sys = probe_system_hardware(path)
        return HardwareProfile(cpu_count=sys.cpu_logical, mem_budget_mb=sys.mem_budget_mb)
    cores = max(1, int(os.cpu_count() or 1))
    mem_mb = 4096.0
    try:
        from indw.tools.reports.benchmark import peak_rss_mb
        rss = float(peak_rss_mb())
        if rss > 256:
            mem_mb = max(mem_mb, rss * 2.5)
    except Exception:
        pass
    env_mem = env_optional_int('INSTANT_MERGE_MEM_BUDGET_MB')
    if env_mem is not None:
        mem_mb = float(env_mem)
    return HardwareProfile(cpu_count=cores, mem_budget_mb=mem_mb)


def _rss_pressure_mb(hw: HardwareProfile) -> float:
    env_val = env_optional_int('INSTANT_LCI_RSS_PRESSURE_MB')
    if env_val is not None:
        return max(512.0, float(env_val))
    return max(512.0, min(hw.mem_budget_mb * 0.72, 2400.0))


def _cache_scale(hw: HardwareProfile, sig: PipelineSignals) -> float:
    pressure = _rss_pressure_mb(hw)
    if sig.rss_mb >= pressure * 0.88:
        return 0.72
    if sig.rss_mb >= pressure * 0.72:
        return 0.85
    if sig.rss_mb < pressure * 0.35 and hw.cpu_count >= 6:
        return 1.12
    return 1.0


def _adaptive_workers(hw: HardwareProfile, sig: PipelineSignals) -> int:
    base = max(1, min(hw.cpu_count, D.DEFAULT_MERGE_WORKERS_CAP))
    pressure = _rss_pressure_mb(hw)
    if sig.rss_mb >= pressure * 0.9:
        return max(1, base - 1)
    if sig.cpu_pct >= 92.0:
        return max(1, base - 1)
    if sig.queue_depth > base * 2 and sig.cpu_pct < 65.0:
        return min(D.DEFAULT_MERGE_WORKERS_CAP, base + 1)
    return base


def _adaptive_chunk(hw: HardwareProfile, sig: PipelineSignals, workers: int) -> int:
    base = D.DEFAULT_MERGE_CHUNK_SIZE
    pressure = _rss_pressure_mb(hw)
    if sig.rss_mb >= pressure * 0.85:
        size = int(base * 0.82)
    elif sig.docs_per_sec >= 0.75 and sig.cpu_pct < 72.0:
        size = int(base * 1.12)
    else:
        size = base
    per_worker = max(D.MIN_MERGE_CHUNK_SIZE, size // max(workers, 1))
    size = max(D.MIN_MERGE_CHUNK_SIZE, min(D.MAX_MERGE_CHUNK_SIZE, per_worker * workers))
    return size


def _adaptive_checkpoint_interval(sig: PipelineSignals) -> int:
    base = D.MERGE_CHECKPOINT_INTERVAL
    if sig.docs_per_sec >= 1.2:
        return max(100, int(base * 1.4))
    if sig.docs_per_sec <= 0.15:
        return max(50, int(base * 0.6))
    return base


def _adaptive_doc_budget(hw: HardwareProfile, sig: PipelineSignals) -> float:
    base = float(D.MERGE_DOC_WALL_BUDGET_SEC)
    if sig.cpu_pct >= 90.0 or sig.rss_mb >= _rss_pressure_mb(hw) * 0.92:
        return max(45.0, base * 0.85)
    if sig.docs_per_sec < 0.2:
        return min(120.0, base * 1.1)
    return base


def build_runtime_policy(
    *,
    workers: int | None = None,
    chunk_size: int | None = None,
    checkpoint_interval: int | None = None,
    signals: PipelineSignals | None = None,
    hw: HardwareProfile | None = None,
) -> RuntimePolicy:
    hw = hw or detect_hardware_profile()
    sig = signals or PipelineSignals()
    source_parts: list[str] = []

    if workers is not None:
        resolved_workers = max(1, int(workers))
        source_parts.append('api_workers')
    else:
        env_w = env_optional_int('INSTANT_MERGE_WORKERS')
        if env_w is not None:
            resolved_workers = max(1, env_w)
            source_parts.append('env_workers')
        else:
            resolved_workers = _adaptive_workers(hw, sig)
            source_parts.append('adaptive_workers')

    if chunk_size is not None:
        resolved_chunk = max(D.MIN_MERGE_CHUNK_SIZE, min(D.MAX_MERGE_CHUNK_SIZE, int(chunk_size)))
        source_parts.append('api_chunk')
    else:
        env_c = env_optional_int('INSTANT_MERGE_CHUNK_SIZE')
        if env_c is not None:
            resolved_chunk = max(D.MIN_MERGE_CHUNK_SIZE, min(D.MAX_MERGE_CHUNK_SIZE, env_c))
            source_parts.append('env_chunk')
        else:
            resolved_chunk = _adaptive_chunk(hw, sig, resolved_workers)
            source_parts.append('adaptive_chunk')

    if checkpoint_interval is not None:
        resolved_ckpt = max(1, int(checkpoint_interval))
        source_parts.append('api_checkpoint')
    else:
        ckpt_env = env_optional_int('INSTANT_MERGE_CHECKPOINT_INTERVAL')
        resolved_ckpt = resolve_explicit_or_env(
            None, 'INSTANT_MERGE_CHECKPOINT_INTERVAL', _adaptive_checkpoint_interval(sig),
        )
        if ckpt_env is not None:
            source_parts.append('env_checkpoint')
        else:
            source_parts.append('adaptive_checkpoint')

    scale = _cache_scale(hw, sig)
    ckpt_save = env_float('INSTANT_MERGE_CHECKPOINT_SEC', D.MERGE_CHECKPOINT_MIN_SAVE_SEC, minimum=1.0)
    metrics_sec = env_float(
        'INSTANT_PIPELINE_METRICS_SNAPSHOT_SEC',
        D.METRICS_SNAPSHOT_INTERVAL_SEC,
        minimum=1.0,
    )
    doc_budget = env_float('INSTANT_MERGE_DOC_WALL_BUDGET_SEC', _adaptive_doc_budget(hw, sig), minimum=30.0)
    doc_max = env_int('INSTANT_MERGE_DOC_MAX_CHARS', D.MAX_CHARS_GATE, minimum=1000)
    batch_timeout = env_float('INSTANT_MERGE_BATCH_TIMEOUT_SEC', D.DEFAULT_MERGE_BATCH_TIMEOUT_SEC, minimum=30.0)
    batch_timeout = min(max(batch_timeout, float(resolved_chunk) / 20.0), 180.0)

    return RuntimePolicy(
        workers=resolved_workers,
        chunk_size=resolved_chunk,
        checkpoint_interval=resolved_ckpt,
        checkpoint_save_sec=ckpt_save,
        metrics_snapshot_sec=metrics_sec,
        doc_wall_budget_sec=doc_budget,
        doc_max_chars=doc_max,
        batch_timeout_sec=batch_timeout,
        evidence_cache_size=max(256, int(D.SEMANTIC_EVIDENCE_CACHE_SIZE * scale)),
        raw_feature_cache_size=max(256, int(D.RAW_FEATURE_CACHE_SIZE * scale)),
        nav_feature_cache_size=max(128, int(D.NAV_FEATURE_CACHE_SIZE * scale)),
        publication_role_cache_size=max(128, int(D.PUBLICATION_ROLE_CACHE_SIZE * scale)),
        structure_cache_size=max(128, int(D.STRUCTURE_CACHE_SIZE * scale)),
        layout_cache_size=max(128, int(D.LAYOUT_CACHE_SIZE * scale)),
        scaffold_cache_size=max(128, int(D.SCAFFOLD_CACHE_SIZE * scale)),
        rss_pressure_mb=_rss_pressure_mb(hw),
        hardware_adapt_enabled=env_bool('INSTANT_LCI_HARDWARE_ADAPT', True),
        lci_num_shards=env_int('INSTANT_LCI_NUM_SHARDS', D.LCI_NUM_SHARDS, minimum=1),
        source='+'.join(source_parts) if source_parts else 'adaptive',
    )


@dataclass
class MergeRuntime:
    policy: RuntimePolicy
    hardware: HardwareProfile = field(default_factory=detect_hardware_profile)
    signals: PipelineSignals = field(default_factory=PipelineSignals)

    @classmethod
    def bootstrap(
        cls,
        *,
        workers: int | None = None,
        chunk_size: int | None = None,
        checkpoint_interval: int | None = None,
        work_dir: Any = None,
    ) -> MergeRuntime:
        hw = detect_hardware_profile(work_dir)
        policy = build_runtime_policy(
            workers=workers,
            chunk_size=chunk_size,
            checkpoint_interval=checkpoint_interval,
            hw=hw,
        )
        from indw.clean.artifact.evidence_cache import bootstrap_session_caches
        bootstrap_session_caches(policy.cache_sizes())
        return cls(policy=policy, hardware=hw)

    def refresh(self, signals: PipelineSignals) -> RuntimePolicy:
        self.signals = signals
        soft = build_runtime_policy(
            workers=self.policy.workers,
            chunk_size=self.policy.chunk_size,
            checkpoint_interval=self.policy.checkpoint_interval,
            signals=signals,
            hw=self.hardware,
        )
        self.policy = replace(
            soft,
            workers=self.policy.workers,
            chunk_size=self.policy.chunk_size,
            source=self.policy.source + '+refresh',
        )
        return self.policy

    def snapshot(self) -> dict[str, Any]:
        return {
            'policy': self.policy.to_dict(),
            'hardware': self.hardware.to_dict(),
            'signals': self.signals.to_dict(),
        }


def active_or_built_policy() -> RuntimePolicy:
    pol = active_policy()
    if pol is not None:
        return pol
    return build_runtime_policy()


def bind_merge_runtime(runtime: MergeRuntime | None) -> None:
    global _BOUND
    _BOUND = runtime
    _tls.runtime = runtime


def get_merge_runtime() -> MergeRuntime | None:
    rt = getattr(_tls, 'runtime', None)
    if rt is not None:
        return rt
    return _BOUND


def active_policy() -> RuntimePolicy | None:
    rt = get_merge_runtime()
    return rt.policy if rt is not None else None
