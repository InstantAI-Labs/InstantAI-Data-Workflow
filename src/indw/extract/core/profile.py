from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

_tls = threading.local()


def ke_profile_enabled() -> bool:
    return os.environ.get('INSTANT_KE_PROFILE', '').strip().lower() in ('1', 'true', 'yes')


@dataclass
class KeOpStats:
    wall_sec: float = 0.0
    cpu_sec: float = 0.0
    calls: int = 0
    payload_bytes: int = 0
    object_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    duplicate_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            'wall_sec': round(self.wall_sec, 6),
            'cpu_sec': round(self.cpu_sec, 6),
            'calls': self.calls,
            'payload_bytes': self.payload_bytes,
            'object_count': self.object_count,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': round(self.cache_hits / total, 4) if total else 0.0,
            'duplicate_calls': self.duplicate_calls,
        }


@dataclass
class KeProfile:
    ops: dict[str, KeOpStats] = field(default_factory=dict)
    _seen_keys: dict[str, set[Any]] = field(default_factory=dict)

    def op(self, name: str) -> KeOpStats:
        st = self.ops.get(name)
        if st is None:
            st = KeOpStats()
            self.ops[name] = st
        return st

    def record(
        self,
        name: str,
        *,
        wall_sec: float = 0.0,
        cpu_sec: float = 0.0,
        payload_bytes: int = 0,
        object_count: int = 0,
        cache_hit: bool | None = None,
        dedupe_key: Any | None = None,
    ) -> None:
        if not ke_profile_enabled():
            return
        st = self.op(name)
        st.wall_sec += wall_sec
        st.cpu_sec += cpu_sec
        st.calls += 1
        st.payload_bytes += payload_bytes
        st.object_count += object_count
        if cache_hit is True:
            st.cache_hits += 1
        elif cache_hit is False:
            st.cache_misses += 1
        if dedupe_key is not None:
            seen = self._seen_keys.setdefault(name, set())
            if dedupe_key in seen:
                st.duplicate_calls += 1
            else:
                seen.add(dedupe_key)

    def ranked(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        total_wall = sum(s.wall_sec for s in self.ops.values()) or 1.0
        for name, st in sorted(self.ops.items(), key=lambda x: -x[1].wall_sec):
            rows.append({
                'op': name,
                **st.to_dict(),
                'pct_pipeline': round(100.0 * st.wall_sec / total_wall, 2),
            })
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {'ops': {k: v.to_dict() for k, v in self.ops.items()}, 'ranked': self.ranked()}


def active_profile() -> KeProfile | None:
    return getattr(_tls, 'profile', None)


def bind_ke_profile(profile: KeProfile | None) -> None:
    _tls.profile = profile


@contextmanager
def ke_profile_session() -> Iterator[KeProfile]:
    prev = active_profile()
    prev_asm = active_unit_assembly_profile()
    prof = KeProfile()
    asm = UnitAssemblyProfile()
    bind_ke_profile(prof)
    bind_unit_assembly_profile(asm)
    try:
        yield prof
    finally:
        bind_ke_profile(prev)
        bind_unit_assembly_profile(prev_asm)


@contextmanager
def ke_timed(op: str, *, payload_bytes: int = 0, object_count: int = 0) -> Iterator[None]:
    if not ke_profile_enabled():
        yield
        return
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        yield
    finally:
        prof = active_profile()
        if prof is not None:
            prof.record(
                op,
                wall_sec=time.perf_counter() - t0,
                cpu_sec=time.process_time() - c0,
                payload_bytes=payload_bytes,
                object_count=object_count,
            )


def ke_record(
    op: str,
    *,
    payload_bytes: int = 0,
    object_count: int = 0,
    cache_hit: bool | None = None,
    dedupe_key: Any | None = None,
) -> None:
    prof = active_profile()
    if prof is None:
        return
    prof.record(
        op,
        payload_bytes=payload_bytes,
        object_count=object_count,
        cache_hit=cache_hit,
        dedupe_key=dedupe_key,
    )


@dataclass
class UnitAssemblyProfile:
    sections: int = 0
    total_wall_sec: float = 0.0
    max_wall_sec: float = 0.0
    avg_wall_sec: float = 0.0
    max_fingerprint: tuple[int, bytes] | None = None
    max_chars: int = 0

    def record(self, *, wall_sec: float, fingerprint: tuple[int, bytes] | None, chars: int) -> None:
        self.sections += 1
        self.total_wall_sec += wall_sec
        if wall_sec > self.max_wall_sec:
            self.max_wall_sec = wall_sec
            self.max_fingerprint = fingerprint
            self.max_chars = chars
        self.avg_wall_sec = self.total_wall_sec / self.sections

    def to_dict(self) -> dict[str, Any]:
        return {
            'sections': self.sections,
            'total_wall_sec': round(self.total_wall_sec, 6),
            'avg_wall_sec': round(self.avg_wall_sec, 6),
            'max_wall_sec': round(self.max_wall_sec, 6),
            'max_chars': self.max_chars,
            'max_fingerprint': (
                f'{self.max_fingerprint[0]}:{self.max_fingerprint[1].hex()[:16]}'
                if self.max_fingerprint else ''
            ),
        }


def active_unit_assembly_profile() -> UnitAssemblyProfile | None:
    return getattr(_tls, 'unit_assembly', None)


def bind_unit_assembly_profile(profile: UnitAssemblyProfile | None) -> None:
    _tls.unit_assembly = profile


def record_clean_unit_section(
    *,
    wall_sec: float,
    fingerprint: tuple[int, bytes] | None,
    chars: int,
) -> None:
    prof = active_unit_assembly_profile()
    if prof is not None:
        prof.record(wall_sec=wall_sec, fingerprint=fingerprint, chars=chars)


@contextmanager
def unit_assembly_profile_session() -> Iterator[UnitAssemblyProfile]:
    prev = active_unit_assembly_profile()
    prof = UnitAssemblyProfile()
    bind_unit_assembly_profile(prof)
    try:
        yield prof
    finally:
        bind_unit_assembly_profile(prev)
