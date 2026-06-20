from __future__ import annotations

from indw.schedule.backends.config import normalize_backend_name, pipeline_execution_backend
from indw.schedule.backends.contract import ExecutionBackend
from indw.schedule.backends.dask import DaskBackend
from indw.schedule.backends.local import LocalBackend
from indw.schedule.backends.multiprocess import MultiprocessBackend
from indw.schedule.backends.thread import ThreadBackend

_BACKENDS: dict[str, ExecutionBackend] = {
    'local': LocalBackend(),
    'thread': ThreadBackend(),
    'multiprocess': MultiprocessBackend(),
    'dask': DaskBackend(),
}


def resolve_execution_backend(name: str | None = None) -> ExecutionBackend:
    key = normalize_backend_name(name) if name else pipeline_execution_backend()
    backend = _BACKENDS.get(key)
    if backend is None:
        backend = _BACKENDS['multiprocess']
    return backend


def backend_topology() -> dict[str, str]:
    return {name: type(b).__name__ for name, b in _BACKENDS.items()}
