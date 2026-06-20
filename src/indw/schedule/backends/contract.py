from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BatchTask(Protocol):
    def done(self) -> bool: ...
    def result(self, timeout: float | None = None) -> dict[str, Any]: ...


@runtime_checkable
class ExecutionSession(Protocol):
    def submit_fast(self, batch: list[dict[str, Any]]) -> BatchTask: ...
    def submit_heavy(self, batch: list[dict[str, Any]]) -> BatchTask: ...
    def run_fallback_merge(self, batch: list[dict[str, Any]]) -> dict[str, Any]: ...
    def active_workers(self) -> int: ...


@runtime_checkable
class ExecutionBackend(Protocol):
    name: str

    def open(
        self,
        worker_init: Any,
        *,
        fast_workers: int,
        heavy_workers: int,
    ) -> ExecutionSession: ...
