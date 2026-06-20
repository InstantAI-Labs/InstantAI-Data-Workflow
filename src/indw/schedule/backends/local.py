from __future__ import annotations

from typing import Any

from indw.schedule.dispatch.workers import (
    WorkerInitBundle,
    init_fast_merge_worker,
    init_merge_worker,
    process_merge_batch,
)
from indw.schedule.stages.pools.chain import process_fast_chain_batch, process_heavy_chain_batch


class _SyncTask:
    __slots__ = ('_result',)

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def done(self) -> bool:
        return True

    def result(self, timeout: float | None = None) -> dict[str, Any]:
        return self._result


class LocalExecutionSession:
    def __init__(self, worker_init: WorkerInitBundle) -> None:
        self._worker_init = worker_init
        self._ready = False

    def _ensure(self) -> None:
        if self._ready:
            return
        init_fast_merge_worker(self._worker_init)
        init_merge_worker(self._worker_init)
        self._ready = True

    def submit_fast(self, batch: list[dict[str, Any]]) -> _SyncTask:
        self._ensure()
        return _SyncTask(process_fast_chain_batch(batch))

    def submit_heavy(self, batch: list[dict[str, Any]]) -> _SyncTask:
        self._ensure()
        return _SyncTask(process_heavy_chain_batch(batch))

    def run_fallback_merge(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure()
        return process_merge_batch(batch)

    def active_workers(self) -> int:
        return 1


class LocalBackend:
    name = 'local'

    def open(
        self,
        worker_init: WorkerInitBundle,
        *,
        fast_workers: int,
        heavy_workers: int,
    ) -> _LocalSessionContext:
        return _LocalSessionContext(worker_init)


class _LocalSessionContext:
    def __init__(self, worker_init: WorkerInitBundle) -> None:
        self._inner = LocalExecutionSession(worker_init)

    def __enter__(self) -> LocalExecutionSession:
        return self._inner

    def __exit__(self, *exc: Any) -> None:
        return None
