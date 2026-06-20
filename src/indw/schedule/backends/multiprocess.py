from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

from indw.schedule.dispatch.workers import (
    WorkerInitBundle,
    init_fast_merge_worker,
    init_merge_worker,
    process_merge_batch,
)
from indw.schedule.stages.pools.chain import process_fast_chain_batch, process_heavy_chain_batch


class _FutureTask:
    __slots__ = ('_fut',)

    def __init__(self, fut: Future) -> None:
        self._fut = fut

    def done(self) -> bool:
        return self._fut.done()

    def result(self, timeout: float | None = None) -> dict[str, Any]:
        return self._fut.result(timeout=timeout)


class _PoolSession:
    def __init__(
        self,
        *,
        fast_executor: ProcessPoolExecutor,
        heavy_executor: ProcessPoolExecutor,
        worker_init: WorkerInitBundle,
    ) -> None:
        self._fast = fast_executor
        self._heavy = heavy_executor
        self._worker_init = worker_init
        self._fallback_ready = False

    def _ensure_fallback(self) -> None:
        if self._fallback_ready:
            return
        init_fast_merge_worker(self._worker_init)
        init_merge_worker(self._worker_init)
        self._fallback_ready = True

    def submit_fast(self, batch: list[dict[str, Any]]) -> _FutureTask:
        return _FutureTask(self._fast.submit(process_fast_chain_batch, batch))

    def submit_heavy(self, batch: list[dict[str, Any]]) -> _FutureTask:
        return _FutureTask(self._heavy.submit(process_heavy_chain_batch, batch))

    def run_fallback_merge(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_fallback()
        return process_merge_batch(batch)

    def active_workers(self) -> int:
        return 0


class MultiprocessBackend:
    name = 'multiprocess'

    def open(
        self,
        worker_init: WorkerInitBundle,
        *,
        fast_workers: int,
        heavy_workers: int,
    ) -> _MultiprocessSession:
        return _MultiprocessSession(
            worker_init,
            fast_workers=fast_workers,
            heavy_workers=heavy_workers,
        )


class _MultiprocessSession:
    def __init__(
        self,
        worker_init: WorkerInitBundle,
        *,
        fast_workers: int,
        heavy_workers: int,
    ) -> None:
        self._worker_init = worker_init
        self._fast_workers = max(fast_workers, 1)
        self._heavy_workers = max(heavy_workers, 1)
        self._mp_ctx = mp.get_context('spawn')
        self._fast_executor: ProcessPoolExecutor | None = None
        self._heavy_executor: ProcessPoolExecutor | None = None
        self._pool: _PoolSession | None = None

    def __enter__(self) -> _PoolSession:
        self._fast_executor = ProcessPoolExecutor(
            max_workers=self._fast_workers,
            mp_context=self._mp_ctx,
            initializer=init_fast_merge_worker,
            initargs=(self._worker_init,),
        )
        self._heavy_executor = ProcessPoolExecutor(
            max_workers=self._heavy_workers,
            mp_context=self._mp_ctx,
            initializer=init_merge_worker,
            initargs=(self._worker_init,),
        )
        self._pool = _PoolSession(
            fast_executor=self._fast_executor,
            heavy_executor=self._heavy_executor,
            worker_init=self._worker_init,
        )
        return self._pool

    def __exit__(self, *exc: Any) -> None:
        if self._fast_executor is not None:
            self._fast_executor.shutdown(wait=True, cancel_futures=False)
        if self._heavy_executor is not None:
            self._heavy_executor.shutdown(wait=True, cancel_futures=False)
