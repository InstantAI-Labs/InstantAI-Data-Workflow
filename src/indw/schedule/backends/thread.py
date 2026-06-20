from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
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


class ThreadExecutionSession:
    def __init__(
        self,
        worker_init: WorkerInitBundle,
        *,
        fast_workers: int,
        heavy_workers: int,
    ) -> None:
        self._worker_init = worker_init
        self._init_lock = threading.Lock()
        self._inited = False
        workers = max(fast_workers, heavy_workers, 1)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='merge-stage')
        self._pending = 0
        self._pending_lock = threading.Lock()

    def _ensure(self) -> None:
        if self._inited:
            return
        with self._init_lock:
            if self._inited:
                return
            init_fast_merge_worker(self._worker_init)
            init_merge_worker(self._worker_init)
            self._inited = True

    def _track(self, fut: Future) -> _FutureTask:
        with self._pending_lock:
            self._pending += 1

        def _done(_f: Future) -> None:
            with self._pending_lock:
                self._pending -= 1

        fut.add_done_callback(_done)
        return _FutureTask(fut)

    def submit_fast(self, batch: list[dict[str, Any]]) -> _FutureTask:
        self._ensure()
        return self._track(self._pool.submit(process_fast_chain_batch, batch))

    def submit_heavy(self, batch: list[dict[str, Any]]) -> _FutureTask:
        self._ensure()
        return self._track(self._pool.submit(process_heavy_chain_batch, batch))

    def run_fallback_merge(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure()
        return process_merge_batch(batch)

    def active_workers(self) -> int:
        with self._pending_lock:
            return self._pending

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)


class ThreadBackend:
    name = 'thread'

    def open(
        self,
        worker_init: WorkerInitBundle,
        *,
        fast_workers: int,
        heavy_workers: int,
    ) -> _ThreadSessionContext:
        return _ThreadSessionContext(
            worker_init,
            fast_workers=fast_workers,
            heavy_workers=heavy_workers,
        )


class _ThreadSessionContext:
    def __init__(
        self,
        worker_init: WorkerInitBundle,
        *,
        fast_workers: int,
        heavy_workers: int,
    ) -> None:
        self._inner = ThreadExecutionSession(
            worker_init,
            fast_workers=fast_workers,
            heavy_workers=heavy_workers,
        )

    def __enter__(self) -> ThreadExecutionSession:
        return self._inner

    def __exit__(self, *exc: Any) -> None:
        self._inner.close()
