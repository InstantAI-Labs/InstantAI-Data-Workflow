from __future__ import annotations

import logging
from typing import Any

from indw.schedule.backends.config import dask_scheduler_address
from indw.schedule.dispatch.workers import (
    WorkerInitBundle,
    init_fast_merge_worker,
    init_merge_worker,
    process_merge_batch,
)
from indw.schedule.stages.pools.chain import process_fast_chain_batch, process_heavy_chain_batch

logger = logging.getLogger(__name__)


def _dask_worker_bootstrap(worker_init: WorkerInitBundle) -> None:
    init_fast_merge_worker(worker_init)
    init_merge_worker(worker_init)


class _DaskTask:
    __slots__ = ('_fut',)

    def __init__(self, fut: Any) -> None:
        self._fut = fut

    def done(self) -> bool:
        return bool(self._fut.done())

    def result(self, timeout: float | None = None) -> dict[str, Any]:
        return self._fut.result(timeout=timeout)


class _DaskSession:
    def __init__(self, client: Any, worker_init: WorkerInitBundle) -> None:
        self._client = client
        self._worker_init = worker_init
        self._fallback_ready = False
        self._pending = 0

    def _ensure_fallback(self) -> None:
        if self._fallback_ready:
            return
        init_fast_merge_worker(self._worker_init)
        init_merge_worker(self._worker_init)
        self._fallback_ready = True

    def submit_fast(self, batch: list[dict[str, Any]]) -> _DaskTask:
        self._pending += 1

        def _done(_f: Any) -> None:
            self._pending = max(0, self._pending - 1)

        fut = self._client.submit(process_fast_chain_batch, batch, pure=False)
        fut.add_done_callback(_done)
        return _DaskTask(fut)

    def submit_heavy(self, batch: list[dict[str, Any]]) -> _DaskTask:
        self._pending += 1

        def _done(_f: Any) -> None:
            self._pending = max(0, self._pending - 1)

        fut = self._client.submit(process_heavy_chain_batch, batch, pure=False)
        fut.add_done_callback(_done)
        return _DaskTask(fut)

    def run_fallback_merge(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_fallback()
        return process_merge_batch(batch)

    def active_workers(self) -> int:
        return self._pending


class DaskBackend:
    name = 'dask'

    def open(
        self,
        worker_init: WorkerInitBundle,
        *,
        fast_workers: int,
        heavy_workers: int,
    ) -> _DaskSessionContext:
        return _DaskSessionContext(worker_init)


class _DaskSessionContext:
    def __init__(self, worker_init: WorkerInitBundle) -> None:
        self._worker_init = worker_init
        self._client: Any = None
        self._session: _DaskSession | None = None

    def __enter__(self) -> _DaskSession:
        try:
            from dask.distributed import Client
        except ImportError as exc:
            raise RuntimeError(
                'dask[distributed] required for INSTANT_PIPELINE_BACKEND=dask; '
                'install with: pip install "dask[distributed]"'
            ) from exc
        addr = dask_scheduler_address()
        if addr:
            self._client = Client(addr, set_as_default=False, timeout=30)
        else:
            from dask.distributed import LocalCluster
            self._cluster = LocalCluster(
                n_workers=2,
                threads_per_worker=1,
                processes=True,
                silence_logs=False,
            )
            self._client = Client(self._cluster, set_as_default=False, timeout=30)
        logger.info(
            'Dask backend connected scheduler=%s workers=%d',
            addr or 'local',
            len(self._client.scheduler_info().get('workers') or {}),
        )
        self._client.run(_dask_worker_bootstrap, self._worker_init, wait=True)
        self._session = _DaskSession(self._client, self._worker_init)
        return self._session

    def __exit__(self, *exc: Any) -> None:
        if self._client is not None:
            self._client.close()
        cluster = getattr(self, '_cluster', None)
        if cluster is not None:
            cluster.close()
