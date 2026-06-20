from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_VALID = frozenset({'local', 'thread', 'multiprocess', 'dask'})


def normalize_backend_name(raw: str) -> str:
    key = raw.strip().lower()
    if key in ('mp', 'process', 'processes', 'spawn'):
        return 'multiprocess'
    if key in ('threads', 'threaded'):
        return 'thread'
    if key in ('sync', 'inline', 'direct'):
        return 'local'
    if key in ('distributed', 'cluster'):
        return 'dask'
    if key in _VALID:
        return key
    return 'multiprocess'


def pipeline_execution_backend() -> str:
    raw = os.environ.get('INSTANT_PIPELINE_BACKEND', '').strip().lower()
    if not raw:
        raw = os.environ.get('INSTANT_PIPELINE_EXECUTOR', 'multiprocess').strip().lower()
    if raw in _VALID or raw in ('mp', 'process', 'threads', 'sync', 'distributed', 'cluster'):
        return normalize_backend_name(raw)
    logger.warning('unknown INSTANT_PIPELINE_BACKEND=%s; using multiprocess', raw)
    return 'multiprocess'


def dask_scheduler_address() -> str | None:
    addr = os.environ.get('INSTANT_DASK_SCHEDULER', '').strip()
    if addr:
        return addr
    return os.environ.get('DASK_SCHEDULER_ADDRESS', '').strip() or None
