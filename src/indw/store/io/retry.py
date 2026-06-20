from __future__ import annotations

import errno
import sqlite3
from pathlib import Path
from typing import Callable, TypeVar

from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_incrementing,
)

T = TypeVar('T')


def _sqlite_locked(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and 'locked' in str(exc).lower()


def _transient_os_write(exc: BaseException) -> bool:
    if not isinstance(exc, OSError):
        return False
    if getattr(exc, 'winerror', None) == 5:
        return True
    return exc.errno in (errno.EACCES,)


def _permission_denied(exc: BaseException) -> bool:
    return isinstance(exc, PermissionError)


def retry_sqlite_locked(
    fn: Callable[[], T],
    *,
    attempts: int = 8,
    backoff_sec: float = 0.05,
    journal_dir: Path | None = None,
) -> T:
    lock_retries = 0
    result: T
    try:
        for attempt in Retrying(
            retry=retry_if_exception(_sqlite_locked),
            stop=stop_after_attempt(max(1, attempts)),
            wait=wait_exponential(multiplier=backoff_sec, max=1.0),
            reraise=True,
        ):
            with attempt:
                try:
                    result = fn()
                except sqlite3.OperationalError as exc:
                    if _sqlite_locked(exc):
                        lock_retries += 1
                    raise
    except sqlite3.OperationalError:
        if journal_dir is not None:
            from indw.tools.metrics.recovery import record_recovery_event
            record_recovery_event(
                journal_dir,
                'sqlite_lock_exhausted',
                retries=lock_retries + 1,
            )
        raise
    if lock_retries > 0 and journal_dir is not None:
        from indw.tools.metrics.recovery import record_recovery_event
        record_recovery_event(journal_dir, 'sqlite_retry', retries=lock_retries)
    return result


def retry_transient_os_write(
    fn: Callable[[], T],
    *,
    attempts: int = 5,
    backoff_sec: float = 0.05,
) -> T:
    result: T
    for attempt in Retrying(
        retry=retry_if_exception(_transient_os_write),
        stop=stop_after_attempt(max(1, attempts)),
        wait=wait_incrementing(start=backoff_sec, increment=backoff_sec),
        reraise=True,
    ):
        with attempt:
            result = fn()
    return result


def retry_permission_denied(
    fn: Callable[[], T],
    *,
    attempts: int = 6,
    backoff_sec: float = 0.15,
) -> T:
    result: T
    for attempt in Retrying(
        retry=retry_if_exception(_permission_denied),
        stop=stop_after_attempt(max(1, attempts)),
        wait=wait_incrementing(start=backoff_sec, increment=backoff_sec),
        reraise=True,
    ):
        with attempt:
            result = fn()
    return result
