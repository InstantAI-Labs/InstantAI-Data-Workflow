from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, TypeVar

from indw.store.io.retry import retry_sqlite_locked

T = TypeVar('T')

DEFAULT_BUSY_TIMEOUT_SEC = 60.0
DEFAULT_LOCK_RETRIES = 8
DEFAULT_LOCK_BACKOFF_SEC = 0.05


def connect_sqlite(
    db_path: str | Path,
    *,
    timeout: float = DEFAULT_BUSY_TIMEOUT_SEC,
    wal: bool = True,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=timeout, check_same_thread=check_same_thread)
    if wal:
        conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute(f'PRAGMA busy_timeout={int(timeout * 1000)}')
    return conn


def run_locked(
    fn: Callable[[], T],
    *,
    retries: int = DEFAULT_LOCK_RETRIES,
    backoff_sec: float = DEFAULT_LOCK_BACKOFF_SEC,
    journal_dir: Path | None = None,
) -> T:
    return retry_sqlite_locked(
        fn,
        attempts=retries,
        backoff_sec=backoff_sec,
        journal_dir=journal_dir,
    )


def sqlite_sidecars(db_path: str | Path) -> tuple[Path, Path]:
    path = Path(db_path)
    return path.with_name(f'{path.name}-wal'), path.with_name(f'{path.name}-shm')


def unlink_sqlite_files(db_path: str | Path) -> list[str]:
    removed: list[str] = []
    for candidate in (Path(db_path), *sqlite_sidecars(db_path)):
        if not candidate.exists():
            continue
        candidate.unlink()
        removed.append(candidate.name)
    return removed
