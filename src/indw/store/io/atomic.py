from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

from indw.store.io.retry import retry_transient_os_write


class DiskFullError(OSError):
    def __init__(self, path: str | Path, *, operation: str = 'write') -> None:
        self.path = Path(path)
        self.operation = operation
        super().__init__(errno.ENOSPC, f'disk full during {operation}: {self.path}')


def is_no_space(exc: BaseException) -> bool:
    if isinstance(exc, DiskFullError):
        return True
    if isinstance(exc, OSError):
        if exc.errno in (errno.ENOSPC, errno.EDQUOT):
            return True
        if getattr(exc, 'winerror', None) in (112, 39):
            return True
    return False


def _tmp_path(path: Path) -> Path:
    return path.with_name(f'.{path.name}.{os.getpid()}.tmp')


def _backup_existing(path: Path) -> None:
    if not path.exists():
        return
    bak = path.with_name(f'{path.name}.bak')
    try:
        shutil.copy2(path, bak)
    except OSError:
        pass


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    backup: bool = True,
    retries: int = 5,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup:
        _backup_existing(path)
    tmp = _tmp_path(path)

    def _write() -> None:
        try:
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError as exc:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if is_no_space(exc):
                raise DiskFullError(path) from exc
            raise

    try:
        retry_transient_os_write(_write, attempts=retries, backoff_sec=0.05)
    except OSError:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = 'utf-8',
    backup: bool = True,
) -> None:
    atomic_write_bytes(path, text.encode(encoding), backup=backup)


def cleanup_stale_temps(directory: str | Path, *, suffix: str = '.tmp') -> int:
    root = Path(directory)
    if not root.is_dir():
        return 0
    removed = 0
    for candidate in root.rglob(f'*{suffix}'):
        if not candidate.is_file():
            continue
        try:
            candidate.unlink()
            removed += 1
        except OSError:
            continue
    return removed
