from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

MERGE_RUN_LOCK = 'merge.run.lock'
DEFAULT_STALE_SEC = 6 * 3600.0

class MergeRunConflictError(RuntimeError):
    pass

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil
        return psutil.Process(pid).is_running()
    except ImportError:
        pass
    except Exception:
        return False
    if os.name == 'nt':
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def read_merge_run_lock(work_dir: Path) -> Optional[dict[str, Any]]:
    path = Path(work_dir) / MERGE_RUN_LOCK
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        return raw if isinstance(raw, dict) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None

def _lock_age_sec(payload: dict[str, Any]) -> float:
    raw = payload.get('started_at')
    if not isinstance(raw, str) or not raw:
        return 0.0
    try:
        started = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
    except (TypeError, ValueError):
        return 0.0

def acquire_merge_run_lock(
    work_dir: Path,
    *,
    owner: str,
    force: bool = False,
    stale_sec: float = DEFAULT_STALE_SEC,
) -> Path:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    path = work / MERGE_RUN_LOCK
    existing = read_merge_run_lock(work)
    if existing:
        pid = int(existing.get('pid', 0) or 0)
        alive = _pid_alive(pid) if pid else False
        age = _lock_age_sec(existing)
        stale = (not alive) or age >= stale_sec
        if not stale:
            if pid == os.getpid():
                if not force:
                    raise MergeRunConflictError(
                        f'Merge lock already held in this process (owner={existing.get("owner")}). '
                        'Release it before starting another merge on the same work dir.'
                    )
            else:
                hint = (
                    'Stop the other process and retry, or use a new validation --run-id.'
                    if force
                    else 'Stop the other process or rerun with --force-lock if stale.'
                )
                raise MergeRunConflictError(
                    'Merge already active for '
                    f'{work} (owner={existing.get("owner")}, pid={pid}, '
                    f'started={existing.get("started_at")}). {hint}'
                )
    payload = {
        'pid': os.getpid(),
        'owner': owner,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'work_dir': str(work.resolve()),
    }
    from indw.store.io.atomic import atomic_write_text
    atomic_write_text(path, json.dumps(payload, indent=2))
    return path

def release_merge_run_lock(work_dir: Path, *, owner: Optional[str] = None) -> None:
    path = Path(work_dir) / MERGE_RUN_LOCK
    if not path.exists():
        return
    if owner is not None:
        existing = read_merge_run_lock(work_dir)
        if existing and str(existing.get('owner')) != owner:
            return
        if existing and int(existing.get('pid', 0) or 0) not in (0, os.getpid()):
            return
    try:
        path.unlink()
    except OSError:
        pass

@contextmanager
def merge_run_lock(
    work_dir: Path,
    *,
    owner: str,
    force: bool = False,
    stale_sec: float = DEFAULT_STALE_SEC,
) -> Iterator[Path]:
    path = acquire_merge_run_lock(work_dir, owner=owner, force=force, stale_sec=stale_sec)
    try:
        yield path
    finally:
        release_merge_run_lock(work_dir, owner=owner)
