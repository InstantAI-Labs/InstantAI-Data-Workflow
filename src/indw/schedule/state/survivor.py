from __future__ import annotations

import mmap
from pathlib import Path
from typing import Any

from indw.schedule.config.tune import ipc_externalize_threshold

_SURVIVOR_MMAP: dict[str, mmap.mmap] = {}


def _store_root(work_dir: Path) -> Path:
    root = Path(work_dir) / '.survivor_store'
    root.mkdir(parents=True, exist_ok=True)
    return root


def _mmap_key(work_dir: Path | str, doc_key: str) -> str:
    return f'{Path(work_dir).resolve()}::{doc_key}'


def externalize_survivor_text(
    work_dir: Path | str,
    *,
    doc_key: str,
    text: str,
    threshold: int | None = None,
) -> tuple[str, str | None]:
    limit = threshold if threshold is not None else ipc_externalize_threshold()
    if len(text) < limit or not doc_key:
        return text, None
    root = _store_root(Path(work_dir))
    path = root / f'{doc_key}.txt'
    if not path.is_file():
        path.write_text(text, encoding='utf-8')
    return '', doc_key


def resolve_survivor_text(
    payload: dict,
    *,
    work_dir: Path | str | None = None,
) -> str:
    inline = str(payload.get('text') or '')
    if inline.strip():
        return inline
    store_key = str(payload.get('text_store_key') or '').strip()
    if not store_key:
        return inline
    wd = work_dir or payload.get('_work_dir')
    if not wd:
        return inline
    path = _store_root(Path(wd)) / f'{store_key}.txt'
    if not path.is_file():
        return inline
    key = _mmap_key(Path(wd), store_key)
    cached = _SURVIVOR_MMAP.get(key)
    if cached is not None:
        try:
            return cached[:].decode('utf-8')
        except Exception:
            pass
    try:
        with path.open('rb') as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        _SURVIVOR_MMAP[key] = mm
        return mm[:].decode('utf-8')
    except Exception:
        return path.read_text(encoding='utf-8')


def release_survivor_mmap(work_dir: Path | str | None = None) -> None:
    if work_dir is None:
        for mm in _SURVIVOR_MMAP.values():
            try:
                mm.close()
            except Exception:
                pass
        _SURVIVOR_MMAP.clear()
        return
    prefix = f'{Path(work_dir).resolve()}::'
    stale = [k for k in _SURVIVOR_MMAP if k.startswith(prefix)]
    for key in stale:
        try:
            _SURVIVOR_MMAP.pop(key).close()
        except Exception:
            pass
