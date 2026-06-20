from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

from indw.store.io.json_codec import dumps, loads


class LocalStageQueue:
    def __init__(self, maxsize: int = 0):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item: Any, *, block: bool = True, timeout: float | None = None) -> None:
        self._q.put(item, block=block, timeout=timeout)

    def get(self, *, block: bool = True, timeout: float | None = None) -> Any:
        return self._q.get(block=block, timeout=timeout)

    def qsize(self) -> int:
        return self._q.qsize()

    def full(self) -> bool:
        return self._q.full()


class FilesystemSpoolQueue:
    def __init__(self, root: Path | str, *, stage: str, maxsize: int = 0):
        self._root = Path(root) / 'queues' / stage
        self._root.mkdir(parents=True, exist_ok=True)
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._seq = 0
        self._pending: list[Path] = []

    def _refresh_pending(self) -> None:
        if self._pending:
            return
        files = sorted(self._root.glob('*.msg'), key=lambda p: p.name)
        self._pending = files

    def put(self, item: Any, *, block: bool = True, timeout: float | None = None) -> None:
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            with self._lock:
                if self._maxsize <= 0 or len(self._pending) < self._maxsize:
                    self._seq += 1
                    path = self._root / f'{self._seq:012d}.msg'
                    path.write_text(dumps(item), encoding='utf-8')
                    self._pending.append(path)
                    return
            if not block:
                raise queue.Full
            if deadline is not None and time.monotonic() >= deadline:
                raise queue.Full
            time.sleep(0.01)

    def get(self, *, block: bool = True, timeout: float | None = None) -> Any:
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            with self._lock:
                self._refresh_pending()
                if self._pending:
                    path = self._pending.pop(0)
                    raw = loads(path.read_text(encoding='utf-8'))
                    path.unlink(missing_ok=True)
                    return raw
            if not block:
                raise queue.Empty
            if deadline is not None and time.monotonic() >= deadline:
                raise queue.Empty
            time.sleep(0.01)

    def qsize(self) -> int:
        with self._lock:
            self._refresh_pending()
            return len(self._pending)

    def full(self) -> bool:
        if self._maxsize <= 0:
            return False
        return self.qsize() >= self._maxsize


class RedisStreamQueue:
    def __init__(
        self,
        *,
        stream: str,
        maxsize: int = 0,
        redis_url: str | None = None,
    ):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError('redis package required for RedisStreamQueue') from exc
        url = redis_url or 'redis://127.0.0.1:6379/0'
        self._client = redis.Redis.from_url(url, decode_responses=False)
        self._stream = stream
        self._group = f'{stream}_cg'
        self._consumer = f'c_{id(self)}'
        self._maxsize = maxsize
        try:
            self._client.xgroup_create(self._stream, self._group, id='0', mkstream=True)
        except Exception:
            pass

    def put(self, item: Any, *, block: bool = True, timeout: float | None = None) -> None:
        if self._maxsize > 0 and self.qsize() >= self._maxsize:
            if not block:
                raise queue.Full
            deadline = time.monotonic() + (timeout or 30.0)
            while self.qsize() >= self._maxsize:
                if time.monotonic() >= deadline:
                    raise queue.Full
                time.sleep(0.05)
        self._client.xadd(self._stream, {b'data': dumps(item).encode('utf-8')})

    def get(self, *, block: bool = True, timeout: float | None = None) -> Any:
        wait_ms = int((timeout or 1.0) * 1000) if block else 1
        rows = self._client.xreadgroup(
            self._group,
            self._consumer,
            {self._stream: '>'},
            count=1,
            block=wait_ms,
        )
        if not rows:
            raise queue.Empty
        _stream, messages = rows[0]
        msg_id, fields = messages[0]
        self._client.xack(self._stream, self._group, msg_id)
        return loads(fields[b'data'].decode('utf-8'))

    def qsize(self) -> int:
        try:
            return int(self._client.xlen(self._stream))
        except Exception:
            return 0

    def full(self) -> bool:
        return self._maxsize > 0 and self.qsize() >= self._maxsize


def make_stage_queue(
    *,
    backend: str,
    stage: str,
    maxsize: int,
    merge_work: Path | str | None = None,
) -> LocalStageQueue | FilesystemSpoolQueue | RedisStreamQueue:
    if backend == 'fs':
        if merge_work is None:
            raise ValueError('merge_work required for filesystem queue backend')
        return FilesystemSpoolQueue(merge_work, stage=stage, maxsize=maxsize)
    if backend in ('redis', 'redis_streams'):
        return RedisStreamQueue(stream=f'instant:{stage}', maxsize=maxsize)
    return LocalStageQueue(maxsize=maxsize)
