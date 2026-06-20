from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def reject_log_path(work_dir: Path) -> Path:
    return Path(work_dir) / 'logs.reject.json'


class MergeRejectLog:

    def __init__(self, work_dir: Path, *, enabled: bool = True, flush_every: int = 25):
        self.path = reject_log_path(work_dir)
        self.enabled = enabled
        self.flush_every = max(1, int(flush_every))
        self._lock = threading.Lock()
        self._pending = 0
        self._fh: Optional[Any] = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        reason: str,
        stage: str,
        source: str = '',
        domain: str = '',
        language: str = '',
        quality_score: float = 0.0,
        chars: int = 0,
        preview: str = '',
        content_type: str = '',
    ) -> None:
        if not self.enabled:
            return
        row = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'reason': str(reason),
            'stage': str(stage),
            'source': str(source),
            'domain': str(domain),
            'language': str(language),
            'quality_score': round(float(quality_score), 4),
            'chars': int(chars),
            'preview': preview[:240],
        }
        if content_type:
            row['content_type'] = content_type
        with self._lock:
            if self._fh is None:
                self._fh = self.path.open('a', encoding='utf-8')
            self._fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            self._pending += 1
            if self._pending >= self.flush_every:
                self._fh.flush()
                self._pending = 0

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None
                self._pending = 0
