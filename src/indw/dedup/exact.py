from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from indw.config.defaults import DEDUP_LOOKUP_CACHE_SIZE, DEDUP_SQLITE_BATCH_SIZE
from indw.dedup.normalize import content_hash, normalize_for_dedup
from indw.dedup.storage import connect_sqlite, run_locked
from indw.store.io.cache import BoundedLRU

_SCHEMA = '\nCREATE TABLE IF NOT EXISTS content_hashes (\n    hash TEXT PRIMARY KEY,\n    source TEXT,\n    first_seen TEXT NOT NULL\n);\nCREATE INDEX IF NOT EXISTS idx_source ON content_hashes(source);\n'


class PersistentHashIndex:

    def __init__(
        self,
        db_path: str | Path,
        *,
        batch_size: int | None = None,
        lookup_cache_size: int | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size if batch_size is not None else DEDUP_SQLITE_BATCH_SIZE
        self._conn = connect_sqlite(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._pending: list[tuple[str, str, str]] = []
        self._pending_hashes: set[str] = set()
        self._pending_ts = ''
        self._pending_by_digest: dict[str, str] = {}
        self._lookup_cache = BoundedLRU(
            max(1024, lookup_cache_size if lookup_cache_size is not None else DEDUP_LOOKUP_CACHE_SIZE),
        )
        self._lookup_cur = self._conn.cursor()
        self.stats = {'inserted': 0, 'duplicates': 0, 'lookups': 0, 'cache_hits': 0}

    def close(self) -> None:
        self.flush()
        self._conn.close()

    def __enter__(self) -> PersistentHashIndex:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.flush()
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def flush(self) -> None:
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        self._pending_hashes = set()
        self._pending_by_digest = {}
        self._pending_ts = ''

        def _write() -> None:
            self._conn.executemany(
                'INSERT OR IGNORE INTO content_hashes (hash, source, first_seen) VALUES (?, ?, ?)',
                pending,
            )
            self._conn.commit()

        run_locked(_write)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def lookup_hash(self, digest: str) -> tuple[bool, str]:
        self.stats['lookups'] += 1
        if digest in self._pending_by_digest:
            self.stats['cache_hits'] += 1
            return True, self._pending_by_digest[digest]
        cached = self._lookup_cache.get(digest)
        if cached is not None:
            self.stats['cache_hits'] += 1
            return cached

        def _lookup() -> tuple[bool, str]:
            self._lookup_cur.execute(
                'SELECT source FROM content_hashes WHERE hash = ? LIMIT 1',
                (digest,),
            )
            row = self._lookup_cur.fetchone()
            if row is None:
                return False, ''
            return True, str(row[0])

        result = _lookup()
        self._lookup_cache.set(digest, result)
        return result

    def contains_hash(self, digest: str) -> bool:
        return self.lookup_hash(digest)[0]

    def hash_source(self, digest: str) -> str:
        found, source = self.lookup_hash(digest)
        return source if found else ''

    def _stage_hash(self, digest: str, *, source: str = '') -> None:
        if not self._pending:
            self._pending_ts = self._now()
        self._pending.append((digest, source, self._pending_ts))
        self._pending_hashes.add(digest)
        self._pending_by_digest[digest] = source
        self._lookup_cache.set(digest, (True, source))
        self.stats['inserted'] += 1
        if len(self._pending) >= self.batch_size:
            self.flush()

    def is_duplicate(self, text: str, *, source: str = '', digest: str | None = None) -> bool:
        digest = digest or content_hash(text)
        found, _ = self.lookup_hash(digest)
        if found:
            self.stats['duplicates'] += 1
            return True
        self._stage_hash(digest, source=source)
        return False

    def register_hash(self, digest: str, *, source: str = '', assume_new: bool = False) -> bool:
        if not assume_new:
            found, _ = self.lookup_hash(digest)
            if found:
                self.stats['duplicates'] += 1
                return False
        self._stage_hash(digest, source=source)
        return True

    def register_text(self, text: str, *, source: str = '') -> bool:
        return self.register_hash(content_hash(text), source=source)

    def count(self) -> int:
        self.flush()
        row = self._conn.execute('SELECT COUNT(*) FROM content_hashes').fetchone()
        return int(row[0]) if row else 0

    def summary(self) -> dict:
        self.flush()
        total = self.count()
        by_source = dict(self._conn.execute('SELECT source, COUNT(*) FROM content_hashes GROUP BY source').fetchall())
        return {
            'db_path': str(self.db_path),
            'unique_hashes': total,
            'by_source': by_source,
            'session_inserted': self.stats['inserted'],
            'session_duplicates': self.stats['duplicates'],
        }

    def ingest_jsonl_hashes(self, jsonl_path: Path, *, source: str = 'import') -> int:
        from indw.dedup.replay import iter_jsonl_text

        added = 0
        for text in iter_jsonl_text(jsonl_path):
            if self.register_text(text, source=source):
                added += 1
        self.flush()
        return added

    @classmethod
    def default_path(cls, work_dir: str | Path) -> Path:
        return Path(work_dir) / 'corpus' / 'dedup_index.sqlite'
