from __future__ import annotations

from pathlib import Path
from typing import Any

from indw.dedup.exact import PersistentHashIndex
from indw.ingest.hash import ExactHashDedup


class ShardedExactDedup:
    def __init__(
        self,
        work_dir: str | Path,
        *,
        shards: int = 2,
        batch_size: int | None = None,
    ):
        self.work_dir = Path(work_dir)
        self.shards = max(1, int(shards))
        self._indexes: list[PersistentHashIndex] = []
        self._dedups: list[ExactHashDedup] = []
        root = self.work_dir / 'corpus' / 'dedup_shards'
        root.mkdir(parents=True, exist_ok=True)
        for i in range(self.shards):
            path = root / f'shard_{i:02d}.sqlite'
            index = PersistentHashIndex(path, batch_size=batch_size)
            self._indexes.append(index)
            self._dedups.append(ExactHashDedup(index))
        self.stats = {'inserted': 0, 'duplicates': 0, 'lookups': 0, 'cache_hits': 0}

    def _shard_idx(self, digest: str) -> int:
        if self.shards <= 1:
            return 0
        prefix = digest[:2] if len(digest) >= 2 else digest
        return int(prefix, 16) % self.shards

    def _pick(self, digest: str) -> ExactHashDedup:
        return self._dedups[self._shard_idx(digest)]

    @property
    def duplicates(self) -> int:
        return sum(d.duplicates for d in self._dedups)

    def is_duplicate(self, text: str, *, source: str = '', digest: str | None = None) -> bool:
        from indw.dedup.normalize import content_hash
        digest = digest or content_hash(text)
        dup = self._pick(digest).is_duplicate(text, source=source, digest=digest)
        self._sync_stats()
        return dup

    def register_hash(self, digest: str, *, source: str = '', assume_new: bool = False) -> bool:
        ok = self._pick(digest).register_hash(digest, source=source, assume_new=assume_new)
        self._sync_stats()
        return ok

    def register_text(self, text: str, *, source: str = '') -> bool:
        from indw.dedup.normalize import content_hash
        digest = content_hash(text)
        ok = self._pick(digest).register_text(text, source=source)
        self._sync_stats()
        return ok

    def flush(self) -> None:
        for index in self._indexes:
            index.flush()

    def close(self) -> None:
        for index in self._indexes:
            index.close()

    def summary(self) -> dict[str, Any]:
        self.flush()
        return {
            'shards': self.shards,
            'session_inserted': self.stats['inserted'],
            'session_duplicates': self.stats['duplicates'],
            'shard_summaries': [d.summary() for d in self._dedups],
        }

    def _sync_stats(self) -> None:
        inserted = duplicates = lookups = cache_hits = 0
        for dedup in self._dedups:
            st = dedup._index.stats
            inserted += int(st.get('inserted', 0))
            duplicates += int(st.get('duplicates', 0))
            lookups += int(st.get('lookups', 0))
            cache_hits += int(st.get('cache_hits', 0))
        self.stats = {
            'inserted': inserted,
            'duplicates': duplicates,
            'lookups': lookups,
            'cache_hits': cache_hits,
        }


def build_exact_dedup(
    work_dir: str | Path,
    *,
    db_path: str | Path | None = None,
    shards: int = 0,
) -> ExactHashDedup | ShardedExactDedup:
    if shards > 1:
        return ShardedExactDedup(work_dir, shards=shards)
    path = db_path or PersistentHashIndex.default_path(work_dir)
    index = PersistentHashIndex(path)
    return ExactHashDedup(index)
