from __future__ import annotations

from typing import Optional

from indw.dedup.exact import PersistentHashIndex
from indw.dedup.normalize import content_hash


class ExactHashDedup:

    def __init__(self, index: Optional[PersistentHashIndex] = None) -> None:
        self._index = index
        self._seen: set[str] = set()
        self.duplicates = 0
        self.kept = 0
        self.skipped_trained = 0

    def is_duplicate(self, text: str, *, source: str = '', digest: str | None = None) -> bool:
        digest = digest or content_hash(text)
        if digest in self._seen:
            self.duplicates += 1
            return True
        if self._index is not None:
            found, hash_source = self._index.lookup_hash(digest)
            if found:
                self._seen.add(digest)
                self.duplicates += 1
                if hash_source in ('trained', 'exported'):
                    self.skipped_trained += 1
                return True
            self._index.register_hash(digest, source=source, assume_new=True)
            self._seen.add(digest)
            self.kept += 1
            return False
        self._seen.add(digest)
        self.kept += 1
        return False

    def seed_text(self, text: str, *, source: str = 'resume') -> None:
        digest = content_hash(text)
        if self._index is not None:
            self._index.register_hash(digest, source=source)
            return
        self._seen.add(digest)

    def has_text(self, text: str) -> bool:
        digest = content_hash(text)
        if self._index is not None:
            found, _ = self._index.lookup_hash(digest)
            return found
        return digest in self._seen

    def seed_text_if_missing(self, text: str, *, source: str = 'resume') -> bool:
        if self.has_text(text):
            return False
        self.seed_text(text, source=source)
        return True

    def __len__(self) -> int:
        if self._index is not None:
            try:
                return self._index.count()
            except Exception:
                return len(self._seen)
        return len(self._seen)
