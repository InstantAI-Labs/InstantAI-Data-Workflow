from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

import numpy as np

from indw.dedup.embed.candidate import BlockingCandidateGenerator
from indw.dedup.embed.config import EmbeddingDedupConfig
from indw.dedup.embed.contracts import DedupDocumentMeta

class BucketAnnIndex:
    def __init__(self, cfg: EmbeddingDedupConfig, generator: BlockingCandidateGenerator):
        self.cfg = cfg
        self.generator = generator
        self._buckets: DefaultDict[tuple, list[tuple[int, DedupDocumentMeta]]] = defaultdict(list)
        self._entries: dict[int, DedupDocumentMeta] = {}
        self._total = 0

    def candidates_for(self, meta: DedupDocumentMeta) -> list[tuple[int, DedupDocumentMeta]]:
        key = self.generator.block_key(meta)
        raw = self._buckets.get(key, [])
        return self.generator.filter_candidates(meta, raw)

    def insert(self, cluster_id: int, meta: DedupDocumentMeta) -> None:
        self._entries[cluster_id] = meta
        key = self.generator.block_key(meta)
        bucket = self._buckets[key]
        bucket.append((cluster_id, meta))
        if len(bucket) > self.cfg.max_bucket_entries:
            del bucket[: len(bucket) - self.cfg.max_bucket_entries]
        self._total += 1
        if self._total > self.cfg.max_global_entries:
            self._evict_oldest()

    def replace(self, cluster_id: int, meta: DedupDocumentMeta) -> None:
        old = self._entries.get(cluster_id)
        if old is not None:
            key = self.generator.block_key(old)
            bucket = self._buckets.get(key, [])
            self._buckets[key] = [(cid, m) for cid, m in bucket if cid != cluster_id]
        self._entries[cluster_id] = meta
        key = self.generator.block_key(meta)
        bucket = self._buckets[key]
        bucket.append((cluster_id, meta))
        if len(bucket) > self.cfg.max_bucket_entries:
            del bucket[: len(bucket) - self.cfg.max_bucket_entries]

    def _evict_oldest(self) -> None:
        if not self._entries:
            return
        drop_id = min(self._entries)
        old = self._entries.pop(drop_id)
        key = self.generator.block_key(old)
        bucket = self._buckets.get(key, [])
        self._buckets[key] = [(cid, m) for cid, m in bucket if cid != drop_id]
        self._total = max(0, self._total - 1)

    @property
    def size(self) -> int:
        return len(self._entries)
