from __future__ import annotations

from indw.dedup.embed.contracts import ClusterMatch, DedupDocumentMeta
from indw.dedup.embed.similarity import VectorSimilarityProvider

class NearDuplicateClusterBuilder:
    def __init__(self, similarity: VectorSimilarityProvider):
        self.similarity = similarity
        self._cluster_size: dict[int, int] = {}
        self._next_id = 1

    def allocate_cluster(self) -> int:
        cid = self._next_id
        self._next_id += 1
        self._cluster_size[cid] = 1
        return cid

    def attach_duplicate(self, cluster_id: int) -> None:
        self._cluster_size[cluster_id] = int(self._cluster_size.get(cluster_id, 1)) + 1

    def find_match(
        self,
        vector,
        candidates: list[tuple[int, DedupDocumentMeta]],
        *,
        threshold: float,
    ) -> ClusterMatch | None:
        if not candidates:
            return None
        best_sim = -1.0
        best: tuple[int, DedupDocumentMeta] | None = None
        for cid, meta in candidates:
            if meta.vector is None:
                continue
            sim = self.similarity.score(vector, meta.vector)
            if sim > best_sim:
                best_sim = sim
                best = (cid, meta)
        if best is None or best_sim < threshold:
            return None
        cid, meta = best
        return ClusterMatch(cluster_id=cid, similarity=best_sim, representative_rank=meta.representative_rank())

    def cluster_sizes(self) -> list[int]:
        return sorted(self._cluster_size.values(), reverse=True)
