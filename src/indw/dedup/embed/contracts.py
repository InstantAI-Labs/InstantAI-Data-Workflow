from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    def embed_batch(
        self,
        texts: list[str],
        *,
        language: str | None = None,
    ) -> list[np.ndarray]: ...

    def healthy(self) -> bool: ...

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def stats(self) -> dict[str, Any]: ...

@runtime_checkable
class SimilarityProvider(Protocol):
    metric: str

    def score(self, a: np.ndarray, b: np.ndarray) -> float: ...

    def pairwise_max(self, query: np.ndarray, candidates: list[np.ndarray]) -> tuple[float, int]: ...

@dataclass
class DedupDocumentMeta:
    text: str
    language: str = 'unknown'
    domain: str = 'unknown'
    quality: float = 0.0
    knowledge: float = 0.0
    utility: float = 0.0
    toxicity: float = 0.0
    artifact: float = 0.0
    length: int = 0
    token_sig: frozenset[str] = field(default_factory=frozenset)
    vector: np.ndarray | None = None

    def representative_rank(self) -> float:
        return (
            self.quality * 0.45
            + self.knowledge * 0.01 * 0.20
            + self.utility * 10.0 * 0.20
            - self.toxicity * 0.10
            - self.artifact * 0.01 * 0.05
        )

@dataclass
class ClusterMatch:
    cluster_id: int
    similarity: float
    representative_rank: float

@dataclass
class EmbeddingDedupDiagnostics:
    candidates_checked: int = 0
    clusters_formed: int = 0
    duplicates_removed: int = 0
    kept: int = 0
    embed_calls: int = 0
    embed_failures: int = 0
    threshold: float = 0.0
    similarity_samples: list[float] = field(default_factory=list)
    cluster_sizes: list[int] = field(default_factory=list)
    processing_ms: float = 0.0
    disabled: bool = False
    disable_reason: str = ''

    def to_dict(self) -> dict[str, Any]:
        sims = self.similarity_samples
        dist: dict[str, Any] = {}
        if sims:
            arr = np.array(sims, dtype=np.float64)
            for p in (10, 50, 90):
                dist[f'p{p}'] = round(float(np.percentile(arr, p)), 4)
            dist['mean'] = round(float(arr.mean()), 4)
        keep_total = max(self.kept + self.duplicates_removed, 1)
        return {
            'embedding_candidates_checked': self.candidates_checked,
            'embedding_clusters': self.clusters_formed,
            'embedding_duplicates_removed': self.duplicates_removed,
            'embedding_kept': self.kept,
            'embedding_keep_rate': round(self.kept / keep_total, 4),
            'embedding_reject_rate': round(self.duplicates_removed / keep_total, 4),
            'embedding_embed_calls': self.embed_calls,
            'embedding_embed_failures': self.embed_failures,
            'embedding_threshold': round(self.threshold, 4),
            'embedding_similarity_distribution': dist,
            'embedding_cluster_sizes': self.cluster_sizes[-20:],
            'embedding_processing_ms': round(self.processing_ms, 2),
            'embedding_disabled': self.disabled,
            'embedding_disable_reason': self.disable_reason,
        }
