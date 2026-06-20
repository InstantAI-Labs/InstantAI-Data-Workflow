from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from indw.dedup.embed.ann import BucketAnnIndex
from indw.dedup.embed.candidate import BlockingCandidateGenerator, build_document_meta
from indw.dedup.embed.cluster import NearDuplicateClusterBuilder
from indw.dedup.embed.config import EmbeddingDedupConfig
from indw.dedup.embed.contracts import EmbeddingDedupDiagnostics
from indw.dedup.embed.providers import load_embedding_provider
from indw.dedup.embed.representative import QualityRepresentativeSelector
from indw.dedup.embed.similarity import VectorSimilarityProvider
from indw.dedup.embed.threshold import AdaptiveSimilarityThreshold

logger = logging.getLogger(__name__)
_WARNED = False

class StreamingEmbeddingDedup:
    def __init__(self, cfg: EmbeddingDedupConfig | None = None):
        global _WARNED
        self.cfg = cfg or EmbeddingDedupConfig()
        self.diagnostics = EmbeddingDedupDiagnostics()
        self.duplicates = 0
        self.kept = 0
        self._disabled = not self.cfg.enabled
        self._provider = None
        self._similarity: VectorSimilarityProvider | None = None
        self._generator: BlockingCandidateGenerator | None = None
        self._index: BucketAnnIndex | None = None
        self._clusters: NearDuplicateClusterBuilder | None = None
        self._selector: QualityRepresentativeSelector | None = None
        self._threshold: AdaptiveSimilarityThreshold | None = None
        if self._disabled:
            return
        try:
            self._provider = load_embedding_provider(
                self.cfg.provider,
                options={
                    'dimension': self.cfg.dimension,
                    'batch_size': self.cfg.batch_size,
                    'device': self.cfg.device,
                    'cache_size': self.cfg.cache_size,
                    **(self.cfg.provider_options or {}),
                },
            )
            if not self._provider.healthy():
                raise RuntimeError(f'provider {self.cfg.provider!r} not healthy')
            self._similarity = VectorSimilarityProvider(self.cfg.similarity)
            self._generator = BlockingCandidateGenerator(self.cfg)
            self._index = BucketAnnIndex(self.cfg, self._generator)
            self._clusters = NearDuplicateClusterBuilder(self._similarity)
            self._selector = QualityRepresentativeSelector(self.cfg.quality_margin)
            self._threshold = AdaptiveSimilarityThreshold(
                min_threshold=self.cfg.min_threshold,
                max_threshold=self.cfg.max_threshold,
            )
        except Exception as exc:
            self._disable(f'init failed: {exc}', warn=not _WARNED)
            _WARNED = True

    def _disable(self, reason: str, *, warn: bool = True) -> None:
        self._disabled = True
        self.diagnostics.disabled = True
        self.diagnostics.disable_reason = reason
        if warn:
            logger.warning('Embedding semantic dedup disabled (%s); pipeline continues', reason)

    @property
    def enabled(self) -> bool:
        return not self._disabled

    def _embed(self, text: str, *, language: str | None) -> np.ndarray | None:
        assert self._provider is not None
        t0 = time.perf_counter()
        self.diagnostics.embed_calls += 1
        try:
            vecs = self._provider.embed_batch([text], language=language)
            self.diagnostics.processing_ms += (time.perf_counter() - t0) * 1000.0
            if not vecs:
                return None
            return vecs[0]
        except Exception as exc:
            self.diagnostics.embed_failures += 1
            self._disable(f'embed failed: {exc}', warn=True)
            return None

    def _meta_from_doc(self, text: str, doc: Any, *, quality_score: float) -> Any:
        utility = 0.0
        tu = getattr(doc, 'training_utility', None)
        if tu is not None:
            utility = float(getattr(tu, 'utility_score', 0.0) or 0.0)
        elif getattr(doc, 'utility_normalized', 0.0):
            utility = float(doc.utility_normalized)
        return build_document_meta(
            text,
            language=str(getattr(doc, 'language', 'unknown') or 'unknown'),
            domain=str(getattr(doc, 'domain', 'unknown') or 'unknown'),
            quality=float(quality_score),
            knowledge=float(getattr(doc, 'knowledge', 0.0) or 0.0),
            utility=utility,
            toxicity=float(getattr(doc, 'toxicity_score', 0.0) or 0.0),
            artifact=float(getattr(doc, 'artifact_contamination', 0.0) or 0.0),
        )

    def should_keep(self, text: str, quality_score: float = 0.0, *, doc: Any = None) -> bool:
        if self._disabled:
            return True
        meta = self._meta_from_doc(text, doc, quality_score=quality_score) if doc is not None else build_document_meta(
            text,
            quality=quality_score,
        )
        return self._evaluate(meta, register_on_keep=False)

    def register(self, text: str, quality_score: float = 0.0, *, doc: Any = None) -> None:
        if self._disabled:
            return
        meta = self._meta_from_doc(text, doc, quality_score=quality_score) if doc is not None else build_document_meta(
            text,
            quality=quality_score,
        )
        self._evaluate(meta, register_on_keep=True)

    def evaluate_and_register(
        self,
        text: str,
        quality_score: float = 0.0,
        *,
        doc: Any = None,
    ) -> bool:
        if self._disabled:
            return True
        meta = self._meta_from_doc(text, doc, quality_score=quality_score) if doc is not None else build_document_meta(
            text,
            quality=quality_score,
        )
        return self._evaluate(meta, register_on_keep=True)

    def _evaluate(self, meta: Any, *, register_on_keep: bool) -> bool:
        assert (
            self._index is not None
            and self._clusters is not None
            and self._selector is not None
            and self._threshold is not None
            and self._provider is not None
        )
        candidates = self._index.candidates_for(meta)
        self.diagnostics.candidates_checked += len(candidates)
        vector = self._embed(meta.text, language=meta.language)
        if vector is None:
            return True
        meta.vector = vector
        threshold = self._threshold.value() if self.cfg.adaptive_threshold else self.cfg.max_threshold
        self.diagnostics.threshold = threshold
        match = self._clusters.find_match(vector, candidates, threshold=threshold)
        if match is not None:
            self.diagnostics.similarity_samples.append(match.similarity)
            if self._selector.should_keep(meta, match):
                if register_on_keep:
                    self._index.replace(match.cluster_id, meta)
                    self.kept += 1
                    self.diagnostics.clusters_formed = self._index.size
                return True
            self._clusters.attach_duplicate(match.cluster_id)
            self._threshold.record_match(match.similarity)
            self.duplicates += 1
            self.diagnostics.duplicates_removed += 1
            return False
        max_sim = 0.0
        if candidates:
            vecs = [c.vector for _, c in candidates if c.vector is not None]
            if vecs:
                max_sim, _ = self._similarity.pairwise_max(vector, vecs)
                self.diagnostics.similarity_samples.append(max_sim)
        self._threshold.record_non_match(max_sim)
        if register_on_keep:
            cid = self._clusters.allocate_cluster()
            self._index.insert(cid, meta)
            self.kept += 1
            self.diagnostics.clusters_formed = self._index.size
        return True

    def summary(self) -> dict[str, Any]:
        self.diagnostics.clusters_formed = self._index.size if self._index is not None else 0
        if self._clusters is not None:
            self.diagnostics.cluster_sizes = self._clusters.cluster_sizes()
        if self._threshold is not None:
            self.diagnostics.threshold = self._threshold.value()
            dist = self._threshold.distribution()
            if dist:
                self.diagnostics.similarity_samples = list(
                    self.diagnostics.similarity_samples[-512:]
                )
        out = self.diagnostics.to_dict()
        out['embedding_duplicates'] = int(self.duplicates)
        out['embedding_kept'] = int(self.kept)
        out['embedding_index_size'] = int(self._index.size if self._index else 0)
        out['embedding_provider'] = getattr(self._provider, 'name', 'disabled')
        if self._provider is not None and hasattr(self._provider, 'stats'):
            try:
                out.update(self._provider.stats())
            except Exception:
                pass
        return out

def create_embedding_dedup(cfg: EmbeddingDedupConfig | None = None, **kwargs: Any) -> StreamingEmbeddingDedup:
    if cfg is None:
        cfg = EmbeddingDedupConfig(**kwargs) if kwargs else EmbeddingDedupConfig()
    return StreamingEmbeddingDedup(cfg)
