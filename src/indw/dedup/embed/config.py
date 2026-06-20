from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from indw.config import defaults as D

@dataclass
class EmbeddingDedupConfig:
    enabled: bool = False
    provider: str = 'hash'
    provider_options: dict[str, Any] | None = None
    similarity: str = 'cosine'
    dimension: int = D.EMBED_DEDUP_DIMENSION
    max_candidates: int = D.EMBED_DEDUP_MAX_CANDIDATES
    max_bucket_entries: int = D.EMBED_DEDUP_MAX_BUCKET_ENTRIES
    max_global_entries: int = D.EMBED_DEDUP_MAX_GLOBAL_ENTRIES
    batch_size: int = D.EMBED_DEDUP_BATCH_SIZE
    device: str = 'auto'
    cache_size: int = D.EMBED_DEDUP_CACHE_SIZE
    adaptive_threshold: bool = True
    min_threshold: float = 0.75
    max_threshold: float = 0.99
    quality_margin: float = 0.05
    same_language_only: bool = True
    block_by_domain: bool = True
    block_by_length: bool = True
    length_bucket_chars: int = 500
    token_overlap_min: float = 0.12
    simhash_bucket_bits: int = 12

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> 'EmbeddingDedupConfig':
        if not raw:
            return cls()
        opts = raw.get('provider_options')
        return cls(
            enabled=bool(raw.get('enabled', False)),
            provider=str(raw.get('provider', raw.get('embedding_provider', 'hash'))),
            provider_options=dict(opts) if isinstance(opts, dict) else None,
            similarity=str(raw.get('similarity', raw.get('similarity_provider', 'cosine'))),
            dimension=int(raw.get('dimension', D.EMBED_DEDUP_DIMENSION)),
            max_candidates=int(raw.get('max_candidates', D.EMBED_DEDUP_MAX_CANDIDATES)),
            max_bucket_entries=int(raw.get('max_bucket_entries', D.EMBED_DEDUP_MAX_BUCKET_ENTRIES)),
            max_global_entries=int(raw.get('max_global_entries', D.EMBED_DEDUP_MAX_GLOBAL_ENTRIES)),
            batch_size=int(raw.get('batch_size', D.EMBED_DEDUP_BATCH_SIZE)),
            device=str(raw.get('device', 'auto')),
            cache_size=int(raw.get('cache_size', D.EMBED_DEDUP_CACHE_SIZE)),
            adaptive_threshold=bool(raw.get('adaptive_threshold', True)),
            min_threshold=float(raw.get('min_threshold', 0.75)),
            max_threshold=float(raw.get('max_threshold', 0.99)),
            quality_margin=float(raw.get('quality_margin', 0.05)),
            same_language_only=bool(raw.get('same_language_only', True)),
            block_by_domain=bool(raw.get('block_by_domain', True)),
            block_by_length=bool(raw.get('block_by_length', True)),
            length_bucket_chars=int(raw.get('length_bucket_chars', 500)),
            token_overlap_min=float(raw.get('token_overlap_min', 0.12)),
            simhash_bucket_bits=int(raw.get('simhash_bucket_bits', 12)),
        )
