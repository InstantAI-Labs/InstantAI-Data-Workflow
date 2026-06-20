from indw.dedup.embed.config import EmbeddingDedupConfig
from indw.dedup.embed.e5 import E5EmbeddingProvider
from indw.dedup.embed.pipeline import StreamingEmbeddingDedup, create_embedding_dedup

__all__ = [
    'EmbeddingDedupConfig',
    'StreamingEmbeddingDedup',
    'create_embedding_dedup',
    'E5EmbeddingProvider',
]
