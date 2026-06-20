from __future__ import annotations

from typing import Any

from indw.dedup.embed.providers import load_embedding_provider

_PROVIDER: Any = None
_BACKEND = 'cpu_hash'


def _provider():
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = load_embedding_provider('hash')
        _PROVIDER.initialize()
    return _PROVIDER


def embed_dedup_batch(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """CPU hash-embedding pool. GPU neural inference uses data.dedup.embed.e5 when embedding extra installed."""
    provider = _provider()
    texts = [str(c.get('text') or '') for c in candidates]
    vectors = provider.embed_batch(texts)
    out: list[dict[str, Any]] = []
    for cand, vec in zip(candidates, vectors):
        row = dict(cand)
        row['embedding'] = vec
        row['_embed_backend'] = _BACKEND
        out.append(row)
    return out


def pool_resource_kind() -> str:
    return _BACKEND
