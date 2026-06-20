from __future__ import annotations

import importlib
import logging
import re
from typing import Any

import numpy as np

from indw.dedup.embed.contracts import EmbeddingProvider

logger = logging.getLogger(__name__)
_WORD = re.compile(r'\w+', re.UNICODE)

class NoopEmbeddingProvider:
    name = 'noop'
    dimension = 0

    def initialize(self) -> None:
        return

    def shutdown(self) -> None:
        return

    def stats(self) -> dict[str, Any]:
        return {'embedding_provider_up': 0.0}

    def embed_batch(self, texts: list[str], *, language: str | None = None) -> list[np.ndarray]:
        return [np.zeros(0, dtype=np.float32) for _ in texts]

    def healthy(self) -> bool:
        return False

class HashEmbeddingProvider:

    name = 'hash'

    def __init__(self, *, dimension: int = 128):
        self.dimension = max(8, int(dimension))
        self._documents = 0

    def initialize(self) -> None:
        return

    def shutdown(self) -> None:
        return

    def stats(self) -> dict[str, Any]:
        return {
            'embedding_provider_up': 1.0,
            'embedding_documents': self._documents,
            'embedding_provider_device': 'cpu',
            'embedding_provider_model': self.name,
        }

    def embed_batch(self, texts: list[str], *, language: str | None = None) -> list[np.ndarray]:
        self._documents += len(texts)
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for tok in _WORD.findall(text.lower())[:2048]:
            h = hash(tok)
            idx = h % self.dimension
            vec[idx] += 1.0 if (h & 1) else -1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def healthy(self) -> bool:
        return True

def load_embedding_provider(
    spec: str,
    *,
    options: dict[str, Any] | None = None,
) -> EmbeddingProvider:
    raw = (spec or 'hash').strip().lower()
    opts = dict(options or {})
    if raw in ('0', 'false', 'off', 'none', 'noop'):
        return NoopEmbeddingProvider()
    if raw == 'hash':
        return HashEmbeddingProvider(dimension=int(opts.get('dimension', 128)))
    if raw in ('multilingual-e5-base', 'e5', 'intfloat/multilingual-e5-base', 'e5-base'):
        from indw.dedup.embed.e5 import E5EmbeddingProvider
        return E5EmbeddingProvider(**opts)
    if ':' in spec:
        module_name, class_name = spec.rsplit(':', 1)
        mod = importlib.import_module(module_name)
        klass = getattr(mod, class_name)
        return klass(**opts)
    raise ValueError(f'unknown embedding provider spec: {spec}')
