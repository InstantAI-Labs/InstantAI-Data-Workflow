from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import unicodedata
from typing import Any

import numpy as np

from indw.store.io.cache import BoundedLRU, CacheStats

logger = logging.getLogger(__name__)
_WS = re.compile(r'\s+')
_CTRL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
_DEFAULT_MODEL = 'intfloat/multilingual-e5-base'
_WARNED = False

def _normalize_e5_text(text: str, *, max_chars: int) -> str:
    if not text:
        return ''
    out = unicodedata.normalize('NFC', text)
    out = _CTRL.sub('', out)
    out = _WS.sub(' ', out).strip()
    if max_chars > 0 and len(out) > max_chars:
        out = out[:max_chars]
    return out

def _format_passage(text: str, *, prefix: str, max_chars: int = 0) -> str:
    body = _normalize_e5_text(text, max_chars=max_chars)
    if not body:
        return ''
    if prefix and not body.startswith(prefix):
        return f'{prefix}{body}'
    return body

def _cache_key(text: str, *, prefix: str) -> str:
    payload = _format_passage(text, prefix=prefix, max_chars=0)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class _LruVectorCache:
    def __init__(self, max_size: int):
        self._disabled = int(max_size) <= 0
        self._stats = CacheStats()
        self._inner: BoundedLRU[np.ndarray] | None = None
        if not self._disabled:
            self._inner = BoundedLRU(max_size, stats=self._stats)

    @property
    def hits(self) -> int:
        return self._stats.hits

    @property
    def misses(self) -> int:
        return self._stats.misses

    def get(self, key: str) -> np.ndarray | None:
        if self._disabled or self._inner is None:
            self._stats.misses += 1
            return None
        return self._inner.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        if self._disabled or self._inner is None:
            return
        self._inner.put(key, vec)

    def clear(self) -> None:
        if self._inner is not None:
            self._inner.clear()


class E5EmbeddingProvider:
    name = 'multilingual-e5-base'
    dimension = 768

    def __init__(
        self,
        *,
        model_id: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        cache_size: int = 8192,
        max_chars: int = 8000,
        max_tokens: int = 512,
        passage_prefix: str = 'passage: ',
        dimension: int | None = None,
    ):
        self.model_id = model_id or _DEFAULT_MODEL
        self.requested_device = (device or 'auto').strip().lower()
        self.batch_size = max(1, int(batch_size))
        self.max_chars = max(256, int(max_chars))
        self.max_tokens = max(32, int(max_tokens))
        self.passage_prefix = passage_prefix
        if dimension is not None:
            self.dimension = int(dimension)
        self._lock = threading.RLock()
        self._model: Any = None
        self._device: str | None = None
        self._initialized = False
        self._healthy = False
        self._shutdown = False
        self._cache = _LruVectorCache(cache_size)
        self._batches = 0
        self._documents = 0
        self._failures = 0
        self._latency_ms = 0.0
        self._init_error = ''

    def initialize(self) -> None:
        global _WARNED
        with self._lock:
            if self._shutdown:
                raise RuntimeError('provider shutdown')
            if self._initialized:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                self._init_error = f'sentence-transformers missing: {exc}'
                if not _WARNED:
                    _WARNED = True
                    logger.warning('%s; embedding dedup disabled', self._init_error)
                raise
            device = self._resolve_device()
            model = SentenceTransformer(self.model_id, device=device)
            model.max_seq_length = self.max_tokens
            if hasattr(model, 'eval'):
                model.eval()
            self._model = model
            self._device = device
            try:
                dim = int(model.get_sentence_embedding_dimension())
                if dim > 0:
                    self.dimension = dim
            except Exception:
                pass
            self._initialized = True
            self._healthy = True

    def _resolve_device(self) -> str:
        req = self.requested_device
        if req in ('auto', ''):
            try:
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
            except Exception:
                pass
            return 'cpu'
        if req.startswith('cuda'):
            try:
                import torch
                if torch.cuda.is_available():
                    return req
            except Exception:
                pass
            return 'cpu'
        return req

    def healthy(self) -> bool:
        if self._shutdown:
            return False
        if not self._initialized:
            try:
                self.initialize()
            except Exception:
                return False
        return self._healthy

    def stats(self) -> dict[str, Any]:
        mem_mb = 0.0
        if self._initialized and self._model is not None:
            try:
                import torch
                if self._device and self._device.startswith('cuda') and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                mem_mb = 0.0
        return {
            'embedding_provider_up': 1.0 if self.healthy() else 0.0,
            'embedding_batches': int(self._batches),
            'embedding_documents': int(self._documents),
            'embedding_latency_ms': round(self._latency_ms, 2),
            'embedding_cache_hits': int(self._cache.hits),
            'embedding_cache_misses': int(self._cache.misses),
            'embedding_failures': int(self._failures),
            'embedding_memory_mb': round(mem_mb, 2),
            'embedding_provider_device': self._device or '',
            'embedding_provider_model': self.model_id,
        }

    def embed_batch(
        self,
        texts: list[str],
        *,
        language: str | None = None,
    ) -> list[np.ndarray]:
        del language
        if not texts:
            return []
        if self._shutdown:
            raise RuntimeError('provider shutdown')
        if not self._initialized:
            self.initialize()
        assert self._model is not None

        keys = [_cache_key(t, prefix=self.passage_prefix) for t in texts]
        out: list[np.ndarray | None] = [None] * len(texts)
        pending_idx: list[int] = []
        pending_texts: list[str] = []
        with self._lock:
            for i, key in enumerate(keys):
                cached = self._cache.get(key)
                if cached is not None:
                    out[i] = cached
                else:
                    pending_idx.append(i)
                    pending_texts.append(texts[i])

        if pending_texts:
            encoded = self._encode_with_recovery(pending_texts)
            with self._lock:
                for pos, vec in zip(pending_idx, encoded):
                    out[pos] = vec
                    if vec is not None and np.linalg.norm(vec) > 0:
                        self._cache.put(keys[pos], vec)

        final: list[np.ndarray] = []
        for i, vec in enumerate(out):
            if vec is None:
                self._failures += 1
                final.append(np.zeros(self.dimension, dtype=np.float32))
            else:
                final.append(vec.astype(np.float32, copy=False))
                self._documents += 1
        return final

    def _encode_with_recovery(self, texts: list[str]) -> list[np.ndarray | None]:
        formatted = [
            _format_passage(t, prefix=self.passage_prefix, max_chars=self.max_chars)
            for t in texts
        ]
        formatted = [t if t else 'passage: ' for t in formatted]
        batch_size = min(self.batch_size, len(formatted))
        results: list[np.ndarray | None] = []
        start = 0
        while start < len(formatted):
            chunk = formatted[start:start + batch_size]
            try:
                vecs = self._encode_chunk(chunk)
                results.extend(vecs)
                start += len(chunk)
            except Exception as exc:
                if self._is_oom(exc) and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    logger.debug('E5 embed OOM; retry batch_size=%d (%s)', batch_size, exc)
                    continue
                if len(chunk) == 1:
                    self._failures += 1
                    logger.debug('E5 embed failed for single document: %s', exc)
                    results.append(None)
                    start += 1
                    continue
                mid = len(chunk) // 2
                first = self._encode_with_recovery(texts[start:start + mid])
                second = self._encode_with_recovery(texts[start + mid:start + len(chunk)])
                results.extend(first)
                results.extend(second)
                start += len(chunk)
        return results

    def _encode_chunk(self, chunk: list[str]) -> list[np.ndarray]:
        assert self._model is not None
        t0 = time.perf_counter()
        with self._lock:
            try:
                import torch
                ctx = torch.inference_mode()
            except Exception:
                from contextlib import nullcontext
                ctx = nullcontext()
            with ctx:
                raw = self._model.encode(
                    chunk,
                    batch_size=min(self.batch_size, len(chunk)),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._latency_ms += elapsed
        self._batches += 1
        if isinstance(raw, np.ndarray):
            if raw.ndim == 1:
                return [raw.astype(np.float32, copy=False)]
            return [raw[i].astype(np.float32, copy=False) for i in range(len(raw))]
        return [np.asarray(v, dtype=np.float32) for v in raw]

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return 'out of memory' in msg or 'cuda' in msg and 'memory' in msg

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._healthy = False
            self._cache.clear()
            model = self._model
            self._model = None
            self._initialized = False
        if model is not None:
            try:
                del model
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
