from __future__ import annotations

import hashlib
import threading
from typing import Any

from indw.store.io.cache import BoundedLRU, CacheStats
from indw.schedule.config.resolve import env_int

TextFingerprint = tuple[int, bytes]


_FP_TEXT_CACHE: BoundedLRU[TextFingerprint] | None = None
_FP_STATS = CacheStats()


def _fp_text_cache() -> BoundedLRU[TextFingerprint]:
    global _FP_TEXT_CACHE
    if _FP_TEXT_CACHE is None:
        with _INIT_LOCK:
            if _FP_TEXT_CACHE is None:
                from indw.config.defaults import TEXT_FINGERPRINT_CACHE_SIZE
                _FP_TEXT_CACHE = _make_cache(
                    'TEXT_FP_TEXT', TEXT_FINGERPRINT_CACHE_SIZE, _FP_STATS,
                )
    return _FP_TEXT_CACHE


def text_fingerprint(text: str) -> TextFingerprint:
    if not text:
        return (0, b'')
    cache = _fp_text_cache()
    hit = cache.get(text)
    if hit is not None:
        return hit
    digest = hashlib.blake2b(
        text.encode('utf-8', 'surrogatepass'),
        digest_size=16,
    ).digest()
    fp: TextFingerprint = (len(text), digest)
    cache.put(text, fp)
    return fp


_EVIDENCE_STATS = CacheStats()
_RAW_STATS = CacheStats()
_NAV_STATS = CacheStats()
_PUB_STATS = CacheStats()
_STRUCTURE_STATS = CacheStats()
_LAYOUT_STATS = CacheStats()
_SCAFFOLD_STATS = CacheStats()
_FILTERS_STATS = CacheStats()

_EVIDENCE_CACHE: BoundedLRU[Any] | None = None
_RAW_CACHE: BoundedLRU[Any] | None = None
_NAV_CACHE: BoundedLRU[Any] | None = None
_PUB_CACHE: BoundedLRU[Any] | None = None
_STRUCTURE_CACHE: BoundedLRU[Any] | None = None
_LAYOUT_CACHE: BoundedLRU[Any] | None = None
_SCAFFOLD_CACHE: BoundedLRU[Any] | None = None
_FILTERS_CACHE: BoundedLRU[Any] | None = None
_INIT_LOCK = threading.Lock()
_CACHE_BOOST = 1


def bootstrap_session_caches(sizes: dict[str, int]) -> None:
    g = globals()
    mapping = {
        'evidence': (_EVIDENCE_STATS, '_EVIDENCE_CACHE'),
        'raw': (_RAW_STATS, '_RAW_CACHE'),
        'nav': (_NAV_STATS, '_NAV_CACHE'),
        'publication': (_PUB_STATS, '_PUB_CACHE'),
        'structure': (_STRUCTURE_STATS, '_STRUCTURE_CACHE'),
        'layout': (_LAYOUT_STATS, '_LAYOUT_CACHE'),
        'scaffold': (_SCAFFOLD_STATS, '_SCAFFOLD_CACHE'),
        'filters': (_FILTERS_STATS, '_FILTERS_CACHE'),
    }
    with _INIT_LOCK:
        for key, (stats, attr) in mapping.items():
            g[attr] = BoundedLRU(max(64, int(sizes.get(key, 256))), stats=stats)


def set_session_cache_boost(boost: int) -> None:
    global _CACHE_BOOST
    _CACHE_BOOST = max(1, min(4, int(boost)))
    with _INIT_LOCK:
        for cache in (
            _EVIDENCE_CACHE, _RAW_CACHE, _NAV_CACHE, _PUB_CACHE,
            _STRUCTURE_CACHE, _LAYOUT_CACHE, _SCAFFOLD_CACHE, _FILTERS_CACHE,
        ):
            if cache is None:
                continue
            base = getattr(cache, '_base_maxsize', cache._maxsize)
            cache._maxsize = max(64, base * _CACHE_BOOST)


def _make_cache(
    attr: str,
    default_size: int,
    stats: CacheStats,
) -> BoundedLRU[Any]:
    size = env_int(f'INSTANT_{attr.upper()}_CACHE_SIZE', default_size, minimum=64)
    cache = BoundedLRU(max(64, size * _CACHE_BOOST), stats=stats)
    cache._base_maxsize = max(64, size)
    return cache


def _evidence_cache() -> BoundedLRU[Any]:
    global _EVIDENCE_CACHE
    if _EVIDENCE_CACHE is None:
        with _INIT_LOCK:
            if _EVIDENCE_CACHE is None:
                from indw.config.defaults import SEMANTIC_EVIDENCE_CACHE_SIZE
                _EVIDENCE_CACHE = _make_cache(
                    'EVIDENCE', SEMANTIC_EVIDENCE_CACHE_SIZE, _EVIDENCE_STATS,
                )
    return _EVIDENCE_CACHE


def _raw_cache() -> BoundedLRU[Any]:
    global _RAW_CACHE
    if _RAW_CACHE is None:
        with _INIT_LOCK:
            if _RAW_CACHE is None:
                from indw.config.defaults import RAW_FEATURE_CACHE_SIZE
                _RAW_CACHE = _make_cache('RAW_FEATURE', RAW_FEATURE_CACHE_SIZE, _RAW_STATS)
    return _RAW_CACHE


def _nav_cache() -> BoundedLRU[Any]:
    global _NAV_CACHE
    if _NAV_CACHE is None:
        with _INIT_LOCK:
            if _NAV_CACHE is None:
                from indw.config.defaults import NAV_FEATURE_CACHE_SIZE
                _NAV_CACHE = _make_cache('NAV_FEATURE', NAV_FEATURE_CACHE_SIZE, _NAV_STATS)
    return _NAV_CACHE


def _pub_cache() -> BoundedLRU[Any]:
    global _PUB_CACHE
    if _PUB_CACHE is None:
        with _INIT_LOCK:
            if _PUB_CACHE is None:
                from indw.config.defaults import PUBLICATION_ROLE_CACHE_SIZE
                _PUB_CACHE = _make_cache(
                    'PUBLICATION_ROLE', PUBLICATION_ROLE_CACHE_SIZE, _PUB_STATS,
                )
    return _PUB_CACHE


def _structure_cache() -> BoundedLRU[Any]:
    global _STRUCTURE_CACHE
    if _STRUCTURE_CACHE is None:
        with _INIT_LOCK:
            if _STRUCTURE_CACHE is None:
                from indw.config.defaults import STRUCTURE_CACHE_SIZE
                _STRUCTURE_CACHE = _make_cache('STRUCTURE', STRUCTURE_CACHE_SIZE, _STRUCTURE_STATS)
    return _STRUCTURE_CACHE


def _layout_cache() -> BoundedLRU[Any]:
    global _LAYOUT_CACHE
    if _LAYOUT_CACHE is None:
        with _INIT_LOCK:
            if _LAYOUT_CACHE is None:
                from indw.config.defaults import LAYOUT_CACHE_SIZE
                _LAYOUT_CACHE = _make_cache('LAYOUT', LAYOUT_CACHE_SIZE, _LAYOUT_STATS)
    return _LAYOUT_CACHE


def _scaffold_cache() -> BoundedLRU[Any]:
    global _SCAFFOLD_CACHE
    if _SCAFFOLD_CACHE is None:
        with _INIT_LOCK:
            if _SCAFFOLD_CACHE is None:
                from indw.config.defaults import SCAFFOLD_CACHE_SIZE
                _SCAFFOLD_CACHE = _make_cache('SCAFFOLD', SCAFFOLD_CACHE_SIZE, _SCAFFOLD_STATS)
    return _SCAFFOLD_CACHE


def _filters_cache() -> BoundedLRU[Any]:
    global _FILTERS_CACHE
    if _FILTERS_CACHE is None:
        with _INIT_LOCK:
            if _FILTERS_CACHE is None:
                from indw.config.defaults import CONTENT_FILTERS_CACHE_SIZE
                _FILTERS_CACHE = _make_cache(
                    'CONTENT_FILTERS', CONTENT_FILTERS_CACHE_SIZE, _FILTERS_STATS,
                )
    return _FILTERS_CACHE


def scaffold_cache_key(text: str, op: str) -> tuple[Any, ...] | None:
    if not text:
        return None
    return (text_fingerprint(text), op)


def cached_scaffold(text: str, op: str, compute: Any) -> Any:
    key = scaffold_cache_key(text, op)
    if key is None:
        return compute()
    cache = _scaffold_cache()
    hit = cache.get(key)
    if hit is not None:
        return hit
    result = compute()
    cache.put(key, result)
    return result


def evidence_cache_key(
    text: str,
    *,
    filters: Any | None,
    duplicate_ratio: float,
    enabled: bool,
    bundle: Any | None,
) -> tuple[Any, ...] | None:
    if bundle is not None or filters is not None or not text:
        return None
    fp = text_fingerprint(text)
    return (fp, float(duplicate_ratio), bool(enabled))


def raw_feature_cache_key(text: str) -> TextFingerprint | None:
    if not text:
        return None
    return text_fingerprint(text)


def nav_feature_cache_key(text: str, position_ratio: float, *, corpus_active: bool) -> tuple[Any, ...] | None:
    if not text or corpus_active:
        return None
    return (text_fingerprint(text), round(float(position_ratio), 4))


def publication_role_cache_key(text: str, position_ratio: float) -> tuple[Any, ...] | None:
    if not text:
        return None
    return (text_fingerprint(text), round(float(position_ratio), 4))


def layout_cache_key(text: str, *, in_fence: bool = False) -> tuple[Any, ...] | None:
    if not text:
        return None
    return (text_fingerprint(text), bool(in_fence))


structure_cache_key = raw_feature_cache_key


def filters_cache_key(
    text: str,
    *,
    words: list[str] | None,
    lines: list[str] | None,
) -> TextFingerprint | None:
    if not text or not text.strip():
        return None
    if words is None and lines is None:
        return text_fingerprint(text)
    if words is not None and lines is not None:
        return text_fingerprint(text)
    return None


def trim_document_caches(*, keep_ratio: float = 0.5) -> None:
    with _INIT_LOCK:
        for cache in (
            _FP_TEXT_CACHE, _EVIDENCE_CACHE, _RAW_CACHE, _NAV_CACHE, _PUB_CACHE,
            _STRUCTURE_CACHE, _LAYOUT_CACHE, _SCAFFOLD_CACHE, _FILTERS_CACHE,
        ):
            if cache is not None:
                cache.trim_to_ratio(keep_ratio)


def clear_document_caches() -> None:
    with _INIT_LOCK:
        for cache in (
            _FP_TEXT_CACHE, _EVIDENCE_CACHE, _RAW_CACHE, _NAV_CACHE, _PUB_CACHE,
            _STRUCTURE_CACHE, _LAYOUT_CACHE, _SCAFFOLD_CACHE, _FILTERS_CACHE,
        ):
            if cache is not None:
                cache.clear()


def reset_evidence_session() -> dict[str, Any]:
    clear_document_caches()
    for stats in (
        _FP_STATS, _EVIDENCE_STATS, _RAW_STATS, _NAV_STATS, _PUB_STATS,
        _STRUCTURE_STATS, _LAYOUT_STATS, _SCAFFOLD_STATS, _FILTERS_STATS,
    ):
        stats.hits = 0
        stats.misses = 0
        stats.evictions = 0
    return session_cache_stats()


def session_cache_stats() -> dict[str, Any]:
    return {
        'text_fingerprint': _FP_STATS.to_dict(),
        'evidence': _EVIDENCE_STATS.to_dict(),
        'raw_features': _RAW_STATS.to_dict(),
        'navigation': _NAV_STATS.to_dict(),
        'publication_roles': _PUB_STATS.to_dict(),
        'structure': _STRUCTURE_STATS.to_dict(),
        'layout': _LAYOUT_STATS.to_dict(),
        'scaffold': _SCAFFOLD_STATS.to_dict(),
        'content_filters': _FILTERS_STATS.to_dict(),
        'evidence_entries': len(_evidence_cache()) if _EVIDENCE_CACHE is not None else 0,
        'raw_entries': len(_raw_cache()) if _RAW_CACHE is not None else 0,
        'nav_entries': len(_nav_cache()) if _NAV_CACHE is not None else 0,
        'pub_entries': len(_pub_cache()) if _PUB_CACHE is not None else 0,
        'structure_entries': len(_structure_cache()) if _STRUCTURE_CACHE is not None else 0,
        'layout_entries': len(_layout_cache()) if _LAYOUT_CACHE is not None else 0,
        'scaffold_entries': len(_scaffold_cache()) if _SCAFFOLD_CACHE is not None else 0,
        'filters_entries': len(_filters_cache()) if _FILTERS_CACHE is not None else 0,
    }


def get_evidence_cache() -> BoundedLRU[Any]:
    return _evidence_cache()


def get_raw_feature_cache() -> BoundedLRU[Any]:
    return _raw_cache()


def get_nav_feature_cache() -> BoundedLRU[Any]:
    return _nav_cache()


def get_publication_role_cache() -> BoundedLRU[Any]:
    return _pub_cache()


def get_structure_cache() -> BoundedLRU[Any]:
    return _structure_cache()


def get_layout_cache() -> BoundedLRU[Any]:
    return _layout_cache()


def get_filters_cache() -> BoundedLRU[Any]:
    return _filters_cache()
