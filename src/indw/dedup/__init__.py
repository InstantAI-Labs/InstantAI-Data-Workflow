_LAZY = {
    'PersistentHashIndex': ('indw.dedup.exact', 'PersistentHashIndex'),
    'StreamingFuzzyDedup': ('indw.dedup.fuzzy', 'StreamingFuzzyDedup'),
    'content_hash': ('indw.dedup.normalize', 'content_hash'),
    'normalize_for_dedup': ('indw.dedup.normalize', 'normalize_for_dedup'),
    'restore_dedup_from_jsonl': ('indw.dedup.replay', 'restore_dedup_from_jsonl'),
    'StreamingSemanticDedup': ('indw.dedup.semantic', 'StreamingSemanticDedup'),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        mod_path, attr = _LAZY[name]
        val = getattr(importlib.import_module(mod_path), attr)
        globals()[name] = val
        return val
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
