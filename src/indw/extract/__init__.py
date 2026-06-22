_LAZY = {
    'extract_knowledge': ('indw.extract.core.units', 'extract_knowledge'),
    'DocumentExecutionContext': ('indw.extract.core.context', 'DocumentExecutionContext'),
    'bind_document_context': ('indw.extract.core.context', 'bind_document_context'),
    'clear_document_context': ('indw.extract.core.context', 'clear_document_context'),
    'get_document_context': ('indw.extract.core.context', 'get_document_context'),
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
