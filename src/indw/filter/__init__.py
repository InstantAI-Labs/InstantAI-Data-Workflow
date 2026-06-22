_LAZY = {
    'QualityGate': ('indw.filter.gate.quality', 'QualityGate'),
    'QualityPipelineConfig': ('indw.filter.spec.quality', 'QualityPipelineConfig'),
    'CorpusDocument': ('indw.filter.spec.document', 'CorpusDocument'),
    'CuratorDecision': ('indw.filter.spec.document', 'CuratorDecision'),
    'Provenance': ('indw.filter.spec.document', 'Provenance'),
    'PipelineAction': ('indw.filter.spec.document', 'PipelineAction'),
    'EXPORT_ACTIONS': ('indw.filter.spec.document', 'EXPORT_ACTIONS'),
    'KEEP_ACTIONS': ('indw.filter.spec.document', 'KEEP_ACTIONS'),
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
