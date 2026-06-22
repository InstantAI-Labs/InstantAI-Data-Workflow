_LAZY = {
    'merge_with_quality': ('indw.schedule.core', 'merge_with_quality'),
    'PipelineRunner': ('indw.schedule.stages.runner', 'PipelineRunner'),
    'PipelineStats': ('indw.schedule.stages.runner', 'PipelineStats'),
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
