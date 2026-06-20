from indw.schedule.core import merge_with_quality

__all__ = ['merge_with_quality', 'PipelineRunner', 'PipelineStats']

_LAZY = {
    'PipelineRunner': ('indw.schedule.stages.runner', 'PipelineRunner'),
    'PipelineStats': ('indw.schedule.stages.runner', 'PipelineStats'),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
