_LAZY = {
    'DatasetDownloader': ('indw.ingest.download', 'DatasetDownloader'),
    'FastDatasetPipeline': ('indw.ingest.run', 'FastDatasetPipeline'),
    'setup_dataset_logging': ('indw.ingest.log', 'setup_dataset_logging'),
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
