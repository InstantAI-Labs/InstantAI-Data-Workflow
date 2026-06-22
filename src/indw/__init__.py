from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('indw')
except PackageNotFoundError:
    __version__ = '0.0.0'

_LAZY_EXPORTS = {
    'CorpusRegistry': ('indw.store.corpus.registry', 'CorpusRegistry'),
    'DatasetDownloader': ('indw.ingest.download', 'DatasetDownloader'),
    'FastDatasetPipeline': ('indw.ingest.run', 'FastDatasetPipeline'),
    'setup_dataset_logging': ('indw.ingest.log', 'setup_dataset_logging'),
    'ScriptProfile': ('indw.filter.language.script', 'ScriptProfile'),
    'analyze_script_profile': ('indw.filter.language.script', 'analyze_script_profile'),
    'MultilingualPolicyConfig': ('indw.filter.language.script_policy', 'MultilingualPolicyConfig'),
    'MixtureOrchestrationConfig': ('indw.schedule.mix.config', 'MixtureOrchestrationConfig'),
    'CorpusMixturePlan': ('indw.schedule.mix.plan', 'CorpusMixturePlan'),
    'adapt_mixture_from_telemetry': ('indw.schedule.mix.telemetry', 'adapt_mixture_from_telemetry'),
    'build_corpus_mixture_plan': ('indw.schedule.mix.mixture_planner', 'build_corpus_mixture_plan'),
    'QualityPipelineConfig': ('indw.filter.spec.quality', 'QualityPipelineConfig'),
    'merge_with_quality': ('indw.schedule.core', 'merge_with_quality'),
    'QualityGate': ('indw.filter.gate.quality', 'QualityGate'),
    'export_token_bins_fast': ('indw.store.export.fast_export', 'export_token_bins_fast'),
    'build_pretrain_dataloader': ('indw.store.export.memmap_stream', 'build_pretrain_dataloader'),
    'build_val_dataloader': ('indw.store.export.memmap_stream', 'build_val_dataloader'),
}

__all__ = [
    '__version__',
    *sorted(_LAZY_EXPORTS),
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        mod_path, attr = _LAZY_EXPORTS[name]
        val = getattr(importlib.import_module(mod_path), attr)
        globals()[name] = val
        return val
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
