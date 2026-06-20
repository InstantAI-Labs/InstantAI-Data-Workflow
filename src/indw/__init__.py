from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('indw')
except PackageNotFoundError:
    __version__ = '0.0.0'

from indw.store.corpus.registry import CorpusRegistry
from indw.ingest.download import DatasetDownloader
from indw.ingest.run import FastDatasetPipeline
from indw.ingest.log import setup_dataset_logging
from indw.filter.language.script import ScriptProfile, analyze_script_profile
from indw.filter.language.script_policy import MultilingualPolicyConfig
from indw.schedule.mix.config import MixtureOrchestrationConfig
from indw.schedule.mix.plan import CorpusMixturePlan
from indw.schedule.mix.telemetry import adapt_mixture_from_telemetry
from indw.schedule.mix.mixture_planner import build_corpus_mixture_plan
from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule import merge_with_quality
from indw.filter.gate.quality import QualityGate
from indw.store.export.fast_export import export_token_bins_fast
from indw.store.export.memmap_stream import build_pretrain_dataloader, build_val_dataloader

__all__ = [
    '__version__',
    'CorpusRegistry',
    'ScriptProfile',
    'analyze_script_profile',
    'MultilingualPolicyConfig',
    'CorpusMixturePlan',
    'MixtureOrchestrationConfig',
    'adapt_mixture_from_telemetry',
    'build_corpus_mixture_plan',
    'QualityGate',
    'QualityPipelineConfig',
    'merge_with_quality',
    'FastDatasetPipeline',
    'DatasetDownloader',
    'setup_dataset_logging',
    'build_pretrain_dataloader',
    'build_val_dataloader',
    'export_token_bins_fast',
]
