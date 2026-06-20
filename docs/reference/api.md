# API reference

Public exports from the `indw` package:

## Corpus and ingest

| Name | Description |
|------|-------------|
| `CorpusRegistry` | Corpus manifest registry |
| `FastDatasetPipeline` | Dataset ingestion pipeline |
| `DatasetDownloader` | Remote dataset download |
| `setup_dataset_logging` | Configure dataset logging |

## Quality and merge

| Name | Description |
|------|-------------|
| `QualityPipelineConfig` | Primary pipeline configuration |
| `QualityGate` | Quality gate evaluation |
| `merge_with_quality` | Main merge entry point |

## Language

| Name | Description |
|------|-------------|
| `ScriptProfile` | Script detection profile |
| `analyze_script_profile` | Analyze text script composition |
| `MultilingualPolicyConfig` | Multilingual policy settings |

## Mixture planning

| Name | Description |
|------|-------------|
| `CorpusMixturePlan` | Mixture plan dataclass |
| `MixtureOrchestrationConfig` | Mixture orchestration config |
| `adapt_mixture_from_telemetry` | Adapt plan from telemetry |
| `build_corpus_mixture_plan` | Build mixture plan |

## Export

| Name | Description |
|------|-------------|
| `export_token_bins_fast` | Fast token bin export |
| `build_pretrain_dataloader` | Pretrain dataloader builder |
| `build_val_dataloader` | Validation dataloader builder |

## Version

```python
import indw
indw.__version__
```

## Submodule imports

```python
from indw.schedule.backends.factory import resolve_execution_backend
from indw.schedule.monitor.audit import sorted_output_hash
from indw.filter.stage0.engine import run_stage0_content_filters
```

Prefer top-level exports for stable integration.
