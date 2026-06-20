# Configuration

INDW is configured through YAML quality files, environment variables, and Python dataclasses.

## Quality pipeline

Primary type: `indw.filter.spec.quality.QualityPipelineConfig`

Example configs ship under `configs/filtering/`:

| File | Typical use |
|------|-------------|
| `quality_fast_first.yaml` | Fast Stage0-first filtering |
| `quality_smoke_5mb.yaml` | Small smoke runs |
| `quality_foundation_en.yaml` | English foundation corpus |

Load from YAML:

```python
import yaml
from pathlib import Path
from indw.filter.spec.quality import QualityPipelineConfig

path = Path("configs/filtering/quality_fast_first.yaml")
cfg = QualityPipelineConfig.from_dict(yaml.safe_load(path.read_text()))
```

Source mix definitions live in `configs/sources/`.

## Execution backend

Set via `INSTANT_PIPELINE_BACKEND`:

| Value | Behavior |
|-------|----------|
| `multiprocess` | Default; process pool |
| `local` | Single process |
| `thread` | Thread pool |
| `dask` | Distributed cluster |

For Dask, also set `INSTANT_DASK_SCHEDULER` or `DASK_SCHEDULER_ADDRESS`.

CLI override: `indw merge ... --backend dask`

## Common runtime variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `INSTANT_MERGE_HW_PROBE` | `1` | Hardware tuning probe |
| `INSTANT_MERGE_STAGE0_AUDIT` | `0` | Stage0 event audit |
| `INSTANT_SKIP_METRICS_PROBE` | `0` | Skip metrics collector |

Full reference: [configuration reference](../reference/configuration.md).
