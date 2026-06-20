# Quickstart

## Prepare input

Place source JSONL files under a raw directory, one subdirectory per source:

```text
raw/
├── source_a/data.jsonl
└── source_b/data.jsonl
```

Each line is a JSON object with at least a `text` field.

## Run merge

```bash
indw merge ./raw ./work/filtered.jsonl --work-dir ./work --workers 2 --fresh
```

| Flag | Purpose |
|------|---------|
| `--work-dir` | Checkpoint, metrics, and intermediate artifacts |
| `--workers` | Parallel worker count |
| `--fresh` | Ignore prior checkpoint state |
| `--backend` | Override `INSTANT_PIPELINE_BACKEND` |

## Python API

```python
from pathlib import Path
from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule import merge_with_quality

merge_with_quality(
    Path("./raw"),
    Path("./work/filtered.jsonl"),
    quality_config=QualityPipelineConfig(),
    work_dir=Path("./work"),
    fresh=True,
    workers=2,
)
```

## Verify output

Work directory contains `filtered.jsonl`, checkpoint files, and optional audit reports. Output hash is stable across worker counts when configuration is unchanged.

```bash
indw test --profile parity
```

## Next steps

- [Configuration](configuration.md) — quality YAML and environment variables
- [Pipeline](../user-guide/pipeline.md) — stage breakdown
- [CLI](../user-guide/cli.md) — all commands
