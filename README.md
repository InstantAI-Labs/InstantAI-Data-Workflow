# INDW

INDW (Instant Data Workflow) is an open-source corpus processing framework for ingestion, quality filtering, semantic cleaning, deduplication, and deterministic merge scheduling.

## Features

- Stage0 fast filtering, PCI gates, ACIM artifact discovery, LCI routing
- Knowledge extraction and semantic cleaning
- Exact, fuzzy, and semantic dedup
- Deterministic merge ordering with hash-verifiable output
- Execution backends: local, thread, multiprocess, Dask

## Installation

```bash
pip install -e ".[dev,language]"
```

Or:

```bash
make install-dev
```

See [installation](docs/data-workflow/getting-started/installation.md) for extras and Docker.

## Quick start

```bash
indw doctor
indw merge ./raw ./out/filtered.jsonl --work-dir ./work --workers 2 --fresh
```

```python
from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule import merge_with_quality

merge_with_quality("./raw", "./out/filtered.jsonl", quality_config=QualityPipelineConfig(), workers=2)
```

## CLI

```bash
indw doctor
indw merge ./raw ./out/filtered.jsonl --work-dir ./work --workers 2 --fresh
indw test --profile parity
indw audit --kind production
indw validate --profile parity
```

See [CLI reference](docs/data-workflow/cli/commands.md).

## Documentation

Full documentation: **[docs/data-workflow/overview.mdx](docs/data-workflow/overview.mdx)**

| Section | Topics |
|---------|--------|
| [Getting started](docs/data-workflow/getting-started/installation.md) | Install, quickstart, configuration |
| [User guide](docs/data-workflow/cli/commands.md) | CLI, pipeline, workflows |
| [Architecture](docs/data-workflow/architecture/overview.md) | Capabilities, execution, scheduler, dedup |
| [Developer](docs/data-workflow/developer/contributing.md) | Contributing, testing, backends |
| [Reference](docs/data-workflow/reference/cli-reference.md) | Commands, config keys, public API |

## Contributing

See [contributing guide](docs/data-workflow/developer/contributing.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
