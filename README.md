# INDW

<p align="left">
  <img src="docs/images/logo.svg" alt="INDW" width="48" />
</p>

**INDW** (Instant Data Workflow) is an open-source framework for production corpus pipelines. It ingests raw document collections, applies multi-tier quality filtering and semantic cleaning, deduplicates content, and produces deterministic merge output suitable for large-scale model training and evaluation.

INDW is designed for operators who need predictable behavior at scale: the same configuration and input corpus yield the same acceptance decisions and output hash whether you run on a laptop, a multi-core server, or a Dask cluster.

## Features

- **Stage0 fast filtering** — reject junk, boilerplate, and low-value documents before expensive processing
- **PCI admission gates** — tiered quality control with configurable thresholds
- **ACIM artifact discovery** — identify and trim structural noise while preserving content
- **LCI routing** — intelligence-guided document routing for heavy stages
- **Knowledge extraction** — structure recovery and section-aware processing
- **Semantic cleaning** — HTML normalization, boilerplate removal, OCR-aware repair
- **Deduplication** — exact, fuzzy (MinHash), and semantic (embedding) modes
- **Deterministic merge** — strict sequence-ordered apply with hash-verifiable JSONL output
- **Execution backends** — `local`, `thread`, `multiprocess`, `dask` without changing pipeline logic

## Architecture

INDW runs a single canonical execution graph. Documents flow through ingest and Stage0, pass admission tiers, enter parallel heavy stage pools (cleaning, extraction, dedup, quality scoring), and are written by an ordered apply coordinator.

```
Ingest → Stage0 → Admission → Heavy pools → Apply → Output
```

Heavy stage logic is backend-agnostic. Worker pools, thread pools, or Dask workers execute the same graph; only the transport layer changes.

![INDW architecture](docs/images/architecture.svg)

See [Architecture overview](docs/architecture/overview.mdx) for capability details.

## Installation

```bash
pip install -e ".[dev,language]"
```

With distributed execution support:

```bash
pip install -e ".[dev,language,distributed]"
```

Or use Make:

```bash
make install-dev
indw doctor
```

Full instructions: [Installation](docs/getting-started/installation.mdx).

## Quick start

Prepare raw JSONL under `raw/<source>/data.jsonl`, then run:

```bash
indw merge ./raw ./out/filtered.jsonl --work-dir ./work --workers 2 --fresh
indw validate
```

Python API:

```python
from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule import merge_with_quality

merge_with_quality(
    "./raw", "./out/filtered.jsonl",
    quality_config=QualityPipelineConfig(),
    work_dir="./work",
    workers=2,
    fresh=True,
)
```

See [Quickstart](docs/getting-started/quickstart.mdx) and [First pipeline](docs/getting-started/first-pipeline.mdx).

## CLI

| Command | Purpose |
|---------|---------|
| `indw doctor` | Verify install and dependencies |
| `indw merge` | Run quality merge on a raw corpus |
| `indw validate` | Parity and acceptance validation |
| `indw test` | Run test profiles (`unit`, `parity`, `critical`) |
| `indw audit` | Operational audit reports |
| `indw benchmark` | Production scale benchmark |

```bash
indw merge ./raw ./out/filtered.jsonl --work-dir ./work --workers 4 --backend multiprocess
indw audit --kind production
```

Reference: [CLI commands](docs/cli/commands.mdx).

## Configuration

INDW accepts YAML quality profiles, environment variables, and Python dataclasses. Example:

```python
import yaml
from pathlib import Path
from indw.filter.spec.quality import QualityPipelineConfig

raw = yaml.safe_load(Path("configs/filtering/quality_fast_first.yaml").read_text())
cfg = QualityPipelineConfig.from_dict(raw)
```

Key environment variables:

| Variable | Purpose |
|----------|---------|
| `INSTANT_PIPELINE_BACKEND` | `local`, `thread`, `multiprocess`, `dask` |
| `INSTANT_DASK_SCHEDULER` | Dask scheduler address |
| `INSTANT_MERGE_HW_PROBE` | Hardware tuning probe |

Precedence: CLI flags override environment; environment overrides YAML defaults.

Details: [Configuration](docs/getting-started/configuration.mdx).

## Pipeline

The merge pipeline is configured through `QualityPipelineConfig`. Stages include preprocessing, Stage0 content filters, admission tiers, semantic cleaning, knowledge extraction, dedup, and quality gates. Survivors are emitted in document sequence order.

Work directories retain checkpoints for resume, resolved configuration snapshots, and optional audit JSON.

Guide: [Filtering](docs/guides/filtering.mdx) · [Scheduler](docs/guides/scheduler.mdx)

## Scaling

Scale horizontally by increasing `--workers` with the `multiprocess` backend, or attach a Dask cluster:

```bash
export INSTANT_PIPELINE_BACKEND=dask
export DASK_SCHEDULER_ADDRESS=tcp://scheduler:8786
indw merge ./raw ./out/filtered.jsonl --workers 8
```

Output hash remains stable across worker counts and backends when configuration is unchanged.

Guides: [Distributed](docs/guides/distributed.mdx) · [Dask](docs/guides/dask.mdx) · [Scaling](docs/architecture/scaling.mdx)

## Documentation

Full documentation: **[docs/README.mdx](docs/README.mdx)**

| Section | Topics |
|---------|--------|
| [Getting started](docs/getting-started/installation.mdx) | Install, quickstart, first pipeline |
| [Guides](docs/guides/ingestion.mdx) | Ingestion through distributed execution |
| [CLI](docs/cli/commands.mdx) | Commands and workflows |
| [Configuration](docs/configuration/pipeline.mdx) | Pipeline, datasets, outputs, env |
| [Architecture](docs/architecture/overview.mdx) | Execution graph, Stage0, scheduler |
| [Developer](docs/developer/contributing.mdx) | Contributing, testing, extensions |
| [Reference](docs/reference/api.mdx) | API, CLI, configuration keys |

## Examples

| Example | Description |
|---------|-------------|
| `examples/merge_local.py` | Local single-worker merge |
| `examples/merge_custom_config.py` | YAML quality profile |
| `examples/merge_custom_output.py` | Custom output and work paths |
| `examples/merge_dask.py` | Dask backend |

## Contributing

Contributions are welcome. Run `indw validate` before submitting pipeline changes.

See [Contributing](docs/developer/contributing.mdx).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Determinism and validation

INDW treats reproducibility as a first-class requirement. The apply coordinator writes survivors in strict document sequence. The canonical output hash (`sorted_output_hash`) is stable across worker counts and execution backends when the quality configuration and input corpus are unchanged.

Run acceptance checks after pipeline changes:

```bash
indw validate
```

This executes the parity suite: local vs multiprocess backend match, workers=1 vs workers=N hash match, and tier admission consistency.

## Optional capabilities

Install extras only when needed:

| Extra | Capability |
|-------|------------|
| `language` | Language detection (langid) |
| `dedup` | Fuzzy MinHash dedup |
| `ann` | ANN index for semantic dedup |
| `embedding` | Sentence-transformer providers |
| `distributed` | Dask cluster backend |
| `monitor` | Prometheus and OpenTelemetry hooks |

```bash
pip install -e ".[all]"
```

## Observability

Merge runs emit stage metrics, scheduler reports, and optional Stage0 audit events. Use `indw audit --kind pipeline` with a populated work directory to inspect architecture and throughput estimates. Production certification uses `indw audit --kind production` and `indw benchmark`.

Work directories are the operational source of truth for a run: resolved config, checkpoints, filtered output, and JSON audit artifacts.

## Support and community

Report issues through the project issue tracker. Security disclosures follow [SECURITY.md](SECURITY.md). Documentation corrections and guide contributions are welcome via pull request; see [Contributing](docs/developer/contributing.mdx).
