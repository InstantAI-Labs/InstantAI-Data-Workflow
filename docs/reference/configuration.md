# Configuration reference

## QualityPipelineConfig

Defined in `indw.filter.spec.quality`. Top-level sections:

| Section | Type | Purpose |
|---------|------|---------|
| `cleaning` | `CleaningConfig` | Semantic clean, artifact discovery |
| `balance` | `BalanceConfig` | Curriculum / domain balance |
| `dedup` | `DedupConfig` | Exact, fuzzy, semantic |
| `synthetic` | `SyntheticDefenseConfig` | Synthetic content defense |
| `gate` | gate settings | PCI thresholds |

Load from dict via `QualityPipelineConfig.from_dict(yaml_dict)`.

## Example YAML locations

```text
configs/
├── filtering/       Quality pipeline configs
├── sources/           Source mix definitions
├── pipeline/          Pipeline defaults
├── language/          Language identification
├── safety/            PII and toxicity
├── licensing/         License rules
└── observability/     Metrics defaults
```

## Environment variables

### Execution

| Variable | Values | Default |
|----------|--------|---------|
| `INSTANT_PIPELINE_BACKEND` | local, thread, multiprocess, dask | multiprocess |
| `INSTANT_DASK_SCHEDULER` | scheduler address | — |
| `DASK_SCHEDULER_ADDRESS` | scheduler address | — |

### Merge runtime

| Variable | Default | Purpose |
|----------|---------|---------|
| `INSTANT_MERGE_HW_PROBE` | `1` | Hardware probe for tuning |
| `INSTANT_MERGE_STAGE0_AUDIT` | `0` | Stage0 audit events |
| `INSTANT_SKIP_METRICS_PROBE` | `0` | Skip metrics probe |

### Intelligence stores

| Variable | Default | Purpose |
|----------|---------|---------|
| `INSTANT_ACIM_VERSION` | `acim-v1` | ACIM store schema |
| `INSTANT_LCI_VERSION` | `lci-v2` | LCI store schema |

Store schema versions are internal format identifiers, not package release versions.

## Internal format versions

Defined in `indw.config.defaults`:

- `MERGE_CHECKPOINT_FORMAT_VERSION`
- `PREPROCESSING_VERSION`
- `TOKEN_SHARD_VERSION`
- `LICENSE_PIPELINE_VERSION`

These govern artifact compatibility, not PyPI releases.

## Package version

Single source: `version` in `pyproject.toml`. Runtime access: `indw.__version__`.
