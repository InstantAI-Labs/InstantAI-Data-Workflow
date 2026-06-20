# Commands reference

## indw CLI

| Command | Description |
|---------|-------------|
| `indw doctor` | Check install and dependencies |
| `indw merge RAW OUT` | Run quality merge |
| `indw validate` | Parity and acceptance validation |
| `indw test [--profile PROFILE]` | Run pytest profiles |
| `indw audit [--kind KIND]` | Run audit report |
| `indw benchmark [--workers "..."]` | Production scale benchmark |

### merge flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--work-dir` | path | — | Work directory for checkpoints and artifacts |
| `--workers` | int | 1 | Worker count |
| `--chunk-size` | int | 500 | Read chunk size |
| `--fresh` | flag | false | Ignore checkpoint, start clean |
| `--backend` | choice | env | local, thread, multiprocess, dask |

### test profiles

`unit`, `critical`, `parity`, `integration`, `smoke`

### audit kinds

| Kind | Purpose |
|------|---------|
| `pipeline` | Architecture and cost audit |
| `dask` | Dask backend integration |
| `production` | Production scale validation |
| `library` | Library adoption analysis |
| `stage0` | Stage0 verification corpus |

### audit flags

| Flag | Applies to | Description |
|------|------------|-------------|
| `--work-dir` | pipeline | Merge work dir with runtime metrics |
| `--workers` | stage0 | Worker count for verification |

## Script entry points

Audit and benchmark scripts are also available under `scripts/` for automation pipelines. Prefer the `indw` CLI for routine use.
