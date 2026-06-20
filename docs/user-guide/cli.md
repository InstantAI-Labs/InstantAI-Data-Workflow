# CLI

The INDW CLI is the primary operator interface.

## Commands

```bash
indw doctor
indw merge RAW OUT [options]
indw validate [pytest args...]
indw test [--profile PROFILE] [pytest args...]
indw audit [--kind KIND] [--work-dir DIR] [--workers N]
indw benchmark [--workers "1 2 4"]
```

## merge

Run the quality merge pipeline on a raw corpus directory.

```bash
indw merge ./raw ./out/filtered.jsonl \
  --work-dir ./work \
  --workers 4 \
  --chunk-size 500 \
  --fresh \
  --backend multiprocess
```

## validate

Run parity and acceptance validation (hash match across workers and backends).

```bash
indw validate
```

Equivalent to `indw test --profile parity`.

## test

Run the test suite with predefined profiles:

| Profile | Scope |
|---------|-------|
| `unit` | Fast tests, parallel |
| `critical` | Production-critical subsystem tests |
| `parity` | Hash and worker parity |
| `integration` | Integration and slow tests |
| `smoke` | End-to-end smoke |

```bash
indw test --profile unit
indw test --profile parity
```

## audit

Run operational audit reports:

| `--kind` | Report |
|----------|--------|
| `pipeline` | Architecture and cost audit |
| `dask` | Dask backend integration |
| `production` | Production scale validation |
| `library` | Library adoption analysis |
| `stage0` | Stage0 verification corpus |

```bash
indw audit --kind production
indw audit --kind stage0 --workers 2
```

## benchmark

Production scale benchmark across worker counts.

## doctor

Check Python version, resolved execution backend, and core dependencies.

See [commands reference](../reference/commands.md) for full flags.
