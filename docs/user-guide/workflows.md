# Workflows

The INDW CLI orchestrates test profiles and audit scripts. Pipeline logic lives in the `indw` package.

## Validation

```bash
indw validate
```

Runs parity tests: worker count hash match, backend parity, tier admission consistency.

## Test profiles

| Profile | Markers | Parallel |
|---------|---------|----------|
| `unit` | `not integration and not slow` | yes |
| `critical` | `critical and not integration` | no |
| `parity` | `integration` | no |
| `integration` | `integration or slow` | no |
| `smoke` | `smoke` | no |

## Audit kinds

| Kind | Purpose |
|------|---------|
| `pipeline` | Architecture and cost audit |
| `dask` | Dask backend integration |
| `production` | Production scale validation |
| `library` | Library adoption analysis |
| `stage0` | Stage0 verification corpus |

## Benchmark

`indw benchmark` runs production scale validation across worker counts and verifies output hash stability.

See [testing](../developer/testing.md) for PR requirements.
