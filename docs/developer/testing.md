# Testing

## Validation

Primary entry point for acceptance checks:

```bash
indw validate
```

## Profiles

```bash
indw test --profile unit
indw test --profile critical
indw test --profile parity
```

| Profile | Purpose |
|---------|---------|
| `unit` | Fast regression, excludes integration/slow |
| `critical` | Production-critical subsystem tests |
| `parity` | Hash and worker/backend parity |
| `integration` | Full integration and slow tests |
| `smoke` | End-to-end smoke |

## Parity tests

Located in `tests/subsystems/`:

- `test_execution_backend.py` — backend factory and local/multiprocess/dask hash match
- `test_stage_pool_parity.py` — workers 1 vs 2 vs 4
- `test_parallel_merge_parity.py` — lenient, discovery, fuzzy configs
- `test_tier_admission_parity.py` — tier gate consistency

Dask full-merge parity is skipped on Windows in CI; run on Linux for full coverage.

## Direct pytest

```bash
pytest tests/subsystems -m critical -v
pytest tests/subsystems/test_merge.py -v
```

Fixtures: `tests/fixtures/pipeline_corpus.py`, `tests/fixtures/redteam/`

## Environment

`tests/conftest.py` sets:

- `INSTANT_MERGE_HW_PROBE=0`
- `INSTANT_PIPELINE_PUSHGATEWAY=off`

## Coverage

Configured in `pyproject.toml` under `[tool.coverage]`. Minimum threshold: 55%.
