# Backends

Execution backends swap process/thread/cluster mechanics without changing pipeline logic.

## Selection

```bash
export INSTANT_PIPELINE_BACKEND=multiprocess   # default
indw merge ./raw ./out.jsonl --backend local
```

| Backend | When to use |
|---------|-------------|
| `local` | Debugging, reproducible single-process traces |
| `thread` | I/O-bound stages, shared memory |
| `multiprocess` | CPU-bound production runs on one machine |
| `dask` | Multi-node or shared cluster execution |

## Dask

Install distributed extra:

```bash
pip install -e ".[distributed]"
```

Point to scheduler:

```bash
export INSTANT_PIPELINE_BACKEND=dask
export DASK_SCHEDULER_ADDRESS=tcp://scheduler:8786
```

Docker compose example: `docker/docker-compose.yml` with `--profile dask`.

Verify integration:

```bash
indw audit --kind dask
indw test --profile parity   # includes dask tests when installed
```

## Factory API

```python
from indw.schedule.backends.factory import resolve_execution_backend

backend = resolve_execution_backend()  # reads INSTANT_PIPELINE_BACKEND
with backend.open(work_spec, fast_workers=4, heavy_workers=2) as session:
    ...
```

Backend names and aliases: `indw.schedule.backends.config.normalize_backend_name`

## Parity invariant

Same quality config and raw input must produce identical output hash across backends and worker counts. Do not merge backend changes that break this invariant.
