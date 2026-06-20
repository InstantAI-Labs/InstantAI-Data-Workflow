# Execution

Merge execution uses a single canonical graph runner. There is no alternate serial or v1 pipelined path.

## Graph runner

Entry: `indw.schedule.graph.runner` → `run_graph_merge`

The runner:

1. Builds stage pools from the quality config
2. Resolves worker allocation
3. Dispatches work through the selected backend
4. Collects survivors in sequence order for apply

## Backends

Backend selection is configuration-only via `INSTANT_PIPELINE_BACKEND`. Stage code is backend-agnostic.

| Backend | Module | Use |
|---------|--------|-----|
| `local` | `schedule/backends/local.py` | Debug, single process |
| `thread` | `schedule/backends/thread.py` | Shared-memory threading |
| `multiprocess` | `schedule/backends/multiprocess.py` | Default production |
| `dask` | `schedule/backends/dask.py` | Distributed cluster |

Factory: `indw.schedule.backends.factory.resolve_execution_backend`

Aliases: `sync` → `local`, `cluster` → `dask`

## Worker dispatch

Worker pools and task submission live in `indw.schedule.dispatch`. The graph runner owns scheduling; backends own process/thread/cluster lifecycle.

## Parity requirement

Output hash must match across:

- `workers=1` vs `workers=N`
- `local` vs `multiprocess`
- `multiprocess` vs `dask` (when Dask is available)

Verified by `indw test --profile parity`.

See [backends](../developer/backends.md) for environment setup.
