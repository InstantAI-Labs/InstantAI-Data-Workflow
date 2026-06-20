# Contributing

## Setup

```bash
make install-dev
indw doctor
```

## Pull requests

1. Scope changes to the relevant `indw` capability package
2. Run `indw validate` for pipeline or scheduler changes
3. Run `indw test --profile critical` for subsystem changes
4. Run `indw audit --kind production` for backend or scaling changes

## Architecture rules

- Pipeline intelligence belongs in `indw.*` packages
- One owner per capability; no duplicate execution paths
- Preserve deterministic ordering, acceptance decisions, and output hashes

## Code style

Match existing subsystem conventions. No per-file copyright headers.

## Versioning

Package version is defined in `pyproject.toml` only.

## Legal

See [LICENSE](../LICENSE) and [NOTICE](../NOTICE).

## Documentation

Update the relevant page under `docs/` when changing user-visible behavior.
