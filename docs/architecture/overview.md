# Architecture overview

INDW is organized by capability. Each Python subpackage under `indw` owns one area of pipeline intelligence.

## Capabilities

| Package | Responsibility |
|---------|----------------|
| `indw.ingest` | Download, HF datasets, format normalization |
| `indw.filter` | Stage0, PCI gates, quality spec, PII, toxicity, licensing |
| `indw.clean` | Semantic cleaning, ACIM artifact discovery, HTML processing |
| `indw.extract` | Knowledge extraction, structure, section recovery |
| `indw.dedup` | Exact, fuzzy, and semantic dedup |
| `indw.schedule` | Merge graph, admission, apply ordering, backends |
| `indw.store` | Corpus registry, IO, export |
| `indw.config` | Config resolution and defaults |
| `indw.tools` | Metrics and audit report builders |

One owner per capability. One implementation per execution path.

## Dependency direction

```text
indw.schedule → indw.filter, indw.clean, indw.extract, indw.dedup, indw.store
indw.filter → indw.clean
indw.extract → indw.clean
```

The scheduler integrates all pipeline stages. Capability packages do not import from the CLI layer.

## Public surface

- **Python API:** `import indw` and `from indw.<capability> import ...`
- **CLI:** `indw` command (`merge`, `test`, `audit`, `benchmark`, `doctor`)

Further reading:

- [Execution](execution.md)
- [Scheduler](scheduler.md)
- [Dedup](dedup.md)
