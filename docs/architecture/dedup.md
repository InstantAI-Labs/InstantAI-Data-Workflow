# Dedup

Dedup runs during the merge heavy stage pool. Three modes are configurable via `QualityPipelineConfig.dedup`.

## Exact dedup

Content-normalized hash lookup. Lowest cost; catches byte-identical documents after normalization.

Owner: `indw.dedup` (exact path)

## Fuzzy dedup

MinHash LSH over shingle sets. Requires optional `dedup` extra (`datasketch`).

Owner: `indw.dedup.fuzzy`, `indw.dedup.backends.fuzzy`

## Semantic dedup

Embedding similarity with optional ANN index. Requires `embedding` and optionally `ann` extras.

Owner: `indw.dedup.embed`

## Semantics

- Dedup decisions are content-based; worker count does not change which documents are removed.
- Representative selection is deterministic for a fixed input order and config.
- Fuzzy and semantic modes can be disabled independently for faster runs.

## Configuration

```python
from indw.filter.spec.quality import DedupConfig

DedupConfig(exact=True, fuzzy=False, semantic=False)
```

YAML equivalents are under the `dedup` key in quality config files.

See [configuration reference](../reference/configuration.md) for field details.
