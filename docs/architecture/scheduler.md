# Scheduler

The scheduler coordinates document flow through admission tiers, heavy stage pools, and ordered apply.

## Admission

Stage0 (`indw.filter.stage0`) performs fast rejection before heavy work.

Tier gates (`indw.schedule.admission`) enforce PCI quality thresholds. Documents that fail early tiers never enter expensive pools.

## Stage pools

Heavy work runs in named pools under `indw.schedule.stages`:

- Preprocess and classification
- Semantic cleaning and artifact discovery (ACIM)
- Knowledge extraction and LCI routing
- Dedup (exact, fuzzy, semantic)
- Quality scoring and rewrite

Pool composition depends on `QualityPipelineConfig`.

## Apply ordering

Survivors are written in strict document sequence. Parallel workers may finish stages out of order; the apply coordinator buffers until the next expected sequence is ready.

This guarantees:

- Deterministic output ordering
- Stable `sorted_output_hash` across parallelism levels

Hash implementation: `indw.schedule.monitor.audit.sorted_output_hash`

## Checkpoints

Merge state persists via `indw.schedule.state.checkpoint`. Format version: `MERGE_CHECKPOINT_FORMAT_VERSION` in `indw.config.defaults`.

Resume is enabled when `fresh=False`.

## Intelligence stores

ACIM and LCI state use versioned stores (`ACIM_STORE_VERSION`, `LCI_STORE_VERSION` in defaults). These are store schema versions, not package release versions.
