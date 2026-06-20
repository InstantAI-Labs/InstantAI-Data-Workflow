# Pipeline

The merge pipeline transforms raw JSONL sources into filtered, deduplicated output through a fixed stage graph.

## Stages

| Phase | Owner | Role |
|-------|-------|------|
| Ingest read | `indw.schedule.read` | Source enumeration, JSONL parsing |
| Stage0 | `indw.filter.stage0` | Fast content filters, early rejection |
| Admission | `indw.schedule.admission` | Tier gates (PCI) |
| Heavy pools | `indw.schedule.stages` | Cleaning, extraction, dedup |
| ACIM / LCI | `indw.clean.artifact`, `indw.schedule.intel` | Artifact discovery, intelligence routing |
| Apply | `indw.schedule.apply` | Ordered survivor write |

Stage logic is identical regardless of execution backend.

## Work directory

A typical `--work-dir` contains:

- `filtered.jsonl` — final output (or path passed to `merge`)
- Checkpoint files for resume
- Stage metrics and optional audit JSON
- Resolved quality config snapshot

Use `--fresh` to discard checkpoint state.

## Determinism

- Documents are applied in sequence order.
- Output hash (`sorted_output_hash`) is stable across worker counts and backends when config is unchanged.
- Acceptance decisions depend on document content and config, not parallelism.

## Resume

Omit `--fresh` to resume from checkpoint. Checkpoint format version is defined in `indw.config.defaults.MERGE_CHECKPOINT_FORMAT_VERSION`.

See [scheduler](../architecture/scheduler.md) for ordering details and [dedup](../architecture/dedup.md) for dedup semantics.
