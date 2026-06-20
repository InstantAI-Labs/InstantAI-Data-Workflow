from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from indw.schedule.state.checkpoint import MergeCheckpoint, count_jsonl_lines

class MergeAccountingError(RuntimeError):
    pass

@dataclass(frozen=True)
class MergeAccountingReport:
    file_lines: int
    checkpoint_kept: int
    checkpoint_scanned: int
    checkpoint_rejected: int
    exact_index_size: int | None
    ok: bool
    violations: tuple[str, ...]

def validate_merge_accounting(
    checkpoint: MergeCheckpoint,
    out_path: Path,
    *,
    exact: Any = None,
    strict: bool = False,
) -> MergeAccountingReport:
    totals = checkpoint.totals()
    file_lines = count_jsonl_lines(out_path)
    cp_kept = int(totals.get('kept', 0))
    cp_scanned = int(totals.get('scanned', 0))
    cp_rejected = int(totals.get('rejected', 0))
    exact_size = len(exact) if exact is not None else None
    violations: list[str] = []

    for name, row in checkpoint.sources.items():
        if row.scanned != row.kept + row.rejected:
            violations.append(
                f'source {name!r}: scanned={row.scanned} != kept={row.kept} + rejected={row.rejected}'
            )

    if cp_scanned != cp_kept + cp_rejected:
        violations.append(
            f'totals: scanned={cp_scanned} != kept={cp_kept} + rejected={cp_rejected}'
        )
    if file_lines != cp_kept:
        violations.append(
            f'filtered.jsonl lines={file_lines} != checkpoint kept={cp_kept}'
        )
    if (
        exact is not None
        and cp_kept > 0
        and exact_size is not None
        and int(getattr(exact, 'kept', 0) or 0) > 0
        and exact_size < cp_kept
    ):
        violations.append(
            f'exact index size={exact_size} < checkpoint kept={cp_kept}'
        )

    report = MergeAccountingReport(
        file_lines=file_lines,
        checkpoint_kept=cp_kept,
        checkpoint_scanned=cp_scanned,
        checkpoint_rejected=cp_rejected,
        exact_index_size=exact_size,
        ok=not violations,
        violations=tuple(violations),
    )
    if strict and violations:
        raise MergeAccountingError('; '.join(violations))
    return report

def assert_merge_output_synced(
    checkpoint: MergeCheckpoint,
    out_path: Path,
    *,
    exact: Any = None,
    context: str = '',
) -> None:
    report = validate_merge_accounting(checkpoint, out_path, exact=exact, strict=False)
    if report.ok:
        return
    prefix = f'{context}: ' if context else ''
    raise MergeAccountingError(prefix + '; '.join(report.violations))

def merge_progress_reject_reasons(
    work_dir: Path,
    gate: Any,
) -> dict[str, int]:
    from indw.schedule.state.checkpoint import load_run_progress

    merged: dict[str, int] = {}
    prior = load_run_progress(work_dir)
    prior_reasons = prior.get('reject_reasons') if isinstance(prior, dict) else None
    if isinstance(prior_reasons, dict):
        for key, value in prior_reasons.items():
            try:
                merged[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    for key, value in dict(gate.stats.reject_reasons).items():
        merged[key] = merged.get(key, 0) + int(value)
    return merged
