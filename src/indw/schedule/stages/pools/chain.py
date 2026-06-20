from __future__ import annotations

import time
from typing import Any, Callable

from indw.schedule.dispatch.alloc import (
    STAGE_FAST_FILTER,
    STAGE_FAST_PREPROCESS,
    STAGE_HIGH_QUALITY,
    STAGE_INTEL_PREVIEW,
    STAGE_INTERMEDIATE,
)


def _cost_row(stage: str, *, entered: int, rejected: int, survived: int, wall_sec: float) -> dict[str, Any]:
    return {
        'stage': stage,
        'entered': entered,
        'rejected': rejected,
        'survived': survived,
        'wall_sec': wall_sec,
    }


def _run_stage(
    fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    batch: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = fn(batch)
    wall = time.perf_counter() - t0
    entered = len(batch)
    terminal = list(result.get('terminal') or [])
    survivors = list(result.get('survivors') or [])
    items = list(result.get('items') or [])
    if terminal:
        rejected = len(terminal)
        survived = len(survivors)
    elif items:
        rejected = sum(1 for it in items if (it.get('cleaning_rejects') or []))
        survived = len(items) - rejected
    else:
        survived = len(survivors)
        rejected = max(entered - survived, 0)
    rows = list(result.get('_cost_rows') or [])
    rows.append(_cost_row(stage, entered=entered, rejected=rejected, survived=survived, wall_sec=wall))
    result['_cost_rows'] = rows
    return result


def process_fast_chain_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    from indw.schedule.stages.pools.preprocess import process_preprocess_batch
    from indw.schedule.stages.pools.filter import process_filter_batch
    from indw.schedule.stages.pools.stage0 import process_stage0_batch

    work_dir = None
    if batch:
        work_dir = batch[0].get('_work_dir')
    for row in batch:
        if work_dir and not row.get('_work_dir'):
            row['_work_dir'] = work_dir

    cost_rows: list[dict[str, Any]] = []
    pre = _run_stage(process_preprocess_batch, batch, STAGE_FAST_PREPROCESS)
    cost_rows.extend(pre.get('_cost_rows') or [])
    terminal = list(pre.get('terminal') or [])
    survivors = list(pre.get('survivors') or [])
    if not survivors:
        return {'terminal': terminal, 'survivors': [], '_cost_rows': cost_rows}

    filt = _run_stage(process_filter_batch, survivors, STAGE_FAST_FILTER)
    cost_rows.extend(filt.get('_cost_rows') or [])
    terminal.extend(filt.get('terminal') or [])
    survivors = list(filt.get('survivors') or [])
    if not survivors:
        return {'terminal': terminal, 'survivors': [], '_cost_rows': cost_rows}

    s0 = _run_stage(process_stage0_batch, survivors, 's2_tier1_gates')
    cost_rows.extend(s0.get('_cost_rows') or [])
    terminal.extend(s0.get('terminal') or [])
    survivors = list(s0.get('survivors') or [])
    return {'terminal': terminal, 'survivors': survivors, '_cost_rows': cost_rows}


def process_heavy_chain_batch(survivors: list[dict[str, Any]]) -> dict[str, Any]:
    from indw.schedule.intel.pools.acim import process_acim_batch
    from indw.schedule.intel.pools.pci import process_pci_batch
    from indw.schedule.stages.pools.clean import process_clean_batch

    if not survivors:
        return {'items': [], 'cleaning_stats': None, '_cost_rows': []}

    cost_rows: list[dict[str, Any]] = []
    pci = _run_stage(process_pci_batch, survivors, STAGE_INTERMEDIATE)
    cost_rows.extend(pci.get('_cost_rows') or [])
    acim_in = list(pci.get('survivors') or [])
    if not acim_in:
        return {'items': [], 'cleaning_stats': None, '_cost_rows': cost_rows}

    acim = _run_stage(process_acim_batch, acim_in, STAGE_INTEL_PREVIEW)
    cost_rows.extend(acim.get('_cost_rows') or [])
    clean_in = list(acim.get('survivors') or [])
    if not clean_in:
        return {'items': [], 'cleaning_stats': None, '_cost_rows': cost_rows}

    clean = _run_stage(process_clean_batch, clean_in, STAGE_HIGH_QUALITY)
    cost_rows.extend(clean.get('_cost_rows') or [])
    clean['items'] = clean.get('items') or []
    clean['_cost_rows'] = cost_rows
    return clean
