from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from indw.clean.gate.evaluate import evaluate_document_gate
from indw.schedule.dispatch.alloc import STAGE_ADMISSION
from indw.filter.stage0.audit import build_report, load_events
from indw.filter.stage0.engine import run_stage0_content_filters


def resolve_survivor_payload_text(payload: dict[str, Any]) -> str:
    text = str(payload.get('text') or '').strip()
    if text:
        return text
    store_key = str(payload.get('text_store_key') or '').strip()
    if not store_key:
        return ''
    from indw.schedule.state.survivor import resolve_survivor_text
    return resolve_survivor_text(payload).strip()


def validate_survivor_payload(payload: dict[str, Any]) -> None:
    trace = list(payload.get('stage_trace') or [])
    if STAGE_ADMISSION not in trace:
        raise ValueError(
            f"survivor seq={payload.get('seq')} missing {STAGE_ADMISSION} in stage_trace",
        )
    text = resolve_survivor_payload_text(payload)
    if not text:
        store_key = str(payload.get('text_store_key') or '').strip()
        detail = f" store_key={store_key!r}" if store_key else ''
        raise ValueError(f"survivor seq={payload.get('seq')} has empty text{detail}")


from indw.schedule.monitor.audit import load_work_json


def _wall_rank(stage_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    stages = stage_metrics.get('stages') or {}
    rows: list[dict[str, Any]] = []
    for name, row in stages.items():
        if not isinstance(row, dict):
            continue
        wall = float(row.get('wall_sec') or 0.0)
        cpu = float(row.get('cpu_sec') or wall)
        in_docs = int(row.get('in_docs') or 0)
        rows.append({
            'stage': name,
            'wall_sec': round(wall, 4),
            'cpu_sec': round(cpu, 4),
            'in_docs': in_docs,
            'docs_per_sec': round(in_docs / max(wall, 1e-9), 3),
            'reject_rate': row.get('reject_rate', 0.0),
        })
    rows.sort(key=lambda r: -r['wall_sec'])
    return rows


def _duplicate_execution(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_seq: dict[int, Counter[str]] = defaultdict(Counter)
    for ev in events:
        seq = ev.get('seq')
        if seq is None:
            continue
        et = str(ev.get('event', ''))
        if et:
            by_seq[int(seq)][et] += 1
    dup_rows = []
    for seq, hits in sorted(by_seq.items()):
        for stage, count in hits.items():
            if stage in ('stage0_fast', 'heavy_enter', 'heavy_exit', 'apply') and count > 1:
                dup_rows.append({'seq': seq, 'stage': stage, 'count': count})
    return {
        'duplicate_events': dup_rows,
        'duplicate_doc_count': len({r['seq'] for r in dup_rows}),
    }


def _late_heavy_reject_analysis(
    events: list[dict[str, Any]],
    *,
    text_by_seq: dict[int, str] | None = None,
) -> dict[str, Any]:
    by_seq: dict[int, dict[str, Any]] = {}
    for ev in events:
        seq = ev.get('seq')
        if seq is None:
            continue
        seq = int(seq)
        row = by_seq.setdefault(seq, {'seq': seq})
        et = str(ev.get('event', ''))
        if et == 'heavy_enter':
            row['heavy_enter'] = True
        elif et == 'heavy_exit':
            row['heavy_exit'] = True
            row['cleaning_rejects'] = int(ev.get('cleaning_rejects') or 0)
            row['chunk_count'] = int(ev.get('chunk_count') or 0)
        elif et == 'apply':
            row['apply_kept'] = bool(ev.get('kept'))
            row['apply_chunks'] = int(ev.get('chunk_count') or 0)

    late: list[dict[str, Any]] = []
    movable: list[dict[str, Any]] = []
    for seq, row in by_seq.items():
        if not row.get('heavy_enter'):
            continue
        rejected_at_heavy = (
            row.get('cleaning_rejects', 0) > 0
            or row.get('chunk_count', 0) == 0
            or row.get('apply_kept') is False
        )
        if not rejected_at_heavy:
            continue
        text = (text_by_seq or {}).get(seq, '')
        stage0_reason = None
        if text:
            stage0_reason = run_stage0_content_filters(
                text,
                meaningful_chars=len(text),
            )
        entry = {
            'seq': seq,
            'cleaning_rejects': row.get('cleaning_rejects', 0),
            'chunk_count': row.get('chunk_count', 0),
            'apply_kept': row.get('apply_kept'),
            'stage0_would_reject': stage0_reason,
        }
        late.append(entry)
        if stage0_reason:
            full = evaluate_document_gate(text)
            if not full.keep:
                movable.append({**entry, 'full_gate_reason': full.reason})
        elif text:
            full = evaluate_document_gate(text)
            if not full.keep:
                entry['full_gate_reject'] = full.reason
                entry['stage0_gap'] = True
                late[-1] = entry
    return {
        'late_heavy_rejects': late,
        'late_heavy_reject_count': len(late),
        'movable_to_stage0': movable,
        'movable_count': len(movable),
        'stage0_gaps': [r for r in late if r.get('stage0_gap')],
        'stage0_gap_count': sum(1 for r in late if r.get('stage0_gap')),
        'semantic_only_heavy_rejects': [
            r.get('full_gate_reject') for r in late if r.get('stage0_gap')
        ],
    }


def build_production_verification_report(
    work_dir: Path,
    *,
    parity: dict[str, Any] | None = None,
    text_by_seq: dict[int, str] | None = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    events = load_events(work_dir)
    stage0 = build_report(
        work_dir,
        scheduler=load_work_json(work_dir / 'pipeline_scheduler_report.json'),
        progress=load_work_json(work_dir / 'pipeline_progress.json'),
        stage_metrics=load_work_json(work_dir / 'stage_metrics.json'),
    )
    scheduler = load_work_json(work_dir / 'pipeline_scheduler_report.json')
    live = load_work_json(work_dir / 'pipeline_live_metrics.json')
    stage_metrics = load_work_json(work_dir / 'stage_metrics.json')
    progress = load_work_json(work_dir / 'pipeline_progress.json')

    wf = stage0.get('waterfall') or {}
    counters = stage0.get('counters') or {}
    input_docs = int(wf.get('input_docs') or counters.get('INPUT_DOCS') or 0)
    survivors = int(wf.get('stage0_survivors') or counters.get('STAGE0_SURVIVORS') or 0)
    heavy_enter = int(wf.get('heavy_entered') or 0)
    heavy_exit = int(wf.get('heavy_completed') or 0)
    final_kept = int(wf.get('final_kept') or 0)

    stage0_reject = int(counters.get('STAGE0_REJECT') or 0)
    prevented = stage0_reject + int(counters.get('STAGE0_ACCEPT') or 0)

    dup = _duplicate_execution(events)
    late = _late_heavy_reject_analysis(events, text_by_seq=text_by_seq)

    sched_live = live.get('scheduler') or {}
    cache_stats = live.get('cache') or stage_metrics.get('cache') or {}

    return {
        'work_dir': str(work_dir),
        'waterfall': {
            'input': input_docs,
            'stage0_rejects': stage0_reject,
            'stage0_terminal_accept': int(counters.get('STAGE0_ACCEPT') or 0),
            'stage0_survivors': survivors,
            'heavy_entered': heavy_enter,
            'heavy_completed': heavy_exit,
            'apply_seen': int(wf.get('apply_seen') or 0),
            'final_kept': final_kept,
            'final_rejected': int(wf.get('final_rejected') or 0),
        },
        'stage0_efficiency': {
            'reject_rate_pct': round(100.0 * stage0_reject / max(input_docs, 1), 2),
            'survivor_rate_pct': round(100.0 * survivors / max(input_docs, 1), 2),
            'heavy_prevention_pct': round(100.0 * prevented / max(input_docs, 1), 2),
            'heavy_admission_rate_pct': round(100.0 * heavy_enter / max(input_docs, 1), 2),
            'heavy_to_kept_yield_pct': round(100.0 * final_kept / max(heavy_enter, 1), 2),
            'deterministic_rejects': int(wf.get('stage0_deterministic_reject') or 0),
        },
        'reject_buckets': {
            k: v for k, v in counters.items() if k.startswith('REJECT_')
        },
        'bottleneck_ranking_by_wall': _wall_rank(stage_metrics),
        'scheduler': {
            'ordering_wait_ms': scheduler.get('ordering_wait_ms', 0),
            'peak_ordering_gap': scheduler.get('peak_ordering_gap', 0),
            'peak_apply_buffer': scheduler.get('peak_apply_buffer', 0),
            'peak_survivor_buffer': sched_live.get('peak_survivor_buffer', 0),
            'peak_heavy_pending': sched_live.get('peak_heavy_pending', 0),
            'reader_block_events': scheduler.get('reader_block_events', 0),
            'heavy_apply_backpressure_events': scheduler.get('heavy_apply_backpressure_events', 0),
            'head_blocked_dispatches': scheduler.get('head_blocked_dispatches', 0),
            'head_priority_dispatches': scheduler.get('head_priority_dispatches', 0),
            'dispatched_past_head': scheduler.get('dispatched_past_head', 0),
            'worker_util_pct': sched_live.get('worker_util_pct', 0),
        },
        'cache_efficiency': cache_stats,
        'duplicate_execution': dup,
        'late_heavy_analysis': late,
        'parity': parity or {},
        'stage0_protection': stage0.get('stage0_protection') or {},
        'wall_time_ms': stage0.get('wall_time_ms') or {},
        'throughput': {
            'docs_per_sec': (progress.get('extra') or {}).get('docs_per_sec')
            or progress.get('docs_per_sec'),
            'merge_wall_sec': stage_metrics.get('merge_wall_sec', 0),
        },
        'architectural_findings': _architectural_findings(
            dup=dup,
            late=late,
            scheduler=scheduler,
            survivors=survivors,
            heavy_enter=heavy_enter,
        ),
    }


def _architectural_findings(
    *,
    dup: dict[str, Any],
    late: dict[str, Any],
    scheduler: dict[str, Any],
    survivors: int,
    heavy_enter: int,
) -> list[str]:
    findings: list[str] = []
    if survivors != heavy_enter:
        findings.append(
            f'stage0_survivors ({survivors}) != heavy_entered ({heavy_enter}) — audit gap or in-flight docs',
        )
    if dup.get('duplicate_doc_count', 0):
        findings.append(
            f"duplicate_execution on {dup['duplicate_doc_count']} seq(s) — likely fast-pool fallback",
        )
    if late.get('movable_count', 0):
        findings.append(
            f"{late['movable_count']} doc(s) entered heavy but stage0 conservative gates would also reject",
        )
    if int(scheduler.get('peak_ordering_gap') or 0) > 0:
        findings.append(
            f"apply head-of-line blocking: peak_ordering_gap={scheduler.get('peak_ordering_gap')}",
        )
    if int(scheduler.get('ordering_wait_ms') or 0) > 1000:
        findings.append(
            f"ordering_wait_ms={scheduler.get('ordering_wait_ms')} — apply stall under parallel load",
        )
    if int(scheduler.get('peak_survivor_buffer') or 0) > 50:
        findings.append('heavy survivor backlog elevated — lane routing or heavy pool saturation')
    return findings


def human_verification_summary(report: dict[str, Any]) -> str:
    wf = report.get('waterfall') or {}
    eff = report.get('stage0_efficiency') or {}
    lines = [
        'Stage 0 Production Verification',
        f"  input={wf.get('input', 0)} "
        f"stage0_reject={wf.get('stage0_rejects', 0)} "
        f"survivors={wf.get('stage0_survivors', 0)} "
        f"heavy_enter={wf.get('heavy_entered', 0)} "
        f"heavy_exit={wf.get('heavy_completed', 0)} "
        f"final_kept={wf.get('final_kept', 0)}",
        f"  stage0_reject_rate={eff.get('reject_rate_pct', 0)}% "
        f"heavy_prevention={eff.get('heavy_prevention_pct', 0)}% "
        f"heavy_admission={eff.get('heavy_admission_rate_pct', 0)}% "
        f"heavy_yield={eff.get('heavy_to_kept_yield_pct', 0)}%",
        f"  duplicate_docs={report.get('duplicate_execution', {}).get('duplicate_doc_count', 0)} "
        f"late_heavy_rejects={report.get('late_heavy_analysis', {}).get('late_heavy_reject_count', 0)} "
        f"movable_to_stage0={report.get('late_heavy_analysis', {}).get('movable_count', 0)}",
    ]
    parity = report.get('parity') or {}
    if parity:
        lines.append(
            f"  parity_hash_match={parity.get('hash_match')} "
            f"seq_hash={parity.get('sequential_hash', '')[:12]} "
            f"par_hash={parity.get('parallel_hash', '')[:12]}",
        )
    findings = report.get('architectural_findings') or []
    if findings:
        lines.append('findings:')
        for f in findings:
            lines.append(f'  - {f}')
    return '\n'.join(lines)
