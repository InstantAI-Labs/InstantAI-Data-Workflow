from __future__ import annotations

import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from indw.schedule.config.resolve import env_flag
from indw.filter.stage0.admission import DOC_TIER_HUGE, DOC_TIER_LARGE
from indw.schedule.state.context import MergeDocumentContext
from indw.schedule.dispatch.lanes import survivor_lane

_AUDIT_ENV = 'INSTANT_MERGE_STAGE0_AUDIT'
_AUDIT_DIR_ENV = 'INSTANT_MERGE_AUDIT_DIR'
_TRACE_LIMIT = 100

_SIZE_BUCKETS = (
    ('0_5k', 0, 5_000),
    ('5_10k', 5_000, 10_000),
    ('10_20k', 10_000, 20_000),
    ('20_50k', 20_000, 50_000),
    ('50_100k', 50_000, 100_000),
    ('100k_plus', 100_000, None),
)


def audit_enabled() -> bool:
    return env_flag(_AUDIT_ENV, default=False)


def bind_audit_dir(path: Path | str) -> None:
    os.environ[_AUDIT_DIR_ENV] = str(path)


def _audit_dir() -> Path | None:
    raw = os.environ.get(_AUDIT_DIR_ENV, '').strip()
    if not raw:
        return None
    return Path(raw)


def _trace_path() -> Path | None:
    base = _audit_dir()
    if base is None:
        return None
    base.mkdir(parents=True, exist_ok=True)
    return base / f'stage0_audit_{os.getpid()}.jsonl'


def _emit(event: dict[str, Any]) -> None:
    if not audit_enabled():
        return
    path = _trace_path()
    if path is None:
        return
    event = dict(event)
    event.setdefault('ts', time.time())
    event.setdefault('pid', os.getpid())
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(event, separators=(',', ':'), default=str) + '\n')


def _reject_bucket(ctx: MergeDocumentContext) -> str | None:
    if ctx.kind == 'blank':
        return 'REJECT_EMPTY'
    if ctx.kind == 'parse_error':
        return 'REJECT_INVALID'
    if ctx.kind == 'empty_text':
        return 'REJECT_EMPTY'
    if not ctx.cleaning_rejects:
        return None
    reason = str(ctx.cleaning_rejects[-1][0] or '')
    if reason in ('too_short',):
        return 'REJECT_TOO_SMALL'
    if reason in ('document_too_large',):
        return 'REJECT_TOO_LARGE'
    if reason in ('exact_doc_dup', 'exact_dup'):
        return 'REJECT_DUPLICATE'
    if reason in ('extreme_corruption', 'ocr_noisy'):
        return 'REJECT_CORRUPTION'
    if reason in ('html_dump',):
        return 'REJECT_HTML'
    if reason in ('navigation_boilerplate', 'disambiguation_page', 'template_page', 'login_page'):
        return 'REJECT_NAV'
    if reason in ('metadata_only',):
        return 'REJECT_METADATA'
    if reason in ('error_page',):
        return 'REJECT_STRUCTURAL'
    if 'language' in reason or reason == 'english_only':
        return 'REJECT_LANGUAGE'
    return 'REJECT_OTHER'


def _size_bucket(chars: int) -> str:
    for name, lo, hi in _SIZE_BUCKETS:
        if hi is None and chars >= lo:
            return name
        if hi is not None and lo <= chars < hi:
            return name
    return '0_5k'


def record_reader_input(*, seq: int, source: str, chars: int | None = None) -> None:
    _emit({
        'event': 'input',
        'seq': int(seq),
        'source': source,
        'chars': chars,
    })


def record_fast_exit(
    ctx: MergeDocumentContext,
    needs_heavy: bool,
    *,
    wall_ms: float,
    path: str = 'fast_pool',
) -> None:
    bucket = _reject_bucket(ctx)
    terminal = not needs_heavy
    lane = survivor_lane({'doc_tier': ctx.doc_tier, 'admission': ctx.admission}) if needs_heavy else ''
    _emit({
        'event': 'stage0_fast',
        'seq': int(ctx.seq),
        'source': ctx.src_name,
        'chars': int(ctx.meaningful_chars or len(ctx.text or '')),
        'doc_tier': ctx.doc_tier,
        'lane': lane,
        'admission': ctx.admission,
        'fast_terminal': terminal,
        'fast_survivor': needs_heavy,
        'reject_bucket': bucket,
        'reject_reason': str(ctx.cleaning_rejects[-1][0] or '') if ctx.cleaning_rejects else '',
        'stage0_accept': terminal and bucket is None,
        'stage0_reject': terminal and bucket is not None,
        'wall_ms': round(wall_ms, 3),
        'path': path,
    })


def record_heavy_enter(
    ctx: MergeDocumentContext,
    *,
    lane: str = '',
    path: str = 'heavy_pool',
) -> None:
    chars = int(ctx.meaningful_chars or len(ctx.text or ''))
    _emit({
        'event': 'heavy_enter',
        'seq': int(ctx.seq),
        'source': ctx.src_name,
        'chars': chars,
        'doc_tier': ctx.doc_tier,
        'lane': lane or survivor_lane({'doc_tier': ctx.doc_tier, 'admission': ctx.admission}),
        'size_bucket': _size_bucket(chars),
        'path': path,
    })


def record_heavy_exit(
    ctx: MergeDocumentContext,
    *,
    wall_ms: float,
    path: str = 'heavy_pool',
) -> None:
    _emit({
        'event': 'heavy_exit',
        'seq': int(ctx.seq),
        'source': ctx.src_name,
        'chars': int(ctx.meaningful_chars or len(ctx.text or '')),
        'doc_tier': ctx.doc_tier,
        'chunk_count': len(ctx.chunks),
        'cleaning_rejects': len(ctx.cleaning_rejects),
        'reject_reasons': [r for r, _ in ctx.cleaning_rejects],
        'wall_ms': round(wall_ms, 3),
        'path': path,
    })


def record_apply(
    line: dict[str, Any],
    *,
    kept: bool,
    wall_ms: float,
) -> None:
    _emit({
        'event': 'apply',
        'seq': int(line['seq']),
        'source': line.get('src_name', ''),
        'kept': bool(kept),
        'kind': line.get('kind', 'processed'),
        'chunk_count': len(line.get('chunks') or []),
        'wall_ms': round(wall_ms, 3),
    })


def collect_event_files(work_dir: Path) -> list[Path]:
    return sorted(work_dir.glob('stage0_audit_*.jsonl'))


def load_events(work_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in collect_event_files(work_dir):
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def build_report(
    work_dir: Path,
    *,
    scheduler: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    stage_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events = load_events(work_dir)
    by_seq: dict[int, dict[str, Any]] = defaultdict(dict)
    counters = Counter()
    reject_counters = Counter()
    wall_by_stage: dict[str, list[float]] = defaultdict(list)
    heavy_chars: list[int] = []
    lane_counts = Counter()
    size_bucket_heavy = Counter()
    size_bucket_input = Counter()
    dup_hits: list[dict[str, Any]] = []
    stage_hits: dict[int, Counter[str]] = defaultdict(Counter)

    for ev in events:
        et = str(ev.get('event', ''))
        seq = ev.get('seq')
        if seq is not None:
            seq = int(seq)
            stage_hits[seq][et] += 1
            row = by_seq[seq]
            row['seq'] = seq
            row['source'] = ev.get('source', row.get('source', ''))
            if ev.get('chars') is not None:
                row['chars'] = ev['chars']
                size_bucket_input[_size_bucket(int(ev['chars']))] += int(et == 'input')
            if ev.get('doc_tier'):
                row['doc_tier'] = ev['doc_tier']
            if ev.get('lane'):
                row['lane'] = ev['lane']
            if ev.get('admission') is not None:
                row['admission'] = ev['admission']
            if et == 'stage0_fast':
                row['fast_terminal'] = ev.get('fast_terminal')
                row['fast_survivor'] = ev.get('fast_survivor')
                if ev.get('reject_bucket'):
                    reject_counters[ev['reject_bucket']] += 1
                if ev.get('stage0_reject'):
                    counters['STAGE0_REJECT'] += 1
                elif ev.get('fast_survivor'):
                    counters['STAGE0_SURVIVORS'] += 1
                else:
                    counters['STAGE0_ACCEPT'] += 1
                wall_by_stage['stage0'].append(float(ev.get('wall_ms') or 0))
            elif et == 'heavy_enter':
                row['heavy_enter'] = True
                lane_counts[str(ev.get('lane') or '')] += 1
                chars = int(ev.get('chars') or 0)
                heavy_chars.append(chars)
                size_bucket_heavy[ev.get('size_bucket') or _size_bucket(chars)] += 1
                counters['HEAVY_ENTER'] += 1
            elif et == 'heavy_exit':
                row['heavy_exit'] = True
                wall_by_stage['heavy'].append(float(ev.get('wall_ms') or 0))
                counters['HEAVY_EXIT'] += 1
            elif et == 'apply':
                row['apply_enter'] = True
                row['apply_exit'] = True
                row['kept'] = ev.get('kept')
                wall_by_stage['apply'].append(float(ev.get('wall_ms') or 0))
                if ev.get('kept'):
                    counters['FINAL_KEPT'] += 1
                else:
                    counters['FINAL_REJECTED'] += 1
            elif et == 'input':
                counters['INPUT_DOCS'] += 1

    for seq, hits in stage_hits.items():
        for stage, count in hits.items():
            if stage in ('stage0_fast', 'heavy_enter', 'heavy_exit', 'apply') and count > 1:
                dup_hits.append({'seq': seq, 'stage': stage, 'count': count})

    input_docs = counters['INPUT_DOCS']
    stage0_reject = counters['STAGE0_REJECT']
    stage0_survivors = counters['STAGE0_SURVIVORS']
    stage0_accept_terminal = counters['STAGE0_ACCEPT']

    trace_rows = []
    for seq in sorted(by_seq)[:_TRACE_LIMIT]:
        row = by_seq[seq]
        trace_rows.append({
            'SEQ': row.get('seq'),
            'SOURCE': row.get('source', ''),
            'CHARS': row.get('chars'),
            'DOC_TIER': row.get('doc_tier', ''),
            'LANE': row.get('lane', ''),
            'ADMISSION': row.get('admission'),
            'FAST_TERMINAL': row.get('fast_terminal'),
            'FAST_SURVIVOR': row.get('fast_survivor'),
            'HEAVY_ENTER': bool(row.get('heavy_enter')),
            'HEAVY_EXIT': bool(row.get('heavy_exit')),
            'APPLY_ENTER': bool(row.get('apply_enter')),
            'APPLY_EXIT': bool(row.get('apply_exit')),
            'KEPT': row.get('kept'),
        })

    heavy_enter = counters['HEAVY_ENTER']
    heavy_exit = counters['HEAVY_EXIT']
    large_lane = lane_counts.get('large', 0)
    huge_lane = lane_counts.get('huge', 0)

    def _wall_summary(name: str) -> dict[str, float]:
        vals = wall_by_stage.get(name) or []
        if not vals:
            return {'count': 0, 'total_ms': 0.0, 'avg_ms': 0.0, 'p95_ms': 0.0}
        return {
            'count': len(vals),
            'total_ms': round(sum(vals), 1),
            'avg_ms': round(statistics.mean(vals), 2),
            'p95_ms': round(_percentile(vals, 95), 2),
        }

    stage_metrics_stages = (stage_metrics or {}).get('stages') or {}
    ke_wall = 0.0
    for name, row in stage_metrics_stages.items():
        if not isinstance(row, dict):
            continue
        if 'ke_' in name or 'semantic' in name or 'cleaning' in name:
            ke_wall += float(row.get('wall_sec') or 0.0)

    bottlenecks = [
        {'name': 'stage0_fast', **_wall_summary('stage0')},
        {'name': 'heavy_total', **_wall_summary('heavy')},
        {'name': 'apply_dedup', **_wall_summary('apply')},
    ]
    if scheduler:
        bottlenecks.append({
            'name': 'ordering_wait',
            'total_ms': float(scheduler.get('ordering_wait_ms') or 0),
            'peak_gap': int(scheduler.get('peak_ordering_gap') or 0),
        })
    bottlenecks.sort(key=lambda r: -float(r.get('total_ms') or r.get('count') or 0))

    prevented_heavy = stage0_reject + stage0_accept_terminal
    unnecessary_large_heavy = sum(
        1 for ev in events
        if ev.get('event') == 'heavy_enter'
        and str(ev.get('doc_tier') or '') in (DOC_TIER_LARGE, DOC_TIER_HUGE)
    )

    report = {
        'work_dir': str(work_dir),
        'event_count': len(events),
        'counters': {
            'INPUT_DOCS': input_docs,
            'STAGE0_ACCEPT': stage0_accept_terminal,
            'STAGE0_REJECT': stage0_reject,
            'REJECT_LANGUAGE': reject_counters['REJECT_LANGUAGE'],
            'REJECT_EMPTY': reject_counters['REJECT_EMPTY'],
            'REJECT_TOO_SMALL': reject_counters['REJECT_TOO_SMALL'],
            'REJECT_TOO_LARGE': reject_counters['REJECT_TOO_LARGE'],
            'REJECT_DUPLICATE': reject_counters['REJECT_DUPLICATE'],
            'REJECT_CORRUPTION': reject_counters.get('REJECT_CORRUPTION', 0),
            'REJECT_HTML': reject_counters.get('REJECT_HTML', 0),
            'REJECT_NAV': reject_counters.get('REJECT_NAV', 0),
            'REJECT_METADATA': reject_counters.get('REJECT_METADATA', 0),
            'REJECT_STRUCTURAL': reject_counters.get('REJECT_STRUCTURAL', 0),
            'REJECT_INVALID': reject_counters['REJECT_INVALID'],
            'REJECT_OTHER': reject_counters['REJECT_OTHER'],
            'STAGE0_SURVIVORS': stage0_survivors,
            'STAGE0_ACCEPTANCE_RATE': round(
                100.0 * stage0_survivors / max(input_docs, 1), 2,
            ),
            'STAGE0_REJECT_RATE': round(
                100.0 * stage0_reject / max(input_docs, 1), 2,
            ),
        },
        'waterfall': {
            'input_docs': input_docs,
            'stage0_deterministic_reject': (
                reject_counters.get('REJECT_CORRUPTION', 0)
                + reject_counters.get('REJECT_HTML', 0)
                + reject_counters.get('REJECT_NAV', 0)
                + reject_counters.get('REJECT_METADATA', 0)
                + reject_counters.get('REJECT_STRUCTURAL', 0)
            ),
            'stage0_survivors': stage0_survivors,
            'heavy_prevented_by_stage0': prevented_heavy,
            'heavy_entered': heavy_enter,
            'heavy_completed': heavy_exit,
            'apply_seen': counters['FINAL_KEPT'] + counters['FINAL_REJECTED'],
            'final_kept': counters['FINAL_KEPT'],
            'final_rejected': counters['FINAL_REJECTED'],
        },
        'heavy_admission': {
            'entered': heavy_enter,
            'completed': heavy_exit,
            'avg_chars': round(statistics.mean(heavy_chars), 1) if heavy_chars else 0,
            'median_chars': round(statistics.median(heavy_chars), 1) if heavy_chars else 0,
            'p95_chars': round(_percentile([float(c) for c in heavy_chars], 95), 1) if heavy_chars else 0,
            'large_lane': large_lane,
            'huge_lane': huge_lane,
        },
        'size_distribution': {
            'input': dict(size_bucket_input),
            'heavy_enter': dict(size_bucket_heavy),
        },
        'trace_first_100': trace_rows,
        'duplicate_execution': dup_hits,
        'wall_time_ms': {
            'stage0': _wall_summary('stage0'),
            'heavy': _wall_summary('heavy'),
            'apply': _wall_summary('apply'),
        },
        'stage0_protection': {
            'docs_prevented_from_heavy': prevented_heavy,
            'prevented_pct': round(100.0 * prevented_heavy / max(input_docs, 1), 2),
            'large_docs_entering_heavy': unnecessary_large_heavy,
            'estimated_heavy_cpu_saved_pct': round(
                100.0 * prevented_heavy / max(input_docs, 1), 2,
            ),
        },
        'scheduler': scheduler or {},
        'progress': progress or {},
        'ke_pipeline_wall_sec': round(ke_wall, 2),
        'bottleneck_ranking': bottlenecks[:10],
        'dead_code_candidates': _dead_code_report(),
        'safe_optimizations': _safe_optimizations(report_context={
            'stage0_survivor_rate': counters['STAGE0_SURVIVORS'] / max(input_docs, 1),
            'heavy_enter': heavy_enter,
            'ordering_gap': int((scheduler or {}).get('peak_ordering_gap') or 0),
            'duplicate_hits': len(dup_hits),
        }),
        'quality_parity': {
            'audit_only': True,
            'acceptance_logic_changed': False,
            'corpus_output_changed': False,
        },
    }
    return report


def _dead_code_report() -> list[dict[str, str]]:
    return [
        {'item': 'process_merge_batch', 'role': 'fast-pool failure fallback only', 'action': 'keep'},
        {'item': 'run_progressive_preprocess', 'role': 'sequential merge entry', 'action': 'keep'},
        {'item': 'preprocess_merge_line', 'role': 'core.py facade', 'action': 'keep'},
        {'item': 'early_language_reject', 'role': 'preprocess_util wrapper', 'action': 'optional_inline'},
        {'item': 'lane_min_seq', 'role': 'per-lane min seq for dispatch', 'action': 'keep'},
    ]


def _safe_optimizations(*, report_context: dict[str, Any]) -> list[str]:
    opts = []
    rate = float(report_context.get('stage0_survivor_rate') or 0)
    if rate > 0.5:
        opts.append('Tighten Stage0 gates: >50% of inputs become heavy survivors — review language/size thresholds')
    if report_context.get('ordering_gap', 0) > 10:
        opts.append('Heavy OOO dispatch; verify heavy_submits >> heavy_workers in scheduler report')
    if report_context.get('duplicate_hits', 0) > 0:
        opts.append('Investigate duplicate stage hits flagged in duplicate_execution')
    if rate < 0.3:
        opts.append('Stage0 filtering effective; focus on heavy per-doc latency not fast reject rate')
    opts.extend([
        'Prep-on-ingest: already removes apply object conversion from hot path',
        'Head-gated heavy dispatch: prevents non-head survivors consuming pool',
        'Document-scoped semantic caches: avoid duplicate KE within doc',
        'Do not parallelize dedup apply without ordered commit redesign',
    ])
    return opts[:10]


def publish_report(work_dir: Path, report: dict[str, Any]) -> Path:
    out = work_dir / 'stage0_audit_report.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return out


def human_summary(report: dict[str, Any]) -> str:
    c = report.get('counters') or {}
    w = report.get('waterfall') or {}
    h = report.get('heavy_admission') or {}
    p = report.get('stage0_protection') or {}
    lines = [
        'Stage0 Audit Summary',
        f"  INPUT_DOCS={c.get('INPUT_DOCS', 0)}",
        f"  STAGE0_REJECT={c.get('STAGE0_REJECT', 0)} STAGE0_SURVIVORS={c.get('STAGE0_SURVIVORS', 0)}",
        f"  survivor_rate={c.get('STAGE0_ACCEPTANCE_RATE', 0)}% reject_rate={c.get('STAGE0_REJECT_RATE', 0)}%",
        f"  REJECT_LANGUAGE={c.get('REJECT_LANGUAGE', 0)} REJECT_TOO_SMALL={c.get('REJECT_TOO_SMALL', 0)}",
        f"  REJECT_TOO_LARGE={c.get('REJECT_TOO_LARGE', 0)} REJECT_EMPTY={c.get('REJECT_EMPTY', 0)}",
        f"  waterfall: input={w.get('input_docs')} -> heavy={w.get('heavy_entered')} -> kept={w.get('final_kept')}",
        f"  heavy: entered={h.get('entered')} median_chars={h.get('median_chars')} large_lane={h.get('large_lane')} huge_lane={h.get('huge_lane')}",
        f"  prevented_from_heavy={p.get('docs_prevented_from_heavy')} ({p.get('prevented_pct')}%)",
        f"  duplicate_hits={len(report.get('duplicate_execution') or [])}",
    ]
    sched = report.get('scheduler') or {}
    if sched:
        lines.append(
            f"  ordering: peak_gap={sched.get('peak_ordering_gap', 0)} "
            f"head_blocked={sched.get('head_blocked_dispatches', 0)}"
        )
    return '\n'.join(lines)
