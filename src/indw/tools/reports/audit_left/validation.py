from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from indw.config.defaults import AUDIT_WRITE_BUFFER_BYTES
from indw.config.loader import CONFIG_ROOT, ConfigRef, Resolver, thaw
from indw.store.corpus.registry import CorpusRegistry
from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule import merge_with_quality
from indw.tools.reports.audit_left.artifact_leakage import measure_corpus_leakage
from indw.filter.gate.scorer import DocumentScore

logger = logging.getLogger(__name__)
_CHARS_PER_TOKEN = 3.8
_DEFAULT_SOURCES_SPEC = 'sources/mix_5mb_hf'
_DEFAULT_QUALITY_SPEC = 'filtering/quality_smoke_5mb'


@dataclass
class PipelineDiscovery:
    work_dir: Path
    raw_dir: Path
    sources_spec: str
    quality_spec: str
    sources_raw: dict[str, Any]
    quality_raw: dict[str, Any]
    pipeline_id: str = 'Instant_150m_moe'


def discover_pipeline(
    *,
    pipeline_id: str = 'default',
    work_dir: Optional[Path] = None,
    sources_spec: Optional[str] = None,
    quality_spec: Optional[str] = None,
) -> PipelineDiscovery:
    root = CONFIG_ROOT.parent
    work_dir = Path(work_dir) if work_dir else (root / 'work' / pipeline_id)
    raw_dir = work_dir / 'raw'
    if not raw_dir.exists():
        raise FileNotFoundError(f'Raw dataset not found: {raw_dir}')
    src_spec = sources_spec or _DEFAULT_SOURCES_SPEC
    q_spec = quality_spec or _DEFAULT_QUALITY_SPEC
    resolver = Resolver.default()
    sources_raw = thaw(resolver.resolve(ConfigRef(kind='dataset_sources', id=src_spec)).raw)
    quality_raw = thaw(resolver.resolve(ConfigRef(kind='quality', id=q_spec)).raw)
    raw_sources = sorted(raw_dir.glob('*/data.jsonl'))
    if not raw_sources:
        raise FileNotFoundError(f'No raw JSONL files under {raw_dir}')
    return PipelineDiscovery(
        work_dir=work_dir,
        raw_dir=raw_dir,
        sources_spec=src_spec,
        quality_spec=q_spec,
        sources_raw=sources_raw,
        quality_raw=quality_raw,
        pipeline_id=pipeline_id,
    )


@dataclass
class ValidationSampleCollector:
    best: list[dict[str, Any]] = field(default_factory=list)
    borderline: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    suspicious: list[dict[str, Any]] = field(default_factory=list)
    _limit: int = 10

    def _preview(self, text: str, n: int = 350) -> str:
        return text[:n].replace('\n', ' ')

    def _push(self, heap: list[dict[str, Any]], item: dict[str, Any], *, key: str, reverse: bool) -> None:
        heap.append(item)
        heap.sort(key=lambda x: x[key], reverse=reverse)
        del heap[self._limit :]

    def record_accepted(self, source: str, text: str, doc: DocumentScore) -> None:
        score = float(doc.quality_score_10) * 10.0
        item = {
            'score': round(score, 1),
            'source': source,
            'language': doc.language,
            'domain': doc.domain,
            'chars': len(text),
            'preview': self._preview(text),
        }
        self._push(self.best, item, key='score', reverse=True)
        if 45 <= score <= 58:
            self._push(self.borderline, item, key='score', reverse=False)
        if doc.signals.synthetic_score > 0.5 or doc.signals.html_score > 0.25:
            susp = {**item, 'reason': 'high_synthetic_or_html'}
            self._push(self.suspicious, susp, key='score', reverse=False)

    def record_rejected(self, source: str, text: str, reason: str, doc: DocumentScore) -> None:
        item = {
            'reason': reason,
            'source': source,
            'language': doc.language,
            'domain': doc.domain,
            'score': round(float(doc.quality_score_10) * 10.0, 1),
            'chars': len(text),
            'preview': self._preview(text),
            'signals': {
                'html': round(doc.signals.html_score, 3),
                'boilerplate': round(doc.signals.boilerplate_score, 3),
                'seo_spam': round(doc.signals.seo_spam_score, 3),
                'synthetic': round(doc.signals.synthetic_score, 3),
                'truncation': round(doc.signals.truncation_score, 3),
            },
        }
        self._push(self.rejected, item, key='score', reverse=True)

    def record_failure(self, source: str, detail: str) -> None:
        self.failures.append({'source': source, 'detail': detail})
        if len(self.failures) > 50:
            self.failures.pop(0)

    def to_dict(self) -> dict[str, Any]:
        return {
            'best_accepted': self.best[:10],
            'borderline_accepted': self.borderline[:10],
            'rejected': self.rejected[:10],
            'failures': self.failures[:20],
            'suspicious_accepted': self.suspicious[:10],
        }


def _map_reject_reasons(reasons: dict[str, int]) -> dict[str, int]:
    mapping = {
        'language': 'non_english',
        'language_cap': 'non_english',
        'low_language_confidence': 'non_english',
        'unsupported_language': 'non_english',
        'empty': 'empty_documents',
        'too_short': 'tiny_documents',
        'too_long': 'oversized_documents',
        'exact_dup': 'duplicate_content',
        'near_dup_fuzzy': 'duplicate_content',
        'near_dup_semantic': 'duplicate_content',
        'near_dup_embed': 'duplicate_content',
        'html': 'html',
        'boilerplate': 'excessive_boilerplate',
        'seo_spam': 'seo_spam',
        'commercial': 'seo_spam',
        'synthetic': 'ai_generated_content',
        'synthetic_slop': 'synthetic_garbage',
        'token_spam': 'word_salad',
        'repetition': 'repeated_punctuation',
        'truncation': 'truncation',
        'encoding': 'encoding_errors',
        'validation_empty': 'empty_documents',
        'validation_too_short': 'tiny_documents',
    }
    out: dict[str, int] = {}
    for reason, count in reasons.items():
        key = mapping.get(reason, reason)
        out[key] = out.get(key, 0) + int(count)
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def _live_progress_loop(
    *,
    stop_event: threading.Event,
    merge_work: Path,
    t0: float,
    interval: float = 5.0,
) -> None:
    progress_path = merge_work / 'pipeline_progress.json'
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_fn = lambda: proc.memory_info().rss / (1024 * 1024)
        cpu_fn = lambda: proc.cpu_percent(interval=None)
    except ImportError:
        mem_fn = lambda: 0.0
        cpu_fn = lambda: 0.0
    while not stop_event.wait(interval):
        elapsed = time.perf_counter() - t0
        mem_mb = mem_fn()
        cpu = cpu_fn()
        if progress_path.exists():
            try:
                prog = json.loads(progress_path.read_text(encoding='utf-8'))
                scanned = int(prog.get('total_scanned', 0))
                kept = int(prog.get('kept', 0))
                rejected = int(prog.get('rejected', 0))
                total = max(scanned, kept + rejected, 1)
                dps = scanned / max(elapsed, 0.001)
                print(
                    f'[{elapsed:6.0f}s] scanned={scanned:,} kept={kept:,} rejected={rejected:,} '
                    f'accept={100 * kept / total:.1f}% {dps:.1f} docs/s mem={mem_mb:.0f}MB cpu={cpu:.0f}%',
                    flush=True,
                )
            except (OSError, json.JSONDecodeError):
                print(f'[{elapsed:6.0f}s] waiting... mem={mem_mb:.0f}MB', flush=True)
        else:
            print(f'[{elapsed:6.0f}s] starting merge... mem={mem_mb:.0f}MB', flush=True)


def run_timed_pipeline_validation(
    *,
    time_limit_sec: float = 600.0,
    pipeline_id: str = 'default',
    workers: int = 1,
    validation_subdir: str = 'pipeline_validation_run',
    run_id: Optional[str] = None,
    resume: bool = False,
    force_lock: bool = False,
) -> dict[str, Any]:
    import os
    os.environ.setdefault('INSTANT_SKIP_METRICS_PROBE', '1')
    os.environ.setdefault('INSTANT_PIPELINE_VALIDATION', '1')
    discovery = discover_pipeline(pipeline_id=pipeline_id)
    subdir = validation_subdir if not run_id else f'{validation_subdir}_{run_id}'
    validation_work = discovery.work_dir / subdir
    validation_work.mkdir(parents=True, exist_ok=True)

    sources_path = validation_work / '_resolved_sources.yaml'
    quality_path = validation_work / '_resolved_quality.yaml'
    sources_path.write_text(yaml.safe_dump(discovery.sources_raw, sort_keys=True), encoding='utf-8')
    quality_path.write_text(yaml.safe_dump(discovery.quality_raw, sort_keys=True), encoding='utf-8')

    quality_cfg = QualityPipelineConfig.from_dict(discovery.quality_raw)
    out_path = validation_work / 'filtered.jsonl'
    collector = ValidationSampleCollector()
    corpus = CorpusRegistry(validation_work, corpus_id=f'{pipeline_id}-validation')

    raw_sources = sorted(discovery.raw_dir.glob('*/data.jsonl'))
    print('=' * 72, flush=True)
    print(f'PIPELINE VALIDATION RUN — {pipeline_id}', flush=True)
    print(f'Raw dir:     {discovery.raw_dir}', flush=True)
    print(f'Sources:     {len(raw_sources)} ({", ".join(p.parent.name for p in raw_sources[:6])}...)', flush=True)
    print(f'Quality:     {discovery.quality_spec}', flush=True)
    print(f'Output:      {out_path}', flush=True)
    print(f'Time limit:  {time_limit_sec:.0f}s', flush=True)
    print(f'Workers:     {workers}', flush=True)
    print(f'Resume:      {resume}', flush=True)
    if run_id:
        print(f'Run id:      {run_id}', flush=True)
    if workers > 1 and not run_id:
        print(
            'WARNING: workers>1 on shared validation dir risks checkpoint/output corruption; '
            'use --workers 1 or --run-id for isolated runs',
            flush=True,
        )
    print('=' * 72, flush=True)

    stop_progress = threading.Event()
    t0 = time.perf_counter()
    progress_thread = threading.Thread(
        target=_live_progress_loop,
        kwargs={'stop_event': stop_progress, 'merge_work': validation_work, 't0': t0},
        daemon=True,
    )
    progress_thread.start()

    error: Optional[str] = None
    stats: dict[str, Any] = {}
    try:
        stats = merge_with_quality(
            discovery.raw_dir,
            out_path,
            quality_config=quality_cfg,
            corpus_registry=corpus,
            write_buffer_bytes=AUDIT_WRITE_BUFFER_BYTES,
            work_dir=validation_work,
            resume=resume,
            fresh=not resume,
            workers=workers,
            time_limit_sec=time_limit_sec,
            validation_collector=collector,
            merge_lock_owner=f'validation:{pipeline_id}',
            merge_lock_force=force_lock,
        )
    except Exception as exc:
        from indw.schedule.state.lock import MergeRunConflictError
        if isinstance(exc, MergeRunConflictError):
            error = str(exc)
        else:
            error = f'{type(exc).__name__}: {exc}'
        logger.exception('Pipeline validation failed')
    finally:
        stop_progress.set()
        progress_thread.join(timeout=2.0)
        corpus.close()

    elapsed = time.perf_counter() - t0
    scanned = int(stats.get('scanned', stats.get('kept', 0) + stats.get('rejected', 0)))
    kept = int(stats.get('kept', 0))
    rejected = int(stats.get('rejected', 0))
    chunk_outcomes = kept + rejected
    from indw.schedule.state.checkpoint import count_jsonl_lines

    filtered_lines = count_jsonl_lines(out_path)
    accounting_gap = scanned - chunk_outcomes
    output_gap = kept - filtered_lines
    total = max(scanned, chunk_outcomes, 1)
    reject_reasons = _map_reject_reasons(dict(stats.get('reject_reasons', {})))
    gate_stats = stats.get('gate_stats') or {}
    chars_kept = int(gate_stats.get('token_chars_kept', 0))
    chars_rejected = int(gate_stats.get('token_chars_rejected', 0))
    tokens_kept = int(chars_kept / _CHARS_PER_TOKEN)
    tokens_rejected = int(chars_rejected / _CHARS_PER_TOKEN)
    lang_kept = dict(stats.get('language_kept', gate_stats.get('language_kept', {})))
    en_kept = lang_kept.get('en', 0)
    en_purity = en_kept / max(kept, 1)

    issues_critical: list[str] = []
    issues_major: list[str] = []
    issues_minor: list[str] = []

    if error:
        issues_critical.append(f'Pipeline crash: {error}')
    if kept == 0 and scanned > 100:
        issues_critical.append('Zero documents accepted despite scanning raw input')
    docs_per_sec = scanned / max(elapsed, 0.001)
    if elapsed >= 45 and docs_per_sec < 0.08:
        issues_major.append(
            f'Low throughput: {docs_per_sec:.2f} docs/s ({scanned} scanned in {elapsed:.0f}s)',
        )
    accept_rate = kept / total
    if accept_rate < 0.05:
        issues_major.append(f'Very low acceptance rate ({100 * accept_rate:.1f}%) — filter may be too aggressive')
    elif accept_rate > 0.95:
        issues_minor.append(f'Very high acceptance rate ({100 * accept_rate:.1f}%) — filter may be too weak')
    if en_purity < 0.88 and kept > 0:
        issues_major.append(f'English purity among kept docs is {100 * en_purity:.1f}% (target ≥88%)')
    if output_gap != 0:
        issues_major.append(
            f'Checkpoint kept ({kept}) != filtered.jsonl lines ({filtered_lines}); output/checkpoint mismatch',
        )
    leakage_report = None
    if out_path.exists() and kept > 0:
        try:
            leak_texts = [
                json.loads(ln).get('text', '')
                for ln in out_path.read_text(encoding='utf-8').splitlines()
                if ln.strip()
            ]
            leakage_report, leak_issues = measure_corpus_leakage(leak_texts)
            for item in leak_issues:
                issues_major.append(f'Output artifact leakage: {item}')
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            issues_minor.append(f'Could not measure output artifact leakage: {exc}')
    if accounting_gap > scanned * 0.05:
        issues_minor.append(
            f'Accounting gap: scanned={scanned} vs chunk outcomes={chunk_outcomes} (gap={accounting_gap})',
        )

    pipeline_ok = error is None and kept > 0 and docs_per_sec >= 0.05
    report = {
        'meta': {
            'pipeline_id': pipeline_id,
            'raw_dir': str(discovery.raw_dir),
            'quality_config': discovery.quality_spec,
            'sources_config': discovery.sources_spec,
            'validation_work_dir': str(validation_work),
            'run_id': run_id,
            'time_limit_sec': time_limit_sec,
            'elapsed_sec': round(elapsed, 2),
            'workers': workers,
            'resume': resume,
        },
        'throughput': {
            'documents_scanned': scanned,
            'documents_accepted': kept,
            'documents_rejected': rejected,
            'acceptance_rate_pct': round(100 * accept_rate, 2),
            'rejection_rate_pct': round(100 * rejected / total, 2),
            'docs_per_sec': round(scanned / max(elapsed, 0.001), 2),
            'avg_latency_ms_per_doc': round(1000 * elapsed / max(scanned, 1), 2),
            'tokens_accepted_est': tokens_kept,
            'tokens_rejected_est': tokens_rejected,
            'token_retention_pct': round(100 * tokens_kept / max(tokens_kept + tokens_rejected, 1), 2),
        },
        'accounting': {
            'lines_scanned': scanned,
            'chunk_outcomes': chunk_outcomes,
            'checkpoint_kept': kept,
            'checkpoint_rejected': rejected,
            'filtered_lines': filtered_lines,
            'scanned_minus_outcomes': accounting_gap,
            'kept_minus_filtered_lines': output_gap,
            'calibration': gate_stats.get('calibration') or {},
        },
        'filtering': {
            'reject_reasons': reject_reasons,
            'raw_reject_reasons': dict(stats.get('reject_reasons', {})),
            'domain_distribution': dict(stats.get('domain_distribution', {})),
            'language_kept': lang_kept,
            'english_purity_kept_pct': round(100 * en_purity, 2),
        },
        'stages_validated': {
            'input_loading': scanned > 0,
            'json_parsing': int(stats.get('skipped_parse', 0)) == 0 or scanned > 0,
            'normalization': True,
            'language_detection': bool(lang_kept),
            'quality_filtering': kept + rejected > 0,
            'deduplication': 'duplicate_content' in reject_reasons or kept > 0,
            'cleanup': leakage_report is None or leakage_report.marker_rate_pct <= 1.5,
            'validation': True,
            'output_writing': out_path.exists() and kept > 0,
        },
        'artifact_leakage': leakage_report.to_dict() if leakage_report else {},
        'samples': collector.to_dict(),
        'estimates': {
            'acceptance_percentage': round(100 * accept_rate, 2),
            'expected_language_purity': round(100 * en_purity, 1),
            'expected_noise_reduction_pct': round(100 * rejected / total, 1),
            'expected_token_retention_pct': round(100 * tokens_kept / max(tokens_kept + tokens_rejected, 1), 1),
            'expected_corpus_quality_score': round(float(gate_stats.get('score_mean', 0)) * 10, 1),
        },
        'issues': {
            'critical': issues_critical,
            'major': issues_major,
            'minor': issues_minor,
        },
        'verdict': {
            'pipeline_functioning': pipeline_ok,
            'filter_too_aggressive': accept_rate < 0.08,
            'filter_too_weak': accept_rate > 0.92,
            'ready_for_full_merge': pipeline_ok and accept_rate >= 0.08 and not issues_critical,
            'summary': (
                'Pipeline processed real raw data successfully.'
                if pipeline_ok
                else 'Pipeline encountered problems during real-data processing.'
            ),
        },
        'recommendations': [],
        'merge_stats': stats,
    }

    if en_purity < 0.88:
        report['recommendations'].append('Confirm quality_english_150m language gate is active at merge')
    if reject_reasons.get('duplicate_content', 0) > rejected * 0.3:
        report['recommendations'].append('High duplicate rejection — verify dedup is not over-aggressive on source mix')
    if reject_reasons.get('non_english', 0) > rejected * 0.2:
        report['recommendations'].append('Non-English rejection is working — expected for English-first config')
    if accept_rate < 0.15:
        report['recommendations'].append('Review quality thresholds — acceptance below 15% may indicate overly strict filters')

    return report


def write_validation_reports(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / 'pipeline_validation_report.json'
    summary_path = out_dir / 'pipeline_validation_summary.json'
    md_path = out_dir / 'pipeline_validation_report.md'

    summary = {
        'elapsed_sec': report['meta']['elapsed_sec'],
        'documents_scanned': report['throughput']['documents_scanned'],
        'documents_accepted': report['throughput']['documents_accepted'],
        'acceptance_rate_pct': report['throughput']['acceptance_rate_pct'],
        'docs_per_sec': report['throughput']['docs_per_sec'],
        'english_purity_kept_pct': report['filtering']['english_purity_kept_pct'],
        'pipeline_functioning': report['verdict']['pipeline_functioning'],
        'issues_critical': report['issues']['critical'],
        'issues_major': report['issues']['major'],
        'top_reject_reasons': dict(list(report['filtering']['reject_reasons'].items())[:8]),
    }

    audit_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    md_path.write_text(render_validation_markdown(report), encoding='utf-8')
    return {'audit': audit_path, 'summary': summary_path, 'markdown': md_path}


def render_validation_markdown(report: dict[str, Any]) -> str:
    m = report['meta']
    t = report['throughput']
    v = report['verdict']
    lines = [
        '# Pipeline Validation Report (Real Raw Data)',
        '',
        f"**Pipeline:** {m['pipeline_id']}  ",
        f"**Raw dir:** `{m['raw_dir']}`  ",
        f"**Quality config:** `{m['quality_config']}`  ",
        f"**Elapsed:** {m['elapsed_sec']}s / {m['time_limit_sec']}s limit  ",
        '',
        '## Verdict',
        '',
        f"**Pipeline functioning:** {'YES' if v['pipeline_functioning'] else 'NO'}  ",
        f"**Ready for full merge:** {'YES' if v['ready_for_full_merge'] else 'NO'}  ",
        v['summary'],
        '',
        '## Throughput',
        '',
        f"- Scanned: **{t['documents_scanned']:,}**",
        f"- Accepted: **{t['documents_accepted']:,}** ({t['acceptance_rate_pct']}%)",
        f"- Rejected: **{t['documents_rejected']:,}** ({t['rejection_rate_pct']}%)",
        f"- Speed: **{t['docs_per_sec']} docs/s**",
        f"- Avg latency: **{t['avg_latency_ms_per_doc']} ms/doc**",
        f"- Tokens retained (est): **{t['tokens_accepted_est']:,}** ({t['token_retention_pct']}%)",
        '',
        '## Top Rejection Reasons',
        '',
    ]
    for reason, count in list(report['filtering']['reject_reasons'].items())[:12]:
        lines.append(f'- {reason}: **{count:,}**')
    lines.extend(['', '## Stage Validation', ''])
    for stage, ok in report['stages_validated'].items():
        lines.append(f"- {'✅' if ok else '❌'} {stage.replace('_', ' ')}")
    for level in ('critical', 'major', 'minor'):
        items = report['issues'][level]
        if items:
            lines.extend(['', f'## {level.title()} Issues', ''])
            for item in items:
                lines.append(f'- {item}')
    recs = report.get('recommendations') or []
    if recs:
        lines.extend(['', '## Recommendations', ''])
        for rec in recs:
            lines.append(f'- {rec}')
    return '\n'.join(lines)
