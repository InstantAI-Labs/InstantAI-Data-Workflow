from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indw.clean.document.code_preservation import audit_code_integrity, preserve_code_blocks
from indw.clean.document.config import CleaningConfig
from indw.clean.corpus import CorpusCleaningPipeline
from indw.clean.semantic.pipeline import get_semantic_clean_report, reset_semantic_clean_report
from indw.tools.reports.fast.analyze import analyze_sample, run_fast_audit
from indw.tools.reports.fast.sample import parse_sample_lines, reservoir_sample_lines
from indw.tools.reports.fast.patterns import _CHARS_PER_TOKEN
from indw.filter.content.domain import domain_from_text
from indw.filter.score.signals import compute_signals
from indw.filter.refine.truncation import analyze_truncation, repair_truncation


def _quality_proxy(text: str, *, source: str = '') -> float:
    if not text or not text.strip():
        return 0.0
    sig = compute_signals(text[:8192])
    positive = (
        sig.formatting_score * 0.15
        + sig.structural_quality * 0.20
        + sig.coherence_score * 0.20
        + sig.factual_density * 0.20
        + sig.educational_value * 0.15
        + (1.0 - sig.boilerplate_score) * 0.10
    )
    negative = (
        sig.truncation_score * 0.25
        + sig.seo_spam_score * 0.20
        + sig.low_information_score * 0.20
        + sig.html_score * 0.15
        + sig.commercial_score * 0.10
        + sig.token_spam_score * 0.10
    )
    domain = domain_from_text(text[:2000], source_hint=source)
    if domain in ('code', 'reasoning', 'wiki', 'docs'):
        positive += 0.05
    return round(max(0.0, min(100.0, (positive - negative * 0.5) * 100)), 2)


@dataclass
class TransformRecord:
    doc_id: str
    source: str
    chars_before: int
    chars_after: int
    tokens_removed: int
    dropped: bool
    drop_reason: str
    artifact_categories: dict[str, int] = field(default_factory=dict)
    truncation_before: str = 'none'
    truncation_after: str = 'none'
    quality_before: float = 0.0
    quality_after: float = 0.0
    preview_before: str = ''
    preview_after: str = ''


def _estimate_tokens(chars: int) -> int:
    return max(0, int(chars / _CHARS_PER_TOKEN))


def _artifact_categories_from_stats(stats_dict: dict[str, Any]) -> dict[str, int]:
    cats: dict[str, int] = {}
    stages = stats_dict.get('stages') or {}
    for stage, data in stages.items():
        if isinstance(data, dict):
            removed = int(data.get('lines_removed', 0) or 0) + int(data.get('chars_removed', 0) // 80)
            if removed:
                cats[stage] = removed
    reasons = stats_dict.get('document_gate_reasons') or {}
    for reason, count in reasons.items():
        cats[f'gate:{reason}'] = int(count)
    return cats


def audit_document_transform(
    text: str,
    *,
    source: str = '',
    pipeline: CorpusCleaningPipeline | None = None,
    apply_truncation_repair: bool = True,
    apply_code_preservation: bool = True,
) -> TransformRecord:
    pipe = pipeline or CorpusCleaningPipeline()
    before = text.strip()
    doc_id = f'{source}:audit'

    score_b = _quality_proxy(before, source=source)
    trunc_b = analyze_truncation(before)

    working = before
    if apply_truncation_repair:
        working, _ = repair_truncation(working)
    if apply_code_preservation:
        working, _ = preserve_code_blocks(working)

    results = pipe.process(working, source=source)
    kept = [r for r in results if not r.dropped and r.text]
    dropped = not kept
    after = kept[0].text if kept else ''
    drop_reason = results[0].drop_reason if dropped and results else ''

    score_a = _quality_proxy(after, source=source) if after else score_b
    trunc_a = analyze_truncation(after) if after else trunc_b

    chars_removed = max(0, len(before) - len(after))
    return TransformRecord(
        doc_id=doc_id,
        source=source,
        chars_before=len(before),
        chars_after=len(after),
        tokens_removed=_estimate_tokens(chars_removed),
        dropped=dropped,
        drop_reason=drop_reason,
        artifact_categories=_artifact_categories_from_stats(pipe.stats.to_dict()),
        truncation_before=trunc_b.severity,
        truncation_after=trunc_a.severity,
        quality_before=score_b,
        quality_after=score_a,
        preview_before=before[:500],
        preview_after=after[:500],
    )


def generate_pipeline_audit_report(
    corpus_path: Path,
    *,
    sample_size: int = 500,
    seed: int = 42,
    cleaning_config: CleaningConfig | None = None,
    include_before_after: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    path = Path(corpus_path)
    total, lines = reservoir_sample_lines(path, sample_size, seed)
    docs = parse_sample_lines(lines)

    cfg = cleaning_config or CleaningConfig()
    if cfg.semantic_cleaning:
        reset_semantic_clean_report()
    pipe = CorpusCleaningPipeline(cfg)
    transforms: list[TransformRecord] = []
    artifact_totals: Counter = Counter()
    trunc_before: Counter = Counter()
    trunc_after: Counter = Counter()
    quality_before: list[float] = []
    quality_after: list[float] = []
    tokens_removed_total = 0
    dropped = 0
    domain_before: Counter = Counter()
    domain_after: Counter = Counter()

    for doc in docs:
        rec = audit_document_transform(
            doc.text,
            source=doc.source,
            pipeline=pipe,
            apply_truncation_repair=getattr(pipe.config, 'truncation_repair', True),
            apply_code_preservation=getattr(pipe.config, 'code_preservation', True),
        )
        transforms.append(rec)
        for cat, n in rec.artifact_categories.items():
            artifact_totals[cat] += n
        trunc_before[rec.truncation_before] += 1
        trunc_after[rec.truncation_after] += 1
        quality_before.append(rec.quality_before)
        quality_after.append(rec.quality_after)
        tokens_removed_total += rec.tokens_removed
        if rec.dropped:
            dropped += 1
        domain_before[domain_from_text(doc.text[:2000], source_hint=doc.source)] += 1
        if rec.preview_after:
            domain_after[domain_from_text(rec.preview_after[:2000], source_hint=doc.source)] += 1

    n = max(len(transforms), 1)
    code_audits = [audit_code_integrity(t.preview_after) for t in transforms if t.preview_after][:50]
    code_valid = sum(a.get('syntax_valid_rate', 0) for a in code_audits) / max(len(code_audits), 1)

    over_filter_risks: list[str] = []
    drop_rate = dropped / n
    if drop_rate > 0.35:
        over_filter_risks.append(f'High drop rate {drop_rate:.1%} — review document_gate thresholds')
    avg_q_delta = sum(quality_after) / n - sum(quality_before) / n
    if avg_q_delta < -5:
        over_filter_risks.append('Average quality score decreased after cleaning — possible knowledge loss')
    protected_domains = {'code', 'reasoning', 'wiki', 'docs'}
    for dom in protected_domains:
        b = domain_before.get(dom, 0) / n
        a = domain_after.get(dom, 0) / n
        if b > 0.05 and a < b * 0.6:
            over_filter_risks.append(f'Domain {dom} reduced {b:.1%} -> {a:.1%}')

    fast = run_fast_audit(path, sample_size=min(sample_size, 3000), seed=seed, skip_line_count=True)
    fast['meta']['total_documents'] = total

    samples = []
    if include_before_after:
        rng = random.Random(seed)
        picks = rng.sample(transforms, min(8, len(transforms)))
        for rec in picks:
            samples.append({
                'source': rec.source,
                'chars_before': rec.chars_before,
                'chars_after': rec.chars_after,
                'tokens_removed': rec.tokens_removed,
                'dropped': rec.dropped,
                'quality_delta': round(rec.quality_after - rec.quality_before, 2),
                'before': rec.preview_before,
                'after': rec.preview_after,
            })

    return {
        'meta': {
            'corpus_path': str(path.resolve()),
            'total_documents': total,
            'sampled_documents': len(transforms),
            'audit_elapsed_sec': round(time.time() - t0, 2),
            'pipeline_config': (cleaning_config or CleaningConfig()).__dict__,
        },
        'processing_summary': {
            'documents_processed': len(transforms),
            'documents_dropped': dropped,
            'drop_rate_pct': round(100 * drop_rate, 2),
            'tokens_removed_estimate': tokens_removed_total,
            'avg_tokens_removed_per_doc': round(tokens_removed_total / n, 1),
            'avg_quality_before': round(sum(quality_before) / n, 2),
            'avg_quality_after': round(sum(quality_after) / n, 2),
            'quality_delta': round(avg_q_delta, 2),
        },
        'artifact_categories': dict(artifact_totals.most_common(20)),
        'truncation': {
            'before': dict(trunc_before),
            'after': dict(trunc_after),
            'repaired_estimate': trunc_before.get('slight', 0) - trunc_after.get('slight', 0),
        },
        'code_integrity': {
            'blocks_sampled': sum(int(a.get('code_blocks', 0)) for a in code_audits),
            'avg_syntax_valid_rate': round(code_valid, 4),
            'language_distribution': dict(
                Counter(
                    lang
                    for a in code_audits
                    for lang, cnt in (a.get('languages') or {}).items()
                    for _ in range(int(cnt))
                ).most_common(12)
            ),
        },
        'domain_distribution': {
            'before': dict(domain_before.most_common()),
            'after': dict(domain_after.most_common()),
        },
        'quality_score_distribution': {
            'before_p50': sorted(quality_before)[len(quality_before) // 2] if quality_before else 0,
            'after_p50': sorted(quality_after)[len(quality_after) // 2] if quality_after else 0,
        },
        'corpus_fast_audit': fast,
        'over_filtering_risks': over_filter_risks,
        'training_quality_impact': {
            'signal_to_noise': 'improved' if avg_q_delta >= 0 else 'degraded',
            'token_efficiency_gain_pct': round(
                100 * tokens_removed_total / max(_estimate_tokens(sum(t.chars_before for t in transforms)), 1),
                2,
            ),
            'estimated_grade': fast.get('verdict', {}).get('grade', '?'),
            'ready_for_training': fast.get('verdict', {}).get('ready_for_training', False),
        },
        'before_after_samples': samples,
        'cleaning_pipeline_stats': pipe.stats.to_dict(),
        'semantic_cleaning_report': (
            get_semantic_clean_report().to_dict() if cfg.semantic_cleaning else None
        ),
    }


def write_pipeline_audit(
    corpus_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 500,
    seed: int = 42,
    cleaning_config: CleaningConfig | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = generate_pipeline_audit_report(
        corpus_path,
        sample_size=sample_size,
        seed=seed,
        cleaning_config=cleaning_config,
    )
    out = output_dir / 'pipeline_audit_report.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    md = output_dir / 'pipeline_audit_report.md'
    md.write_text(_render_audit_md(report), encoding='utf-8')
    return out


def _render_audit_md(report: dict[str, Any]) -> str:
    m = report['meta']
    ps = report['processing_summary']
    tq = report['training_quality_impact']
    lines = [
        '# Pipeline Audit Report',
        '',
        f"**Corpus:** `{m.get('corpus_path')}`",
        f"**Documents:** {m['total_documents']:,} (sampled {m['sampled_documents']:,})",
        f"**Runtime:** {m['audit_elapsed_sec']}s",
        '',
        '## Processing Summary',
        '',
        f"- Drop rate: **{ps['drop_rate_pct']}%**",
        f"- Tokens removed (est.): **{ps['tokens_removed_estimate']:,}**",
        f"- Quality: {ps['avg_quality_before']} → {ps['avg_quality_after']} (Δ {ps['quality_delta']})",
        '',
        '## Training Impact',
        '',
        f"- Signal/noise: **{tq['signal_to_noise']}**",
        f"- Token efficiency gain: **{tq['token_efficiency_gain_pct']}%**",
        f"- Corpus grade: **{tq['estimated_grade']}**",
        '',
    ]
    risks = report.get('over_filtering_risks') or []
    if risks:
        lines.extend(['## Over-filtering Risks', ''])
        for r in risks:
            lines.append(f'- {r}')
    samples = report.get('before_after_samples') or []
    if samples:
        lines.extend(['', '## Before/After Samples', ''])
        for s in samples[:3]:
            lines.append(f"### {s['source']} (−{s['tokens_removed']} tokens)")
            lines.append(f"**Before:** {s['before'][:200]}...")
            lines.append(f"**After:** {s['after'][:200]}...")
            lines.append('')
    return '\n'.join(lines)
