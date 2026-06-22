from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import Any
from indw.tools.reports.fast.stats import wilson_ci

def write_quality_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        'corpus': report['meta'].get('corpus_path'),
        'total_documents': report['meta']['total_documents'],
        'sampled_documents': report['meta']['sampled_documents'],
        'estimated_tokens': report['basic_statistics']['estimated_total_tokens'],
        'scores': report['scores'],
        'verdict': report['verdict']['grade'],
        'ready_for_training': report['verdict']['ready_for_training'],
        'english_rate_pct': report['language']['english_sample_rate_pct'],
        'major_issues': report['verdict']['major_issues'],
    }

def render_markdown_report(report: dict[str, Any]) -> str:
    m = report['meta']
    s = report['scores']
    v = report['verdict']
    lines = [
        '# Corpus Quality Audit Report',
        '',
        f"**Corpus:** `{m.get('corpus_path')}`  ",
        f"**File size:** {m.get('file_size_gb')} GB  ",
        f"**Total documents:** {m['total_documents']:,}  ",
        f"**Sampled:** {m['sampled_documents']:,} ({m['confidence_level']} CI)  ",
        f"**Runtime:** {m.get('total_elapsed_sec')}s  ",
        '',
        '## Verdict',
        '',
        f"**Grade: {v['grade']}** — {'READY' if v['ready_for_training'] else 'NOT READY'} for training  ",
        f"Overall quality: **{s['overall_dataset_quality']}/100**  ",
        '',
        '## Scores (0–100)',
        '',
        '| Metric | Score |',
        '|--------|-------|',
    ]
    for k, val in s.items():
        lines.append(f'| {k.replace("_", " ").title()} | {val} |')
    lines.extend(['', '## Basic Statistics', ''])
    bs = report['basic_statistics']
    lines.append(f"- Estimated tokens: **{bs['estimated_total_tokens']:,}**")
    lines.append(f"- Avg doc length: {bs['avg_doc_length_chars']:,} chars")
    lines.append(f"- Median / p90 / p99: {bs['median_chars']:,} / {bs['p90_chars']:,} / {bs['p99_chars']:,}")
    lines.extend(['', '## Language', ''])
    lang = report['language']
    lines.append(f"- English: **{lang['english_sample_rate_pct']}%** (sample)")
    lines.append(f"- Mixed language: {lang['mixed_language_sample_rate_pct']}%")
    lines.append(f"- Recommendation: {lang['recommendation']}")
    lines.extend(['', '## Filtering Remnants (sample rates)', ''])
    for name, data in report['filtering_validation'].items():
        if isinstance(data, dict) and 'sample_rate_pct' in data:
            lines.append(f"- {name}: {data['sample_rate_pct']}% [{data['ci95_low_pct']}–{data['ci95_high_pct']}]")
    lines.extend(['', '## Truncation', ''])
    tr = report['truncation']
    lines.append(f"- Low (none): {tr['none_pct']}%")
    lines.append(f"- Medium (slight): {tr['medium_slight_pct']}%")
    lines.append(f"- High (heavy): {tr['high_heavy_pct']}%")
    lines.extend(['', '## Pipeline Validation', ''])
    for k, ok in report['pipeline_validation'].items():
        lines.append(f"- {'✅' if ok else '❌'} {k.replace('_', ' ')}")
    if v['major_issues']:
        lines.extend(['', '## Major Issues', ''])
        for issue in v['major_issues']:
            lines.append(f'- {issue}')
    if v['minor_issues']:
        lines.extend(['', '## Minor Issues', ''])
        for issue in v['minor_issues']:
            lines.append(f'- {issue}')
    fixes = [f for f in v.get('suggested_fixes', []) if f]
    if fixes:
        lines.extend(['', '## Suggested Fixes', ''])
        for fix in fixes:
            lines.append(f'- {fix}')
    lines.extend(['', '## Best Documents (top 3)', ''])
    for doc in report['best_documents'][:3]:
        lines.append(f"**Score {doc['score']}** — {doc['preview'][:200]}...")
    lines.extend(['', '## Worst Documents (top 3)', ''])
    for doc in report['worst_documents'][:3]:
        lines.append(f"**Score {doc['score']}** — {doc['preview'][:200]}...")
    return '\n'.join(lines)
