from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

@dataclass
class LanguageRunStats:
    documents_scanned: int = 0
    mixed_language_count: int = 0
    unknown_count: int = 0
    rejected: int = 0
    early_rejected: int = 0
    score_rejected: int = 0
    balancer_rejected: int = 0
    detection_calls: int = 0
    detection_cpu_sec: float = 0.0
    skipped_post_clean: int = 0
    confidence_sum: float = 0.0
    language_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    reject_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    reject_by_stage: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    confidence_histogram: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_detection(self, cpu_sec: float) -> None:
        self.detection_calls += 1
        self.detection_cpu_sec += max(0.0, cpu_sec)

    def record_early_reject(self, reason: str) -> None:
        self.early_rejected += 1
        self.rejected += 1
        self.reject_by_stage['early_merge'] += 1
        if reason:
            self.reject_reasons[str(reason)] += 1

    def record_score_reject(self, reason: str) -> None:
        self.score_rejected += 1
        self.rejected += 1
        self.reject_by_stage['quality_score'] += 1
        if reason:
            self.reject_reasons[str(reason)] += 1

    def record_balancer_reject(self) -> None:
        self.balancer_rejected += 1
        self.rejected += 1
        self.reject_by_stage['lang_balancer'] += 1
        self.reject_reasons['language_cap'] += 1

    def record(self, assessment: Any) -> None:
        self.documents_scanned += 1
        self.confidence_sum += float(assessment.confidence)
        self.language_counts[str(assessment.primary_language)] += 1
        if assessment.mixed_language:
            self.mixed_language_count += 1
        if assessment.primary_language in ('und', 'unknown'):
            self.unknown_count += 1
        if assessment.should_reject:
            self.rejected += 1
            if assessment.reject_reason:
                self.reject_reasons[str(assessment.reject_reason)] += 1
        bucket = 'high' if assessment.confidence >= 0.85 else 'mid' if assessment.confidence >= 0.55 else 'low'
        self.confidence_histogram[bucket] += 1

    @property
    def average_confidence(self) -> float:
        return self.confidence_sum / max(self.documents_scanned, 1)

    def language_distribution(self) -> dict[str, float]:
        total = max(self.documents_scanned, 1)
        return {k: round(v / total, 4) for k, v in sorted(self.language_counts.items(), key=lambda kv: -kv[1])}

    def to_summary_dict(self) -> dict[str, Any]:
        n = max(self.documents_scanned, 1)
        return {
            'documents_scanned': self.documents_scanned,
            'language_distribution': self.language_distribution(),
            'confidence_distribution': dict(self.confidence_histogram),
            'mixed_language_rate': round(self.mixed_language_count / n, 4),
            'unknown_language_rate': round(self.unknown_count / n, 4),
            'rejected_documents': self.rejected,
            'early_rejected': self.early_rejected,
            'score_rejected': self.score_rejected,
            'balancer_rejected': self.balancer_rejected,
            'detection_calls': self.detection_calls,
            'detection_cpu_sec': round(self.detection_cpu_sec, 4),
            'skipped_post_clean': self.skipped_post_clean,
            'reject_by_stage': dict(self.reject_by_stage),
            'average_confidence': round(self.average_confidence, 4),
            'reject_reasons': dict(self.reject_reasons),
        }

def write_language_reports(
    stats: LanguageRunStats,
    *,
    output_dir: str | Path,
    report_detail: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    summary = stats.to_summary_dict()
    dist_path = out / 'language_distribution.json'
    dist_path.write_text(json.dumps({'created_at': now, **summary}, indent=2), encoding='utf-8')
    report = {'created_at': now, 'summary': summary, 'detail': report_detail or {}}
    report_path = out / 'language_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    hist_path = out / 'language_history.json'
    history: list[dict[str, Any]] = []
    if hist_path.exists():
        try:
            loaded = json.loads(hist_path.read_text(encoding='utf-8'))
            if isinstance(loaded, list):
                history = loaded
        except json.JSONDecodeError:
            history = []
    history.append({'created_at': now, **summary})
    if len(history) > 200:
        history = history[-200:]
    hist_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
    return {
        'language_report': report_path,
        'language_distribution': dist_path,
        'language_history': hist_path,
    }
