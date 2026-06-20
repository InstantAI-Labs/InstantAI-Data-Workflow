from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from indw.filter.toxicity.config import DEFAULT_CATEGORIES

@dataclass
class ToxicityRunStats:
    documents_scanned: int = 0
    accepted: int = 0
    reviewed: int = 0
    rejected: int = 0
    hard_rejected: int = 0
    toxicity_sum: float = 0.0
    category_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    reject_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    bands: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, *, final_score: float, band: str, reason: Optional[str], ml_top: str) -> None:
        self.documents_scanned += 1
        self.toxicity_sum += final_score
        self.bands[band] += 1
        if band == 'accept':
            self.accepted += 1
        elif band == 'review':
            self.reviewed += 1
        elif band == 'hard_reject':
            self.hard_rejected += 1
            self.rejected += 1
        elif band == 'reject':
            self.rejected += 1
        if reason:
            self.reject_reasons[reason] += 1
        key = ml_top.replace('sexual', 'sexual_abuse') if ml_top == 'sexual' else ml_top
        if key:
            self.category_counts[key] += 1

    @property
    def average_toxicity_score(self) -> float:
        return self.toxicity_sum / max(self.documents_scanned, 1)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            'documents_scanned': self.documents_scanned,
            'accepted': self.accepted,
            'reviewed': self.reviewed,
            'rejected': self.rejected,
            'hard_rejected': self.hard_rejected,
            'average_toxicity_score': round(self.average_toxicity_score, 4),
            'category_counts': {k: self.category_counts.get(k, 0) for k in DEFAULT_CATEGORIES},
            'reject_reasons': dict(self.reject_reasons),
            'bands': dict(self.bands),
        }

def write_toxicity_reports(
    stats: ToxicityRunStats,
    *,
    output_dir: str | Path,
    report_detail: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    summary = stats.to_summary_dict()
    summary_path = out / 'toxicity_summary.json'
    summary_path.write_text(json.dumps({'created_at': now, **summary}, indent=2), encoding='utf-8')
    report = {
        'created_at': now,
        'summary': summary,
        'detail': report_detail or {},
    }
    report_path = out / 'toxicity_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    hist_path = out / 'toxicity_history.json'
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
        'toxicity_report': report_path,
        'toxicity_summary': summary_path,
        'toxicity_history': hist_path,
    }
