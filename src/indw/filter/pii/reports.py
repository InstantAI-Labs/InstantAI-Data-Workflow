from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

@dataclass
class PiiRunStats:
    documents_scanned: int = 0
    entities_detected: int = 0
    secrets_detected: int = 0
    redactions_applied: int = 0
    accepted: int = 0
    redacted: int = 0
    rejected: int = 0
    hard_rejected: int = 0
    pii_sum: float = 0.0
    bands: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, assessment: Any) -> None:
        self.documents_scanned += 1
        self.pii_sum += assessment.risk.pii_score
        self.bands[assessment.risk.band] += 1
        self.entities_detected += len(assessment.entities.entities)
        self.secrets_detected += len(assessment.secrets.spans)
        if assessment.redacted_text is not None:
            self.redactions_applied += 1
        band = assessment.risk.band
        if band == 'accept':
            self.accepted += 1
        elif band == 'redact':
            self.redacted += 1
        elif band == 'hard_reject':
            self.hard_rejected += 1
            self.rejected += 1
        elif band == 'reject':
            self.rejected += 1
        if assessment.risk.reason:
            self.reasons[assessment.risk.reason] += 1

    @property
    def average_pii_score(self) -> float:
        return self.pii_sum / max(self.documents_scanned, 1)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            'documents_scanned': self.documents_scanned,
            'entities_detected': self.entities_detected,
            'secrets_detected': self.secrets_detected,
            'redactions_applied': self.redactions_applied,
            'accepted': self.accepted,
            'redacted': self.redacted,
            'rejected_documents': self.rejected,
            'hard_rejected_documents': self.hard_rejected,
            'average_pii_score': round(self.average_pii_score, 4),
            'bands': dict(self.bands),
            'reasons': dict(self.reasons),
        }

def write_pii_reports(
    stats: PiiRunStats,
    *,
    output_dir: str | Path,
    report_detail: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    summary = stats.to_summary_dict()
    summary_path = out / 'pii_summary.json'
    summary_path.write_text(json.dumps({'created_at': now, **summary}, indent=2), encoding='utf-8')
    report = {'created_at': now, 'summary': summary, 'detail': report_detail or {}}
    report_path = out / 'pii_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    hist_path = out / 'pii_history.json'
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
        'pii_report': report_path,
        'pii_summary': summary_path,
        'pii_history': hist_path,
    }
