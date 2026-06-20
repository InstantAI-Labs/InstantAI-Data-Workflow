from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from indw.filter.license.schema import LICENSE_CATEGORIES, PIPELINE_VERSION

@dataclass
class LicenseRunStats:
    documents_scanned: int = 0
    kept: int = 0
    flagged: int = 0
    removed: int = 0
    unknown_license: int = 0
    attribution_required: int = 0
    by_license: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_domain: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_language: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_document_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    remove_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    flag_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_chars: int = 0
    est_tokens: int = 0

    def record(self, assessment: Any, *, text_len: int = 0, kept: bool = True) -> None:
        self.documents_scanned += 1
        lic = getattr(assessment, 'license', 'Unknown') or 'Unknown'
        self.by_license[lic] += 1
        dom = getattr(assessment, 'domain', '') or 'unknown'
        self.by_domain[dom] += 1
        lang = getattr(assessment, 'language', '') or 'unknown'
        self.by_language[lang] += 1
        dtype = getattr(assessment, 'document_type', 'unknown') or 'unknown'
        self.by_document_type[dtype] += 1
        src = getattr(assessment, 'source', '') or 'unknown'
        self.by_source[src] += 1
        if lic == 'Unknown':
            self.unknown_license += 1
        if getattr(assessment, 'attribution_required', False):
            self.attribution_required += 1
        action = getattr(assessment, 'filter_action', 'KEEP')
        if action == 'REMOVE':
            self.removed += 1
            reason = getattr(assessment, 'reject_reason', '') or getattr(assessment, 'filter_reason', '')
            if reason:
                self.remove_reasons[reason] += 1
        elif action == 'FLAG':
            self.flagged += 1
            reason = getattr(assessment, 'filter_reason', '')
            if reason:
                self.flag_reasons[reason] += 1
        if kept and action != 'REMOVE':
            self.kept += 1
            self.total_chars += text_len
            self.est_tokens += int(text_len / 3.8)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            'version': PIPELINE_VERSION,
            'documents_scanned': self.documents_scanned,
            'kept': self.kept,
            'flagged': self.flagged,
            'removed': self.removed,
            'unknown_license': self.unknown_license,
            'attribution_required': self.attribution_required,
            'by_license': {k: self.by_license[k] for k in LICENSE_CATEGORIES if self.by_license.get(k)},
            'by_license_other': {
                k: v for k, v in self.by_license.items() if k not in LICENSE_CATEGORIES
            },
            'by_domain': dict(sorted(self.by_domain.items(), key=lambda x: -x[1])[:50]),
            'by_language': dict(sorted(self.by_language.items(), key=lambda x: -x[1])[:30]),
            'by_document_type': dict(self.by_document_type),
            'by_source': dict(sorted(self.by_source.items(), key=lambda x: -x[1])[:30]),
            'remove_reasons': dict(self.remove_reasons),
            'flag_reasons': dict(self.flag_reasons),
            'total_chars': self.total_chars,
            'est_tokens': self.est_tokens,
            'generated_at': datetime.now(timezone.utc).isoformat(),
        }

def write_license_reports(
    stats: LicenseRunStats,
    *,
    output_dir: Path,
    report_detail: Optional[dict[str, Any]] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = stats.to_summary_dict()
    if report_detail:
        summary['detail'] = report_detail
    out = output_dir / 'license_audit_report.json'
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    history_dir = output_dir / 'history'
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    hist = history_dir / f'run_{stamp}.json'
    hist.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return out
