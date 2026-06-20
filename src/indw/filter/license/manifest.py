from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from indw.filter.license.reports import LicenseRunStats
from indw.filter.license.schema import PIPELINE_VERSION, PROVENANCE_JSON_SCHEMA

def build_dataset_manifest(
    stats: LicenseRunStats,
    *,
    corpus_path: Optional[Path] = None,
    pipeline_version: str = PIPELINE_VERSION,
    filtering_decisions: Optional[dict[str, Any]] = None,
    source_distribution: Optional[dict[str, float]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    summary = stats.to_summary_dict()
    license_dist = {
        k: {
            'count': summary['by_license'].get(k, 0),
            'fraction': round(
                summary['by_license'].get(k, 0) / max(stats.documents_scanned, 1),
                6,
            ),
        }
        for k in summary.get('by_license', {})
    }
    manifest: dict[str, Any] = {
        'version': pipeline_version,
        'schema': PROVENANCE_JSON_SCHEMA['$id'],
        'build_date': datetime.now(timezone.utc).isoformat(),
        'pipeline_version': pipeline_version,
        'corpus_path': str(corpus_path) if corpus_path else '',
        'total_documents': stats.kept,
        'total_tokens': stats.est_tokens,
        'total_chars': stats.total_chars,
        'license_distribution': license_dist,
        'source_distribution': source_distribution or {
            k: round(v / max(stats.kept, 1), 6) for k, v in stats.by_source.items()
        },
        'document_type_distribution': dict(stats.by_document_type),
        'language_distribution': dict(stats.by_language),
        'filtering_decisions': filtering_decisions or {
            'kept': stats.kept,
            'flagged': stats.flagged,
            'removed': stats.removed,
            'remove_reasons': dict(stats.remove_reasons),
            'flag_reasons': dict(stats.flag_reasons),
            'unknown_licenses': stats.unknown_license,
            'attribution_required': stats.attribution_required,
        },
        'domains_top': dict(list(sorted(stats.by_domain.items(), key=lambda x: -x[1]))[:25]),
    }
    if extra:
        manifest.update(extra)
    return manifest

def write_dataset_manifest(
    stats: LicenseRunStats,
    path: Path,
    **kwargs: Any,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_dataset_manifest(stats, corpus_path=path.parent / 'filtered.jsonl', **kwargs)
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return path
