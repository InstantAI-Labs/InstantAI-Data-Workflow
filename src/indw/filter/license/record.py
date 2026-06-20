from __future__ import annotations

from typing import Any, Optional

from indw.filter.license.detector import LicenseAssessment
from indw.filter.license.schema import PROVENANCE_FIELDS

def extract_row_provenance(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get('meta') if isinstance(row.get('meta'), dict) else {}
    return {
        'source': str(row.get('source') or meta.get('source') or ''),
        'url': str(row.get('url') or meta.get('url') or ''),
        'domain': str(row.get('domain') or meta.get('domain') or ''),
        'license': str(row.get('license') or meta.get('license') or meta.get('declared_license') or ''),
        'crawl_date': str(row.get('crawl_date') or meta.get('crawl_date') or ''),
        'language': str(row.get('language') or meta.get('language') or ''),
        'document_type': str(row.get('document_type') or meta.get('document_type') or ''),
        'repo_license_text': str(meta.get('repo_license_text') or ''),
        'hf_id': str(meta.get('hf_id') or ''),
    }

def build_provenance_record(
    text: str,
    assessment: LicenseAssessment,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        'text': text,
        'source': assessment.source,
        'url': assessment.url,
        'domain': assessment.domain,
        'license': assessment.license,
        'license_confidence': round(assessment.license_confidence, 4),
        'crawl_date': assessment.crawl_date,
        'language': assessment.language,
        'document_type': assessment.document_type,
        'copyright_status': assessment.copyright_status,
        'attribution_required': assessment.attribution_required,
    }
    if extra:
        record.update({k: v for k, v in extra.items() if k not in record or k == 'meta'})
    return record

def is_legacy_text_only_row(row: dict[str, Any]) -> bool:
    keys = set(row.keys()) - {'text'}
    return not keys or keys <= {'meta'}

def row_to_training_record(
    row: dict[str, Any],
    *,
    assessment: Optional[LicenseAssessment] = None,
    text: Optional[str] = None,
) -> dict[str, Any]:
    content = text if text is not None else str(row.get('text') or '')
    prov = extract_row_provenance(row)
    if assessment is not None:
        return build_provenance_record(content, assessment)
    return {
        'text': content,
        'source': prov['source'],
        'url': prov['url'],
        'domain': prov['domain'],
        'license': prov['license'] or 'Unknown',
        'license_confidence': float(row.get('license_confidence', 0.0) or 0.0),
        'crawl_date': prov['crawl_date'],
        'language': prov['language'] or str(row.get('language') or ''),
        'document_type': prov['document_type'] or str(row.get('document_type') or 'unknown'),
        'copyright_status': str(row.get('copyright_status') or 'unknown'),
        'attribution_required': bool(row.get('attribution_required', False)),
    }

def provenance_field_names() -> tuple[str, ...]:
    return PROVENANCE_FIELDS
