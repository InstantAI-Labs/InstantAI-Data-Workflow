from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from indw.filter.license.classifier import (
    classify_book_copyright,
    classify_document_type,
    classify_news_signals,
)
from indw.filter.license.normalize import detect_license_in_text, normalize_license_string
from indw.filter.license.policy import (
    LicenseFilterDecision,
    attribution_required_for_license,
    copyright_status_for_license,
    decide_license_filter,
)
from indw.filter.license.config import LicensePolicyConfig
from indw.filter.license.source_policy import (
    extract_domain,
    is_government_domain,
    lookup_domain_license,
    lookup_source_license,
    merge_license_candidates,
    parse_repo_license_file,
    resolve_license_source_policy,
)
from indw.filter.license.schema import DocumentType, FilterAction

@dataclass
class LicenseAssessment:
    license: str = 'Unknown'
    license_confidence: float = 0.0
    license_origin: str = 'none'
    source: str = ''
    url: str = ''
    domain: str = ''
    crawl_date: str = ''
    language: str = ''
    document_type: DocumentType = 'unknown'
    copyright_status: str = 'unknown'
    attribution_required: bool = False
    filter_action: FilterAction = 'FLAG'
    filter_reason: str = 'unknown_license'
    reject_reason: str = ''
    flags: list[str] = field(default_factory=list)
    paywall_detected: bool = False
    drm_detected: bool = False
    redistribution_prohibited: bool = False
    proprietary_notice: bool = False
    syndicated_news: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'license': self.license,
            'license_confidence': round(self.license_confidence, 4),
            'license_origin': self.license_origin,
            'source': self.source,
            'url': self.url,
            'domain': self.domain,
            'crawl_date': self.crawl_date,
            'language': self.language,
            'document_type': self.document_type,
            'copyright_status': self.copyright_status,
            'attribution_required': self.attribution_required,
            'filter_action': self.filter_action,
            'filter_reason': self.filter_reason,
            'reject_reason': self.reject_reason,
            'flags': list(self.flags),
            'paywall_detected': self.paywall_detected,
            'drm_detected': self.drm_detected,
            'redistribution_prohibited': self.redistribution_prohibited,
            'proprietary_notice': self.proprietary_notice,
            'syndicated_news': self.syndicated_news,
        }

class LicenseDetector:
    def __init__(self, policy: Optional[LicensePolicyConfig] = None):
        self.policy = policy or LicensePolicyConfig()

    def assess(
        self,
        text: str,
        *,
        source: str = '',
        url: str = '',
        domain: str = '',
        language: str = '',
        crawl_date: str = '',
        declared_license: str = '',
        repo_license_text: str = '',
        hf_id: str = '',
        piracy_score: float = 0.0,
        meta: Optional[dict[str, Any]] = None,
    ) -> LicenseAssessment:
        meta = meta or {}
        url = url or str(meta.get('url') or '')
        domain = domain or str(meta.get('domain') or '') or extract_domain(url)
        crawl_date = crawl_date or str(meta.get('crawl_date') or '') or datetime.now(timezone.utc).strftime('%Y-%m-%d')
        declared = declared_license or str(meta.get('license') or meta.get('declared_license') or '')

        document_type = classify_document_type(
            source=source,
            url=url,
            domain=domain,
            text=text,
        )

        candidates: list[tuple[str, float, str]] = []

        if declared:
            lic, conf = normalize_license_string(declared)
            if lic != 'Unknown':
                candidates.append((lic, conf, 'source_declared'))

        src_lic, src_conf = lookup_source_license(source, hf_id=hf_id or str(meta.get('hf_id') or ''))
        if src_lic != 'Unknown':
            candidates.append((src_lic, src_conf, 'source_declared'))

        dom_lic, dom_conf = lookup_domain_license(domain)
        if dom_lic != 'Unknown':
            candidates.append((dom_lic, dom_conf, 'domain'))

        if repo_license_text:
            repo_lic, repo_conf = parse_repo_license_file(repo_license_text)
            if repo_lic != 'Unknown':
                candidates.append((repo_lic, repo_conf, 'repo_file'))

        if document_type == 'book':
            book_lic, book_conf = classify_book_copyright(text, domain=domain)
            if book_lic:
                candidates.append((book_lic, book_conf, 'text_explicit'))

        text_lic, text_conf = detect_license_in_text(text)
        if text_lic != 'Unknown':
            candidates.append((text_lic, text_conf, 'text_explicit'))

        license_label, license_conf, license_origin = merge_license_candidates(candidates)

        patterns = resolve_license_source_policy().patterns
        paywall = bool(patterns['paywall'].search(text[:6000]))
        drm = bool(patterns['drm'].search(text[:6000]))
        redist = bool(patterns['redistribution_prohibited'].search(text[:8000]))
        proprietary = bool(patterns['proprietary_notice'].search(text[:6000]))
        news = classify_news_signals(text)
        syndicated = news['syndicated']

        decision = decide_license_filter(
            license_label=license_label,
            license_confidence=license_conf,
            document_type=document_type,
            policy=self.policy,
            paywall_detected=paywall,
            drm_detected=drm,
            redistribution_prohibited=redist,
            piracy_score=piracy_score,
            proprietary_notice=proprietary,
            syndicated_news=syndicated,
            is_government=is_government_domain(domain) or document_type == 'government',
        )

        status = copyright_status_for_license(license_label)
        if decision.copyright_status != 'unknown':
            status = decision.copyright_status

        return LicenseAssessment(
            license=license_label,
            license_confidence=license_conf,
            license_origin=license_origin,
            source=source,
            url=url,
            domain=domain,
            crawl_date=crawl_date,
            language=language,
            document_type=document_type,
            copyright_status=status,
            attribution_required=decision.attribution_required or attribution_required_for_license(license_label),
            filter_action=decision.action,
            filter_reason=decision.reason,
            reject_reason=decision.reject_reason,
            flags=list(decision.flags),
            paywall_detected=paywall,
            drm_detected=drm,
            redistribution_prohibited=redist,
            proprietary_notice=proprietary,
            syndicated_news=syndicated,
        )
