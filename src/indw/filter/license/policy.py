from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from indw.filter.license.config import LicensePolicyConfig
from indw.filter.license.schema import CopyrightStatus, FilterAction

@dataclass
class LicenseFilterDecision:
    action: FilterAction = 'KEEP'
    reason: str = ''
    copyright_status: CopyrightStatus = 'unknown'
    attribution_required: bool = False
    flags: list[str] = field(default_factory=list)
    reject_reason: str = ''

    @property
    def should_reject(self) -> bool:
        return self.action == 'REMOVE'

    @property
    def should_flag(self) -> bool:
        return self.action == 'FLAG'

def copyright_status_for_license(license_label: str) -> CopyrightStatus:
    if license_label in ('Public Domain', 'CC0'):
        return 'clear'
    if license_label in ('CC-BY', 'MIT', 'Apache-2.0', 'BSD'):
        return 'attribution_required'
    if license_label == 'CC-BY-SA':
        return 'share_alike'
    if license_label in ('Proprietary', 'Restricted', 'GPL'):
        return 'prohibited'
    if license_label == 'Unknown':
        return 'unknown'
    return 'restricted'

def attribution_required_for_license(license_label: str) -> bool:
    return license_label in (
        'CC-BY',
        'CC-BY-SA',
        'CC-BY-NC',
        'MIT',
        'Apache-2.0',
        'BSD',
        'MPL',
        'LGPL',
    )

def decide_license_filter(
    *,
    license_label: str,
    license_confidence: float,
    document_type: str,
    policy: LicensePolicyConfig,
    paywall_detected: bool = False,
    drm_detected: bool = False,
    redistribution_prohibited: bool = False,
    piracy_score: float = 0.0,
    proprietary_notice: bool = False,
    syndicated_news: bool = False,
    is_government: bool = False,
) -> LicenseFilterDecision:
    flags: list[str] = []
    status = copyright_status_for_license(license_label)
    attribution = attribution_required_for_license(license_label)

    if piracy_score >= 0.18 and policy.reject_pirated_books:
        return LicenseFilterDecision(
            action='REMOVE',
            reason='pirated_content',
            copyright_status='prohibited',
            attribution_required=False,
            flags=['pirated_content'],
            reject_reason='pirated_content',
        )

    if policy.reject_paywalled and paywall_detected:
        return LicenseFilterDecision(
            action='REMOVE',
            reason='paywalled',
            copyright_status='restricted',
            attribution_required=attribution,
            flags=['paywalled'],
            reject_reason='paywalled',
        )

    if policy.reject_drm and drm_detected:
        return LicenseFilterDecision(
            action='REMOVE',
            reason='drm_protected',
            copyright_status='prohibited',
            attribution_required=False,
            flags=['drm_protected'],
            reject_reason='drm_protected',
        )

    if policy.reject_redistribution_prohibited and redistribution_prohibited:
        return LicenseFilterDecision(
            action='REMOVE',
            reason='redistribution_prohibited',
            copyright_status='prohibited',
            attribution_required=False,
            flags=['redistribution_prohibited'],
            reject_reason='redistribution_prohibited',
        )

    conf_ok = license_confidence >= policy.min_confidence_for_reject

    if (
        policy.reject_proprietary
        and license_label == 'Proprietary'
        and (conf_ok or proprietary_notice)
    ):
        return LicenseFilterDecision(
            action='REMOVE',
            reason='proprietary_license',
            copyright_status='prohibited',
            attribution_required=False,
            flags=['proprietary'],
            reject_reason='proprietary_license',
        )

    if policy.reject_restricted and license_label == 'Restricted' and conf_ok:
        return LicenseFilterDecision(
            action='REMOVE',
            reason='restricted_license',
            copyright_status='restricted',
            attribution_required=attribution,
            flags=['restricted'],
            reject_reason='restricted_license',
        )

    if (
        policy.reject_incompatible_repos
        and document_type == 'code_repository'
        and license_label in policy.incompatible_repo_licenses
        and conf_ok
    ):
        return LicenseFilterDecision(
            action='REMOVE',
            reason='incompatible_repo_license',
            copyright_status='prohibited',
            attribution_required=False,
            flags=['incompatible_repo'],
            reject_reason='incompatible_repo_license',
        )

    if license_label in policy.remove_licenses and conf_ok:
        if license_label == 'GPL' and document_type == 'code_repository':
            return LicenseFilterDecision(
                action='REMOVE',
                reason='incompatible_repo_license',
                copyright_status='prohibited',
                attribution_required=False,
                flags=['gpl_repo'],
                reject_reason='incompatible_repo_license',
            )

    if is_government and policy.allow_government:
        return LicenseFilterDecision(
            action='KEEP',
            reason='government_publication',
            copyright_status='clear',
            attribution_required=False,
            flags=['government'],
        )

    if license_label == 'CC-BY-SA' and policy.allow_wikipedia_compatible:
        if document_type in ('wiki', 'web'):
            return LicenseFilterDecision(
                action='KEEP',
                reason='wikipedia_compatible',
                copyright_status='share_alike',
                attribution_required=True,
                flags=['share_alike'],
            )

    if license_label in policy.keep_licenses:
        return LicenseFilterDecision(
            action='KEEP',
            reason='permitted_license',
            copyright_status=status,
            attribution_required=attribution,
            flags=flags,
        )

    if license_label == 'CC-BY-SA' and policy.allow_cc_by_sa:
        return LicenseFilterDecision(
            action='KEEP',
            reason='share_alike_permitted',
            copyright_status='share_alike',
            attribution_required=True,
            flags=['share_alike'],
        )

    if license_label == 'Unknown' and policy.flag_unknown:
        flags.append('unknown_license')
        return LicenseFilterDecision(
            action='FLAG',
            reason='unknown_license',
            copyright_status='unknown',
            attribution_required=False,
            flags=flags,
        )

    if attribution and policy.flag_attribution_required:
        flags.append('attribution_required')
        return LicenseFilterDecision(
            action='FLAG',
            reason='attribution_required',
            copyright_status=status,
            attribution_required=True,
            flags=flags,
        )

    if license_label in policy.flag_licenses:
        flags.append(f'license_{license_label.lower().replace("-", "_")}')
        return LicenseFilterDecision(
            action='FLAG',
            reason=f'flagged_license_{license_label}',
            copyright_status=status,
            attribution_required=attribution,
            flags=flags,
        )

    if syndicated_news and document_type == 'news':
        flags.append('syndicated_news')
        return LicenseFilterDecision(
            action='FLAG',
            reason='syndicated_news',
            copyright_status=status,
            attribution_required=True,
            flags=flags,
        )

    return LicenseFilterDecision(
        action='FLAG',
        reason='unknown_license',
        copyright_status='unknown',
        attribution_required=False,
        flags=['unknown_license'],
    )
