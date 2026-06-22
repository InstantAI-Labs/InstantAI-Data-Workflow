from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from functools import lru_cache
from typing import Any, Optional

from indw.config.loader import ConfigRef, Resolver

DEFAULT_LICENSE_SPEC = 'licensing/default'

def _require_frozenset(raw: dict[str, Any], key: str) -> frozenset[str]:
    val = raw.get(key)
    if not val:
        raise ValueError(f'{key} required in license policy config')
    return frozenset(str(x) for x in val)

def _frozenset_or(raw: dict[str, Any], key: str, default: frozenset[str]) -> frozenset[str]:
    if key not in raw:
        return default
    val = raw.get(key)
    if not val:
        return default
    return frozenset(str(x) for x in val)

@dataclass
class LicensePolicyConfig:
    enabled: bool = True
    reject_proprietary: bool = True
    reject_restricted: bool = True
    reject_paywalled: bool = True
    reject_drm: bool = True
    reject_redistribution_prohibited: bool = True
    reject_pirated_books: bool = True
    reject_incompatible_repos: bool = True
    flag_unknown: bool = True
    flag_attribution_required: bool = True
    allow_cc_by_sa: bool = True
    allow_government: bool = True
    allow_wikipedia_compatible: bool = True
    include_provenance_in_jsonl: bool = False
    min_confidence_for_reject: float = 0.70
    incompatible_repo_licenses: frozenset[str] = field(default_factory=frozenset)
    keep_licenses: frozenset[str] = field(default_factory=frozenset)
    flag_licenses: frozenset[str] = field(default_factory=frozenset)
    remove_licenses: frozenset[str] = field(default_factory=frozenset)
    output_dir: str = 'licensing'

    @classmethod
    def resolve(cls, spec: Optional[str] = None) -> LicensePolicyConfig:
        return deepcopy(_resolve_license_cached(spec or DEFAULT_LICENSE_SPEC))

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> LicensePolicyConfig:
        if not raw:
            return cls.resolve()
        defaults = cls()
        return cls(
            enabled=bool(raw.get('enabled', True)),
            reject_proprietary=bool(raw.get('reject_proprietary', True)),
            reject_restricted=bool(raw.get('reject_restricted', True)),
            reject_paywalled=bool(raw.get('reject_paywalled', True)),
            reject_drm=bool(raw.get('reject_drm', True)),
            reject_redistribution_prohibited=bool(
                raw.get('reject_redistribution_prohibited', True)
            ),
            reject_pirated_books=bool(raw.get('reject_pirated_books', True)),
            reject_incompatible_repos=bool(raw.get('reject_incompatible_repos', True)),
            flag_unknown=bool(raw.get('flag_unknown', True)),
            flag_attribution_required=bool(raw.get('flag_attribution_required', True)),
            allow_cc_by_sa=bool(raw.get('allow_cc_by_sa', True)),
            allow_government=bool(raw.get('allow_government', True)),
            allow_wikipedia_compatible=bool(raw.get('allow_wikipedia_compatible', True)),
            include_provenance_in_jsonl=bool(raw.get('include_provenance_in_jsonl', False)),
            min_confidence_for_reject=float(raw.get('min_confidence_for_reject', defaults.min_confidence_for_reject)),
            incompatible_repo_licenses=_frozenset_or(raw, 'incompatible_repo_licenses', defaults.incompatible_repo_licenses),
            keep_licenses=_frozenset_or(raw, 'keep_licenses', defaults.keep_licenses),
            flag_licenses=_frozenset_or(raw, 'flag_licenses', defaults.flag_licenses),
            remove_licenses=_frozenset_or(raw, 'remove_licenses', defaults.remove_licenses),
            output_dir=str(raw.get('output_dir') or 'licensing'),
        )

@lru_cache(maxsize=8)
def _resolve_license_cached(spec: str) -> LicensePolicyConfig:
    resolved = Resolver.default().resolve(ConfigRef(kind='safety', id=spec))
    raw = dict(resolved.raw)
    if not raw:
        return LicensePolicyConfig()
    return LicensePolicyConfig.from_dict(raw)
