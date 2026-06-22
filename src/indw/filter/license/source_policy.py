from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from indw.config.loader import ConfigRef, Resolver, thaw

DEFAULT_SOURCE_POLICY_SPEC = 'licensing/sources'

def _license_map(raw: dict[str, Any]) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for key, entry in (raw or {}).items():
        if not isinstance(entry, dict):
            raise ValueError(f'license entry for {key} must be a mapping')
        if 'license' not in entry or 'confidence' not in entry:
            raise ValueError(f'license entry for {key} requires license and confidence')
        out[str(key)] = (str(entry['license']), float(entry['confidence']))
    return out

def _compile_patterns(raw: dict[str, Any]) -> dict[str, re.Pattern[str]]:
    patterns = raw.get('text_patterns') or {}
    if not patterns:
        raise ValueError('text_patterns required in license source policy')
    return {name: re.compile(str(expr)) for name, expr in patterns.items()}

@dataclass(frozen=True)
class LicenseSourcePolicy:
    source_licenses: dict[str, tuple[str, float]]
    domain_licenses: dict[str, tuple[str, float]]
    gov_tld_suffixes: tuple[str, ...]
    repo_hosts: frozenset[str]
    book_hosts: frozenset[str]
    pd_book_domains: frozenset[str]
    wikipedia_prefix_license: str
    wikipedia_prefix_confidence: float
    government_domain_license: str
    government_domain_confidence: float
    patterns: dict[str, re.Pattern[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LicenseSourcePolicy:
        wiki = raw.get('wikipedia_prefix') or {}
        gov = raw.get('government_domain') or {}
        if 'license' not in wiki or 'confidence' not in wiki:
            raise ValueError('wikipedia_prefix requires license and confidence')
        if 'license' not in gov or 'confidence' not in gov:
            raise ValueError('government_domain requires license and confidence')
        gov_suffixes = raw.get('gov_tld_suffixes')
        repo_hosts = raw.get('repo_hosts')
        book_hosts = raw.get('book_hosts')
        if not gov_suffixes or not repo_hosts or not book_hosts:
            raise ValueError('gov_tld_suffixes, repo_hosts, and book_hosts are required')
        return cls(
            source_licenses=_license_map(raw.get('source_licenses') or {}),
            domain_licenses=_license_map(raw.get('domain_licenses') or {}),
            gov_tld_suffixes=tuple(str(x) for x in gov_suffixes),
            repo_hosts=frozenset(str(x) for x in repo_hosts),
            book_hosts=frozenset(str(x) for x in book_hosts),
            pd_book_domains=frozenset(str(x) for x in (raw.get('pd_book_domains') or [])),
            wikipedia_prefix_license=str(wiki['license']),
            wikipedia_prefix_confidence=float(wiki['confidence']),
            government_domain_license=str(gov['license']),
            government_domain_confidence=float(gov['confidence']),
            patterns=_compile_patterns(raw),
        )

@lru_cache(maxsize=1)
def resolve_license_source_policy() -> LicenseSourcePolicy:
    resolved = Resolver.default().resolve(ConfigRef(kind='safety', id=DEFAULT_SOURCE_POLICY_SPEC))
    return LicenseSourcePolicy.from_dict(thaw(resolved.raw))

def extract_domain(url: str) -> str:
    if not url:
        return ''
    raw = url.strip()
    if not raw.startswith(('http://', 'https://')):
        raw = 'https://' + raw.lstrip('/')
    try:
        host = urlparse(raw).netloc.lower()
    except ValueError:
        return ''
    return host[4:] if host.startswith('www.') else host

def lookup_source_license(source: str, *, hf_id: str = '') -> tuple[str, float]:
    pol = resolve_license_source_policy()
    if source in pol.source_licenses:
        return pol.source_licenses[source]
    if hf_id in pol.source_licenses:
        return pol.source_licenses[hf_id]
    key = source.lower().replace('_', '-')
    if key.startswith('wikipedia'):
        return pol.wikipedia_prefix_license, pol.wikipedia_prefix_confidence
    return 'Unknown', 0.0

def lookup_domain_license(domain: str) -> tuple[str, float]:
    pol = resolve_license_source_policy()
    if not domain:
        return 'Unknown', 0.0
    d = domain.lower()
    if d in pol.domain_licenses:
        return pol.domain_licenses[d]
    for suffix in pol.gov_tld_suffixes:
        if d.endswith(suffix):
            return pol.government_domain_license, pol.government_domain_confidence
    for host, lic in pol.domain_licenses.items():
        if d == host or d.endswith('.' + host):
            return lic
    return 'Unknown', 0.0

def is_government_domain(domain: str) -> bool:
    d = (domain or '').lower()
    return any(d.endswith(sfx) for sfx in resolve_license_source_policy().gov_tld_suffixes)

def is_repo_host(domain: str) -> bool:
    d = (domain or '').lower()
    hosts = resolve_license_source_policy().repo_hosts
    return d in hosts or any(d.endswith('.' + h) for h in hosts)

def merge_license_candidates(
    candidates: list[tuple[str, float, str]],
) -> tuple[str, float, str]:
    best = ('Unknown', 0.0, 'none')
    priority = {'source_declared': 3, 'domain': 2, 'repo_file': 2, 'text_explicit': 1, 'none': 0}
    for lic, conf, origin in candidates:
        if conf <= 0:
            continue
        cur_pri = priority.get(best[2], 0)
        new_pri = priority.get(origin, 0)
        if conf > best[1] or (conf == best[1] and new_pri > cur_pri):
            best = (lic, conf, origin)
    return best

def source_meta_from_yaml_entry(entry: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if entry.get('license'):
        meta['declared_license'] = str(entry['license'])
    if entry.get('hf_id'):
        meta['hf_id'] = str(entry['hf_id'])
    if entry.get('url'):
        meta['url'] = str(entry['url'])
    if entry.get('domain'):
        meta['domain'] = str(entry['domain'])
    return meta

def parse_repo_license_file(content: str) -> tuple[str, float]:
    from indw.filter.license.normalize import detect_license_in_text, normalize_license_string

    if not content:
        return 'Unknown', 0.0
    lic, conf = detect_license_in_text(content[:8000])
    if lic != 'Unknown':
        return lic, conf
    first_line = content.strip().splitlines()[0] if content.strip() else ''
    return normalize_license_string(first_line)
