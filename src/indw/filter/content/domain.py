from __future__ import annotations

import re
from functools import lru_cache

from indw.config.loader import ConfigRef, Resolver

DEFAULT_DOMAIN_MAP_SPEC = 'sources/domain_map'

@lru_cache(maxsize=1)
def _domain_policy() -> dict:
    try:
        resolved = Resolver.default().resolve(ConfigRef(kind='safety', id=DEFAULT_DOMAIN_MAP_SPEC))
        return dict(resolved.raw)
    except (FileNotFoundError, KeyError, OSError):
        return {}

def _prefix_domains() -> dict[str, str]:
    raw = _domain_policy().get('prefix_domains') or {}
    return {str(k).lower(): str(v) for k, v in raw.items()}

def _structural_policy() -> dict:
    return dict(_domain_policy().get('structural_domains') or {})

def _prefix_matches_source(key: str, prefix: str) -> bool:
    idx = key.find(prefix)
    if idx < 0:
        return False
    before = key[idx - 1] if idx > 0 else '-'
    after_idx = idx + len(prefix)
    after = key[after_idx] if after_idx < len(key) else '-'
    return before in '-/' and after in '-/'

def domain_from_source(source_name: str) -> str:
    return _domain_from_source_cached(source_name.lower().replace('_', '-'))


@lru_cache(maxsize=512)
def _domain_from_source_cached(key: str) -> str:
    best: tuple[int, str] | None = None
    for prefix, domain in _prefix_domains().items():
        if _prefix_matches_source(key, prefix):
            if best is None or len(prefix) > best[0]:
                best = (len(prefix), domain)
    structural = _structural_policy()
    return best[1] if best else str(structural.get('default', 'web'))

def domain_from_text(text: str, *, source_hint: str = '') -> str:
    if source_hint:
        hinted = domain_from_source(source_hint)
        structural = _structural_policy()
        if hinted != structural.get('default', 'web'):
            return hinted
    code_threshold = int(_structural_policy().get('code_line_threshold', 3))
    code_lines = len(re.findall(r'^\s*(def |class |import |#include)', text, re.M))
    if code_lines >= code_threshold or '```' in text:
        return 'code'
    if re.search(r'<thoughts>|<reasoning>|<plan>', text):
        return 'reasoning'
    qa_threshold = int(_structural_policy().get('qa_question_threshold', 3))
    if text.count('?') >= qa_threshold and text.count('\n') < 40:
        return 'qa'
    if re.search(r'^#{1,3}\s', text, re.M):
        return 'docs'
    return domain_from_source(source_hint) if source_hint else str(_structural_policy().get('default', 'web'))
