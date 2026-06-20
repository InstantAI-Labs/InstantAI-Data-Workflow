from __future__ import annotations

import re
from typing import Optional

from indw.filter.license.source_policy import (
    extract_domain,
    is_government_domain,
    is_repo_host,
    resolve_license_source_policy,
)
from indw.filter.license.schema import DocumentType

_WIKI_MARKERS = re.compile(
    r'(?i)\b(?:wikipedia|wikimedia|wikidata|mediawiki)\b',
)

def classify_document_type(
    *,
    source: str = '',
    url: str = '',
    domain: str = '',
    text: str = '',
) -> DocumentType:
    pol = resolve_license_source_policy()
    dom = domain or extract_domain(url)
    src = (source or '').lower()
    sample = (text or '')[:4000]
    news_pat = pol.patterns['news_host']

    if is_repo_host(dom) or 'github' in src or 'gitlab' in src or 'sourceforge' in src:
        return 'code_repository'
    if dom in pol.book_hosts or 'gutenberg' in src or 'archive.org' in dom:
        return 'book'
    if is_government_domain(dom):
        return 'government'
    if _WIKI_MARKERS.search(sample) or 'wikipedia' in dom or src.startswith('wikipedia'):
        return 'wiki'
    if news_pat.search(dom) or news_pat.search(sample[:800]):
        return 'news'
    if re.search(r'(?i)\b(?:abstract|doi:|arxiv:|journal|proceedings)\b', sample):
        return 'academic'
    if re.search(r'(?i)\b(?:posted by|upvoted|subreddit|stack\s+overflow)\b', sample):
        return 'forum'
    if dom:
        return 'web'
    return 'unknown'

def classify_book_copyright(text: str, *, domain: str = '') -> tuple[Optional[str], float]:
    pol = resolve_license_source_policy()
    sample = (text or '')[:8000]
    pd_markers = pol.patterns['pd_book_markers']
    commercial = pol.patterns['commercial_book']
    if pd_markers.search(sample) or domain in pol.pd_book_domains:
        return 'Public Domain', 0.94
    if commercial.search(sample) and not pd_markers.search(sample):
        return 'Proprietary', 0.72
    return None, 0.0

def classify_news_signals(text: str) -> dict[str, bool]:
    syndicated = resolve_license_source_policy().patterns['syndicated']
    sample = (text or '')[:6000]
    return {
        'syndicated': bool(syndicated.search(sample)),
        'has_copyright_notice': bool(
            re.search(r'(?i)(?:©|copyright)\s*\d{4}', sample)
        ),
    }
