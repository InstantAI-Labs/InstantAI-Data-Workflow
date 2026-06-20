from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

from indw.clean.document.stats import StageStats

_TAG_DENSITY = re.compile(r'<[a-zA-Z!/]')
_NUMERIC_ENTITY = re.compile(r'&#(\d+);')
_HEX_ENTITY = re.compile(r'&#x([0-9a-fA-F]+);')
_MAX_UNICODE = 0x10FFFF

_FALLBACK_SCRIPT_STYLE = re.compile(r'<(script|style|noscript)[^>]*>.*?</\1>', re.I | re.S)
_FALLBACK_BREAK = re.compile(r'<br\s*/?>', re.I)
_FALLBACK_TAG = re.compile(r'<[^>]+>')
_FALLBACK_ENTITIES = {
    '&nbsp;': ' ',
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#39;': "'",
    '&apos;': "'",
}

_TRAFILATURA_OPTS: dict[str, Any] = {
    'include_comments': False,
    'include_tables': True,
    'include_images': False,
    'include_links': False,
    'deduplicate': True,
    'favor_precision': True,
    'no_fallback': False,
}


@dataclass
class HtmlExtractMeta:
    title: str = ''
    author: str = ''
    date: str = ''
    url: str = ''
    hostname: str = ''
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    extractor: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'title': self.title,
            'author': self.author,
            'date': self.date,
            'url': self.url,
            'hostname': self.hostname,
            'categories': list(self.categories),
            'tags': list(self.tags),
            'extractor': self.extractor,
        }


def looks_like_html(text: str) -> bool:
    if not text or '<' not in text:
        return False
    return _TAG_DENSITY.search(text) is not None


def _decode_codepoint(raw: str, *, base: int = 10) -> str:
    try:
        code = int(raw, base)
    except ValueError:
        return ''
    if 0 <= code <= _MAX_UNICODE:
        try:
            return chr(code)
        except (ValueError, OverflowError):
            pass
    return ''


def _replace_numeric_entity(match: re.Match[str]) -> str:
    return _decode_codepoint(match.group(1))


def _replace_hex_entity(match: re.Match[str]) -> str:
    return _decode_codepoint(match.group(1), base=16)


def _fallback_regex_clean(text: str) -> str:
    out = _FALLBACK_SCRIPT_STYLE.sub(' ', text)
    out = _FALLBACK_BREAK.sub('\n', out)
    out = _FALLBACK_TAG.sub(' ', out)
    for entity, repl in _FALLBACK_ENTITIES.items():
        out = out.replace(entity, repl)
    out = _NUMERIC_ENTITY.sub(_replace_numeric_entity, out)
    out = _HEX_ENTITY.sub(_replace_hex_entity, out)
    return html.unescape(out)


def _capture_metadata(raw_html: str) -> HtmlExtractMeta:
    meta = HtmlExtractMeta(extractor='trafilatura')
    try:
        from trafilatura.metadata import extract_metadata
    except ImportError:
        return meta
    try:
        parsed = extract_metadata(raw_html)
    except Exception:
        return meta
    if parsed is None:
        return meta
    meta.title = str(getattr(parsed, 'title', '') or '')
    meta.author = str(getattr(parsed, 'author', '') or '')
    meta.date = str(getattr(parsed, 'date', '') or '')
    meta.url = str(getattr(parsed, 'url', '') or '')
    meta.hostname = str(getattr(parsed, 'hostname', '') or '')
    cats = getattr(parsed, 'categories', None) or []
    tags = getattr(parsed, 'tags', None) or []
    meta.categories = [str(c) for c in cats if c]
    meta.tags = [str(t) for t in tags if t]
    return meta


def _collapse_consecutive_duplicate_lines(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    out: list[str] = []
    prev_key = ''
    for ln in lines:
        key = ln.strip()
        if key and key == prev_key:
            continue
        out.append(ln)
        if key:
            prev_key = key
    return '\n'.join(out)


def _extract_with_trafilatura(text: str) -> tuple[str | None, HtmlExtractMeta]:
    try:
        import trafilatura
    except ImportError:
        return None, HtmlExtractMeta(extractor='unavailable')
    meta = _capture_metadata(text)
    try:
        body = trafilatura.extract(text, **_TRAFILATURA_OPTS)
    except Exception:
        return None, meta
    if body and body.strip():
        meta.extractor = 'trafilatura'
        return _collapse_consecutive_duplicate_lines(body.strip()), meta
    return None, meta


def extract_html_document(
    text: str,
    *,
    metadata_out: list[HtmlExtractMeta] | None = None,
) -> str:
    if not text or not looks_like_html(text):
        return text
    extracted, meta = _extract_with_trafilatura(text)
    if metadata_out is not None:
        metadata_out.append(meta)
    if extracted:
        return extracted
    fallback = _fallback_regex_clean(text)
    if metadata_out is not None and metadata_out:
        metadata_out[-1] = HtmlExtractMeta(extractor='regex_fallback')
    return fallback


def clean_html(
    text: str,
    *,
    stats: StageStats | None = None,
    metadata_out: list[HtmlExtractMeta] | None = None,
) -> str:
    if not text or not looks_like_html(text):
        return text
    before = len(text)
    out = extract_html_document(text, metadata_out=metadata_out)
    if stats is not None:
        stats.in_docs += 1
        stats.chars_removed += max(0, before - len(out))
        stats.out_docs += 1
    return out
