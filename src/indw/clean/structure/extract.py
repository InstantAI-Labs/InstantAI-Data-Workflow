from __future__ import annotations

import re

from indw.clean.document.stats import StageStats

_GLUE_TITLE = re.compile(
    r'^([A-Z][A-Za-z0-9\'\-\.]+(?:\s+[A-Z][A-Za-z0-9\'\-\.]+){0,8})'
    r'(?:Article|Category|Portal|Template|File|Help:)'
    r'(?:\s*(?:Free\s+Pass|Main\s+Page|Special:))?\s*'
    r'(.+)$',
    re.S,
)
_NAV_GLUE = re.compile(
    r'(?i)\b(?:article\s+free\s+pass|main\s+page|special:|category:|portal:)\b'
)
_SECTION_HEADING = re.compile(r'(?m)^(?:#{1,6}\s+.+|={2,}\s*.+\s*={2,})\s*$')
_WIKI_SECTION = re.compile(r'(?m)^==\s*(.+?)\s*==\s*$')

def _dedupe_title_in_body(text: str) -> str:
    m = re.match(r'(?is)^title\s*:\s*(.+?)\s*\n+\s*(.*)$', text)
    if not m:
        return text
    title = m.group(1).strip()
    body = m.group(2).strip()
    if not body:
        return text
    title_low = title.lower()
    lines = body.splitlines()
    cleaned: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            cleaned.append('')
            continue
        low = stripped.lower()
        if i == 0:
            if low == title_low:
                continue
            if low.startswith(f'{title_low} '):
                remainder = stripped[len(title):].strip(' .:;-')
                if re.match(r'(?i)^(are|is|was|were)\s+', remainder):
                    remainder = re.sub(r'(?i)^(are|is|was|were)\s+', '', remainder)
                if remainder:
                    cleaned.append(remainder[0].upper() + remainder[1:] if remainder else remainder)
                continue
            if low.startswith(title_low) and len(stripped) < len(title) + 80:
                remainder = stripped[len(title):].strip(' .:;-')
                if remainder:
                    cleaned.append(remainder if remainder[0].isupper() else f'The {title} {remainder}')
                    continue
        cleaned.append(line)
    return f'Title: {title}\n\n' + '\n'.join(cleaned).strip()

def _normalize_headings(text: str) -> str:
    def _wiki_heading(m: re.Match[str]) -> str:
        return f'\n## {m.group(1).strip()}\n'

    text = _WIKI_SECTION.sub(_wiki_heading, text)
    return text

def extract_structure(
    text: str,
    *,
    source: str = '',
    stats: StageStats | None = None,
    generic_only: bool = False,
) -> str:
    if not text or not text.strip():
        return text
    out = text.strip()
    if generic_only:
        out = _normalize_headings(out)
        out = re.sub(r'\n{3,}', '\n\n', out).strip()
        if stats is not None:
            stats.in_docs += 1
            stats.out_docs += 1 if out else 0
        return out

    if re.match(r'(?i)^title\s*:', out):
        out = _dedupe_title_in_body(out)
        out = _normalize_headings(out)
        return out

    m = _GLUE_TITLE.match(out)
    if m:
        title = m.group(1).strip()
        body = m.group(2).strip()
        body = _NAV_GLUE.sub(' ', body)
        body = '\n'.join(re.sub(r'[ \t]+', ' ', ln).strip() for ln in body.splitlines() if ln.strip())
        if body.lower().startswith(f'{title.lower()} {title.lower()}'):
            body = body[len(title):].strip(' .:;-')
        elif body.lower().startswith(title.lower()):
            remainder = body[len(title):].strip(' .:;-')
            if remainder and remainder[0].islower():
                body = f'The {title} {remainder}'
            else:
                body = remainder or body
        out = f'Title: {title}\n\n{body}'

    lines = out.split('\n', 1)
    if len(lines) == 2 and not out.lower().startswith('title:'):
        first, body = lines[0].strip(), lines[1].strip()
        title_like = (
            3 <= len(first.split()) <= 14
            and len(first) < 140
            and not first.endswith(('.', '!', '?'))
            and not first.startswith(('#', '```', 'http'))
            and not _NAV_GLUE.search(first)
            and not re.match(r'(?i)^(?:question|answer|q|a)\s*:', first)
        )
        if title_like and body and not body.lower().startswith(first.lower()[: min(20, len(first))]):
            out = f'Title: {first}\n\n{body}'

    out = _dedupe_title_in_body(out)
    out = _normalize_headings(out)
    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    if stats is not None:
        stats.in_docs += 1
        stats.out_docs += 1 if out else 0
    return out
