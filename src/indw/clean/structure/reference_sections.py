from __future__ import annotations

import re

from indw.clean.document.stats import StageStats

_SECTION_DROP = re.compile(
    r'(?im)^\s*(?:={1,6}\s*)?'
    r'(?:references|external\s+links|see\s+also|bibliography|further\s+reading|notes|sources)'
    r'(?:\s*={1,6})?\s*$'
)
_CATEGORY_LINE = re.compile(r'(?im)^\s*categories?\s*:\s*.+$')
_PORTAL_LINE = re.compile(r'(?im)^\s*portal\s*:\s*.+$')
_STUB_LINE = re.compile(r'(?i)\b(?:this\s+article\s+is\s+a\s+stub|stub\s+template)\b')
_NAV_TEMPLATE = re.compile(r'(?i)\{\{[^{}]*(?:navbox|sidebar|infobox\s+cell)[^{}]*\}\}')
_CITE_HEAVY = re.compile(r'\[\d{1,4}\]')

def has_reference_tail_markers(text: str) -> bool:
    if not text:
        return False
    sample = text[:8000]
    return bool(
        _SECTION_DROP.search(sample)
        or _CATEGORY_LINE.search(sample)
        or _NAV_TEMPLATE.search(sample)
    )

def _thin_citations(text: str, *, max_per_100_chars: int = 3) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        cites = len(_CITE_HEAVY.findall(line))
        if cites == 0:
            lines.append(line)
            continue
        density = cites / max(len(line), 1) * 100
        if density > max_per_100_chars and len(line) < 400:
            cleaned = _CITE_HEAVY.sub('', line)
            cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
            if cleaned:
                lines.append(cleaned)
            continue
        if cites > 8:
            cleaned = _CITE_HEAVY.sub('', line)
            lines.append(re.sub(r'\s{2,}', ' ', cleaned).strip())
            continue
        lines.append(line)
    return '\n'.join(lines)

def clean_reference_sections(text: str, *, stats: StageStats | None = None) -> str:
    if not text:
        return text
    out = _NAV_TEMPLATE.sub(' ', text)
    lines: list[str] = []
    skip_rest = False
    removed = 0
    for line in out.splitlines():
        if _SECTION_DROP.match(line.strip()):
            skip_rest = True
            removed += 1
            continue
        if skip_rest:
            removed += 1
            continue
        if _CATEGORY_LINE.match(line) or _PORTAL_LINE.match(line):
            removed += 1
            continue
        if _STUB_LINE.search(line) and len(line.strip()) < 120:
            removed += 1
            continue
        lines.append(line)
    out = '\n'.join(lines)
    out = re.sub(
        r'(?is)\b(?:={1,6}\s*)?(?:references|external\s+links|see\s+also|bibliography|further\s+reading|notes|sources)'
        r'(?:\s*={1,6})?\b.*$',
        '',
        out,
    )
    out = re.sub(r'(?i)\s*Categories:\s*[^\n.]+', '', out)
    out = re.sub(r'(?i)\s*Portal:\s*[^\n.]+', '', out)
    out = re.sub(r'(?i)\s*This article is a stub\.?', '', out)
    out = _thin_citations(out)
    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    if stats is not None:
        stats.lines_removed += removed
        stats.in_docs += 1
        stats.out_docs += 1 if out else 0
    return out
