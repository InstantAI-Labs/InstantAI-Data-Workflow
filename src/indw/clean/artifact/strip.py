from __future__ import annotations

import re
import unicodedata

from indw.clean.document.normalize import repair_formatting
from indw.clean.document.patterns import _CONTROL, _HTML_TAG, _INVISIBLE, _MULTI_BLANK, _REPEAT_PUNCT, _WORD

_GARBAGE_TOKEN = re.compile(
    r'(?i)(?<![\w])('
    r'yughvn|asdfgh|qwerty|zxcvzx|zxzxzx|kjhgfds|'
    r'cv\s+bb|xx+\s+yy+|'
    r'[bcdfghjklmnpqrstvwxyz]{5,}|'
    r'(?:[a-z]\s+){3,}[a-z]'
    r')(?![\w])'
)

_HTML_ARTIFACT = re.compile(
    r'(?i)(?:'
    r'\b(?:id|class|style|onclick|onload|data-[a-z0-9_-]+|aria-[a-z0-9_-]+)\s*=\s*["\'][^"\']*["\']|'
    r'<\s*/?\s*[a-z][a-z0-9]*\b[^>]*>|'
    r'&(?:nbsp|amp|lt|gt|quot|apos|#\d+|#x[0-9a-f]+);'
    r')'
)

_CSS_JS_REMNANT = re.compile(
    r'(?i)(?:'
    r'\{[^}]{0,200}(?:color|font-size|margin|padding|display|background|width|height)\s*:[^}]+\}|'
    r'\b(?:function|var|let|const)\s+\w+\s*[=(]|'
    r'document\.(?:getElementById|querySelector|cookie)|'
    r'window\.(?:location|open)|'
    r'(?:console|alert)\s*\('
    r')'
)

_BOILERPLATE_SECTION = re.compile(
    r'(?im)^\s*(?:'
    r'see\s+also|external\s+links?|further\s+reading|references|bibliography|'
    r'last\s+edited|last\s+modified|citation\s+needed|stub\s+article|'
    r'categories\s*:|hidden\s+categories\s*:|'
    r'navigation\s+menu|edit\s+this\s+page|view\s+history|talk\s*:\s*'
    r')\s*:?\s*$'
)

_CORRUPTED_INSERT = re.compile(
    r'(?i)(?<![\w])('
    r'[bcdfghjklmnpqrstvwxyz]{4,}(?:\s+[bcdfghjklmnpqrstvwxyz]{2,})+|'
    r'yughvn|asdfgh|cv\s+bb|zxzxzx'
    r')(?![\w])'
)

def _dedupe_sentences(text: str) -> str:
    parts = re.split(r'([.!?]+\s+)', text)
    if len(parts) < 3:
        return text
    seen: set[str] = set()
    out: list[str] = []
    i = 0
    while i < len(parts):
        sentence = parts[i].strip()
        delim = parts[i + 1] if i + 1 < len(parts) else ''
        key = re.sub(r'\s+', ' ', sentence.lower())[:200]
        if sentence and key not in seen:
            seen.add(key)
            out.append(sentence + delim)
        elif not sentence and delim:
            out.append(delim)
        i += 2 if i + 1 < len(parts) else 1
    return ''.join(out)

def _strip_boilerplate_sections(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    skip_rest = False
    for line in lines:
        stripped = line.strip()
        if _BOILERPLATE_SECTION.match(stripped):
            skip_rest = True
            continue
        if skip_rest and not stripped:
            continue
        if skip_rest and stripped and not line.startswith(' '):
            if len(_WORD.findall(stripped)) >= 8:
                skip_rest = False
            else:
                continue
        if not skip_rest:
            out.append(line)
    return '\n'.join(out).strip()

def clean_document_artifacts(
    text: str,
    *,
    preserve_code_fences: bool = True,
    html_already_extracted: bool = False,
) -> tuple[str, dict[str, int]]:
    stats = {
        'garbage_tokens': 0,
        'html_artifacts': 0,
        'corrupted_insertions': 0,
        'format_repairs': 0,
        'paragraphs_deduped': 0,
    }
    if not text:
        return text, stats

    out = text
    if not html_already_extracted and '<' in out:
        from indw.clean.document.html import clean_html
        out = clean_html(out)
    out = _HTML_ARTIFACT.sub(' ', out)
    out = _CSS_JS_REMNANT.sub(' ', out)
    if '<' in out:
        out = _HTML_TAG.sub(' ', out)

    before_g = len(_GARBAGE_TOKEN.findall(out))
    out = _GARBAGE_TOKEN.sub('', out)
    stats['garbage_tokens'] = before_g

    before_c = len(_CORRUPTED_INSERT.findall(out))
    out = _CORRUPTED_INSERT.sub('', out)
    stats['corrupted_insertions'] = before_c

    from indw.clean.document.dedup import dedupe_paragraphs

    out = _strip_boilerplate_sections(out)
    prev_len = len(out)
    out, _ = dedupe_paragraphs(out)
    if len(out) < prev_len:
        stats['paragraphs_deduped'] += 1
    out = _dedupe_sentences(out)

    out = unicodedata.normalize('NFC', out)
    out = out.replace('\ufffd', '')
    out = _CONTROL.sub('', out)
    out = _INVISIBLE.sub('', out)
    out = _REPEAT_PUNCT.sub(r'\1', out)
    out = _MULTI_BLANK.sub('\n\n', out)
    out = re.sub(r'[^\S\n]{2,}', ' ', out)

    out, repairs = repair_formatting(out, preserve_code_fences=preserve_code_fences)
    stats['format_repairs'] = repairs

    return out.strip(), stats
