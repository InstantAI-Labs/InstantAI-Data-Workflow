from __future__ import annotations

import re
import unicodedata

_CTRL_MINIMAL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
_WS_COLLAPSE = re.compile(r'\s+')
_CODE_FENCE = re.compile(r'(```[\s\S]*?```)', re.M)
_ESCAPED_MD = re.compile(r'\\([_*#\[\]])')
_ORPHAN_ESCAPE = re.compile(r'(?<![\\])\\(?![\\nrt"\'/])')
_BROKEN_LINK = re.compile(r'\[([^\]]*)\]\(\s*\)')
_INVISIBLE = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad]')
_SPACE_NL = re.compile(r'[ \t]+\n')
_MULTI_SPACE = re.compile(r'[^\S\n]{2,}')
_REPEAT_PUNCT = re.compile(r'([!?.,;:])\1{2,}')
_MULTI_NL = re.compile(r'\n{3,}')
_TRAIL_BRACKET = re.compile(r'^\s*[\]\u3011]\s*')
_COMMA_JOIN = re.compile(r'\n{2,}\s*([,;:])\s*')
_COLON_DOT = re.compile(r'(:\s*)\.\s*([a-z])', re.I)
_EMPTY_LIST = re.compile(r'(?m)^(\s*)[-*+]\s*$')
_SMART_QUOTES = str.maketrans(
    {
        '\u2018': "'",
        '\u2019': "'",
        '\u201a': "'",
        '\u201b': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u201e': '"',
        '\u201f': '"',
        '\u00ab': '"',
        '\u00bb': '"',
        '\u2039': "'",
        '\u203a': "'",
    }
)

def meaningful_char_count(text: str) -> int:
    return sum(1 for c in text if c.isalnum())


def minimal_normalize_detail(text: str) -> tuple[str, int]:
    if not text:
        return text, 0
    out: list[str] = []
    meaningful = 0
    in_ws = False
    for c in text:
        oc = ord(c)
        if 0 <= oc <= 8 or oc in (11, 12) or 14 <= oc <= 31:
            continue
        if c.isspace():
            if out and not in_ws:
                out.append(' ')
                in_ws = True
            continue
        in_ws = False
        out.append(c)
        if c.isalnum():
            meaningful += 1
    normalized = ''.join(out).strip()
    return normalized, meaningful


def minimal_normalize(text: str) -> str:
    normalized, _ = minimal_normalize_detail(text)
    return normalized

def _normalize_chunk(text: str) -> str:
    chunk = unicodedata.normalize('NFC', text)
    chunk = _INVISIBLE.sub('', chunk)
    chunk = chunk.translate(_SMART_QUOTES)
    chunk = chunk.replace('\r\n', '\n').replace('\r', '\n')
    chunk = _SPACE_NL.sub('\n', chunk)
    chunk = _MULTI_SPACE.sub(' ', chunk)
    chunk = _REPEAT_PUNCT.sub(r'\1', chunk)
    chunk = _MULTI_NL.sub('\n\n', chunk)
    return chunk.strip()

def repair_formatting(text: str, *, preserve_code_fences: bool = True) -> tuple[str, int]:
    if not text:
        return text, 0
    repairs = 0
    out = text

    def _fix_chunk(chunk: str) -> str:
        nonlocal repairs
        before = chunk
        chunk = _ESCAPED_MD.sub(r'\1', chunk)
        chunk = _ORPHAN_ESCAPE.sub('', chunk)
        chunk = _BROKEN_LINK.sub(r'\1', chunk)
        chunk = _TRAIL_BRACKET.sub('', chunk)
        chunk = _COMMA_JOIN.sub(r'\1 ', chunk)
        chunk = _COLON_DOT.sub(r'\1\2', chunk)
        chunk = _EMPTY_LIST.sub('', chunk)
        if chunk != before:
            repairs += 1
        return chunk

    if not preserve_code_fences or '```' not in out:
        return _fix_chunk(out), repairs

    parts: list[str] = []
    pos = 0
    fence_count = out.count('```')
    if fence_count % 2 == 1:
        out = out + '\n```'
        repairs += 1
    for match in _CODE_FENCE.finditer(out):
        if match.start() > pos:
            parts.append(_fix_chunk(out[pos:match.start()]))
        parts.append(match.group(1))
        pos = match.end()
    if pos < len(out):
        parts.append(_fix_chunk(out[pos:]))
    return '\n\n'.join(part for part in parts if part), repairs

def normalize_text(text: str, *, preserve_code_fences: bool = True, repair: bool = True) -> str:
    if not text:
        return text
    text = _CTRL_MINIMAL.sub('', text)
    if repair:
        text, _ = repair_formatting(text, preserve_code_fences=preserve_code_fences)
    if not preserve_code_fences or '```' not in text:
        return _normalize_chunk(text)

    parts: list[str] = []
    pos = 0
    for match in _CODE_FENCE.finditer(text):
        if match.start() > pos:
            parts.append(_normalize_chunk(text[pos:match.start()]))
        parts.append(match.group(1))
        pos = match.end()
    if pos < len(text):
        parts.append(_normalize_chunk(text[pos:]))
    return '\n\n'.join(part for part in parts if part)
