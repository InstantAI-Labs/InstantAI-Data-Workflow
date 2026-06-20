from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_BACKTICK_GLUE = re.compile(r'`([^`]{1,40})`')
_MULTI_SPACE = re.compile(r'[ \t]{2,}')
_HYPHEN_BREAK = re.compile(r'(\w)-\n(\w)')
_GLUE_PUNCT = re.compile(r'(\w)[\u00b7\u2022\u2024](\w)')
_DICT_TITLE = re.compile(
    r'(?i)(?:title|definition|pronunciation|etymology)\s*:\s*',
)
_WEBSTER_PREFIX = re.compile(
    r'(?is)^(?:from the collaborative international dictionary[^:]{0,80}:\s*)+',
)

@dataclass
class OcrNormalizeStats:
    chars_repaired: int = 0
    lines_merged: int = 0
    tokens_fixed: int = 0
    prefixes_stripped: int = 0

def _repair_unicode(text: str) -> tuple[str, int]:
    if not text:
        return text, 0
    before = len(text)
    out = unicodedata.normalize('NFKC', text)
    out = out.replace('\ufffd', '')
    repaired = max(0, before - len(out))
    return out, repaired

def _merge_hyphen_breaks(text: str) -> tuple[str, int]:
    merged = 0
    out = text
    while True:
        new = _HYPHEN_BREAK.sub(r'\1\2', out)
        if new == out:
            break
        merged += 1
        out = new
    return out, merged

def _fix_glued_tokens(text: str) -> tuple[str, int]:
    fixed = 0
    out = _GLUE_PUNCT.sub(r'\1 \2', text)
    if out != text:
        fixed += 1
    out2 = _BACKTICK_GLUE.sub(r'\1', out)
    if out2 != out:
        fixed += len(_BACKTICK_GLUE.findall(out))
    return out2, fixed

def _strip_dictionary_ui(text: str) -> tuple[str, int]:
    stripped = 0
    out = _WEBSTER_PREFIX.sub('', text.strip())
    if out != text.strip():
        stripped += 1
    lines: list[str] = []
    for ln in out.splitlines():
        if _DICT_TITLE.match(ln.strip()) and len(ln.split()) <= 8:
            stripped += 1
            continue
        lines.append(ln)
    return '\n'.join(lines), stripped

def _collapse_whitespace(text: str) -> str:
    lines = [_MULTI_SPACE.sub(' ', ln.rstrip()) for ln in text.splitlines()]
    out = '\n'.join(lines)
    return re.sub(r'\n{3,}', '\n\n', out)

def normalize_ocr_text(text: str) -> tuple[str, OcrNormalizeStats]:
    if not text or not text.strip():
        return text, OcrNormalizeStats()

    stats = OcrNormalizeStats()
    out, n = _repair_unicode(text)
    stats.chars_repaired += n

    out, n = _merge_hyphen_breaks(out)
    stats.lines_merged += n

    out, n = _fix_glued_tokens(out)
    stats.tokens_fixed += n

    out, n = _strip_dictionary_ui(out)
    stats.prefixes_stripped += n

    out = _collapse_whitespace(out)
    return out.strip(), stats
