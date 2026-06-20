from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable
from indw.clean.artifact.registry import line_is_artifact
from indw.clean.document.patterns import _CODE_FENCE, _METADATA_LINE, _UI_LINE, _WORD
from indw.clean.meta.patterns import (
    _ADA_COMMENT_META,
    _ADA_LICENSE_BLOCK,
    _AI_PROMPT_LINE,
    _AI_TRAINING_BODY,
    _BOILERPLATE_LINE,
    _CODE_AUTHOR_META,
    _CODE_COPYRIGHT,
    _CODE_FENCE,
    _CODE_GENERATED,
    _CODE_LICENSE_BLOCK,
    _COLLAPSED_CODE_START,
    _COPYRIGHT_BLOCK,
    _COPYRIGHT_LINE,
    _COT_MARKERS,
    _EDITORIAL_LINE,
    _EMAIL_LINE,
    _ENCYCLOPEDIA_CHROME,
    _FORUM_ANSWER_TRANSITION,
    _FORUM_BULLET_COMMENT,
    _FORUM_LINE,
    _FORUM_QUOTE_PREFIX,
    _FORUM_STATS_LINE,
    _FORUM_TIMESTAMP,
    _FORUM_USERNAME_GREETING,
    _FRONT_MATTER,
    _HEADER_LINE,
    _INFORMATIVE_COMMENT,
    _INLINE_LICENSE_CHUNK,
    _INSTRUCTION_LABEL,
    _INSTRUCTION_SCAFFOLD,
    _KNOWLEDGE_LABELS,
    _LEGAL_FOOTER,
    _LICENSE_LINE,
    _METADATA_LINE,
    _META_LABELS,
    _README_EMPTY_HEADER,
    _REPO_LINE,
    _SOCIAL_PROMO_PREFIX,
    _TOC_ONLY_LINE,
    _TRAILING_FORUM_BLOCK,
    _UI_LINE,
    _VENDOR_NOTICE_LINE,
    _WIKI_NAV_INLINE,
    _WORD,
)
from indw.clean.meta.stats import MetadataCleanStats

def _protect_code_fences(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}
    idx = 0

    def _stash(match: re.Match[str]) -> str:
        nonlocal idx
        key = f'\x00FENCE{idx}\x00'
        placeholders[key] = match.group(0)
        idx += 1
        return key

    return _CODE_FENCE.sub(_stash, text), placeholders

def _restore_code_fences(text: str, placeholders: dict[str, str]) -> str:
    out = text
    for key, val in placeholders.items():
        out = out.replace(key, val)
    return out

def _strip_inline_license_runs(text: str, *, max_chunks: int = 120) -> tuple[str, int]:
    from indw.clean.document.license import strip_collapsed_inline_license
    return strip_collapsed_inline_license(text, max_chunks=max_chunks)

def _strip_leading_license_preamble(text: str, *, max_scan_lines: int = 120) -> tuple[str, int]:
    lines = text.splitlines()
    if not lines:
        return text, 0

    removed = 0
    idx = 0
    while idx < min(len(lines), max_scan_lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        is_dash = bool(re.match(r'^[-=]{8,}$', stripped))
        is_ada_comment = bool(_ADA_COMMENT_META.match(lines[idx]))
        is_copyright = bool(_COPYRIGHT_LINE.match(stripped))
        is_vendor = bool(_VENDOR_NOTICE_LINE.match(stripped))
        is_repo = bool(_REPO_LINE.match(stripped))
        if is_dash or is_ada_comment or is_copyright or is_vendor or is_repo:
            removed += 1
            idx += 1
            continue
        if _LICENSE_LINE.search(stripped) and (len(stripped) < 240 or idx < 20):
            removed += 1
            idx += 1
            continue
        if re.search(r'(?i)^use,?\s+copy,?\s+modify', stripped):
            removed += 1
            idx += 1
            continue
        break

    if removed == 0:
        return text, 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return '\n'.join(lines[idx:]), removed

def _is_forum_metadata_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _FORUM_LINE.match(stripped) or _FORUM_STATS_LINE.match(stripped):
        return True
    if _FORUM_USERNAME_GREETING.match(stripped):
        return True
    if stripped.startswith('•') and re.search(_FORUM_TIMESTAMP, stripped):
        return True
    if re.search(rf'(?i)–\s*[\w.-]+\s+{_FORUM_TIMESTAMP}', stripped):
        return True
    if _FORUM_QUOTE_PREFIX.search(stripped):
        return True
    return False

def _strip_forum_quote_blocks(text: str) -> tuple[str, int]:
    removed = 0
    while True:
        m = _FORUM_QUOTE_PREFIX.search(text)
        if not m:
            break
        start = m.start()
        rest = text[m.end():]
        trans = _FORUM_ANSWER_TRANSITION.search(rest)
        if trans and trans.start() > 16:
            text = text[:start] + rest[trans.start():]
        else:
            dot = rest.find('. ', 24, min(len(rest), 500))
            text = text[:start] + (rest[dot + 2:] if dot > 0 else rest[min(len(rest), 280):])
        removed += 1
    return text, removed

def _strip_forum_inline_artifacts(text: str) -> tuple[str, int]:
    removed = 0
    out = text
    out, n = _FORUM_BULLET_COMMENT.subn(' ', out)
    removed += n
    out = _WIKI_NAV_INLINE.sub('', out)
    if _WIKI_NAV_INLINE.search(text):
        removed += 1
    out, quote_removed = _strip_forum_quote_blocks(out)
    removed += quote_removed
    out = re.sub(r'[ \t]{2,}', ' ', out)
    out = re.sub(r' *\n *', '\n', out)
    return out.strip(), removed

def forum_contamination_hits(text: str) -> int:
    if not text:
        return 0
    hits = len(_FORUM_BULLET_COMMENT.findall(text))
    hits += len(_FORUM_QUOTE_PREFIX.findall(text))
    hits += len(_WIKI_NAV_INLINE.findall(text))
    hits += sum(1 for ln in text.splitlines() if _is_forum_metadata_line(ln))
    return hits

def _strip_trailing_forum_footer(text: str) -> tuple[str, int]:
    scan = text[-1400:] if len(text) > 1400 else text
    match = _TRAILING_FORUM_BLOCK.search(scan)
    if not match:
        return text, 0
    tail = match.group(0)
    if len(tail) > max(len(text) * 0.35, 900):
        return text, 0
    start = len(text) - len(scan) + match.start()
    if start < 40:
        return text, 0
    return text[:start].rstrip(), 1

def _strip_trailing_license_footer(text: str, *, max_scan_lines: int = 40) -> tuple[str, int]:
    lines = text.splitlines()
    if not lines:
        return text, 0
    removed = 0
    end = len(lines)
    scanned = 0
    while end > 0 and scanned < max_scan_lines:
        stripped = lines[end - 1].strip()
        if not stripped:
            end -= 1
            scanned += 1
            continue
        if (
            _COPYRIGHT_LINE.match(stripped)
            or _ADA_COMMENT_META.match(lines[end - 1])
            or _VENDOR_NOTICE_LINE.match(stripped)
            or (_LICENSE_LINE.search(stripped) and len(stripped) < 240)
        ):
            removed += 1
            end -= 1
            scanned += 1
            continue
        break
    if removed == 0:
        return text, 0
    return '\n'.join(lines[:end]).rstrip(), removed

def apply_discovery_line_drops(
    text: str,
    *,
    trim: bool = False,
    shadow: bool = True,
    discovery_engine: Any | None = None,
) -> tuple[str, int]:
    if shadow or not trim:
        return text, 0
    try:
        eng = discovery_engine
        if eng is None:
            from indw.clean.artifact.discovery_engine import get_discovery_engine
            eng = get_discovery_engine()
        report = eng.discover(text)
        if report.trim and not report.shadow:
            removed = report.chars_removed
            return report.trim.text, removed
    except Exception:
        return text, 0
    return text, 0

def _drop_matching_lines(
    text: str,
    patterns: list[re.Pattern[str]],
    *,
    on_match: Callable[[], None] | None = None,
) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if line_is_artifact(stripped) or any(p.match(stripped) for p in patterns):
            if on_match:
                on_match()
            continue
        kept.append(line)
    return '\n'.join(kept)

def _drop_forum_metadata_lines(text: str, stats: MetadataCleanStats) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        if _is_forum_metadata_line(line):
            stats.email_forum_removed += 1
            continue
        kept.append(line)
    return '\n'.join(kept)

def clean_code_comments(code: str) -> tuple[str, int]:
    removed = 0
    lines = code.splitlines()
    out: list[str] = []
    in_block = False
    block_buf: list[str] = []

    def _flush_block() -> None:
        nonlocal removed
        block = '\n'.join(block_buf)
        if block.strip() and not _INFORMATIVE_COMMENT.search(block):
            removed += 1
            block_buf.clear()
            return
        out.extend(block_buf)
        block_buf.clear()

    for line in lines:
        stripped = line.strip()
        if in_block:
            block_buf.append(line)
            if '*/' in stripped:
                in_block = False
                _flush_block()
            continue
        if stripped.startswith('/*') and not stripped.endswith('*/'):
            in_block = True
            block_buf = [line]
            continue
        if _CODE_LICENSE_BLOCK.match(stripped):
            removed += 1
            continue
        if (
            _CODE_COPYRIGHT.match(line)
            or _CODE_GENERATED.match(line)
            or _CODE_AUTHOR_META.match(line)
        ) and not _INFORMATIVE_COMMENT.search(line):
            removed += 1
            continue
        out.append(line)
    if in_block:
        _flush_block()
    return '\n'.join(out), removed

def _clean_code_fences(text: str, stats: MetadataCleanStats) -> str:
    def _repl(match: re.Match[str]) -> str:
        block = match.group(0)
        fence_match = re.match(r'^```(\w*)', block)
        lang = fence_match.group(1) if fence_match else ''
        inner = re.sub(r'^```\w*\n?', '', block)
        inner = re.sub(r'\n?```$', '', inner)
        cleaned, n = clean_code_comments(inner)
        stats.comment_blocks_removed += n
        prefix = f'```{lang}\n' if lang else '```\n'
        return prefix + cleaned + '\n```'

    return _CODE_FENCE.sub(_repl, text)
