from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass, field

_CODE_FENCE = re.compile(r'(```[\s\S]*?```)', re.M)
_HEADING = re.compile(
    r'(?m)^(?:#{1,6}\s+.+|={2,}\s*.+\s*={2,}|Title:|Question:|Answer:|Additional Answer:)\s*$'
)
_TABLE_ROW = re.compile(r'^\s*\|?.+\|.+\|?\s*$')
_LIST_LINE = re.compile(r'^\s*(?:[-*+•]|\d+[.)])\s+', re.M)
_URL = re.compile(r'https?://\S+|www\.\S+', re.I)
_HTML_TAG = re.compile(r'<[a-z][^>]*>|</[a-z]+>|&[a-z]+;|&#\d+;', re.I)


@dataclass(frozen=True)
class LayoutVector:
    line_count: int = 0
    avg_len: float = 0.0
    digit_ratio: float = 0.0
    punct_ratio: float = 0.0
    url_ratio: float = 0.0
    list_ratio: float = 0.0
    fence_ratio: float = 0.0
    alpha_ratio: float = 0.0

    def bucket(self) -> str:
        return (
            f'{self.line_count}:{int(self.avg_len // 8)}:'
            f'{int(self.digit_ratio * 10)}:{int(self.punct_ratio * 10)}:'
            f'{int(self.url_ratio * 10)}:{int(self.list_ratio * 10)}'
        )

    def to_tuple(self) -> tuple[float, ...]:
        return (
            float(self.line_count),
            self.avg_len,
            self.digit_ratio,
            self.punct_ratio,
            self.url_ratio,
            self.list_ratio,
            self.fence_ratio,
            self.alpha_ratio,
        )


@dataclass
class DocumentUnit:
    kind: str
    text: str
    start: int
    end: int
    unit_id: str = ''
    parent_id: str | None = None
    layout: LayoutVector = field(default_factory=LayoutVector)
    in_fence: bool = False

    def __post_init__(self) -> None:
        if not self.unit_id:
            self.unit_id = uuid.uuid4().hex[:12]


@dataclass
class DecomposedDocument:
    text: str
    units: list[DocumentUnit] = field(default_factory=list)
    char_count: int = 0

    def lines(self) -> list[DocumentUnit]:
        return [u for u in self.units if u.kind == 'line']

    def blocks(self) -> list[DocumentUnit]:
        return [u for u in self.units if u.kind in ('block', 'paragraph', 'section')]


def normalize_ws(text: str) -> str:
    t = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', ' ', t.strip())


def compute_layout(
    text: str,
    *,
    in_fence: bool = False,
    lines: list[str] | None = None,
) -> LayoutVector:
    if not text:
        return LayoutVector()
    if lines is None:
        from indw.clean.artifact.evidence_cache import get_layout_cache, layout_cache_key
        key = layout_cache_key(text, in_fence=in_fence)
        if key is not None:
            cache = get_layout_cache()
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = _compute_layout_impl(text, in_fence=in_fence, lines=None)
            cache.put(key, result)
            return result
    return _compute_layout_impl(text, in_fence=in_fence, lines=lines)


def _compute_layout_impl(
    text: str,
    *,
    in_fence: bool = False,
    lines: list[str] | None = None,
) -> LayoutVector:
    if lines is None:
        lines = text.splitlines() or [text]
    lc = max(len(lines), 1)
    lens = [len(ln) for ln in lines]
    avg = sum(lens) / lc
    cc = max(len(text), 1)
    digits = alpha = punct = 0
    for c in text:
        if c.isdigit():
            digits += 1
        elif c.isalpha():
            alpha += 1
        elif not c.isspace():
            punct += 1
    url_chars = sum(len(m.group(0)) for m in _URL.finditer(text))
    fence_chars = len(_CODE_FENCE.findall(text)) * 3
    list_lines = sum(1 for ln in lines if _LIST_LINE.match(ln))
    return LayoutVector(
        line_count=lc,
        avg_len=avg,
        digit_ratio=digits / cc,
        punct_ratio=punct / cc,
        url_ratio=url_chars / cc,
        list_ratio=list_lines / lc,
        fence_ratio=fence_chars / cc if in_fence else 0.0,
        alpha_ratio=alpha / cc,
    )


def _classify_block(text: str, layout: LayoutVector) -> str:
    stripped = text.strip()
    if _HTML_TAG.search(stripped) and layout.punct_ratio > 0.08:
        return 'html_remnant'
    if _HEADING.match(stripped):
        return 'header'
    if layout.list_ratio >= 0.6 and layout.line_count >= 2:
        return 'list'
    table_lines = sum(1 for ln in text.splitlines() if _TABLE_ROW.match(ln))
    if table_lines >= 2 and table_lines / max(layout.line_count, 1) >= 0.5:
        return 'table'
    if layout.line_count == 1 and layout.avg_len < 120:
        return 'line'
    if layout.line_count >= 2:
        return 'paragraph'
    return 'block'


def _split_blocks(text: str) -> list[tuple[str, int, int, bool]]:
    if '```' not in text:
        parts: list[tuple[str, int, int, bool]] = []
        pos = 0
        for m in re.finditer(r'\n\s*\n', text):
            chunk = text[pos:m.start()]
            if chunk.strip():
                parts.append((chunk.strip(), pos, m.start(), False))
            pos = m.end()
        if pos < len(text) and text[pos:].strip():
            parts.append((text[pos:].strip(), pos, len(text), False))
        if not parts and text.strip():
            parts.append((text.strip(), 0, len(text), False))
        return parts

    parts = []
    pos = 0
    for match in _CODE_FENCE.finditer(text):
        if match.start() > pos:
            chunk = text[pos:match.start()]
            cpos = pos
            for m in re.finditer(r'\n\s*\n', chunk):
                sub = chunk[: m.start()].strip()
                if sub:
                    parts.append((sub, cpos, cpos + m.start(), False))
                cpos += m.end()
            tail = chunk[cpos - pos :].strip() if cpos - pos < len(chunk) else chunk.strip()
            if tail:
                parts.append((tail, cpos, match.start(), False))
        parts.append((match.group(1).strip(), match.start(), match.end(), True))
        pos = match.end()
    if pos < len(text):
        chunk = text[pos:]
        if chunk.strip():
            parts.append((chunk.strip(), pos, len(text), False))
    return parts


def decompose_document(text: str) -> DecomposedDocument:
    if not text or not text.strip():
        return DecomposedDocument(text=text or '', char_count=0)

    src = text
    units: list[DocumentUnit] = []
    blocks = _split_blocks(src)
    doc_len = len(src)

    for block_text, start, end, in_fence in blocks:
        layout = compute_layout(block_text, in_fence=in_fence)
        kind = 'code' if in_fence else _classify_block(block_text, layout)
        block_id = uuid.uuid4().hex[:12]
        units.append(
            DocumentUnit(
                kind=kind,
                text=block_text,
                start=start,
                end=end,
                unit_id=block_id,
                layout=layout,
                in_fence=in_fence,
            )
        )
        if in_fence:
            continue
        rel = 0
        for line in block_text.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                rel += len(line) + 1
                continue
            ls = start + rel
            le = ls + len(line)
            rel += len(line) + 1
            llayout = compute_layout(line_stripped)
            lkind = 'header' if _HEADING.match(line_stripped) else 'line'
            units.append(
                DocumentUnit(
                    kind=lkind,
                    text=line_stripped,
                    start=ls,
                    end=le,
                    parent_id=block_id,
                    layout=llayout,
                )
            )

    for u in units:
        if u.kind in ('header',) and u.start < doc_len * 0.12:
            u.kind = 'header'
        elif u.start >= doc_len * 0.9 and u.layout.line_count <= 2:
            u.kind = 'footer'

    return DecomposedDocument(text=src, units=units, char_count=doc_len)


def position_bin(char_offset: int, doc_len: int) -> int:
    if doc_len <= 0:
        return 2
    r = char_offset / doc_len
    if r < 0.10:
        return 0
    if r < 0.25:
        return 1
    if r < 0.75:
        return 2
    if r < 0.90:
        return 3
    return 4
