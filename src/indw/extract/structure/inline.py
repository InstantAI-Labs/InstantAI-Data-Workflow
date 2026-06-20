from __future__ import annotations

import re
from dataclasses import dataclass

from indw.clean.artifact.evidence_engine import compute_semantic_evidence
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator

_PIPE = re.compile(r'\|')
_CTA_BREAK = re.compile(r'(?<=[a-z])(?:\s|[\u00d7×•·|])+(?=[A-Z])')
_INLINE_BREAK = re.compile(r'[.!?](?=\S)')
_CODE_FENCE_SPLIT = re.compile(r'(```[\s\S]*?```)', re.M)
_PAGE_OF = re.compile(r'(?i)\bpage\s+\d+\s+of\s+\d+\b')
_DIGIT = re.compile(r'\d+')

@dataclass
class InlineStructuralStats:
    spans_removed: int = 0
    chars_removed: int = 0

def _is_pipe_cluster(span: str) -> bool:
    if '|' not in span:
        return False
    if any(tok in span for tok in ('<=', '>=', '->', '::', '&&', '||', '!=', '==')):
        return False
    parts = [p.strip() for p in span.split('|')]
    if len(parts) < 2 or len(parts) > 6 or len(span) > 120:
        return False
    if any('$' in p or '\\' in p for p in parts):
        return False
    if max(len(p.split()) for p in parts) > 5:
        return False
    return sum(len(p) for p in parts) / len(parts) <= 22

def _word_bounds(
    text: str,
    idx: int,
    *,
    direction: int,
    max_words: int,
    stop_at_newline: bool = False,
) -> int:
    pos = idx
    words = 0
    while 0 <= pos < len(text) and words < max_words:
        if stop_at_newline:
            nl = text.find('\n', pos if direction > 0 else max(0, pos - 1))
            if direction > 0 and nl >= 0 and nl < pos:
                return pos
            if direction < 0 and nl >= 0 and nl < pos:
                return pos
        if direction < 0:
            while pos > 0 and text[pos - 1].isspace():
                if stop_at_newline and text[pos - 1] == '\n':
                    return pos
                pos -= 1
            if pos <= 0:
                break
            while pos > 0 and (text[pos - 1].isalnum() or text[pos - 1] in "'-_"):
                pos -= 1
            words += 1
            while pos > 0 and text[pos - 1].isspace():
                if stop_at_newline and text[pos - 1] == '\n':
                    return pos
                pos -= 1
        else:
            while pos < len(text) and text[pos].isspace():
                if stop_at_newline and text[pos] == '\n':
                    return pos
                pos += 1
            if pos >= len(text):
                break
            while pos < len(text) and (text[pos].isalnum() or text[pos] in "'-_"):
                pos += 1
            words += 1
            while pos < len(text) and text[pos].isspace():
                if stop_at_newline and text[pos] == '\n':
                    return pos
                pos += 1
    return pos

def _expand_pipe_cluster(text: str, pipe_idx: int) -> tuple[int, int] | None:
    left = _word_bounds(text, pipe_idx, direction=-1, max_words=1, stop_at_newline=True)
    right = _word_bounds(text, pipe_idx + 1, direction=1, max_words=3, stop_at_newline=True)
    span = text[left:right].strip()
    if '|' not in span or '\n' in span:
        return None
    if _is_pipe_cluster(span):
        return left, right
    baseline = AdaptiveBaselineEstimator()
    pos = (left + right) / (2 * max(len(text), 1))
    if _span_noise_score(span, position_ratio=pos) >= 0.32:
        return left, right
    return None

def _pipe_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _PIPE.finditer(text):
        expanded = _expand_pipe_cluster(text, m.start())
        if expanded:
            spans.append(expanded)
    return spans

def _span_noise_score(span: str, *, position_ratio: float) -> float:
    s = span.strip()
    if not s or len(s) < 4:
        return 0.0
    baseline = AdaptiveBaselineEstimator()
    ev = compute_semantic_evidence(s)
    nav = ev.negative.get('navigational', 0.0) + ev.negative.get('transactional', 0.0)
    promo = ev.negative.get('promotional', 0.0) + ev.negative.get('administrative', 0.0)
    noise = baseline.baseline(list(ev.negative.values()) or [0.0])
    edge = 0.0
    if position_ratio < 0.12 or position_ratio > 0.88:
        edge = baseline.baseline([0.2, 1.0 - position_ratio if position_ratio > 0.5 else position_ratio])
    short = 1.0 if len(s.split()) <= 8 and len(s) < 100 else 0.0
    digit = min(1.0, len(_DIGIT.findall(s)) / max(len(s.split()), 1))
    return min(1.0, noise * 0.4 + nav * 0.25 + promo * 0.15 + edge * 0.1 + short * 0.05 + digit * 0.05)

def _align_word_edges(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return start, end

def _vote_scaffold_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    doc_len = max(len(text), 1)
    for m in re.finditer(r'\d+', text):
        left = max(0, m.start() - 18)
        right = min(len(text), m.end() + 18)
        for sep in '.!?':
            idx = text.find(sep, left, right)
            if idx > left:
                right = idx
        left, right = _align_word_edges(text, left, right)
        if right <= left:
            continue
        chunk = text[left:right].strip()
        if len(chunk) < 12 or len(chunk) > 72:
            continue
        if _PAGE_OF.search(chunk):
            continue
        digits = len(_DIGIT.findall(chunk))
        if digits < 2:
            continue
        if re.search(r'(?i)\b(?:minute|hour|day|week|month|year)s?\b', chunk) and digits == 1:
            continue
        words = chunk.split()
        if words and len(words[0]) > 8:
            continue
        short_w = sum(1 for w in words if len(w) <= 5) / max(len(words), 1)
        if short_w < 0.65:
            continue
        pos = (left + right) / (2 * doc_len)
        noise = _span_noise_score(chunk, position_ratio=pos)
        if noise < 0.32:
            continue
        ev = compute_semantic_evidence(chunk)
        if ev.utility > 0.18:
            continue
        words = text[left:right].split()
        while words and len(words[0]) > 5:
            left += len(words[0])
            while left < right and text[left].isspace():
                left += 1
            words = words[1:]
        if left >= right:
            continue
        spans.append((left, right))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    extended: list[tuple[int, int]] = []
    for start, end in merged:
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        probe = ''
        for w in text[cursor:].split()[:4]:
            probe = f'{probe} {w}'.strip()
            ev = compute_semantic_evidence(probe)
            if ev.utility > 0.14 or len(probe) > 26:
                break
            if _span_noise_score(probe, position_ratio=0.04) < 0.30:
                break
            cursor += len(w)
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
        extended.append((start, cursor))
    return extended

def _prefix_nav_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    line_end = text.find('\n')
    head = text if line_end < 0 else text[:line_end]
    if not head.strip():
        return spans
    cut = -1
    page = _PAGE_OF.search(head)
    if page:
        cut = page.end()
        while cut < len(head) and head[cut] in ' )':
            cut += 1
    elif head.count(':') >= 1 and len(head.split()) <= 28:
        if re.match(r'(?i)^\s*(?:system|user|assistant|human)\s*:', head):
            return spans
        colon_idx = head.find(':')
        after = head[colon_idx + 1:colon_idx + 4].strip() if colon_idx >= 0 else ''
        if colon_idx <= 30 and after and not after[0].isdigit():
            nav = _span_noise_score(head, position_ratio=0.02)
            if nav >= 0.4:
                cut = head.find(':') + 1
                rest = head[cut:].strip()
                paren = rest.find(')')
                if paren >= 0:
                    cut = head.find(rest) + paren + 1
                else:
                    cut = min(len(head), head.find(':') + 48)
    if cut > 0 and cut < len(head):
        tail = head[cut:].strip()
        if tail and _span_noise_score(tail, position_ratio=0.08) < 0.28:
            return spans
        spans.append((0, cut))
    return spans

def _suffix_nav_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in reversed(list(re.finditer(r'\.\s+', text))):
        start = m.end()
        tail = text[start:].strip()
        if not tail or len(tail) < 12 or len(tail) > 220:
            continue
        if re.search(r'(?m)^\s*(?:def |class |import |from |#include|function\s+\w)', tail):
            continue
        if _span_noise_score(tail, position_ratio=0.94) >= 0.48:
            lead = text[max(0, m.start() - 40):m.start() + 1]
            tail_ev = compute_semantic_evidence(tail)
            if tail_ev.utility > 0.16 and _span_noise_score(tail, position_ratio=0.94) < 0.58:
                continue
            if lead and _span_noise_score(lead, position_ratio=0.5) < _span_noise_score(tail, position_ratio=0.94):
                while start < len(text) and text[start].isspace():
                    start += 1
                spans.append((start, len(text)))
                break
    return spans

def _prefix_cta_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _CTA_BREAK.finditer(text):
        head = text[:m.start()].strip()
        if not head or len(head) > 56:
            continue
        ev = compute_semantic_evidence(head)
        promo = (
            ev.negative.get('promotional', 0.0)
            + ev.negative.get('navigational', 0.0)
            + ev.negative.get('transactional', 0.0)
        )
        if promo < 0.35:
            continue
        if _span_noise_score(head, position_ratio=0.03) < 0.38:
            continue
        spans.append((0, m.start()))
        break
    return spans

def _inline_clause_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    doc_len = max(len(text), 1)
    for m in _INLINE_BREAK.finditer(text):
        start = m.end()
        end = min(len(text), start + 58)
        for m2 in _CTA_BREAK.finditer(text, start, end):
            end = m2.start()
            break
        chunk = text[start:end].strip()
        if len(chunk) < 10:
            continue
        digits = len(_DIGIT.findall(chunk))
        if digits < 1 and '|' not in chunk:
            continue
        pos = (start + end) / (2 * doc_len)
        if _span_noise_score(chunk, position_ratio=pos) < 0.33:
            continue
        ev = compute_semantic_evidence(chunk)
        if ev.utility > 0.2:
            continue
        spans.append((start, end))
    return spans

def _leading_noise_prefix_spans(text: str) -> list[tuple[int, int]]:
    words = text.split()
    if len(words) < 5:
        return []
    cursor = 0
    probe = ''
    for w in words[:5]:
        probe = f'{probe} {w}'.strip()
        ev = compute_semantic_evidence(probe)
        if ev.utility > 0.14:
            break
        if _span_noise_score(probe, position_ratio=0.03) < 0.28:
            break
        cursor += len(w)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    if cursor <= 0 or cursor >= len(text) * 0.40:
        return []
    rest = text[cursor:].strip()
    if not rest:
        return []
    if compute_semantic_evidence(rest).utility <= compute_semantic_evidence(probe).utility * 1.05:
        return []
    return [(0, cursor)]

def _forum_ui_bridge_spans(text: str) -> list[tuple[int, int]]:
    from indw.extract.roles.forum import score_answer_substance

    spans: list[tuple[int, int]] = []
    if not text or len(text) < 24:
        return spans
    for m in re.finditer(r'[\?\!]\s+', text):
        bridge_start = m.end()
        if bridge_start >= len(text) - 8:
            continue
        words = text[bridge_start:].split()
        if len(words) < 2:
            continue
        idx = 0
        while idx < min(4, len(words)) and score_answer_substance(words[idx]) < 0.12:
            idx += 1
        if idx == 0 or not _forum_remnant_words(words[:idx]):
            continue
        if idx > 2 and len(' '.join(words[:idx])) > 18:
            idx = 2
        rest = ' '.join(words[idx:idx + 12])
        if score_answer_substance(rest) < 0.25:
            continue
        cursor = bridge_start + len(' '.join(words[:idx]))
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        spans.append((bridge_start, cursor))
    return spans

def _forum_remnant_words(words: list[str]) -> bool:
    if not words:
        return False
    return all(w.islower() and w.isalpha() and len(w) <= 9 for w in words)

def _leading_low_substance_spans(text: str) -> list[tuple[int, int]]:
    from indw.extract.roles.forum import score_answer_substance

    words = text.split()
    if len(words) < 3:
        return []
    idx = 0
    while idx < min(2, len(words)) and score_answer_substance(words[idx]) < 0.12:
        idx += 1
    if idx == 0 or not _forum_remnant_words(words[:idx]):
        return []
    rest = ' '.join(words[idx:idx + 12])
    if score_answer_substance(rest) < 0.25:
        return []
    cursor = len(' '.join(words[:idx]))
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return [(0, cursor)]

def _collect_structural_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    spans.extend(_leading_low_substance_spans(text))
    spans.extend(_leading_noise_prefix_spans(text))
    spans.extend(_forum_ui_bridge_spans(text))
    spans.extend(_pipe_spans(text))
    spans.extend(_vote_scaffold_spans(text))
    spans.extend(_inline_clause_spans(text))
    spans.extend(_prefix_nav_spans(text))
    spans.extend(_prefix_cta_spans(text))
    spans.extend(_suffix_nav_spans(text))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged

def _prose_segments(text: str, *, preserve_code_fences: bool = True):
    if not text:
        return
    if not preserve_code_fences or '```' not in text:
        yield 0, len(text), text
        return
    pos = 0
    for match in _CODE_FENCE_SPLIT.finditer(text):
        if match.start() > pos:
            yield pos, match.start(), text[pos:match.start()]
        pos = match.end()
    if pos < len(text):
        yield pos, len(text), text[pos:]

def _strip_segment(text: str) -> tuple[str, InlineStructuralStats]:
    stats = InlineStructuralStats()
    spans = _collect_structural_spans(text)
    if not spans:
        return text, stats
    parts: list[str] = []
    last = 0
    for start, end in spans:
        if start < last:
            continue
        parts.append(text[last:start])
        stats.spans_removed += 1
        stats.chars_removed += end - start
        last = end
    parts.append(text[last:])
    cleaned = ''.join(parts)
    cleaned = re.sub(r'([a-z]{2,})([A-Z][a-z])', r'\1 \2', cleaned)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    cleaned = re.sub(r' *\n *', '\n', cleaned)
    cleaned = re.sub(r'(?m)^\s+$', '', cleaned)
    return cleaned.strip(), stats

def strip_inline_structural(
    text: str,
    *,
    preserve_code_fences: bool = True,
) -> tuple[str, InlineStructuralStats]:
    if not text or not text.strip():
        return text, InlineStructuralStats()
    if not preserve_code_fences or '```' not in text:
        return _strip_segment(text)

    parts: list[str] = []
    total = InlineStructuralStats()
    pos = 0
    for match in _CODE_FENCE_SPLIT.finditer(text):
        if match.start() > pos:
            chunk, st = _strip_segment(text[pos:match.start()])
            parts.append(chunk)
            total.spans_removed += st.spans_removed
            total.chars_removed += st.chars_removed
        parts.append(match.group(1))
        pos = match.end()
    if pos < len(text):
        chunk, st = _strip_segment(text[pos:])
        parts.append(chunk)
        total.spans_removed += st.spans_removed
        total.chars_removed += st.chars_removed
    return ''.join(parts), total
