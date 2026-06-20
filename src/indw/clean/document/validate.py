from __future__ import annotations

import json
import re
from typing import Any, Optional

from indw.config.defaults import CHUNK_MIN_QUALITY_SCORE_100, MIN_CHARS_FINAL, MIN_CHARS_GATE
from indw.clean.document.config import CleaningConfig
from indw.clean.document.normalize import meaningful_char_count
from indw.clean.document.patterns import _CONTROL
from indw.clean.document.segment import _token_estimate


def meaningful_char_floor(
    cfg: CleaningConfig,
    *,
    gate_approved: bool = False,
    quality_score: float = 0.0,
    gate_min_chars: int = MIN_CHARS_GATE,
) -> int:
    base = int(cfg.min_chars_after_clean)
    if not gate_approved:
        return base
    if quality_score >= 0.62:
        return max(MIN_CHARS_FINAL, min(MIN_CHARS_GATE, base // 2))
    if quality_score >= 0.50:
        return MIN_CHARS_GATE
    if quality_score >= 0.40:
        return max(MIN_CHARS_FINAL, int(gate_min_chars))
    relaxed = max(MIN_CHARS_GATE, int(MIN_CHARS_GATE + quality_score * 80))
    return min(base, relaxed)


def validate_chunk(
    text: str,
    cfg: CleaningConfig,
    *,
    quality_score: float = 0.0,
    quality_score_10: float = 0.0,
    min_quality_score_10: float | None = None,
    min_quality_score_100: float = CHUNK_MIN_QUALITY_SCORE_100,
    skip_quality_check: bool = False,
    gate_approved: bool = False,
    gate_min_chars: int = MIN_CHARS_GATE,
) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, 'empty'
    if _CONTROL.search(text):
        return False, 'control_chars'
    if '\ufffd' in text:
        return False, 'broken_encoding'
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        return False, 'encoding'

    meaningful = meaningful_char_count(text)
    score = quality_score if quality_score > 0.0 else quality_score_10 / 10.0
    floor = meaningful_char_floor(
        cfg,
        gate_approved=gate_approved,
        quality_score=score,
        gate_min_chars=gate_min_chars,
    )
    if meaningful < floor:
        return False, 'min_meaningful_chars'

    tokens = _token_estimate(text, cfg)
    overlap_headroom = 1.0 + max(0.0, min(cfg.chunk_overlap_ratio, 0.2))
    max_tokens_allowed = int(cfg.max_tokens * overlap_headroom * 1.15)
    if tokens > max_tokens_allowed:
        return False, 'max_tokens'

    if not skip_quality_check:
        floor_10 = min_quality_score_10 if min_quality_score_10 is not None else min_quality_score_100 / 10.0
        if floor_10 > 0 and quality_score_10 < floor_10:
            return False, 'quality_below_threshold'

    return True, ''


def validate_jsonl_record(
    row: dict[str, Any],
    *,
    text_key: str = 'text',
) -> tuple[bool, str]:
    if not isinstance(row, dict):
        return False, 'not_object'
    text = row.get(text_key) or row.get('content') or ''
    if not isinstance(text, str) or not text.strip():
        return False, 'empty_text'
    return True, ''


def validate_jsonl_line(line: str, *, text_key: str = 'text') -> tuple[bool, str, Optional[dict[str, Any]]]:
    if not line or not line.strip():
        return False, 'blank_line', None
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return False, 'malformed_json', None
    ok, reason = validate_jsonl_record(row, text_key=text_key)
    if not ok:
        return False, reason, row
    return True, '', row
