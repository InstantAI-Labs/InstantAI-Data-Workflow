from __future__ import annotations

import re

from indw.clean.artifact.engine import get_artifact_engine
from indw.clean.artifact.registry import line_is_artifact
from indw.clean.document.patterns import (
    _ACK_LINE,
    _METADATA_LINE,
    _MODERATOR_LINE,
    _PIPE_NAV,
    _REPLY_ACK_LINE,
    _UI_LINE,
)
from indw.clean.document.stats import StageStats

_SOCIAL_BUTTON = re.compile(r'(?i)\b(?:tweet|retweet|like|pin\s+it|share\s+on)\b')

def _drop_lines(
    text: str,
    *,
    patterns: list[re.Pattern[str]],
    stats: StageStats | None = None,
) -> str:
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append('')
            continue
        drop = line_is_artifact(stripped) or any(p.search(stripped) for p in patterns)
        if not drop and _PIPE_NAV.match(stripped) and len(stripped) < 120:
            drop = True
        if not drop and _SOCIAL_BUTTON.search(stripped) and len(stripped) < 80:
            drop = True
        if drop:
            removed += 1
            continue
        kept.append(line)
    out = '\n'.join(kept)
    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    if stats is not None:
        stats.lines_removed += removed
        stats.in_docs += 1
        stats.out_docs += 1 if out else 0
        if not out:
            stats.dropped += 1
    return out

def remove_ui_noise(text: str, *, stats: StageStats | None = None) -> str:
    engine = get_artifact_engine()
    working, inline_stats = engine.strip_inline(text, preserve_code_fences=True)
    if stats is not None:
        stats.lines_removed += inline_stats.spans_removed
    working = _drop_lines(working, patterns=[_UI_LINE], stats=stats)
    return working

def remove_metadata(text: str, *, stats: StageStats | None = None) -> str:
    return _drop_lines(text, patterns=[_METADATA_LINE], stats=stats)

def remove_low_value_lines(
    text: str,
    *,
    drop_ack: bool = True,
    drop_mod: bool = True,
    drop_quotes: bool = True,
    stats: StageStats | None = None,
) -> str:
    patterns: list[re.Pattern[str]] = []
    if drop_ack:
        patterns.append(_ACK_LINE)
    if drop_mod:
        patterns.append(_MODERATOR_LINE)
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append('')
            continue
        drop = any(p.search(stripped) for p in patterns)
        if drop_ack and _REPLY_ACK_LINE.search(stripped):
            drop = True
        if drop_quotes and stripped.startswith('>') and len(stripped) < 500:
            drop = True
        if drop:
            removed += 1
            continue
        kept.append(line)
    out = '\n'.join(kept)
    out = re.sub(
        r'(?i)\s*(?:comment|reply|meta)\s*:\s*(?:\+1|thanks?|thank\s+you|thx|ty|nice|helpful|this\s+worked)\b[^.\n!?]*',
        '',
        out,
    )
    engine = get_artifact_engine()
    out, inline_stats = engine.strip_inline(out, preserve_code_fences=True)
    removed += inline_stats.spans_removed
    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    if stats is not None:
        stats.lines_removed += removed
    return out
