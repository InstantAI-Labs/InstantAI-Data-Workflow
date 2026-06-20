from __future__ import annotations

import re

from dataclasses import dataclass, field

from indw.clean.artifact.confidence import FusedConfidence
from indw.clean.artifact.decompose import DocumentUnit
from indw.clean.artifact.safeguards import is_protected_unit

@dataclass
class TrimPolicy:
    shadow: bool = True
    max_trim_ratio: float = 0.40
    min_trim_confidence: float = 0.92
    medium_trim_confidence: float = 0.72

@dataclass
class TrimResult:
    text: str
    chars_removed: int = 0
    units_removed: int = 0
    shadow_only: bool = True
    removed_spans: list[tuple[int, int]] = field(default_factory=list)
    protected_skips: int = 0

def safe_trim_fragments(
    text: str,
    units: list[DocumentUnit],
    fused: list[tuple[DocumentUnit, FusedConfidence]],
    *,
    policy: TrimPolicy,
) -> TrimResult:
    would_trim = [
        (u, f) for u, f in fused
        if f.would_trim and not is_protected_unit(u.text, kind=u.kind, in_fence=u.in_fence)
    ]
    protected = sum(
        1 for u, f in fused
        if f.would_trim and is_protected_unit(u.text, kind=u.kind, in_fence=u.in_fence)
    )

    if policy.shadow or not text:
        spans = [(u.start, u.end) for u, _ in would_trim]
        return TrimResult(text=text, shadow_only=True, removed_spans=spans, protected_skips=protected)

    remove_units = [
        u for u, f in would_trim
        if f.trim_tier == 'high' and u.kind in ('line', 'footer', 'header', 'paragraph')
        or f.trim_tier == 'medium' and u.kind in ('line', 'footer', 'header')
    ]
    if not remove_units:
        return TrimResult(text=text, shadow_only=False, protected_skips=protected)

    remove_chars = sum(u.end - u.start for u in remove_units)
    if remove_chars / max(len(text), 1) > policy.max_trim_ratio:
        return TrimResult(text=text, shadow_only=False, protected_skips=protected)

    spans = sorted([(u.start, u.end) for u in remove_units], reverse=True)
    out = text
    removed = 0
    for start, end in spans:
        if start < 0 or end > len(out):
            continue
        out = out[:start] + out[end:]
        removed += end - start

    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    return TrimResult(
        text=out,
        chars_removed=removed,
        units_removed=len(remove_units),
        shadow_only=False,
        removed_spans=[(u.start, u.end) for u in remove_units],
        protected_skips=protected,
    )
