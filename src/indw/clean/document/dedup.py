from __future__ import annotations

import re

from indw.dedup.normalize import normalize_for_dedup
from indw.clean.document.stats import StageStats

def dedupe_paragraphs(text: str, *, stats: StageStats | None = None) -> tuple[str, float]:
    if not text:
        return text, 0.0
    blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    if len(blocks) < 2:
        return text, 0.0
    seen: set[str] = set()
    kept: list[str] = []
    dupes = 0
    for block in blocks:
        norm = normalize_for_dedup(block)
        if not norm or len(norm) < 24:
            kept.append(block)
            continue
        if norm in seen:
            dupes += 1
            continue
        seen.add(norm)
        kept.append(block)
    out = '\n\n'.join(kept)
    ratio = dupes / max(len(blocks), 1)
    if stats is not None:
        stats.in_docs += 1
        stats.out_docs += 1 if out else 0
        stats.dropped += dupes
    return out, ratio
