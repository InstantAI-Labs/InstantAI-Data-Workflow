from __future__ import annotations

import math
from collections import Counter

from typing import Any

from indw.schedule.intel.pci import FingerprintBundle


def char_entropy_norm(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    ent = -sum((c / n) * math.log2(c / n) for c in counts.values())
    max_ent = math.log2(max(1, min(256, len(counts))))
    if max_ent <= 0:
        return 0.0
    return min(1.0, ent / max_ent)


_CODE_MARKERS = ('def ', 'class ', 'import ', '```', 'function ', 'const ')


def structural_complexity(
    text: str,
    fp: FingerprintBundle,
    *,
    scan: Any | None = None,
) -> float:
    if scan is not None:
        line_count = scan.line_count
        code_hits = scan.code_hits
        sections = scan.section_count
        uniq_lens = scan.uniq_section_lens
    else:
        line_count = 0
        code_hits = 0
        for ln in text.splitlines():
            stripped = ln.strip()
            if not stripped:
                continue
            line_count += 1
            if any(m in stripped for m in _CODE_MARKERS):
                code_hits += 1
        if line_count == 0:
            return 0.0
        section_lens = [len(p.split()) for p in text.split('\n\n') if p.strip()]
        sections = len(section_lens)
        uniq_lens = len(set(section_lens))
    if line_count == 0:
        return 0.0
    lines_denom = fp.lines if fp.lines > 0 else line_count
    return min(1.0, (
        min(1.0, sections / 24.0) * 0.28
        + min(1.0, code_hits / max(lines_denom, 1)) * 0.32
        + min(1.0, fp.lines / 240.0) * 0.18
        + min(1.0, uniq_lens / 18.0) * 0.12
        + min(1.0, fp.chars / 48000.0) * 0.10
    ))


def layout_complexity(fp: FingerprintBundle) -> float:
    return min(1.0, (
        (1.0 if fp.wrapper.endswith(fp.structural[:4]) else 0.15)
        + min(1.0, fp.lines / 120.0) * 0.45
        + min(1.0, fp.chars / 24000.0) * 0.40
    ))


def novelty_score(*, observation_count: int, verified: bool) -> float:
    if observation_count <= 0:
        return 1.0
    base = 1.0 / (1.0 + math.log1p(observation_count))
    if verified:
        return max(0.05, base * 0.35)
    return min(1.0, base)


def family_confidence(*, observation_count: int, verified_count: int) -> float:
    if observation_count < 3:
        return 0.0
    verify_ratio = verified_count / max(observation_count, 1)
    mass = min(1.0, math.log1p(observation_count) / math.log1p(48))
    return min(1.0, verify_ratio * 0.55 + mass * 0.45)
