from __future__ import annotations

import re

from indw.clean.document.license import clean_license_content
from indw.clean.semantic.section_artifacts import (
    find_promotional_tail_start,
    line_should_remove,
    score_section_artifact,
)
from indw.filter.score.artifacts import analyze_artifact_signals

_WS = re.compile(r'[ \t]+')

_REMOVE_ROLES = frozenset({
    'navigation', 'footer', 'contact', 'promotional', 'related_content',
})
_TRIM_ROLES = frozenset({
    'metadata', 'author_info', 'legal',
})
_BODY_ROLES = frozenset({
    'title', 'introduction', 'body', 'examples', 'references', 'code', 'table',
})

def _strip_promotional_tail(text: str) -> tuple[str, int]:
    cut = find_promotional_tail_start(text)
    if cut is None or cut <= 0:
        return text, 0
    return text[:cut].rstrip(), len(text) - cut

def _filter_lines(
    lines: list[str],
    *,
    role: str,
    position_ratio: float,
    preserve_educational: bool,
) -> tuple[list[str], dict[str, int]]:
    kept: list[str] = []
    removed: dict[str, int] = {}
    n = max(len(lines), 1)
    for i, line in enumerate(lines):
        pos = position_ratio + (i / n) * 0.06
        drop, kind = line_should_remove(
            line,
            position_ratio=pos,
            section_role=role,
            preserve_educational=preserve_educational,
        )
        if drop:
            removed[kind] = removed.get(kind, 0) + 1
            continue
        kept.append(line.rstrip())
    return kept, removed

def clean_section_text(
    text: str,
    *,
    role: str = 'body',
    position_ratio: float = 0.5,
    preserve_educational: bool = True,
) -> tuple[str, dict[str, int]]:
    if not text or not text.strip():
        return '', {}

    stats: dict[str, int] = {}

    if role in _REMOVE_ROLES:
        profile = score_section_artifact(text, position_ratio=position_ratio, section_role=role)
        kind, score = profile.dominant()
        if score >= 0.38 and not preserve_educational:
            stats[kind] = stats.get(kind, 0) + 1
            return '', stats
        lines, removed = _filter_lines(
            text.splitlines(),
            role=role,
            position_ratio=position_ratio,
            preserve_educational=False,
        )
        for k, v in removed.items():
            stats[k] = stats.get(k, 0) + v
        out = '\n'.join(lines).strip()
        return out, stats

    if role == 'legal':
        cleaned, _ = clean_license_content(text, enabled=True)
        stats['legal'] = stats.get('legal', 0) + max(0, len(text) - len(cleaned))
        return cleaned.strip(), stats

    if role in _TRIM_ROLES:
        lines, removed = _filter_lines(
            text.splitlines(),
            role=role,
            position_ratio=position_ratio,
            preserve_educational=preserve_educational,
        )
        for k, v in removed.items():
            stats[k] = stats.get(k, 0) + v
        out = '\n'.join(lines).strip()
        if not out and preserve_educational:
            return text.strip(), stats
        return out, stats

    lines = text.splitlines()
    if role in _BODY_ROLES:
        filtered, removed = _filter_lines(
            lines,
            role=role,
            position_ratio=position_ratio,
            preserve_educational=preserve_educational,
        )
        for k, v in removed.items():
            stats[k] = stats.get(k, 0) + v
        lines = filtered

        ocr_removed = 0
        kept_lines: list[str] = []
        for ln in lines:
            sig = analyze_artifact_signals(ln)
            if sig.ocr_corruption >= 0.78 and len(ln.split()) >= 5:
                ocr_removed += 1
                stats['ocr_noise'] = stats.get('ocr_noise', 0) + 1
                continue
            if sig.dict_ui >= 0.88 and len(ln) < 140:
                ocr_removed += 1
                stats['ocr_noise'] = stats.get('ocr_noise', 0) + 1
                continue
            kept_lines.append(ln)
        lines = kept_lines

    out = '\n'.join(lines)
    if role in _BODY_ROLES:
        trimmed, ncut = _strip_promotional_tail(out)
        if ncut > 0:
            stats['promotional'] = stats.get('promotional', 0) + 1
            out = trimmed

    out = re.sub(r'\n{3,}', '\n\n', out)
    if role in ('navigation', 'footer'):
        out = _WS.sub(' ', out)

    if role in ('body', 'introduction', 'title', 'examples', 'references'):
        cleaned, _ = clean_license_content(out, enabled=True)
        out = cleaned

    return out.strip(), stats
