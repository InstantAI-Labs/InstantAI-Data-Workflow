from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from indw.clean.document.code_preservation import (
    detect_code_language,
    validate_code_syntax,
)
from indw.clean.document.patterns import _CODE_FENCE
from indw.filter.license.normalize import detect_license_in_text, normalize_license_string

_FENCE_PLACEHOLDER = re.compile(r'\x00FENCE\d+\x00')
_CHARS_PER_TOKEN = 3.8
_REMOVE_CONFIDENCE = 0.82
_REVIEW_CONFIDENCE = 0.55

_LICENSE_URL = re.compile(
    r'https?://(?:www\.)?(?:opensource\.org|gnu\.org/licenses|apache\.org/licenses|'
    r'creativecommons\.org|spdx\.org/licenses)[^\s\])>]*',
    re.I,
)
_SPDX_LINE = re.compile(
    r'(?im)^\s*(?:#|//|--|/\*|\*)?\s*SPDX-License-Identifier\s*:\s*([^\n*/]+)',
)
def _is_license_boilerplate_block(span: str) -> bool:
    if not span or len(span.strip()) < 40:
        return False
    if _DISCUSSING_LICENSE.search(span):
        return False
    legal = (
        len(_LEGAL_GRANT.findall(span))
        + len(_WARRANTY.findall(span))
        + len(_NUMBERED_CONDITION.findall(span))
    )
    if legal >= 2:
        return True
    if _LEGAL_GRANT.search(span) and len(span) > 180:
        return True
    if legal >= 1 and len(_LICENSE_NAME.findall(span)) >= 1 and len(span) > 120:
        return True
    if re.search(r'(?i)(?:copyright|©|\(c\))\s*.*(?:all\s+rights\s+reserved|permission\s+is\s+hereby)', span):
        return True
    if re.search(r'(?i)gnu\s+(?:general|lesser)\s+public\s+license', span) and len(span) > 80:
        return True
    return False

_LICENSE_HEADER = re.compile(
    r'(?im)^\s*(?:'
    r'(?:the\s+)?(?:mit|apache|bsd|isc|mozilla)\s+license|'
    r'gnu\s+(?:general|lesser)\s+public\s+license|'
    r'creative\s+commons(?:\s+attribution)?'
    r')\s*$',
)

_COPYRIGHT_BLOCK = re.compile(
    r'(?is)(?:'
    r'(?:^|\n)\s*permission\s+is\s+hereby\s+granted[\s\S]*?(?:\n\n|\Z)|'
    r'(?:^|\n)\s*redistribution\s+and\s+use\s+in\s+source\s+and\s+binary\s+forms[\s\S]*?(?:\n\n|\Z)|'
    r'(?:^|\n)\s*gnu\s+(?:general|lesser)\s+public\s+license[\s\S]*?(?:\n\n|\Z)'
    r')',
)

def _find_license_preamble_span(text: str) -> tuple[int, int] | None:
    lines = text.splitlines(keepends=True)
    header_idx: int | None = None
    for i, line in enumerate(lines):
        if _LICENSE_HEADER.match(line.strip()) or _COPYRIGHT_LINE.match(line.strip()):
            header_idx = i
            break
        if _SPDX_LINE.search(line):
            header_idx = i
            break
    if header_idx is None:
        return None
    offset = sum(len(lines[j]) for j in range(header_idx))
    end = offset
    saw_grant = False
    for j in range(header_idx, len(lines)):
        ln = lines[j]
        stripped = ln.strip()
        end += len(ln)
        if _LEGAL_GRANT.search(stripped):
            saw_grant = True
        if not stripped:
            if saw_grant and j > header_idx + 4:
                break
            continue
        if _CODE_SYNTAX.match(ln) and not _COMMENT_PREFIX.match(ln):
            end -= len(ln)
            break
        if _FENCE_PLACEHOLDER.search(ln) or stripped.startswith('```'):
            end -= len(ln)
            break
        if saw_grant and _WARRANTY.search(stripped):
            if j + 1 < len(lines) and not lines[j + 1].strip():
                end += len(lines[j + 1])
            break
        if saw_grant and j > header_idx + 8 and not _LEGAL_GRANT.search(stripped):
            if not _LICENSE_NAME.search(stripped) and not _NUMBERED_CONDITION.search(stripped):
                if _WARRANTY.search('\n'.join(lines[header_idx:j + 1])):
                    break
        if j > header_idx + 80:
            break
    span = text[offset:end]
    if _is_license_boilerplate_block(span):
        return offset, end
    if header_idx > 0 and not _SPDX_LINE.search(lines[header_idx]):
        prior = ''.join(lines[:header_idx])
        if len(prior.split()) > 12 and not _is_license_boilerplate_block(prior):
            return None
    if _LICENSE_HEADER.match(lines[header_idx].strip()) and _LEGAL_GRANT.search(span):
        return offset, end
    if _SPDX_LINE.search(lines[header_idx]) and len(span) < 200:
        return offset, end
    return None

_COPYRIGHT_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'copyright\s*(?:©|\(c\)|\(C\))?|'
    r'©\s*\d{4}|'
    r'all\s+rights\s+reserved'
    r')\b'
)
_WARRANTY = re.compile(
    r'(?i)\b(?:'
    r'without\s+(?:any\s+)?warranty|disclaimer\s+of\s+warranty|'
    r'as\s+is["\']?\s*,?\s*without|'
    r'fitness\s+for\s+a\s+particular\s+purpose|merchantability|'
    r'liable\s+for\s+any\s+(?:damages|claim)'
    r')\b'
)
_LEGAL_GRANT = re.compile(
    r'(?i)\b(?:'
    r'permission\s+is\s+hereby\s+granted|redistribution\s+and\s+use|'
    r'provided\s+that\s+the\s+following\s+conditions|'
    r'subject\s+to\s+the\s+terms\s+of\s+the\s+(?:mit|apache|bsd|gpl|isc)\s+license|'
    r'licensed\s+under\s+the\s+(?:mit|apache|bsd|gpl|lgpl|mpl|isc)'
    r')\b'
)
_LICENSE_NAME = re.compile(
    r'(?i)\b(?:'
    r'mit\s+license|apache\s+license(?:\s+version\s+2\.0)?|bsd\s+license|'
    r'isc\s+license|gnu\s+(?:general|lesser)\s+public\s+license|'
    r'mozilla\s+public\s+license|creative\s+commons|'
    r'cc[\s-]?by(?:[\s-]?nc)?(?:[\s-]?sa)?|'
    r'lgpl|gpl[\s-]?(?:2|3)|mpl[\s-]?2'
    r')\b'
)
_NUMBERED_CONDITION = re.compile(r'(?m)^\s*\d+[\.\)]\s+(?:redistributions?|this\s+software|the\s+above)')
_COLLAPSED_CODE_START = re.compile(
    r'(?i)(?:'
    r'\bwith\s+[A-Z][\w.]*(?:\.[A-Z][\w.]*)*\s*;|'
    r'\bpackage(?:\s+body)?\s+[A-Z][\w.]*|'
    r'\b(?:procedure|function)\s+[A-Z][\w.]*|'
    r'\b#pragma\b|\b#include\s+[<"]|'
    r'\b(?:def|class|import|from|namespace)\s+[A-Za-z_]'
    r')',
)
_INLINE_LEGAL_SIGNAL = re.compile(
    r'(?i)(?:'
    r'apache\.org/licenses|gnu\.org/licenses|opensource\.org/licenses|'
    r'compliance\s+with\s+the\s+license|'
    r'without\s+warranties|without\s+any\s+warranty|'
    r'permission\s+is\s+hereby|'
    r'spdx-license-identifier|'
    r'copyright\s*(?:\(c\)|©)?|'
    r'limitations\s+under\s+the\s+license|'
    r'licensed\s+under\s+the|'
    r'obtain\s+a\s+copy\s+of\s+the\s+license|'
    r'all\s+rights\s+reserved|'
    r'redistribution\s+and\s+use'
    r')',
)
_INLINE_LICENSE_CHUNK = re.compile(
    r'(?is)(?:'
    r'[-=]{8,}\s*|'
    r'--\s*(?:'
    r'copyright|©|\(c\)|all\s+rights\s+reserved|'
    r'(?:gnu|apache|mit|bsd|lgpl|mpl|zlib|isc)\s+license|'
    r'permission\s+is\s+hereby|redistribution|'
    r'spdx-license-identifier|generated\s+(?:automatically|by)|'
    r'free\s+software\s+foundation|gnat\s+library|gnarl|'
    r'licensed\s+under|do\s+not\s+edit|'
    r'written\s+by|author\s*:|version\s*:|'
    r'\$revision\s*:|see\s+license\s+for\s+details|'
    r'limitations\s+under\s+the\s+license|'
    r'compliance\s+with\s+the\s+license|'
    r'without\s+warranties'
    r')[^-]*'
    r')'
)
_COMMENT_PREFIX = re.compile(r'^\s*(?:#|//|--|/\*|\*)\s*')
_EDUCATIONAL = re.compile(
    r'(?i)\b(?:'
    r'example|tutorial|implement|algorithm|parameter|returns?|raises?|'
    r'@param|@return|@example|usage|note\s+that|because|therefore|'
    r'explain|function|class|method|api|how\s+to|step\s+\d|'
    r'popular\s+because|often\s+choose|developers?\s+often|is\s+popular|'
    r'discuss|overview|compared\s+to|commonly\s+used'
    r')\b'
)
_DISCUSSING_LICENSE = re.compile(
    r'(?i)(?:'
    r'(?:the|an?)\s+\w+\s+license\s+is\s+(?:popular|common|permissive)|'
    r'license\s+is\s+popular\s+because|developers?\s+often\s+choose|'
    r'compared\s+to\s+(?:gpl|mit|apache|bsd)'
    r')'
)
_SPDX_OR_COPYRIGHT_LINE = re.compile(
    r'(?i)^\s*(?:#|//|--|/\*|\*)?\s*(?:'
    r'SPDX-License-Identifier\s*:|'
    r'copyright\s*(?:©|\(c\)|\(C\))?|'
    r'licensed\s+under\s+the|all\s+rights\s+reserved'
    r')'
)
_CODE_SYNTAX = re.compile(
    r'^\s*(?:def |class |import |from |package |procedure |function |'
    r'#include|public |private |fn |func |const |let |var )',
    re.M,
)
_INFORMATIVE_DOC = re.compile(
    r'(?i)(?:'
    r'"""[\s\S]{20,}"""|'
    r"'''[\s\S]{20,}'''|"
    r'@\w+\s+\S|'
    r'\b(?:explain|because|note\s+that|important|warning|example|usage)\b'
    r')'
)

@dataclass
class LicenseRegion:
    start: int
    end: int
    license_type: str = 'Unknown'
    confidence: float = 0.0
    region_kind: str = 'block'
    action: Literal['remove', 'preserve', 'review'] = 'preserve'
    reason: str = ''

@dataclass
class LicenseCleanStats:
    license_detected: str = 'Unknown'
    license_confidence: float = 0.0
    chars_before: int = 0
    chars_after: int = 0
    tokens_removed: int = 0
    regions_removed: int = 0
    regions_preserved_uncertain: int = 0
    regions_flagged_review: int = 0
    syntax_checks: int = 0
    syntax_failures: int = 0
    license_types_found: dict[str, int] = field(default_factory=dict)
    samples: list[dict[str, str]] = field(default_factory=list)

    @property
    def chars_removed(self) -> int:
        return max(0, self.chars_before - self.chars_after)

    def to_dict(self) -> dict:
        return {
            'license_detected': self.license_detected,
            'license_confidence': round(self.license_confidence, 4),
            'chars_before': self.chars_before,
            'chars_after': self.chars_after,
            'chars_removed': self.chars_removed,
            'tokens_removed': self.tokens_removed,
            'regions_removed': self.regions_removed,
            'regions_preserved_uncertain': self.regions_preserved_uncertain,
            'regions_flagged_review': self.regions_flagged_review,
            'syntax_checks': self.syntax_checks,
            'syntax_failures': self.syntax_failures,
            'license_types_found': dict(self.license_types_found),
            'samples': self.samples[:6],
        }

def _score_region(text: str, *, offset: int = 0) -> tuple[float, str, list[str]]:
    if not text or not text.strip():
        return 0.0, 'Unknown', []
    sample = text[:4000]
    signals: list[str] = []
    score = 0.0

    lic_type, lic_conf = detect_license_in_text(sample)
    if lic_type != 'Unknown':
        score += 0.35 * lic_conf
        signals.append(f'detected:{lic_type}')

    legal_hits = len(_LEGAL_GRANT.findall(sample)) + len(_WARRANTY.findall(sample))
    legal_hits += len(_LICENSE_NAME.findall(sample))
    legal_hits += len(_LICENSE_URL.findall(sample))
    legal_hits += len(_NUMBERED_CONDITION.findall(sample))
    spdx = _SPDX_LINE.search(sample)
    if spdx:
        legal_hits += 2
        canon, _ = normalize_license_string(spdx.group(1).strip())
        if canon != 'Unknown':
            lic_type = lic_type if lic_type != 'Unknown' else canon

    edu_hits = len(_EDUCATIONAL.findall(sample))
    code_hits = len(_CODE_SYNTAX.findall(sample)) + len(_FENCE_PLACEHOLDER.findall(sample))
    doc_hits = 1 if _INFORMATIVE_DOC.search(sample) else 0
    words = max(len(sample.split()), 1)

    legal_density = min(1.0, legal_hits / max(words / 12, 1))
    edu_density = min(1.0, (edu_hits + code_hits * 2 + doc_hits * 2) / max(words / 20, 1))

    score += legal_density * 0.45
    score -= edu_density * 0.40
    if code_hits >= 2 and edu_hits >= 1:
        score -= 0.25
        signals.append('educational_code')
    if doc_hits:
        score -= 0.15
        signals.append('informative_doc')

    comment_lines = sum(1 for ln in sample.splitlines() if _COMMENT_PREFIX.match(ln))
    line_count = max(len([ln for ln in sample.splitlines() if ln.strip()]), 1)
    if comment_lines / line_count > 0.7 and legal_hits >= 1:
        score += 0.12
        signals.append('comment_license_block')

    if re.search(r'(?i)copyright\s*(?:©|\(c\))', sample) and not _DISCUSSING_LICENSE.search(sample):
        score += 0.10
        signals.append('copyright')
    if 'all rights reserved' in sample.lower() and edu_hits == 0:
        score += 0.08

    if _DISCUSSING_LICENSE.search(sample):
        score -= 0.35
        signals.append('discussing_license')
    if edu_hits >= 2 and legal_hits <= 2:
        score -= 0.20

    score = max(0.0, min(1.0, score))
    if lic_type == 'Unknown' and legal_hits >= 2:
        lic_type = 'Unknown-Legal'
    return score, lic_type, signals

def _iter_blocks(text: str) -> list[tuple[int, int, str, str]]:
    blocks: list[tuple[int, int, str, str]] = []
    pos = 0
    for match in _CODE_FENCE.finditer(text):
        if match.start() > pos:
            prose = text[pos:match.start()]
            for m in re.finditer(r'(?:\n\s*\n|^)(.+?)(?=\n\s*\n|\Z)', prose, re.S):
                chunk = m.group(1)
                if chunk.strip():
                    start = pos + m.start(1)
                    blocks.append((start, start + len(chunk), chunk, 'prose'))
        inner = match.group(0)
        lang_m = re.match(r'^```(\w*)', inner)
        lang = lang_m.group(1) if lang_m else ''
        body = re.sub(r'^```\w*\n?', '', inner)
        body = re.sub(r'\n?```\s*$', '', body)
        blocks.append((match.start(), match.end(), body, f'code:{lang or "unknown"}'))
        pos = match.end()
    for match in _FENCE_PLACEHOLDER.finditer(text):
        if match.start() > pos:
            prose = text[pos:match.start()]
            for m in re.finditer(r'(?:\n\s*\n|^)(.+?)(?=\n\s*\n|\Z)', prose, re.S):
                chunk = m.group(1)
                if chunk.strip():
                    start = pos + m.start(1)
                    blocks.append((start, start + len(chunk), chunk, 'prose'))
        blocks.append((match.start(), match.end(), match.group(0), 'code:fenced'))
        pos = max(pos, match.end())
    if pos < len(text):
        prose = text[pos:]
        for m in re.finditer(r'(?:\n\s*\n|^)(.+?)(?=\n\s*\n|\Z)', prose, re.S):
            chunk = m.group(1)
            if chunk.strip():
                start = pos + m.start(1)
                blocks.append((start, start + len(chunk), chunk, 'prose'))
    if not blocks and text.strip():
        blocks.append((0, len(text), text, 'prose'))
    return blocks

def _iter_comment_runs(lines: list[str]) -> list[tuple[int, int, str]]:
    runs: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        if not _COMMENT_PREFIX.match(lines[i]) and not lines[i].strip().startswith('/*'):
            i += 1
            continue
        start = i
        buf = [lines[i]]
        i += 1
        while i < len(lines):
            ln = lines[i]
            if not ln.strip():
                break
            if _COMMENT_PREFIX.match(ln) or ln.strip().startswith('*') or '*/' in ln:
                buf.append(ln)
                i += 1
                if '*/' in ln:
                    break
                continue
            break
        runs.append((start, i, '\n'.join(buf)))
    return runs

def detect_license_regions(
    text: str,
    *,
    remove_confidence: float = _REMOVE_CONFIDENCE,
    review_confidence: float = _REVIEW_CONFIDENCE,
) -> list[LicenseRegion]:
    if not text or not text.strip():
        return []

    regions: list[LicenseRegion] = []
    doc_lic, doc_conf = detect_license_in_text(text[:16000])
    preamble = _find_license_preamble_span(text)

    lines = text.splitlines(keepends=True)
    line_pos = 0
    for i, line in enumerate(lines):
        line_start = line_pos
        line_end = line_pos + len(line)
        line_pos = line_end
        if not _SPDX_OR_COPYRIGHT_LINE.match(line):
            continue
        if preamble and line_start >= preamble[0] and line_end <= preamble[1]:
            continue
        conf, lic, signals = _score_region(line)
        if 'discussing_license' in signals:
            continue
        if _SPDX_LINE.search(line) or _COPYRIGHT_LINE.match(line.strip()):
            action: Literal['remove', 'preserve', 'review'] = 'remove'
            conf = max(conf, 0.92)
        else:
            action = 'remove' if conf >= remove_confidence else 'review'
        regions.append(LicenseRegion(
            start=line_start, end=line_end,
            license_type=lic, confidence=conf,
            region_kind='license_line', action=action,
            reason='spdx_or_copyright_line',
        ))

    for start, end, block, kind in _iter_blocks(text):
        if kind.startswith('code:'):
            lang_hint = kind.split(':', 1)[1]
            lines = block.splitlines()
            for run_start, run_end, run_text in _iter_comment_runs(lines):
                conf, lic, signals = _score_region(run_text)
                if conf < review_confidence:
                    continue
                line_off = sum(len(lines[j]) + 1 for j in range(run_start))
                abs_start = start + line_off
                abs_end = abs_start + len(run_text)
                action: Literal['remove', 'preserve', 'review'] = 'preserve'
                if conf >= remove_confidence and 'educational_code' not in signals:
                    action = 'remove'
                elif conf >= review_confidence:
                    action = 'review'
                regions.append(LicenseRegion(
                    start=abs_start, end=abs_end,
                    license_type=lic if lic != 'Unknown' else doc_lic,
                    confidence=conf, region_kind='code_comment',
                    action=action, reason=','.join(signals[:4]),
                ))
            block_conf, block_lic, block_signals = _score_region(block)
            if block_conf >= review_confidence and 'educational_code' not in block_signals:
                action = 'remove' if block_conf >= remove_confidence else 'review'
                if action != 'preserve':
                    regions.append(LicenseRegion(
                        start=start, end=end,
                        license_type=block_lic if block_lic != 'Unknown' else doc_lic,
                        confidence=block_conf, region_kind='code_block',
                        action=action, reason=','.join(block_signals[:4]),
                    ))
            continue

        code_lines = [
            ln for ln in block.splitlines()
            if ln.strip() and _CODE_SYNTAX.match(ln) and not _COMMENT_PREFIX.match(ln)
        ]
        if code_lines:
            continue

        conf, lic, signals = _score_region(block)
        if conf < review_confidence:
            continue
        action = 'remove' if conf >= remove_confidence else 'review'
        if 'educational_code' in signals or 'informative_doc' in signals:
            if conf < 0.90:
                action = 'review' if conf >= review_confidence else 'preserve'
        regions.append(LicenseRegion(
            start=start, end=end,
            license_type=lic if lic != 'Unknown' else doc_lic,
            confidence=max(conf, doc_conf * 0.5 if doc_lic != 'Unknown' else 0),
            region_kind='prose', action=action, reason=','.join(signals[:4]),
        ))

    if preamble:
        pstart, pend = preamble
        span = text[pstart:pend]
        conf, lic, signals = _score_region(span)
        regions.append(LicenseRegion(
            start=pstart, end=pend,
            license_type=lic if lic != 'Unknown' else doc_lic,
            confidence=max(conf, 0.90), region_kind='license_preamble',
            action='remove', reason='license_preamble,' + ','.join(signals[:3]),
        ))
    for m in _COPYRIGHT_BLOCK.finditer(text):
        span = m.group(0)
        if not _is_license_boilerplate_block(span):
            continue
        if preamble and m.start() >= preamble[0] and m.end() <= preamble[1]:
            continue
        conf, lic, signals = _score_region(span)
        regions.append(LicenseRegion(
            start=m.start(), end=m.end(),
            license_type=lic if lic != 'Unknown' else doc_lic,
            confidence=max(conf, 0.88), region_kind='regex_block',
            action='remove', reason='copyright_block,' + ','.join(signals[:3]),
        ))

    for m in re.finditer(
        r'(?is)(?:^|\n\n)\s*gnu\s+(?:general|lesser)\s+public\s+license[\s\S]*?(?=\n\n[A-Z]|\Z)',
        text,
    ):
        span = m.group(0)
        if not _is_license_boilerplate_block(span):
            continue
        conf, lic, signals = _score_region(span)
        regions.append(LicenseRegion(
            start=m.start(), end=m.end(),
            license_type=lic if lic != 'Unknown' else 'GPL',
            confidence=max(conf, 0.90), region_kind='gpl_block',
            action='remove', reason='gpl_block,' + ','.join(signals[:3]),
        ))

    for m in _LICENSE_URL.finditer(text):
        line_start = text.rfind('\n', 0, m.start()) + 1
        line_end = text.find('\n', m.end())
        if line_end < 0:
            line_end = len(text)
        line = text[line_start:line_end]
        conf, lic, _ = _score_region(line)
        if conf >= review_confidence or _LICENSE_NAME.search(line):
            regions.append(LicenseRegion(
                start=line_start, end=line_end,
                license_type=lic, confidence=max(conf, 0.75),
                region_kind='license_url_line', action='remove',
                reason='license_url',
            ))

    regions.sort(key=lambda r: (r.start, -(r.end - r.start)))
    merged: list[LicenseRegion] = []
    for reg in regions:
        if merged and reg.start < merged[-1].end:
            prev = merged[-1]
            if reg.confidence > prev.confidence:
                merged[-1] = LicenseRegion(
                    start=min(prev.start, reg.start),
                    end=max(prev.end, reg.end),
                    license_type=reg.license_type if reg.license_type != 'Unknown' else prev.license_type,
                    confidence=max(prev.confidence, reg.confidence),
                    region_kind=reg.region_kind,
                    action=reg.action if reg.confidence >= prev.confidence else prev.action,
                    reason=reg.reason,
                )
            continue
        merged.append(reg)
    return merged

def strip_collapsed_inline_license(text: str, *, max_chunks: int = 120) -> tuple[str, int]:
    if not text or text.count('\n') > 12 or '```' in text:
        return text, 0

    out = text.strip()
    removed = 0

    while removed < max_chunks:
        match = _INLINE_LICENSE_CHUNK.match(out)
        if not match:
            break
        out = out[match.end():].lstrip()
        removed += 1

    code_m = _COLLAPSED_CODE_START.search(out)
    if code_m and code_m.start() >= 20:
        prefix = out[:code_m.start()].strip()
        legal_hits = len(_INLINE_LEGAL_SIGNAL.findall(prefix))
        ada_comments = len(re.findall(r'--\s*\S', prefix))
        should_strip = (
            legal_hits >= 1
            or 'apache.org/licenses' in prefix.lower()
            or ada_comments >= 3
            or (ada_comments >= 2 and len(prefix) > 120 and _WARRANTY.search(prefix))
        )
        if should_strip:
            out = out[code_m.start():].lstrip()
            out = re.sub(r'^[-=]{10,}\s*', '', out)
            removed += max(1, len(prefix))

    if removed > 0:
        out = re.sub(r'^[.\s]+', '', out)
    return out, removed

def _apply_removals(text: str, regions: list[LicenseRegion]) -> tuple[str, list[LicenseRegion]]:
    removed: list[LicenseRegion] = []
    preserved_review: list[LicenseRegion] = []
    out_parts: list[str] = []
    pos = 0
    for reg in sorted(regions, key=lambda r: r.start):
        if reg.action == 'preserve':
            continue
        if reg.action == 'review':
            preserved_review.append(reg)
            continue
        if reg.start < pos:
            continue
        out_parts.append(text[pos:reg.start])
        removed.append(reg)
        pos = reg.end
    out_parts.append(text[pos:])
    result = ''.join(out_parts)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip(), removed

def _validate_code_blocks(text: str, stats: LicenseCleanStats) -> str:
    def _repl(match: re.Match[str]) -> str:
        block = match.group(0)
        lang_m = re.match(r'^```(\w*)', block)
        hint = lang_m.group(1) if lang_m else ''
        inner = re.sub(r'^```\w*\n?', '', block)
        inner = re.sub(r'\n?```\s*$', '', inner)
        lang = detect_code_language(inner, hint=hint)
        stats.syntax_checks += 1
        ok, issues = validate_code_syntax(inner, lang)
        if not ok and 'truncated' not in ''.join(issues):
            stats.syntax_failures += 1
        prefix = f'```{hint}\n' if hint else '```\n'
        return prefix + inner + '\n```'

    return _CODE_FENCE.sub(_repl, text)

def clean_license_content(
    text: str,
    *,
    enabled: bool = True,
    remove_confidence: float = _REMOVE_CONFIDENCE,
    review_confidence: float = _REVIEW_CONFIDENCE,
    validate_syntax: bool = True,
    max_samples: int = 3,
) -> tuple[str, LicenseCleanStats]:
    stats = LicenseCleanStats(chars_before=len(text or ''))
    if not enabled or not text or not text.strip():
        stats.chars_after = len(text or '')
        return text, stats

    regions = detect_license_regions(
        text,
        remove_confidence=remove_confidence,
        review_confidence=review_confidence,
    )
    for reg in regions:
        if reg.license_type and reg.license_type not in ('Unknown', 'Unknown-Legal'):
            stats.license_types_found[reg.license_type] = (
                stats.license_types_found.get(reg.license_type, 0) + 1
            )

    if regions:
        best = max(regions, key=lambda r: r.confidence)
        stats.license_detected = best.license_type
        stats.license_confidence = best.confidence
        doc_lic, doc_conf = detect_license_in_text(text[:16000])
        if doc_conf > stats.license_confidence and doc_lic != 'Unknown':
            stats.license_detected = doc_lic
            stats.license_confidence = doc_conf

    before_snip = text[:300]
    cleaned, removed = _apply_removals(text, regions)
    collapsed, collapsed_removed = strip_collapsed_inline_license(cleaned)
    if collapsed_removed > 0:
        cleaned = collapsed
        stats.regions_removed += 1
        stats.tokens_removed += int(collapsed_removed / _CHARS_PER_TOKEN)
    stats.regions_removed += len(removed)
    stats.regions_flagged_review = sum(1 for r in regions if r.action == 'review')
    stats.regions_preserved_uncertain = stats.regions_flagged_review

    if validate_syntax and '```' in cleaned:
        cleaned = _validate_code_blocks(cleaned, stats)

    stats.chars_after = len(cleaned)
    stats.tokens_removed = int(stats.chars_removed / _CHARS_PER_TOKEN)

    if stats.chars_removed > 0 and len(stats.samples) < max_samples:
        stats.samples.append({
            'license': stats.license_detected,
            'confidence': str(round(stats.license_confidence, 3)),
            'tokens_removed': str(stats.tokens_removed),
            'before': before_snip,
            'after': cleaned[:300],
        })

    return cleaned, stats
