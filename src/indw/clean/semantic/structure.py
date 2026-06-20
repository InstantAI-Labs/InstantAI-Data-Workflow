from __future__ import annotations

import re
from dataclasses import dataclass, field

from indw.clean.artifact.decompose import LayoutVector, compute_layout
from indw.clean.semantic.fingerprints import SemanticFingerprintMatcher
from indw.clean.semantic.section_artifacts import score_section_artifact
from indw.extract.nav.context import get_navigation_context, score_navigation_role
from indw.clean.artifact.evidence import DocumentFeatureExtractor
from indw.clean.artifact.evidence_engine import compute_semantic_evidence

_FENCE = re.compile(r'```[\s\S]*?```', re.M)
_HEADING = re.compile(r'^(?:#{1,6}\s+.+|={2,}\s*.+\s*={2,})\s*$', re.M)
_PIPE_NAV = re.compile(r'\|')
_TABLE_ROW = re.compile(r'^\s*\|?.+\|.+\|?\s*$')

SECTION_ROLES = (
    'title',
    'introduction',
    'body',
    'code',
    'examples',
    'references',
    'table',
    'metadata',
    'navigation',
    'footer',
    'contact',
    'legal',
    'author_info',
    'related_content',
    'promotional',
)

@dataclass
class SemanticSection:
    text: str
    start: int
    end: int
    position_ratio: float = 0.0
    structural_kind: str = 'body'
    section_role: str = 'body'
    in_fence: bool = False
    unit_id: str = ''
    layout: LayoutVector = field(default_factory=LayoutVector)

    @property
    def kind(self) -> str:
        return self.section_role

def _is_nav_pipe_block(text: str, layout: LayoutVector) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or len(lines) > 6:
        return False
    pipe_lines = sum(1 for ln in lines if _PIPE_NAV.search(ln))
    if pipe_lines < len(lines):
        return False
    avg_len = sum(len(ln) for ln in lines) / len(lines)
    return avg_len < 140 and layout.list_ratio < 0.7

def _is_real_table(text: str) -> bool:
    rows = [ln for ln in text.splitlines() if _TABLE_ROW.match(ln)]
    if len(rows) < 2:
        return False
    if _is_nav_pipe_block(text, compute_layout(text)):
        return False
    return True

def infer_section_role(
    text: str,
    *,
    layout: LayoutVector,
    position_ratio: float,
    in_fence: bool = False,
    structural_kind: str = 'body',
) -> str:
    if in_fence or structural_kind == 'code':
        return 'code'

    raw = DocumentFeatureExtractor().extract(text)
    fp = SemanticFingerprintMatcher().match(text)
    ev = compute_semantic_evidence(text)

    if _is_real_table(text):
        return 'table'
    if _is_nav_pipe_block(text, layout):
        return 'navigation'

    if position_ratio < 0.16 and layout.line_count <= 6 and layout.avg_len > 50:
        return 'introduction'
    if raw.line_count <= 2 and raw.nav_line_ratio < 0.12 and len(text) < 160 and ev.utility > 0.04:
        return 'introduction'
    if position_ratio < 0.22 and raw.word_count > 12 and raw.nav_line_ratio < 0.25 and ev.utility > 0.12:
        return 'body'
    if ev.utility > 0.18 and raw.fence_char_ratio > 0.03:
        return 'body'
    if ev.utility > 0.20 and position_ratio > 0.30 and raw.nav_line_ratio < 0.35:
        return 'body'
    if ev.quality.technical > 0.22 and ev.utility > 0.12 and raw.nav_line_ratio < 0.2:
        return 'body'

    nav = score_navigation_role(
        text,
        position_ratio=position_ratio,
        layout=layout,
        ctx=get_navigation_context(),
    )
    nav_role, nav_conf = nav.dominant()
    if nav.is_navigation(threshold=0.50) and nav.article < 0.38 and nav_role in (
        'navigation', 'breadcrumb', 'menu', 'pagination', 'sidebar', 'sitemap',
    ):
        return 'navigation'
    if nav_role == 'footer' and nav.footer > 0.48:
        return 'footer'

    if raw.contact_token_ratio > 0.10 or raw.schedule_token_ratio > 0.08:
        if layout.line_count <= 8 or position_ratio > 0.55:
            return 'contact'

    if structural_kind == 'header' or _HEADING.match(text.strip()):
        if position_ratio < 0.12:
            return 'title'
        if position_ratio > 0.82:
            return 'related_content'
        if ev.utility > 0.12 or raw.word_count > 20:
            return 'body'
        return 'body'

    if position_ratio > 0.88 and (fp.get('footer', 0.0) > 0.55 or layout.line_count <= 3):
        return 'footer'
    if position_ratio < 0.10 and fp.get('navigation', 0.0) > 0.55 and layout.list_ratio > 0.25:
        return 'navigation'
    if fp.get('license', 0.0) > 0.72 and raw.structured_line_ratio > 0.35 and raw.word_count < 120:
        return 'legal'
    if fp.get('seo', 0.0) > 0.65 and layout.line_count <= 5 and position_ratio < 0.2:
        return 'metadata'
    if position_ratio < 0.16 and layout.line_count <= 6 and layout.avg_len > 50:
        return 'introduction'
    if position_ratio > 0.80 and layout.list_ratio > 0.45:
        return 'related_content'
    if position_ratio > 0.74 and layout.avg_len < 240 and layout.line_count <= 4:
        return 'author_info'

    art = score_section_artifact(text, position_ratio=position_ratio, section_role='body')
    if position_ratio > 0.70 and art.promotional >= 0.40:
        return 'promotional'
    if position_ratio > 0.78 and art.contact >= 0.42:
        return 'contact'

    if raw.step_line_hits >= 2 or (raw.copula_def_hits >= 1 and layout.list_ratio > 0.2):
        if 0.12 < position_ratio < 0.75:
            return 'examples'
    if raw.citation_hits >= 2 and position_ratio > 0.62:
        return 'references'
    if position_ratio < 0.14 and art.news_meta >= 0.38:
        return 'metadata'

    return 'body'

def _split_prose(
    chunk: str,
    base_offset: int,
    total_len: int,
    min_chars: int,
) -> list[tuple[str, int, int]]:
    if not chunk.strip():
        return []
    paras = re.split(r'\n\s*\n+', chunk)
    out: list[tuple[str, int, int]] = []
    local = 0
    for para in paras:
        para = para.strip()
        if not para:
            local += 2
            continue
        if len(para) < min_chars and out:
            prev_text, prev_start, _ = out[-1]
            out[-1] = (f'{prev_text}\n\n{para}', prev_start, base_offset + local + len(para))
            local += len(para) + 2
            continue
        start = base_offset + local
        end = start + len(para)
        out.append((para, start, end))
        local += len(para) + 2
    return out

def segment_sections(text: str, *, min_section_chars: int = 80) -> list[SemanticSection]:
    if not text or not text.strip():
        return []

    total = max(len(text), 1)
    raw_parts: list[tuple[str, int, int, bool]] = []
    cursor = 0

    for m in _FENCE.finditer(text):
        if m.start() > cursor:
            raw_parts.extend([
                (t, s, e, False)
                for t, s, e in _split_prose(text[cursor:m.start()], cursor, total, min_section_chars)
            ])
        raw_parts.append((m.group(0), m.start(), m.end(), True))
        cursor = m.end()

    if cursor < len(text):
        raw_parts.extend([
            (t, s, e, False)
            for t, s, e in _split_prose(text[cursor:], cursor, total, min_section_chars)
        ])

    if not raw_parts:
        raw_parts = [(text.strip(), 0, len(text), False)]

    sections: list[SemanticSection] = []
    for part_text, start, end, in_fence in raw_parts:
        layout = compute_layout(part_text, in_fence=in_fence)
        skind = 'code' if in_fence else ('table' if _is_real_table(part_text) else 'paragraph')
        if _HEADING.match(part_text.strip()):
            skind = 'header'
        role = infer_section_role(
            part_text,
            layout=layout,
            position_ratio=start / total,
            in_fence=in_fence,
            structural_kind=skind,
        )
        sections.append(SemanticSection(
            text=part_text,
            start=start,
            end=end,
            position_ratio=start / total,
            structural_kind=skind,
            section_role=role,
            in_fence=in_fence,
            layout=layout,
        ))

    return sections

def segment_document(text: str, *, min_chunk_chars: int = 120) -> list[SemanticSection]:
    return segment_sections(text, min_section_chars=min(min_chunk_chars, 80))
