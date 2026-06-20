from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indw.clean.artifact.decompose import compute_layout
from indw.clean.semantic.structure import infer_section_role
from indw.extract.sections.boundaries import (
    boundary_cut_strength,
    decompose_span_sections,
    detect_boundaries,
    split_at_boundaries,
    _decompose_spans,
)

@dataclass
class RecoveredSection:
    text: str
    start: int
    end: int
    position_ratio: float = 0.0
    structural_role: str = 'body'
    layout_kind: str = 'paragraph'
    in_fence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'structural_role': self.structural_role,
            'position_ratio': round(self.position_ratio, 4),
            'layout_kind': self.layout_kind,
            'chars': len(self.text),
            'preview': self.text[:160],
        }

def recover_structure(text: str, *, min_section_chars: int = 60) -> list[RecoveredSection]:
    if not text or not text.strip():
        return []

    cuts = detect_boundaries(text, min_section_chars=min_section_chars)
    cut_strength = boundary_cut_strength(text, cuts)
    parts = split_at_boundaries(text, cuts, min_section_chars=min_section_chars, cut_strength=cut_strength)
    if len(parts) <= 1:
        span_parts = decompose_span_sections(text, min_section_chars=min_section_chars)
        if len(span_parts) > 1:
            parts = span_parts
    total = max(len(text), 1)
    sections: list[RecoveredSection] = []

    for chunk, start, end in parts:
        layout = compute_layout(chunk)
        pos = (start + end) / (2 * total)
        skind = 'table' if layout.list_ratio > 0.55 and layout.line_count >= 3 else 'paragraph'
        role = infer_section_role(
            chunk,
            layout=layout,
            position_ratio=pos,
            in_fence=False,
            structural_kind=skind,
        )
        sections.append(RecoveredSection(
            text=chunk,
            start=start,
            end=end,
            position_ratio=pos,
            structural_role=role,
            layout_kind=skind,
        ))
    return sections
