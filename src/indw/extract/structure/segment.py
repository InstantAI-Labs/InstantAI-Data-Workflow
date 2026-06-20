from __future__ import annotations

from dataclasses import dataclass, field

from indw.clean.artifact.decompose import DocumentUnit, decompose_document
from indw.clean.semantic.structure import SemanticSection, infer_section_role, segment_sections
from indw.clean.artifact.evidence_engine import compute_semantic_evidence
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator

@dataclass
class UnifiedSegment:
    text: str
    start: int
    end: int
    kind: str
    role: str
    position_ratio: float = 0.0
    in_fence: bool = False
    unit_id: str = ''
    artifact_score: float = 0.0
    knowledge_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            'kind': self.kind,
            'role': self.role,
            'position_ratio': round(self.position_ratio, 4),
            'artifact_score': round(self.artifact_score, 4),
            'knowledge_score': round(self.knowledge_score, 4),
            'preview': self.text[:160],
        }

@dataclass
class SegmentedDocument:
    segments: list[UnifiedSegment] = field(default_factory=list)
    content_spans: list[tuple[int, int]] = field(default_factory=list)
    artifact_spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def content_text(self) -> str:
        parts = [self.segments[i].text for i, _ in enumerate(self.content_spans) if self.segments]
        if not parts and self.segments:
            parts = [s.text for s in self.segments if s.knowledge_score >= s.artifact_score]
        return '\n\n'.join(p.strip() for p in parts if p.strip())

def segment_document(text: str, *, min_section_chars: int = 60) -> SegmentedDocument:
    if not text or not text.strip():
        return SegmentedDocument()

    doc = decompose_document(text)
    sem_sections = segment_sections(text, min_section_chars=min_section_chars)
    role_by_range: dict[tuple[int, int], str] = {
        (s.start, s.end): s.section_role for s in sem_sections
    }

    out = SegmentedDocument()
    doc_len = max(len(text), 1)
    for unit in doc.units:
        pos = (unit.start + unit.end) / (2 * doc_len)
        role = unit.kind
        for (s, e), r in role_by_range.items():
            if unit.start >= s and unit.end <= e + 1:
                role = r
                break
        if role in ('line', 'paragraph', 'block'):
            role = infer_section_role(
                unit.text,
                layout=unit.layout,
                position_ratio=pos,
                in_fence=unit.in_fence,
                structural_kind=unit.kind,
            )
        out.segments.append(
            UnifiedSegment(
                text=unit.text,
                start=unit.start,
                end=unit.end,
                kind=unit.kind,
                role=role,
                position_ratio=pos,
                in_fence=unit.in_fence,
                unit_id=unit.unit_id,
            )
        )
    return out

def score_segments(
    segmented: SegmentedDocument,
    doc_text: str,
    *,
    unit_scores: dict[str, float] | None = None,
) -> SegmentedDocument:
    baseline = AdaptiveBaselineEstimator()
    unit_scores = unit_scores or {}
    content: list[tuple[int, int]] = []
    artifact: list[tuple[int, int]] = []

    for i, seg in enumerate(segmented.segments):
        if seg.in_fence:
            seg.artifact_score = 0.0
            seg.knowledge_score = 1.0
            content.append((i, i))
            continue

        fused = unit_scores.get(seg.unit_id)
        if fused is not None:
            seg.artifact_score = fused
            seg.knowledge_score = 1.0 - fused
        else:
            ev = compute_semantic_evidence(seg.text)
            noise = baseline.baseline(list(ev.negative.values()) or [0.0])
            know = ev.utility
            struct_noise = 0.0
            if seg.role in ('navigation', 'footer', 'contact', 'promotional', 'related_content', 'metadata'):
                struct_noise = baseline.baseline([0.35, seg.position_ratio])
            seg.artifact_score = min(1.0, noise * 0.55 + struct_noise * 0.45)
            seg.knowledge_score = know
            if ev.preserve:
                seg.artifact_score *= 0.25

        trim_thr = baseline.baseline([seg.artifact_score, 0.42, 1.0 - seg.knowledge_score])
        if seg.artifact_score >= trim_thr and seg.knowledge_score < baseline.baseline([seg.knowledge_score, 0.22]):
            artifact.append((i, i))
        else:
            content.append((i, i))

    segmented.content_spans = content
    segmented.artifact_spans = artifact
    return segmented
