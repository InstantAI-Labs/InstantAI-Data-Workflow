from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indw.clean.artifact.decompose import compute_layout
from indw.extract.nav.context import (
    NavigationContext,
    NavigationRoleScore,
    get_navigation_context,
    score_navigation_role,
)
from indw.extract.structure.recovery import recover_structure

@dataclass
class NavigationAuditSpan:
    text: str
    start: int
    end: int
    role: str
    confidence: float
    position_ratio: float

@dataclass
class NavigationAuditReport:
    leaks: list[NavigationAuditSpan] = field(default_factory=list)
    sections_scored: int = 0
    nav_sections_removed: int = 0
    knowledge_sections_kept: int = 0
    precision: float = 1.0
    recall: float = 0.0
    false_positive_rate: float = 0.0
    knowledge_retention: float = 0.0
    navigation_leakage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'leak_count': len(self.leaks),
            'sections_scored': self.sections_scored,
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'false_positive_rate': round(self.false_positive_rate, 4),
            'knowledge_retention': round(self.knowledge_retention, 4),
            'navigation_leakage': round(self.navigation_leakage, 4),
            'leaks': [
                {
                    'role': s.role,
                    'confidence': round(s.confidence, 4),
                    'preview': s.text[:120],
                }
                for s in self.leaks[:12]
            ],
        }

def structural_nav_spans(
    text: str,
    *,
    threshold: float = 0.42,
    ctx: NavigationContext | None = None,
) -> list[NavigationAuditSpan]:
    if not text or not text.strip():
        return []
    ctx = ctx or get_navigation_context()
    total = max(len(text), 1)
    spans: list[NavigationAuditSpan] = []

    sections = recover_structure(text, min_section_chars=30)
    if len(sections) <= 1:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        cursor = 0
        for ln in lines:
            idx = text.find(ln, cursor)
            if idx < 0:
                idx = cursor
            pos = (idx + len(ln) / 2) / total
            scored = score_navigation_role(ln, position_ratio=pos, ctx=ctx)
            role, conf = scored.dominant()
            if scored.is_navigation(threshold=threshold) and conf >= threshold:
                spans.append(NavigationAuditSpan(
                    text=ln, start=idx, end=idx + len(ln),
                    role=role, confidence=conf, position_ratio=pos,
                ))
            cursor = idx + len(ln)
        return spans

    for sec in sections:
        scored = score_navigation_role(
            sec.text,
            position_ratio=sec.position_ratio,
            layout=compute_layout(sec.text),
            ctx=ctx,
        )
        role, conf = scored.dominant()
        if scored.is_navigation(threshold=threshold) and conf >= threshold:
            spans.append(NavigationAuditSpan(
                text=sec.text, start=sec.start, end=sec.end,
                role=role, confidence=conf, position_ratio=sec.position_ratio,
            ))
    return spans

def audit_output_navigation(
    original: str,
    output: str,
    *,
    threshold: float = 0.42,
    ctx: NavigationContext | None = None,
) -> NavigationAuditReport:
    ctx = ctx or get_navigation_context()
    report = NavigationAuditReport()
    if not output or not output.strip():
        report.knowledge_retention = 0.0
        return report

    leaks = structural_nav_spans(output, threshold=threshold, ctx=ctx)
    report.leaks = leaks
    report.navigation_leakage = len(leaks) / max(1, len(output.splitlines()))
    report.sections_scored = len(recover_structure(output, min_section_chars=30)) or 1
    report.knowledge_retention = len(output) / max(len(original), 1)

    input_nav = structural_nav_spans(original, threshold=threshold, ctx=ctx)
    output_nav = leaks
    true_nav = len(input_nav)
    detected = len(output_nav)
    if true_nav > 0:
        report.recall = 1.0 - min(1.0, detected / true_nav)
    else:
        report.recall = 1.0 if detected == 0 else 0.0
    report.precision = 1.0 if detected == 0 else max(0.0, 1.0 - detected / max(detected, 1))
    report.false_positive_rate = report.navigation_leakage
    return report

def score_section_navigation(
    text: str,
    *,
    position_ratio: float = 0.5,
    ctx: NavigationContext | None = None,
) -> NavigationRoleScore:
    return score_navigation_role(text, position_ratio=position_ratio, ctx=ctx)
