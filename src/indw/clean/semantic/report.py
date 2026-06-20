from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

@dataclass
class SemanticCleanReport:
    documents_processed: int = 0
    sections_total: int = 0
    sections_removed: int = 0
    sections_downweighted: int = 0
    sections_kept: int = 0
    sections_cleaned: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    quality_scores: list[float] = field(default_factory=list)
    section_classifications: Counter[str] = field(default_factory=Counter)
    routing_decisions: Counter[str] = field(default_factory=Counter)
    artifact_categories: Counter[str] = field(default_factory=Counter)
    false_positive_samples: list[dict[str, Any]] = field(default_factory=list)
    false_negative_samples: list[dict[str, Any]] = field(default_factory=list)
    before_after: list[dict[str, Any]] = field(default_factory=list)
    threshold_snapshot: dict[str, float] = field(default_factory=dict)

    def observe_doc(
        self,
        *,
        before: str,
        after: str,
        routing_actions: list[tuple[str, str, float, str]],
        section_roles: list[str],
        utility: float,
        samples: list[dict[str, Any]] | None = None,
        chunk_actions: list[tuple[str, str, float]] | None = None,
        dominant_labels: list[str] | None = None,
        artifact_categories: dict[str, int] | None = None,
        ocr_repairs: int = 0,
    ) -> None:
        self.documents_processed += 1
        self.tokens_before += len(before.split())
        self.tokens_after += len(after.split())
        self.quality_scores.append(utility)

        if chunk_actions and not routing_actions:
            routing_actions = [(a, l, c, '') for a, l, c in chunk_actions]
        if dominant_labels and not section_roles:
            section_roles = dominant_labels

        for role in section_roles:
            self.section_classifications[role] += 1

        if artifact_categories:
            for cat, n in artifact_categories.items():
                self.artifact_categories[cat] += n
        if ocr_repairs:
            self.artifact_categories['ocr_repair'] += ocr_repairs

        for action, role, _conf, _reason in routing_actions:
            self.sections_total += 1
            self.routing_decisions[action] += 1
            if action == 'REMOVE':
                self.sections_removed += 1
            elif action == 'DOWNWEIGHT':
                self.sections_downweighted += 1
            elif action == 'KEEP_AFTER_CLEANING':
                self.sections_cleaned += 1
                self.sections_kept += 1
            else:
                self.sections_kept += 1

        if samples:
            for s in samples:
                if s.get('kind') == 'fp' and len(self.false_positive_samples) < 50:
                    self.false_positive_samples.append(s)
                elif s.get('kind') == 'fn' and len(self.false_negative_samples) < 50:
                    self.false_negative_samples.append(s)
        if len(self.before_after) < 30:
            self.before_after.append({
                'before_chars': len(before),
                'after_chars': len(after),
                'before_preview': before[:280],
                'after_preview': after[:280],
            })

    def to_dict(self) -> dict[str, Any]:
        qs = self.quality_scores
        dist = {}
        if qs:
            ordered = sorted(qs)
            dist = {
                'p25': round(ordered[len(ordered) // 4], 4),
                'p50': round(ordered[len(ordered) // 2], 4),
                'p75': round(ordered[(3 * len(ordered)) // 4], 4),
                'mean': round(sum(qs) / len(qs), 4),
            }
        tok_save = 0.0
        if self.tokens_before:
            tok_save = round(1.0 - self.tokens_after / self.tokens_before, 4)
        return {
            'documents_processed': self.documents_processed,
            'sections': {
                'total': self.sections_total,
                'removed': self.sections_removed,
                'downweighted': self.sections_downweighted,
                'kept': self.sections_kept,
                'cleaned': self.sections_cleaned,
            },
            'section_classifications': dict(self.section_classifications),
            'routing_decisions': dict(self.routing_decisions),
            'quality_score_distribution': dist,
            'artifact_categories': dict(self.artifact_categories),
            'token_savings_ratio': tok_save,
            'tokens_before': self.tokens_before,
            'tokens_after': self.tokens_after,
            'knowledge_retention_estimate': round(
                1.0 - self.sections_removed / max(self.sections_total, 1), 4,
            ),
            'boilerplate_removal_ratio': round(
                self.sections_removed / max(self.sections_total, 1), 4,
            ),
            'thresholds': self.threshold_snapshot,
            'false_positive_samples': self.false_positive_samples[:20],
            'false_negative_samples': self.false_negative_samples[:20],
            'before_after_examples': self.before_after[:15],
        }
