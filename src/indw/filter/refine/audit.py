from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

@dataclass
class RefineAuditReport:
    documents: int = 0
    documents_kept: int = 0
    route_counts: Counter[str] = field(default_factory=Counter)
    corpus_partitions: Counter[str] = field(default_factory=Counter)
    content_categories: Counter[str] = field(default_factory=Counter)
    artifact_categories: Counter[str] = field(default_factory=Counter)
    metadata_fields_removed: Counter[str] = field(default_factory=Counter)
    before_after_examples: list[dict[str, Any]] = field(default_factory=list)
    _weights: list[float] = field(default_factory=list)
    _quality_scores: list[float] = field(default_factory=list)
    _chars_before: list[int] = field(default_factory=list)
    _chars_after: list[int] = field(default_factory=list)
    sections_cleaned: int = 0

    def observe(
        self,
        *,
        route: str,
        corpus_partition: str,
        sample_weight: float,
        quality_composite: float,
        kept: bool = True,
        removed_meta_fields: list[str] | None = None,
        text_preview: str = '',
        chars_before: int = 0,
        chars_after: int = 0,
        artifact_signals: dict[str, float] | None = None,
        content_category: str = '',
        text_modified: bool = False,
    ) -> None:
        self.documents += 1
        if kept:
            self.documents_kept += 1
        self.route_counts[route] += 1
        self.corpus_partitions[corpus_partition] += 1
        self._weights.append(sample_weight)
        self._quality_scores.append(quality_composite)
        if text_modified:
            self.sections_cleaned += 1

        if chars_before > 0:
            self._chars_before.append(chars_before)
            self._chars_after.append(chars_after)

        if content_category:
            self.content_categories[content_category] += 1

        if removed_meta_fields:
            for f in removed_meta_fields:
                self.metadata_fields_removed[f] += 1

        if artifact_signals:
            for k, v in artifact_signals.items():
                if k == 'severity':
                    continue
                if v >= 0.35:
                    self.artifact_categories[k] += 1

        if (
            kept
            and chars_before > 0
            and chars_after < chars_before * 0.92
            and len(self.before_after_examples) < 15
        ):
            self.before_after_examples.append({
                'route': route,
                'chars_before': chars_before,
                'chars_after': chars_after,
                'token_savings_est': round((chars_before - chars_after) / 3.8),
                'preview_before': text_preview[:160],
            })

    def finalize(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        qs = self._quality_scores
        qdist: dict[str, float] = {}
        if qs:
            ordered = sorted(qs)
            qdist = {
                'p25': round(ordered[len(ordered) // 4], 2),
                'p50': round(ordered[len(ordered) // 2], 2),
                'p75': round(ordered[(3 * len(ordered)) // 4], 2),
                'min': round(ordered[0], 2),
                'max': round(ordered[-1], 2),
            }
        weights = sorted(self._weights) if self._weights else []
        wdist = {}
        if weights:
            n = len(weights)
            wdist = {
                'p25': round(weights[n // 4], 4),
                'p50': round(weights[n // 2], 4),
                'p75': round(weights[(3 * n) // 4], 4),
            }
        token_savings = sum(
            max(0, b - a) for b, a in zip(self._chars_before, self._chars_after)
        )
        retention = self.documents_kept / max(self.documents, 1)
        return {
            'documents': self.documents,
            'documents_kept': self.documents_kept,
            'knowledge_retention_ratio': round(retention, 4),
            'sections_cleaned': self.sections_cleaned,
            'routing_decisions': dict(self.route_counts),
            'corpus_partitions': dict(self.corpus_partitions),
            'content_category_balance': dict(self.content_categories.most_common(20)),
            'artifact_categories': dict(self.artifact_categories.most_common(12)),
            'sample_weight_distribution': wdist,
            'quality_score_distribution': qdist,
            'token_savings_est': int(token_savings / 3.8),
            'metadata_fields_removed': dict(self.metadata_fields_removed.most_common(30)),
            'before_after_examples': self.before_after_examples[:12],
        }
