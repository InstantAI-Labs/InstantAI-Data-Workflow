from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class DiscoveryValidation:
    docs_processed: int = 0
    units_scored: int = 0
    would_trim_units: int = 0
    trimmed_units: int = 0
    protected_skips: int = 0
    chars_removed: int = 0
    knowledge_preserved: int = 0
    false_trim_candidates: int = 0
    legacy_disagreements: int = 0
    promoted_hits: int = 0

    def record_report(self, report: Any, *, trimmed: bool = False) -> None:
        self.docs_processed += 1
        leg = float(getattr(report, 'legacy_ratio', 0.0) or 0.0)
        disc = float(getattr(report, 'discovery_ratio', 0.0) or 0.0)
        if abs(disc - leg) > 0.15:
            self.legacy_disagreements += 1
        if trimmed:
            self.chars_removed += int(getattr(report, 'chars_removed', 0) or 0)
        for unit in getattr(report, 'unit_scores', []) or []:
            self.units_scored += 1
            conf = unit.confidence
            if conf.would_trim:
                self.would_trim_units += 1
                if conf.knowledge_confidence >= 0.55:
                    self.false_trim_candidates += 1
                else:
                    self.knowledge_preserved += 1
            if conf.artifact_confidence >= 0.5 and conf.frequency_confidence >= 0.3:
                self.promoted_hits += 1

    def to_dict(self) -> dict[str, Any]:
        n = max(self.docs_processed, 1)
        u = max(self.units_scored, 1)
        return {
            'docs_processed': self.docs_processed,
            'units_scored': self.units_scored,
            'would_trim_units': self.would_trim_units,
            'trimmed_units': self.trimmed_units,
            'protected_skips': self.protected_skips,
            'chars_removed': self.chars_removed,
            'legacy_disagreement_rate': round(self.legacy_disagreements / n, 4),
            'false_trim_candidate_rate': round(self.false_trim_candidates / max(self.would_trim_units, 1), 4),
            'knowledge_preservation_rate': round(self.knowledge_preserved / u, 4),
            'promoted_hit_rate': round(self.promoted_hits / u, 4),
            'avg_chars_removed_per_doc': round(self.chars_removed / n, 1),
        }
