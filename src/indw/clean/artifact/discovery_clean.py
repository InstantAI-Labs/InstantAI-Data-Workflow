from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from indw.clean.artifact.discovery_config import DiscoveryConfig
from indw.clean.artifact.discovery_engine import ArtifactDiscoveryEngine, ArtifactReport
from indw.clean.artifact.safeguards import is_protected_unit
from indw.clean.artifact.trim import TrimPolicy, safe_trim_fragments
from indw.clean.gate.evaluate import compute_artifact_ratio

@dataclass
class DynamicCleanResult:
    text: str
    report: ArtifactReport
    chars_removed: int = 0
    legacy_ratio: float = 0.0
    mode: str = 'shadow'

    def to_dict(self) -> dict[str, Any]:
        return {
            'mode': self.mode,
            'chars_removed': self.chars_removed,
            'legacy_ratio': round(self.legacy_ratio, 4),
            'discovery_ratio': round(self.report.discovery_ratio, 4),
            'would_trim_units': sum(1 for u in self.report.unit_scores if u.confidence.would_trim),
            'shadow': self.report.shadow,
        }

def apply_dynamic_artifact_clean(
    text: str,
    engine: ArtifactDiscoveryEngine,
    *,
    doc_id: str = '',
    legacy_fallback: bool = True,
) -> DynamicCleanResult:
    cfg = engine.config
    legacy_ratio, _ = compute_artifact_ratio(text, include_discovery=False)
    report = engine.discover(text, doc_id=doc_id, legacy_ratio=legacy_ratio)

    if cfg.shadow or not cfg.trim:
        return DynamicCleanResult(
            text=text,
            report=report,
            legacy_ratio=legacy_ratio,
            mode='shadow',
        )

    out = engine.apply_trim(text, report)
    removed = len(text) - len(out)
    return DynamicCleanResult(
        text=out,
        report=report,
        chars_removed=removed,
        legacy_ratio=legacy_ratio,
        mode='primary' if cfg.primary else 'trim',
    )
