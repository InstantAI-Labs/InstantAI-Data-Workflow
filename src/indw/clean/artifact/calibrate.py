from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indw.clean.artifact.discovery_config import DiscoveryConfig
from indw.clean.artifact.discovery_corpus import CorpusStatsAccumulator
from indw.clean.artifact.discovery_registry import DynamicArtifactRegistry

@dataclass
class ShadowDisagreement:
    doc_id: str = ''
    legacy_ratio: float = 0.0
    discovery_ratio: float = 0.0
    delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'legacy_ratio': round(self.legacy_ratio, 4),
            'discovery_ratio': round(self.discovery_ratio, 4),
            'delta': round(self.delta, 4),
        }

@dataclass
class CalibrationReport:
    batch_id: int = 0
    docs_seen: int = 0
    promoted: int = 0
    demoted: int = 0
    registry_size: int = 0
    shadow_disagreements: list[ShadowDisagreement] = field(default_factory=list)
    trim_threshold: float = 0.92

    def to_dict(self) -> dict[str, Any]:
        return {
            'batch_id': self.batch_id,
            'docs_seen': self.docs_seen,
            'promoted': self.promoted,
            'demoted': self.demoted,
            'registry_size': self.registry_size,
            'trim_threshold': self.trim_threshold,
            'shadow_disagreements': [d.to_dict() for d in self.shadow_disagreements[-500:]],
        }

def batch_calibrate(
    accumulator: CorpusStatsAccumulator,
    registry: DynamicArtifactRegistry,
    config: DiscoveryConfig,
    *,
    corpus_dir: str = '',
    shadow_disagreements: list[ShadowDisagreement] | None = None,
) -> CalibrationReport:
    accumulator.end_batch(decay=config.decay)
    cal = registry.calibrate(accumulator)
    report = CalibrationReport(
        batch_id=accumulator.batch_id,
        docs_seen=accumulator.docs_seen,
        promoted=cal['promoted'],
        demoted=cal['demoted'],
        registry_size=cal['total'],
        shadow_disagreements=shadow_disagreements or [],
        trim_threshold=config.min_trim_confidence,
    )
    if corpus_dir:
        out = Path(corpus_dir) / 'discovery_calibration.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = []
        if out.exists():
            try:
                existing = json.loads(out.read_text(encoding='utf-8'))
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(report.to_dict())
        out.write_text(json.dumps(existing[-50:], indent=2), encoding='utf-8')
    return report
