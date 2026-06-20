from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indw.clean.artifact.calibrate import (
    CalibrationReport,
    ShadowDisagreement,
    batch_calibrate,
)
from indw.clean.artifact.confidence import ConfidenceFusion, FusedConfidence
from indw.clean.artifact.discovery_config import DiscoveryConfig
from indw.clean.artifact.discovery_corpus import CorpusStatsAccumulator, CorpusStatsStore, fragment_key
from indw.clean.artifact.decompose import DecomposedDocument, decompose_document
from indw.clean.artifact.discovery_registry import DynamicArtifactRegistry
from indw.clean.artifact.trim import TrimPolicy, TrimResult, safe_trim_fragments

@dataclass
class UnitScore:
    unit_id: str
    kind: str
    text: str
    start: int
    end: int
    confidence: FusedConfidence = field(default_factory=FusedConfidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            'unit_id': self.unit_id,
            'kind': self.kind,
            'text': self.text[:200],
            'start': self.start,
            'end': self.end,
            **self.confidence.to_dict(),
        }

@dataclass
class ArtifactReport:
    doc_id: str = ''
    artifact_ratio: float = 0.0
    legacy_ratio: float = 0.0
    discovery_ratio: float = 0.0
    unit_scores: list[UnitScore] = field(default_factory=list)
    trim: TrimResult | None = None
    shadow: bool = True
    chars_removed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'artifact_ratio': round(self.artifact_ratio, 4),
            'legacy_ratio': round(self.legacy_ratio, 4),
            'discovery_ratio': round(self.discovery_ratio, 4),
            'shadow': self.shadow,
            'chars_removed': self.chars_removed,
            'would_trim_units': sum(1 for u in self.unit_scores if u.confidence.would_trim),
            'unit_scores': [u.to_dict() for u in self.unit_scores[:200]],
            'trim_spans': self.trim.removed_spans if self.trim else [],
        }

_ENGINES: dict[str, ArtifactDiscoveryEngine] = {}

class ArtifactDiscoveryEngine:
    def __init__(self, config: DiscoveryConfig | None = None) -> None:
        self.config = config or DiscoveryConfig()
        db_path = None
        if self.config.corpus_dir:
            db_path = Path(self.config.corpus_dir) / 'artifact_discovery.sqlite'
        self._store = CorpusStatsStore(db_path)
        self.accumulator = CorpusStatsAccumulator(self._store)
        self.registry = DynamicArtifactRegistry(config=self.config)
        self.fusion = ConfidenceFusion(
            min_trim_confidence=self.config.min_trim_confidence,
            medium_trim_confidence=self.config.medium_trim_confidence,
        )
        self._shadow_disagreements: list[ShadowDisagreement] = []
        self._last_report: ArtifactReport | None = None
        from indw.clean.artifact.discovery_validation import DiscoveryValidation
        self.validation = DiscoveryValidation()

    def discover(
        self,
        text: str,
        *,
        doc_id: str = '',
        legacy_ratio: float | None = None,
    ) -> ArtifactReport:
        if not self.config.enabled or not text or not text.strip():
            return ArtifactReport(doc_id=doc_id, shadow=self.config.shadow)

        if len(text) < self.config.min_doc_chars:
            ratio = self.registry.artifact_ratio(text, self.accumulator) if self.accumulator.docs_seen else 0.0
            return ArtifactReport(
                doc_id=doc_id,
                artifact_ratio=ratio,
                discovery_ratio=ratio,
                legacy_ratio=legacy_ratio or 0.0,
                shadow=self.config.shadow,
            )

        doc = decompose_document(text)
        self.accumulator.observe_document(doc, doc_id=doc_id)

        key_counts: dict[str, int] = {}
        for unit in doc.units:
            if unit.in_fence:
                continue
            k = fragment_key(unit.text, unit.layout)
            key_counts[k] = key_counts.get(k, 0) + 1

        fused_pairs = self.fusion.fuse_document(
            doc.units, self.registry, self.accumulator, text, key_counts=key_counts,
        )
        unit_scores = [
            UnitScore(
                unit_id=unit.unit_id,
                kind=unit.kind,
                text=unit.text,
                start=unit.start,
                end=unit.end,
                confidence=conf,
            )
            for unit, conf in fused_pairs
        ]

        discovery_ratio = self.registry.artifact_ratio(text, self.accumulator)
        leg = legacy_ratio if legacy_ratio is not None else 0.0
        if self.accumulator.docs_seen > 1 or self.registry.promoted_entries():
            artifact_ratio = max(discovery_ratio, discovery_ratio * 0.85 + leg * 0.15)
        else:
            artifact_ratio = leg

        policy = TrimPolicy(
            shadow=self.config.shadow or not self.config.trim,
            max_trim_ratio=self.config.max_trim_ratio,
            min_trim_confidence=self.config.min_trim_confidence,
            medium_trim_confidence=self.config.medium_trim_confidence,
        )
        trim = safe_trim_fragments(text, doc.units, fused_pairs, policy=policy)
        out_text = trim.text

        if legacy_ratio is not None:
            delta = abs(discovery_ratio - legacy_ratio)
            if delta > 0.15:
                self._shadow_disagreements.append(
                    ShadowDisagreement(
                        doc_id=doc_id,
                        legacy_ratio=legacy_ratio,
                        discovery_ratio=discovery_ratio,
                        delta=delta,
                    )
                )

        report = ArtifactReport(
            doc_id=doc_id,
            artifact_ratio=artifact_ratio,
            legacy_ratio=leg,
            discovery_ratio=discovery_ratio,
            unit_scores=unit_scores,
            trim=trim,
            shadow=policy.shadow,
            chars_removed=trim.chars_removed,
        )
        self._last_report = report
        trimmed = not policy.shadow and self.config.trim
        self.validation.record_report(report, trimmed=trimmed)
        if report.trim:
            self.validation.trimmed_units += report.trim.units_removed
            self.validation.protected_skips += report.trim.protected_skips
        return report

    def apply_trim(self, text: str, report: ArtifactReport) -> str:
        if report.trim and not report.shadow:
            return report.trim.text
        return text

    def document_artifact_ratio(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        return self.registry.artifact_ratio(text, self.accumulator)

    def audit_flags(self, text: str) -> list[str]:
        return self.registry.audit_flags(text, self.accumulator)

    def scan_text(self, text: str) -> dict[str, int]:
        return self.registry.scan_text(text, self.accumulator)

    def end_batch(self) -> CalibrationReport:
        return batch_calibrate(
            self.accumulator,
            self.registry,
            self.config,
            corpus_dir=self.config.corpus_dir,
            shadow_disagreements=self._shadow_disagreements,
        )

    def calibration_report(self) -> CalibrationReport:
        return CalibrationReport(
            batch_id=self.accumulator.batch_id,
            docs_seen=self.accumulator.docs_seen,
            registry_size=len(self.registry._entries),
            shadow_disagreements=list(self._shadow_disagreements),
            trim_threshold=self.config.min_trim_confidence,
        )

    def close(self) -> None:
        self._store.close()

def get_discovery_engine(
    config: DiscoveryConfig | None = None,
    *,
    corpus_dir: str = '',
) -> ArtifactDiscoveryEngine:
    cfg = config or DiscoveryConfig()
    if corpus_dir and not cfg.corpus_dir:
        cfg.corpus_dir = corpus_dir
    key = cfg.corpus_dir or '__default__'
    if key not in _ENGINES:
        _ENGINES[key] = ArtifactDiscoveryEngine(cfg)
    elif config is not None:
        eng = _ENGINES[key]
        eng.config = cfg
    return _ENGINES[key]

def reset_discovery_engines() -> None:
    for eng in _ENGINES.values():
        eng.close()
    _ENGINES.clear()
