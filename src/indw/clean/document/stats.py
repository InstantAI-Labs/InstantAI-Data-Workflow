from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class StageStats:
    in_docs: int = 0
    out_docs: int = 0
    dropped: int = 0
    lines_removed: int = 0
    chars_removed: int = 0
    chunks_removed: int = 0
    chunks_downweighted: int = 0
    wall_sec: float = 0.0
    calls: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return {
            'in_docs': self.in_docs,
            'out_docs': self.out_docs,
            'dropped': self.dropped,
            'lines_removed': self.lines_removed,
            'chars_removed': self.chars_removed,
            'chunks_removed': self.chunks_removed,
            'chunks_downweighted': self.chunks_downweighted,
            'wall_sec': round(self.wall_sec, 4),
            'calls': self.calls,
        }

    @contextmanager
    def timed(self) -> Iterator[None]:
        t0 = time.perf_counter()
        self.calls += 1
        try:
            yield
        finally:
            self.wall_sec += time.perf_counter() - t0


@dataclass
class KnowledgeExtractionStats:
    structure_recovery: StageStats = field(default_factory=StageStats)
    aggregation: StageStats = field(default_factory=StageStats)
    section_classify: StageStats = field(default_factory=StageStats)
    section_quality: StageStats = field(default_factory=StageStats)
    boundary_role: StageStats = field(default_factory=StageStats)
    unit_clean: StageStats = field(default_factory=StageStats)
    unit_assembly: StageStats = field(default_factory=StageStats)
    serialization: StageStats = field(default_factory=StageStats)

    ke_ops: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            'structure_recovery': self.structure_recovery.to_dict(),
            'aggregation': self.aggregation.to_dict(),
            'section_classify': self.section_classify.to_dict(),
            'section_quality': self.section_quality.to_dict(),
            'boundary_role': self.boundary_role.to_dict(),
            'unit_clean': self.unit_clean.to_dict(),
            'unit_assembly': self.unit_assembly.to_dict(),
            'serialization': self.serialization.to_dict(),
        }
        if self.ke_ops:
            out['ke_ops'] = self.ke_ops
        return out


@dataclass
class CleaningStats:
    html: StageStats = field(default_factory=StageStats)
    ui_noise: StageStats = field(default_factory=StageStats)
    metadata: StageStats = field(default_factory=StageStats)
    boilerplate: StageStats = field(default_factory=StageStats)
    source_processing: StageStats = field(default_factory=StageStats)
    structure: StageStats = field(default_factory=StageStats)
    compression: StageStats = field(default_factory=StageStats)
    conversation: StageStats = field(default_factory=StageStats)
    deduplication: StageStats = field(default_factory=StageStats)
    segmentation: StageStats = field(default_factory=StageStats)
    quality_filter: StageStats = field(default_factory=StageStats)
    document_gate: StageStats = field(default_factory=StageStats)
    artifacts: StageStats = field(default_factory=StageStats)
    code_preservation: StageStats = field(default_factory=StageStats)
    semantic_cleaning: StageStats = field(default_factory=StageStats)
    truncation_repair: StageStats = field(default_factory=StageStats)
    discovery: StageStats = field(default_factory=StageStats)
    knowledge_extraction: KnowledgeExtractionStats = field(default_factory=KnowledgeExtractionStats)
    document_gate_reasons: dict[str, int] = field(default_factory=dict)
    discovery_reports: list[dict[str, Any]] = field(default_factory=list)
    discovery_shadow_disagreements: int = 0
    input_documents: int = 0
    output_chunks: int = 0
    dropped_chunks: int = 0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            'input_documents': self.input_documents,
            'output_chunks': self.output_chunks,
            'dropped_chunks': self.dropped_chunks,
            'stages': {
                'html': self.html.to_dict(),
                'ui_noise': self.ui_noise.to_dict(),
                'metadata': self.metadata.to_dict(),
                'boilerplate': self.boilerplate.to_dict(),
                'source_processing': self.source_processing.to_dict(),
                'structure': self.structure.to_dict(),
                'compression': self.compression.to_dict(),
                'conversation': self.conversation.to_dict(),
                'deduplication': self.deduplication.to_dict(),
                'segmentation': self.segmentation.to_dict(),
                'quality_filter': self.quality_filter.to_dict(),
                'document_gate': self.document_gate.to_dict(),
                'artifacts': self.artifacts.to_dict(),
                'code_preservation': self.code_preservation.to_dict(),
                'semantic_cleaning': self.semantic_cleaning.to_dict(),
                'truncation_repair': self.truncation_repair.to_dict(),
                'discovery': self.discovery.to_dict(),
            },
            'knowledge_extraction': self.knowledge_extraction.to_dict(),
            'document_gate_reasons': dict(self.document_gate_reasons),
            'discovery_shadow_disagreements': self.discovery_shadow_disagreements,
            'discovery_report_count': len(self.discovery_reports),
        }
        try:
            from indw.clean.artifact.evidence_cache import session_cache_stats
            out['evidence_cache'] = session_cache_stats()
        except Exception:
            pass
        return out
