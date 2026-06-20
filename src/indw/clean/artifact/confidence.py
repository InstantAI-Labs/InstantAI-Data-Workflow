from __future__ import annotations

from dataclasses import dataclass

from indw.clean.artifact.discovery_corpus import CorpusStatsAccumulator
from indw.clean.artifact.decompose import DocumentUnit, position_bin
from indw.clean.artifact.discovery_registry import ArtifactEntry, DynamicArtifactRegistry
from indw.clean.artifact.safeguards import is_protected_unit
from indw.clean.document.value import analyze_content_value

@dataclass
class FusedConfidence:
    artifact_confidence: float = 0.0
    knowledge_confidence: float = 0.0
    frequency_confidence: float = 0.0
    position_confidence: float = 0.0
    structural_confidence: float = 0.0
    novelty_confidence: float = 0.0
    repetition_confidence: float = 0.0
    coverage_confidence: float = 0.0
    entropy_confidence: float = 0.0
    trim_tier: str = 'keep'
    would_trim: bool = False

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            'artifact_confidence': round(self.artifact_confidence, 4),
            'knowledge_confidence': round(self.knowledge_confidence, 4),
            'frequency_confidence': round(self.frequency_confidence, 4),
            'position_confidence': round(self.position_confidence, 4),
            'structural_confidence': round(self.structural_confidence, 4),
            'novelty_confidence': round(self.novelty_confidence, 4),
            'repetition_confidence': round(self.repetition_confidence, 4),
            'coverage_confidence': round(self.coverage_confidence, 4),
            'entropy_confidence': round(self.entropy_confidence, 4),
            'trim_tier': self.trim_tier,
            'would_trim': self.would_trim,
        }

class ConfidenceFusion:
    def __init__(
        self,
        *,
        min_trim_confidence: float = 0.92,
        medium_trim_confidence: float = 0.72,
        knowledge_dampen: float = 0.55,
    ) -> None:
        self.min_trim_confidence = min_trim_confidence
        self.medium_trim_confidence = medium_trim_confidence
        self.knowledge_dampen = knowledge_dampen

    def fuse_unit(
        self,
        unit: DocumentUnit,
        entry: ArtifactEntry | None,
        *,
        doc_text: str,
        count_in_doc: int = 1,
        doc_len: int = 0,
    ) -> FusedConfidence:
        if entry is None:
            return FusedConfidence()

        if is_protected_unit(unit.text, kind=unit.kind, in_fence=unit.in_fence):
            return FusedConfidence(knowledge_confidence=1.0, trim_tier='protected')

        ctx_start = max(0, unit.start - 200)
        ctx_end = min(len(doc_text), unit.end + 200)
        context = doc_text[ctx_start:ctx_end]
        cv = analyze_content_value(context)
        knowledge = cv.overall_value_score
        if cv.evidence and cv.evidence.preserve:
            knowledge = max(knowledge, 0.85)

        repetition = 1.0 - min(1.0, 1.0 / max(count_in_doc, 1))
        coverage = entry.frequency_confidence
        entropy_signal = 1.0 - entry.novelty_confidence

        artifact = entry.artifact_confidence
        artifact = min(1.0, artifact * 0.75 + repetition * 0.15 + coverage * 0.10)
        if knowledge > self.knowledge_dampen:
            artifact *= max(0.0, 1.0 - (knowledge - self.knowledge_dampen))

        tier = 'keep'
        would_trim = False
        if artifact >= self.min_trim_confidence and knowledge < self.knowledge_dampen and entry.novelty_confidence < 0.35:
            if not unit.in_fence and unit.kind != 'code':
                tier = 'high'
                would_trim = True
        elif artifact >= self.medium_trim_confidence and knowledge < 0.45 and entry.novelty_confidence < 0.25:
            bin_idx = position_bin(unit.start, max(doc_len, 1))
            if bin_idx in (0, 4) and unit.kind in ('line', 'header', 'footer'):
                tier = 'medium'
                would_trim = True

        return FusedConfidence(
            artifact_confidence=artifact,
            knowledge_confidence=knowledge,
            frequency_confidence=entry.frequency_confidence,
            position_confidence=entry.position_confidence,
            structural_confidence=entry.structural_confidence,
            novelty_confidence=entry.novelty_confidence,
            repetition_confidence=repetition,
            coverage_confidence=coverage,
            entropy_confidence=entropy_signal,
            trim_tier=tier,
            would_trim=would_trim,
        )

    def fuse_document(
        self,
        units: list[DocumentUnit],
        registry: DynamicArtifactRegistry,
        accumulator: CorpusStatsAccumulator,
        doc_text: str,
        *,
        key_counts: dict[str, int] | None = None,
    ) -> list[tuple[DocumentUnit, FusedConfidence]]:
        counts: dict[str, int] = key_counts or {}
        doc_len = len(doc_text)
        out: list[tuple[DocumentUnit, FusedConfidence]] = []
        for unit in units:
            from indw.clean.artifact.discovery_corpus import fragment_key

            key = fragment_key(unit.text, unit.layout)
            cnt = counts.get(key, 1)
            entry = registry.lookup(unit.text, accumulator, layout=unit.layout, count_in_doc=cnt)
            fused = self.fuse_unit(
                unit, entry, doc_text=doc_text, count_in_doc=cnt, doc_len=doc_len,
            )
            out.append((unit, fused))
        return out
