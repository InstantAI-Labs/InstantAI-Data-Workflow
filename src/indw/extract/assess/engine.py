from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indw.clean.artifact.discovery_engine import ArtifactDiscoveryEngine
from indw.extract.assess.doc_type import DocumentClassProfile, classify_document
from indw.extract.assess.feedback import SelfLearningFeedback
from indw.extract.structure.inline import strip_inline_structural
from indw.extract.assess.quality import AdaptiveQualityAssessment, assess_quality
from indw.extract.structure.segment import (
    SegmentedDocument,
    score_segments,
    segment_document,
)
from indw.extract.structure.analyze import StructuralProfile, analyze_structure
from indw.extract.nav.template import TemplateMiner, TemplateProfile
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator

@dataclass
class CleanStats:
    units_removed: int = 0
    chars_removed: int = 0
    spans_removed: int = 0

@dataclass
class UnderstandingReport:
    structural: StructuralProfile = field(default_factory=StructuralProfile)
    template: TemplateProfile = field(default_factory=TemplateProfile)
    doc_class: DocumentClassProfile = field(default_factory=DocumentClassProfile)
    quality: AdaptiveQualityAssessment = field(default_factory=AdaptiveQualityAssessment)
    artifact_ratio: float = 0.0
    navigation_ratio: float = 0.0
    metadata_ratio: float = 0.0
    template_ratio: float = 0.0
    content_ratio: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ui_noise_ratio(self) -> float:
        return min(
            1.0,
            max(
                self.artifact_ratio,
                self.navigation_ratio,
                self.template_ratio,
                self.quality.boilerplate_ratio,
                self.quality.template_ratio,
            ),
        )

    @property
    def combined_density(self) -> float:
        return self.ui_noise_ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            'artifact_ratio': round(self.artifact_ratio, 4),
            'navigation_ratio': round(self.navigation_ratio, 4),
            'metadata_ratio': round(self.metadata_ratio, 4),
            'template_ratio': round(self.template_ratio, 4),
            'content_ratio': round(self.content_ratio, 4),
            'ui_noise_ratio': round(self.ui_noise_ratio, 4),
            'doc_class': self.doc_class.to_dict(),
            'quality': self.quality.to_dict(),
            'structural': self.structural.to_dict(),
        }

_ENGINES: dict[str, DocumentUnderstandingEngine] = {}

class DocumentUnderstandingEngine:
    def __init__(
        self,
        *,
        discovery: ArtifactDiscoveryEngine | None = None,
        corpus_dir: str = '',
        max_remove_ratio: float = 0.42,
    ) -> None:
        self.discovery = discovery
        self.corpus_dir = corpus_dir
        self.max_remove_ratio = max_remove_ratio
        acc = discovery.accumulator if discovery is not None else None
        self.template_miner = TemplateMiner(acc)
        self.feedback = SelfLearningFeedback(corpus_dir=corpus_dir)

    def analyze(self, text: str) -> UnderstandingReport:
        if not text or not text.strip():
            return UnderstandingReport()

        structural = analyze_structure(text)
        template = self.template_miner.analyze(text)
        doc_class = classify_document(text)
        quality = assess_quality(text, structural=structural, template=template, doc_class=doc_class)
        segmented = score_segments(segment_document(text), text)
        scored = segmented

        n = max(len(scored.segments), 1)
        artifact_units = len(scored.artifact_spans)
        discovery_ratio = 0.0
        if self.discovery is not None:
            discovery_ratio = self.discovery.document_artifact_ratio(text)

        return UnderstandingReport(
            structural=structural,
            template=template,
            doc_class=doc_class,
            quality=quality,
            artifact_ratio=max(
                discovery_ratio,
                artifact_units / n,
                structural.boilerplate_density,
            ),
            navigation_ratio=structural.navigation_density,
            metadata_ratio=structural.metadata_density,
            template_ratio=template.template_density,
            content_ratio=structural.content_density,
            segments=[s.to_dict() for s in scored.segments[:40]],
        )

    def clean(
        self,
        text: str,
        *,
        preserve_code_fences: bool = True,
        doc_id: str = '',
    ) -> tuple[str, CleanStats]:
        if not text or not text.strip():
            return text, CleanStats()

        stats = CleanStats()
        working, inline_stats = strip_inline_structural(
            text,
            preserve_code_fences=preserve_code_fences,
        )
        stats.spans_removed += inline_stats.spans_removed
        stats.chars_removed += inline_stats.chars_removed
        unit_scores: dict[str, float] = {}

        if self.discovery is not None and self.discovery.config.enabled:
            from indw.clean.gate.evaluate import compute_artifact_ratio
            leg, _ = compute_artifact_ratio(working, include_discovery=False)
            report = self.discovery.discover(working, doc_id=doc_id, legacy_ratio=leg)
            if report.trim and not report.shadow and self.discovery.config.trim:
                working = self.discovery.apply_trim(working, report)
                stats.chars_removed += report.chars_removed
            for u in report.unit_scores:
                if u.confidence.would_trim:
                    unit_scores[u.unit_id] = u.confidence.artifact_confidence

        segmented = segment_document(working)
        scored = score_segments(segmented, working, unit_scores=unit_scores)

        remove_idx = {i for i, _ in scored.artifact_spans}
        if not remove_idx:
            return working, stats

        max_remove = max(1, round(len(scored.segments) * self.max_remove_ratio))
        if len(remove_idx) > max_remove:
            ranked = sorted(
                ((i, scored.segments[i]) for i in remove_idx),
                key=lambda x: x[1].artifact_score - x[1].knowledge_score,
                reverse=True,
            )
            remove_idx = {i for i, _ in ranked[:max_remove]}

        parts: list[str] = []
        for i, seg in enumerate(scored.segments):
            if i in remove_idx:
                stats.units_removed += 1
                stats.spans_removed += 1
                stats.chars_removed += len(seg.text)
                continue
            if seg.text.strip():
                parts.append(seg.text.strip())

        out = '\n\n'.join(parts).strip()
        if not out and quality_preserve(scored):
            return working, stats
        return out or working, stats

    def record_leakage(self, text: str, *, reason: str = 'output_leakage') -> None:
        self.feedback.record_leakage(text, reason=reason)
        if self.discovery is not None:
            self.feedback.apply_to_registry(self.discovery.registry)
            self.feedback.save()

    def ui_noise_ratio(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        from indw.extract.structure.inline import (
            _collect_structural_spans,
            _prose_segments,
        )

        noise_chars = 0
        prose_chars = 0
        for _s, _e, segment in _prose_segments(text, preserve_code_fences=True):
            if not segment.strip():
                continue
            prose_chars += len(segment)
            spans = _collect_structural_spans(segment)
            noise_chars += sum(e - s for s, e in spans)
        if prose_chars <= 0:
            return 0.0
        return min(1.0, noise_chars / prose_chars)

    def artifact_ratio(self, text: str) -> float:
        return self.analyze(text).artifact_ratio

def quality_preserve(scored: SegmentedDocument) -> bool:
    return any(s.knowledge_score > 0.45 for s in scored.segments)

def get_understanding_engine(
    *,
    discovery: ArtifactDiscoveryEngine | None = None,
    corpus_dir: str = '',
) -> DocumentUnderstandingEngine:
    key = corpus_dir or (discovery.config.corpus_dir if discovery else '__default__')
    if key not in _ENGINES:
        _ENGINES[key] = DocumentUnderstandingEngine(discovery=discovery, corpus_dir=corpus_dir or key)
    elif discovery is not None:
        _ENGINES[key].discovery = discovery
    return _ENGINES[key]

def reset_understanding_engines() -> None:
    _ENGINES.clear()
