from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indw.clean.document.license import strip_collapsed_inline_license
from indw.clean.semantic.embedded import strip_embedded_artifacts
from indw.clean.semantic.classifier import ChunkClassification, SemanticChunkClassifier
from indw.clean.semantic.clean import clean_section_text
from indw.clean.semantic.config import SemanticCleaningConfig
from indw.clean.semantic.ocr_normalize import normalize_ocr_text
from indw.clean.semantic.report import SemanticCleanReport
from indw.clean.semantic.routing import RoutingDecision, SectionRouter
from indw.clean.semantic.structure import SemanticSection, segment_sections
from indw.clean.semantic.thresholds import get_threshold_calibrator

@dataclass
class SemanticCleanResult:
    text: str
    removed_chunks: int = 0
    downweighted_chunks: int = 0
    kept_chunks: int = 0
    cleaned_chunks: int = 0
    utility: float = 0.0
    artifact_categories: dict[str, int] = field(default_factory=dict)
    ocr_repairs: int = 0
    classifications: list[ChunkClassification] = field(default_factory=list)
    routing_decisions: list[RoutingDecision] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)

_GLOBAL_REPORT = SemanticCleanReport()

def get_semantic_clean_report() -> SemanticCleanReport:
    return _GLOBAL_REPORT

def reset_semantic_clean_report() -> None:
    global _GLOBAL_REPORT
    _GLOBAL_REPORT = SemanticCleanReport()

class SemanticCleaningPipeline:
    def __init__(self, config: SemanticCleaningConfig | None = None):
        self.config = config or SemanticCleaningConfig()
        self._classifier = SemanticChunkClassifier()
        self._calibrator = get_threshold_calibrator()
        self._router = SectionRouter(self.config, self._calibrator)

    def _apply_routing(
        self,
        section: SemanticSection,
        decision: RoutingDecision,
    ) -> tuple[str | None, dict[str, int]]:
        action = decision.action
        preserve = bool(
            decision.classification
            and decision.classification.evidence
            and decision.classification.evidence.preserve
        )
        edu = decision.signals.educational_value + decision.signals.technical_value

        if action == 'REMOVE':
            know = decision.signals.knowledge_value
            if section.section_role in ('body', 'introduction', 'title', 'code', 'examples', 'references'):
                if edu >= 0.14 or know >= 0.20:
                    text, stats = clean_section_text(
                        section.text,
                        role=section.section_role,
                        position_ratio=section.position_ratio,
                        preserve_educational=True,
                    )
                    return (text or None), stats
            return None, {}

        text = section.text.strip()
        if action == 'KEEP_AFTER_CLEANING':
            text, stats = clean_section_text(
                text,
                role=section.section_role,
                position_ratio=section.position_ratio,
                preserve_educational=True,
            )
            if not text:
                return None, stats
            return text, stats

        if action == 'KEEP' and section.section_role in (
            'body', 'introduction', 'title', 'examples', 'references',
        ):
            text, stats = clean_section_text(
                text,
                role=section.section_role,
                position_ratio=section.position_ratio,
                preserve_educational=True,
            )
            return (text or None), stats

        if action == 'DOWNWEIGHT':
            text, stats = clean_section_text(
                text,
                role=section.section_role,
                position_ratio=section.position_ratio,
                preserve_educational=preserve or edu >= 0.15,
            )
            return (text or None), stats

        return text, {}

    def process(self, text: str) -> SemanticCleanResult:
        cfg = self.config
        if not cfg.enabled or not text or not text.strip():
            return SemanticCleanResult(text=text.strip() if text else '')

        before = text
        working, ocr_stats = normalize_ocr_text(text)
        working, _ = strip_collapsed_inline_license(working)
        working, _embedded = strip_embedded_artifacts(working, config=cfg)
        sections = segment_sections(working, min_section_chars=60)
        if not sections:
            return SemanticCleanResult(text=working, ocr_repairs=ocr_stats.tokens_fixed)

        kept_parts: list[str] = []
        classifications: list[ChunkClassification] = []
        routing_decisions: list[RoutingDecision] = []
        artifact_totals: dict[str, int] = {}
        removed = downweighted = kept = cleaned = 0
        utilities: list[float] = []
        labels: list[str] = []
        routing_actions: list[tuple[str, str, float, str]] = []
        samples: list[dict[str, Any]] = []

        for section in sections:
            cls = self._classifier.classify(
                section.text,
                position_ratio=section.position_ratio,
                in_fence=section.in_fence,
                enabled=cfg.enabled,
            )
            decision = self._router.route(
                section.text,
                cls,
                section_role=section.section_role,
                position_ratio=section.position_ratio,
                in_fence=section.in_fence,
            )

            classifications.append(cls)
            routing_decisions.append(decision)
            utilities.append(cls.utility)
            labels.append(section.section_role)

            edu = decision.signals.educational_value + decision.signals.technical_value
            routing_actions.append((
                decision.action,
                section.section_role,
                decision.confidence,
                decision.reason,
            ))

            out_text, art_stats = self._apply_routing(section, decision)
            for k, v in art_stats.items():
                artifact_totals[k] = artifact_totals.get(k, 0) + v

            if out_text is None:
                removed += 1
                artifact_totals[section.section_role] = artifact_totals.get(section.section_role, 0) + 1
                if cfg.record_samples and len(samples) < cfg.max_samples_per_doc and edu >= 0.15:
                    samples.append({
                        'kind': 'fp',
                        'action': decision.action,
                        'role': section.section_role,
                        'reason': decision.reason,
                        'utility': cls.utility,
                        'preview': section.text[:240],
                    })
                continue

            if decision.action == 'DOWNWEIGHT':
                downweighted += 1
            elif decision.action == 'KEEP_AFTER_CLEANING':
                cleaned += 1
                kept += 1
            else:
                kept += 1

            kept_parts.append(out_text)

        out = '\n\n'.join(kept_parts).strip()
        out, _ = normalize_ocr_text(out)
        if not out:
            out = working

        max_remove = max(1, round(len(sections) * cfg.max_remove_ratio))
        if removed > max_remove and len(sections) > 2 and kept_parts:
            restored: list[str] = []
            for section, decision in zip(sections, routing_decisions):
                if decision.action != 'REMOVE':
                    continue
                edu = decision.signals.educational_value + decision.signals.technical_value
                know = decision.signals.knowledge_value
                if edu >= 0.20 or know >= 0.28:
                    restored.append(section.text.strip())
            if restored:
                out = '\n\n'.join([*kept_parts, *restored]).strip()
            removed = len(sections) - len(kept_parts) - len(restored)
            downweighted = min(downweighted, len(sections))
            cleaned = min(cleaned, len(kept_parts))
            kept = len(kept_parts) + len(restored)

        utility = sum(utilities) / max(len(utilities), 1)
        result = SemanticCleanResult(
            text=out,
            removed_chunks=removed,
            downweighted_chunks=downweighted,
            kept_chunks=kept,
            cleaned_chunks=cleaned,
            utility=utility,
            artifact_categories=artifact_totals,
            ocr_repairs=ocr_stats.tokens_fixed + ocr_stats.lines_merged,
            classifications=classifications,
            routing_decisions=routing_decisions,
            samples=samples,
        )

        report = get_semantic_clean_report()
        report.threshold_snapshot = self._calibrator.snapshot()
        report.observe_doc(
            before=before,
            after=out,
            routing_actions=routing_actions,
            section_roles=labels,
            utility=utility,
            samples=samples,
            artifact_categories=artifact_totals,
            ocr_repairs=result.ocr_repairs,
        )
        return result

def clean_document_semantic(
    text: str,
    config: SemanticCleaningConfig | None = None,
) -> SemanticCleanResult:
    return SemanticCleaningPipeline(config).process(text)
