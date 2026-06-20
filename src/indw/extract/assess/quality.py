from __future__ import annotations

from dataclasses import dataclass, field

from indw.extract.assess.doc_type import DocumentClassProfile, classify_document
from indw.extract.structure.analyze import StructuralProfile, analyze_structure
from indw.extract.nav.template import TemplateProfile, TemplateMiner
from indw.clean.artifact.evidence_engine import compute_semantic_evidence
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator
from indw.clean.document.value import analyze_content_value

@dataclass
class AdaptiveQualityAssessment:
    educational_value: float = 0.0
    knowledge_density: float = 0.0
    redundancy: float = 0.0
    information_content: float = 0.0
    structural_quality: float = 0.0
    boilerplate_ratio: float = 0.0
    template_ratio: float = 0.0
    noise_ratio: float = 0.0
    coherence: float = 0.0
    context_completeness: float = 0.0
    preserve: bool = False
    discard_reason: str = ''

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}

def assess_quality(
    text: str,
    *,
    structural: StructuralProfile | None = None,
    template: TemplateProfile | None = None,
    doc_class: DocumentClassProfile | None = None,
) -> AdaptiveQualityAssessment:
    if not text or not text.strip():
        return AdaptiveQualityAssessment()

    structural = structural or analyze_structure(text)
    template = template or TemplateMiner().analyze(text)
    doc_class = doc_class or classify_document(text)
    cv = analyze_content_value(text)
    ev = compute_semantic_evidence(text)
    baseline = AdaptiveBaselineEstimator()

    noise = baseline.baseline(list(ev.negative.values()) or [0.0])
    out = AdaptiveQualityAssessment(
        educational_value=cv.educational_score,
        knowledge_density=cv.overall_value_score,
        redundancy=ev.redundancy,
        information_content=structural.information_density,
        structural_quality=structural.paragraph_quality_mean,
        boilerplate_ratio=structural.boilerplate_density,
        template_ratio=template.template_density,
        noise_ratio=noise,
        coherence=baseline.baseline([structural.sentence_completeness_mean, cv.overall_value_score]),
        context_completeness=baseline.baseline([
            structural.content_density,
            1.0 - structural.navigation_density,
            cv.overall_value_score,
        ]),
        preserve=bool(ev.preserve),
        discard_reason=ev.discard_reason or '',
    )
    if cv.evidence and cv.evidence.preserve:
        out.preserve = True
    if doc_class.primary.startswith('noise_') and out.knowledge_density < baseline.baseline([out.knowledge_density, 0.2]):
        out.noise_ratio = max(out.noise_ratio, doc_class.confidence)
    return out
