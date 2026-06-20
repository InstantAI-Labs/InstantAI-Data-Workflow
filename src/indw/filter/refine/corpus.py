from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from indw.filter.content.code import analyze_code, analyze_code_dump
from indw.filter.refine.truncation import analyze_truncation
from indw.filter.score.signals import QualitySignals, compute_signals
from indw.clean.artifact.evidence_util import _mean
from indw.clean.document.value import (
    DocumentAnalysisBundle,
    DocumentIntentScores,
    analyze_content_value,
    build_analysis_bundle,
    is_information_rich,
)
Grade = Literal['A', 'B', 'C', 'D', 'E']

@dataclass
class DocumentAnalysisCache:
    signals: QualitySignals | None = None
    content_value: Any | None = None
    bundle: DocumentAnalysisBundle | None = None

    def bundle_for(self, text: str) -> DocumentAnalysisBundle:
        if self.bundle is None:
            self.bundle = build_analysis_bundle(text)
        return self.bundle

    def signals_for(self, text: str) -> QualitySignals:
        if self.signals is None:
            self.signals = self.bundle_for(text).signals(text)
        return self.signals

    def content_value_for(self, text: str, *, source: str = '') -> Any:
        if self.content_value is None:
            self.content_value = analyze_content_value(
                text, source=source, bundle=self.bundle_for(text),
            )
        return self.content_value

@dataclass
class DocumentQualityMetrics:
    coherence: float = 0.0
    knowledge_density: float = 0.0
    educational_value: float = 0.0
    factual_density: float = 0.0
    code_quality: float = 0.0
    language_quality: float = 0.0
    truncation_probability: float = 0.0
    code_dump_probability: float = 0.0
    overall_quality: float = 0.0
    content_category: str = 'blog'
    grade: Grade = 'C'

    def to_dict(self) -> dict[str, float | str]:
        return {
            'coherence': round(self.coherence, 2),
            'knowledge_density': round(self.knowledge_density, 2),
            'educational_value': round(self.educational_value, 2),
            'factual_density': round(self.factual_density, 2),
            'code_quality': round(self.code_quality, 2),
            'language_quality': round(self.language_quality, 2),
            'truncation_probability': round(self.truncation_probability, 4),
            'code_dump_probability': round(self.code_dump_probability, 4),
            'overall_quality': round(self.overall_quality, 2),
            'content_category': self.content_category,
            'grade': self.grade,
        }

@dataclass
class RefineCorpusStats:
    documents_processed: int = 0
    documents_kept: int = 0
    documents_removed: int = 0
    documents_modified: int = 0
    truncated_trimmed: int = 0
    truncated_removed: int = 0
    code_dumps_removed: int = 0
    code_blocks_stripped: int = 0
    knowledge_poor_removed: int = 0
    curator_removed: int = 0
    curator_keep_clean: int = 0
    curator_keep: int = 0
    license_blocks_removed: int = 0
    generated_code_removed: int = 0
    chars_before: int = 0
    chars_after: int = 0
    knowledge_density_before_sum: float = 0.0
    knowledge_density_after_sum: float = 0.0
    overall_quality_before_sum: float = 0.0
    overall_quality_sum: float = 0.0
    drop_reasons: dict[str, int] = field(default_factory=dict)
    category_kept: dict[str, int] = field(default_factory=dict)

    def record_drop(self, reason: str) -> None:
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1

    @property
    def avg_knowledge_density_before(self) -> float:
        n = max(self.documents_processed, 1)
        return self.knowledge_density_before_sum / n

    @property
    def avg_knowledge_density_after(self) -> float:
        n = max(self.documents_kept, 1)
        return self.knowledge_density_after_sum / n

    @property
    def knowledge_density_improvement(self) -> float:
        return self.avg_knowledge_density_after - (
            self.knowledge_density_before_sum / max(self.documents_kept, 1)
        )

    @property
    def token_reduction(self) -> int:
        return int(max(0, self.chars_before - self.chars_after) / 3.8)

    @property
    def corpus_grade(self) -> Grade:
        avg = self.overall_quality_sum / max(self.documents_kept, 1)
        return score_to_grade(avg)

def score_to_grade(score: float) -> Grade:
    if score >= 75:
        return 'A'
    if score >= 60:
        return 'B'
    if score >= 45:
        return 'C'
    if score >= 30:
        return 'D'
    return 'E'

def compute_knowledge_density(
    text: str,
    signals: QualitySignals,
    *,
    content_value: Any = None,
    source: str = '',
    intent: DocumentIntentScores | None = None,
    bundle: DocumentAnalysisBundle | None = None,
    code_dump: CodeDumpResult | None = None,
) -> float:
    del intent
    from indw.filter.content.code import generated_code_score

    ctx = bundle or build_analysis_bundle(text)
    cv = content_value or analyze_content_value(text, source=source, bundle=ctx)
    evidence = cv.evidence or ctx.evidence(text)
    from indw.clean.artifact.evidence_util import evidence_margin

    margin = evidence_margin(evidence.utility, evidence.threshold, evidence.uncertainty)
    density = (evidence.utility + max(0.0, margin)) * 100.0
    if evidence.preserve or is_information_rich(cv, text=text):
        density = max(density, evidence.semantic_strength * evidence.coherence * 100.0)
    if code_dump is not None and code_dump.classification == 'educational_code':
        density += 14.0
    if signals.code_density > 0.20:
        density -= generated_code_score(text) * 14.0
    return max(0.0, min(100.0, density))

def compute_document_metrics(
    text: str,
    *,
    signals: QualitySignals | None = None,
    source: str = '',
    truncation: TruncationResult | None = None,
    code_dump: CodeDumpResult | None = None,
    cache: DocumentAnalysisCache | None = None,
) -> DocumentQualityMetrics:
    if cache is not None and signals is None:
        sig = cache.signals_for(text)
    else:
        sig = signals or compute_signals(text)
    if cache is not None:
        ctx = cache.bundle_for(text)
        cv = cache.content_value_for(text, source=source)
    else:
        ctx = build_analysis_bundle(text)
        cv = analyze_content_value(text, source=source, bundle=ctx)
    code_sig = analyze_code(text) if sig.code_density > 0.12 else None
    trunc = truncation or analyze_truncation(text)
    dump = code_dump or analyze_code_dump(text)

    knowledge = compute_knowledge_density(
        text, sig, content_value=cv, source=source,
        intent=ctx.intent(cv, text=text), bundle=ctx, code_dump=dump,
    )
    evidence = cv.evidence or ctx.evidence(text)
    q = evidence.quality

    def _channel(*values: float) -> float:
        vals = [v for v in values if v > 0]
        if not vals:
            return 0.0
        peak = max(vals)
        return min(100.0, peak * 100.0 if peak <= 1.0 else peak)

    educational = _channel(q.educational, sig.educational_value)
    factual = _channel(q.reference, sig.factual_density, evidence.information_density)
    coherence = _channel(q.coherence, sig.coherence_score, evidence.coherence)
    language_q = _channel(sig.alpha_ratio, sig.burstiness_score, sig.structural_quality, 1.0 - sig.synthetic_score)
    code_q = 0.0
    if code_sig is not None:
        code_q = _channel(code_sig.educational_score, code_sig.syntax_balance, 1.0 - code_sig.generated_score)

    channels = [knowledge, educational, factual, coherence, language_q, code_q]
    active = [c for c in channels if c > 0]
    overall = _mean(active) if active else 0.0
    overall -= _mean([trunc.probability, dump.probability]) * 12.0
    overall = max(0.0, min(100.0, overall))

    return DocumentQualityMetrics(
        coherence=coherence,
        knowledge_density=knowledge,
        educational_value=educational,
        factual_density=factual,
        code_quality=code_q,
        language_quality=language_q,
        truncation_probability=trunc.probability,
        code_dump_probability=dump.probability,
        overall_quality=overall,
        content_category=getattr(cv, 'category', 'blog'),
        grade=score_to_grade(overall),
    )
