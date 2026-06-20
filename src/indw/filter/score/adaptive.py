from __future__ import annotations

from typing import Optional

from indw.filter.content.code import CodeQualitySignals
from indw.filter.score.signals import QualitySignals
from indw.clean.artifact.evidence import AdaptiveBaselineEstimator
from indw.clean.document.value import ContentValueSignals, is_information_rich

def adaptive_document_score(
    signals: QualitySignals,
    *,
    domain: str,
    code: Optional[CodeQualitySignals],
    content_value: Optional[ContentValueSignals] = None,
    multilingual_quality: float = 0.0,
    text: str = '',
) -> float:
    baseline = AdaptiveBaselineEstimator()
    positive = [
        min(1.0, signals.char_entropy / 5.0),
        signals.word_diversity,
        1.0 - max(signals.line_repetition, signals.char_repetition),
        signals.formatting_score,
        1.0 - signals.html_score,
        1.0 - signals.injection_score,
        signals.alpha_ratio,
        signals.structural_quality,
        signals.coherence_score,
        signals.factual_density,
        signals.educational_value,
    ]
    negative = [
        signals.token_spam_score,
        signals.reasoning_repetition,
        signals.truncation_score,
        signals.boilerplate_score,
        signals.commercial_score,
        signals.seo_spam_score,
        signals.low_information_score,
        signals.template_synthetic_score,
        signals.hallucination_risk_score,
        signals.software_piracy_score,
    ]
    if domain == 'reasoning':
        positive.append(signals.reasoning_density)
    if domain == 'code' and code:
        positive.append(code.educational_score)
        negative.append(code.duplicate_line_ratio)

    score = baseline.baseline(positive) * (1.0 - baseline.baseline(negative))

    if content_value is not None and content_value.evidence is not None:
        ev = content_value.evidence
        score = baseline.baseline([score, ev.utility, ev.semantic_strength * ev.coherence])
        if ev.preserve:
            score = baseline.baseline([score, ev.utility + (1.0 - ev.uncertainty) * 0.1])
    elif content_value is not None:
        score = baseline.baseline([score, content_value.overall_value_score])
        if is_information_rich(content_value, text=text):
            score = baseline.baseline([score, content_value.overall_value_score])

    if multilingual_quality > 0.0:
        score = max(score, multilingual_quality * 0.15 + score * 0.85)
    return max(0.0, min(1.0, score))
