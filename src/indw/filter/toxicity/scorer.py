from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from indw.filter.toxicity.config import ToxicityPolicyConfig
from indw.filter.toxicity.context import ContextResult
from indw.filter.toxicity.rule_scorer import CategoryScores
from indw.filter.toxicity.patterns import PatternEngineResult
from indw.filter.toxicity.rules import RuleEngineResult

_REASON_MAP = {
    'hate': 'hate_speech',
    'harassment': 'harassment',
    'violence': 'violence',
    'sexual_abuse': 'sexual_abuse',
    'sexual': 'sexual_abuse',
    'extremism': 'extremism',
    'self_harm': 'self_harm',
    'extremist_slogan': 'extremism',
    'harassment_spam': 'harassment',
    'profanity_spam': 'harassment',
    'direct_threat': 'violence',
    'self_harm_directive': 'self_harm',
}

@dataclass
class FinalToxicityScore:
    rule_score: float
    pattern_score: float
    classifier_score: float
    context_score: float
    final_toxicity_score: float
    band: str
    toxicity_reason: Optional[str]
    should_reject: bool
    should_hard_reject: bool

    @property
    def toxicity_score(self) -> float:
        return self.classifier_score

    def to_public_dict(self) -> dict:
        return {
            'toxicity_score': round(self.final_toxicity_score, 4),
            'toxicity_reason': self.toxicity_reason,
        }

def combine_scores(
    *,
    rule: RuleEngineResult,
    pattern: PatternEngineResult,
    ml: CategoryScores,
    context: ContextResult,
    policy: ToxicityPolicyConfig,
) -> FinalToxicityScore:
    weights = policy.scoring_weights
    w_clf = float(weights.get('classifier', weights.get('ml', 0.72)))
    w_rule = float(weights.get('rule', 0.08))
    w_pattern = float(weights.get('pattern', 0.08))
    w_ctx = float(weights.get('context', weights.get('context_toxic', 0.12)))
    clf_score = ml.toxicity_score
    ctx_boost = context.context_score if context.context == 'toxic' else 0.0
    weighted = (
        w_clf * clf_score
        + w_rule * rule.rule_score
        + w_pattern * pattern.pattern_score
        + w_ctx * ctx_boost
    )
    raw = min(1.0, max(weighted, clf_score * 0.92, rule.rule_score * 0.5, pattern.pattern_score * 0.5))
    if context.context == 'educational' and context.confidence >= policy.educational_min_confidence:
        raw *= max(0.0, 1.0 - policy.educational_dampening * context.confidence)
    final = max(0.0, min(1.0, raw))
    if (
        context.context in ('educational', 'neutral')
        and context.educational_score >= 0.45
        and rule.rule_score < 0.8
        and pattern.pattern_score < 0.8
        and clf_score < 0.72
    ):
        final = min(final, policy.rejection.review_max - 0.01)
    if (
        context.context == 'educational'
        and context.confidence >= policy.educational_min_confidence
        and rule.rule_score < 0.8
        and pattern.pattern_score < 0.8
        and clf_score < 0.75
    ):
        final = min(final, policy.rejection.accept_max + 0.12)
    rej = policy.rejection
    band = rej.band(final)
    reason = _resolve_reason(ml, rule, pattern, final)
    hard = band == 'hard_reject'
    reject = band in ('reject', 'hard_reject')
    if policy.max_toxicity_score is not None and final > policy.max_toxicity_score:
        reject = True
        if final >= rej.hard_reject_min:
            hard = True
    return FinalToxicityScore(
        rule_score=rule.rule_score,
        pattern_score=pattern.pattern_score,
        classifier_score=clf_score,
        context_score=context.context_score,
        final_toxicity_score=final,
        band=band,
        toxicity_reason=reason if reject else None,
        should_reject=reject,
        should_hard_reject=hard,
    )

def _resolve_reason(
    ml: CategoryScores,
    rule: RuleEngineResult,
    pattern: PatternEngineResult,
    final: float,
) -> Optional[str]:
    if final <= 0.2:
        return None
    top = ml.top_category()
    if ml.toxicity_score >= 0.35:
        return _REASON_MAP.get(top, top)
    if rule.top_category:
        return _REASON_MAP.get(rule.top_category, rule.top_category)
    if pattern.top_category:
        return _REASON_MAP.get(pattern.top_category, pattern.top_category)
    return _REASON_MAP.get(top, top)
