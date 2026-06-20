from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from indw.filter.toxicity.config import ToxicityPolicyConfig
from indw.filter.toxicity.context import ContextClassifier, ContextResult
from indw.filter.toxicity.rule_scorer import CategoryScores, RuleBasedToxicityScorer
from indw.filter.toxicity.patterns import PatternEngine, PatternEngineResult
from indw.filter.toxicity.rules import RuleEngine, RuleEngineResult
from indw.filter.toxicity.scorer import FinalToxicityScore, combine_scores

@dataclass
class ToxicityAssessment:
    rule: RuleEngineResult
    pattern: PatternEngineResult
    ml: CategoryScores
    context: ContextResult
    final: FinalToxicityScore

    @property
    def toxicity_score(self) -> float:
        return self.final.final_toxicity_score

    @property
    def toxicity_reason(self) -> Optional[str]:
        return self.final.toxicity_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            'rule_score': round(self.rule.rule_score, 4),
            'pattern_score': round(self.pattern.pattern_score, 4),
            'classifier': self.ml.to_dict(),
            'context': self.context.to_dict(),
            'final_toxicity_score': round(self.final.final_toxicity_score, 4),
            'band': self.final.band,
            'toxicity_score': round(self.final.final_toxicity_score, 4),
            'toxicity_reason': self.final.toxicity_reason,
        }

class ToxicityDetector:
    def __init__(self, policy: Optional[ToxicityPolicyConfig] = None):
        self.policy = policy or ToxicityPolicyConfig.resolve()
        raw = self.policy.rules_raw()
        self._rules = RuleEngine(raw)
        self._patterns = PatternEngine(raw)
        self._scorer = RuleBasedToxicityScorer()
        self._context = ContextClassifier(self.policy.context)

    def assess(
        self,
        text: str,
        *,
        factual_density: float = 0.0,
        educational_value: float = 0.0,
    ) -> ToxicityAssessment:
        if not self.policy.enabled:
            empty_rule = RuleEngineResult()
            empty_pat = PatternEngineResult()
            empty_ml = CategoryScores()
            empty_ctx = ContextResult()
            final = combine_scores(
                rule=empty_rule,
                pattern=empty_pat,
                ml=empty_ml,
                context=empty_ctx,
                policy=self.policy,
            )
            return ToxicityAssessment(empty_rule, empty_pat, empty_ml, empty_ctx, final)
        rule = self._rules.evaluate(text)
        pattern = self._patterns.evaluate(text)
        rule_cats = [h.category for h in rule.hits]
        pat_cats = [h.category for h in pattern.hits]
        ml = self._scorer.predict(
            text,
            rule_hits=rule_cats,
            pattern_hits=pat_cats,
            rule_score=rule.rule_score,
            pattern_score=pattern.pattern_score,
        )
        context = self._context.classify(
            text,
            classifier_scores=ml.to_dict(),
            rule_score=rule.rule_score,
            pattern_score=pattern.pattern_score,
            factual_density=factual_density,
            educational_value=educational_value,
        )
        final = combine_scores(rule=rule, pattern=pattern, ml=ml, context=context, policy=self.policy)
        return ToxicityAssessment(rule, pattern, ml, context, final)
