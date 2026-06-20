from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from indw.store.eval.compare import CorpusComparison
from indw.store.eval.config import CorpusEvaluationConfig
from indw.store.eval.scoring import CorpusScoreResult

DecisionLabel = Literal['ACCEPT', 'REVIEW', 'REJECT']

@dataclass
class AcceptanceDecision:
    decision: DecisionLabel = 'REVIEW'
    promotion_band: str = 'REVIEW'
    confidence: float = 0.0
    corpus_score: int = 0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'decision': self.decision,
            'promotion_band': self.promotion_band,
            'confidence': round(self.confidence, 4),
            'corpus_score': self.corpus_score,
            'reasons': self.reasons,
        }

def decide_acceptance(
    score: CorpusScoreResult,
    comparison: CorpusComparison,
    *,
    config: CorpusEvaluationConfig,
    has_previous: bool,
) -> AcceptanceDecision:
    prom = config.promotion
    reasons: list[str] = []
    for sig in comparison.improvements:
        reasons.append(f'{sig.metric}_{sig.direction}')
    critical_regressions = [s for s in comparison.regressions if s.kind in ('quality', 'dedup', 'safety')]
    for sig in comparison.regressions:
        reasons.append(f'{sig.metric}_{sig.direction}')

    s = score.corpus_score
    if s >= prom.promote_min_score:
        band = 'PROMOTE'
    elif s >= prom.review_min_score:
        band = 'REVIEW'
    else:
        band = 'REJECT'

    if not has_previous:
        if s >= prom.review_min_score and not critical_regressions:
            decision = 'ACCEPT'
            confidence = min(0.96, 0.86 + s / 200.0)
            reasons.append('first_corpus_version')
        elif critical_regressions:
            decision = 'REJECT'
            confidence = 0.88
        else:
            decision = 'REJECT'
            confidence = 0.82
    elif critical_regressions:
        decision = 'REJECT'
        confidence = min(0.99, 0.75 + 0.05 * len(critical_regressions))
    elif band == 'REJECT':
        decision = 'REJECT'
        confidence = 0.85
    elif band == 'PROMOTE' and not critical_regressions:
        decision = 'ACCEPT'
        confidence = min(0.99, 0.88 + s / 1000.0)
    elif band == 'REVIEW' and comparison.improvements and not critical_regressions:
        if s >= prom.review_min_score + 6 and len(comparison.improvements) >= 2:
            decision = 'ACCEPT'
            confidence = min(0.94, 0.82 + len(comparison.improvements) * 0.04)
        else:
            decision = 'REVIEW'
            confidence = 0.78
    else:
        decision = 'REVIEW'
        confidence = 0.78
    if s >= prom.promote_min_score and decision == 'ACCEPT':
        reasons.append('corpus_score_promote')
    if critical_regressions:
        reasons = [f'regression_{s.metric}' for s in critical_regressions] + reasons

    return AcceptanceDecision(
        decision=decision,
        promotion_band=band,
        confidence=confidence,
        corpus_score=s,
        reasons=sorted(set(reasons))[:12],
    )
