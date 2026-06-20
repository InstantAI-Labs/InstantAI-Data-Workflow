from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from indw.filter.toxicity.config import ContextConfig

ContextLabel = Literal['educational', 'toxic', 'neutral']

@dataclass
class ContextResult:
    context: ContextLabel = 'neutral'
    confidence: float = 0.0
    educational_score: float = 0.0
    toxic_direct_score: float = 0.0
    context_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'context': self.context,
            'confidence': round(self.confidence, 4),
            'educational_score': round(self.educational_score, 4),
            'toxic_direct_score': round(self.toxic_direct_score, 4),
            'context_score': round(self.context_score, 4),
        }

class ContextClassifier:
    def __init__(self, config: ContextConfig):
        self.config = config

    def classify(
        self,
        text: str,
        *,
        classifier_scores: Optional[dict[str, float]] = None,
        rule_score: float = 0.0,
        pattern_score: float = 0.0,
        factual_density: float = 0.0,
        educational_value: float = 0.0,
    ) -> ContextResult:
        if not text:
            return ContextResult()
        signal_boost = 0.0
        if classifier_scores:
            signal_boost = max(classifier_scores.values()) * 0.25
        journalistic = bool(
            re.search(
                r'(?i)\b(news report|peer.reviewed|according to the|historical record|tribunal documented|'
                r'this paper presents|legal judgment|investigators revealed)\b',
                text,
            )
        )
        edu_signal = min(
            1.0,
            factual_density * 0.4
            + educational_value * 0.4
            + (0.35 if journalistic else 0.0)
            + (0.15 if re.search(r'(?i)\b(abstract|methodology|according to|historically)\b', text) else 0.0),
        )
        toxic_direct = min(1.0, signal_boost + max(rule_score, pattern_score) * 0.5)
        if journalistic and edu_signal >= 0.38 and toxic_direct < 0.4:
            return ContextResult(
                context='educational',
                confidence=min(1.0, edu_signal),
                educational_score=edu_signal,
                toxic_direct_score=toxic_direct,
                context_score=max(0.0, 1.0 - toxic_direct),
            )
        if rule_score >= 0.85 or pattern_score >= 0.85:
            return ContextResult(
                context='toxic',
                confidence=min(1.0, max(rule_score, pattern_score)),
                educational_score=edu_signal,
                toxic_direct_score=toxic_direct,
                context_score=toxic_direct,
            )
        if edu_signal >= 0.42 and toxic_direct < 0.35 and max(rule_score, pattern_score) < 0.5:
            return ContextResult(
                context='educational',
                confidence=min(1.0, edu_signal),
                educational_score=edu_signal,
                toxic_direct_score=toxic_direct,
                context_score=1.0 - toxic_direct,
            )
        if toxic_direct >= 0.5:
            return ContextResult(
                context='toxic',
                confidence=toxic_direct,
                educational_score=edu_signal,
                toxic_direct_score=toxic_direct,
                context_score=toxic_direct,
            )
        return ContextResult(
            context='neutral',
            confidence=0.5,
            educational_score=edu_signal,
            toxic_direct_score=toxic_direct,
            context_score=0.25,
        )
