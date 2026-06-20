from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

@dataclass
class PatternHit:
    category: str
    weight: float
    severity: float

@dataclass
class PatternEngineResult:
    pattern_score: float = 0.0
    hits: list[PatternHit] = field(default_factory=list)
    top_category: str = ''

_CATEGORY_TO_REASON = {
    'direct_threat': 'violence',
    'self_harm_directive': 'self_harm',
    'recruitment': 'extremism',
    'threats': 'violence',
    'self_harm_encouragement': 'self_harm',
    'harassment_insult': 'harassment',
    'hate_extermination': 'hate',
}

class PatternEngine:
    def __init__(self, policy_raw: dict[str, Any]):
        self._compiled: list[tuple[str, float, float, re.Pattern[str]]] = []
        fast = policy_raw.get('fast_rules') or {}
        for category, spec in (fast.get('patterns') or {}).items():
            if not isinstance(spec, Mapping):
                continue
            weight = float(spec.get('weight', 0.9))
            severity = float(spec.get('severity', weight))
            for pat in spec.get('patterns') or []:
                try:
                    self._compiled.append((str(category), weight, severity, re.compile(str(pat))))
                except re.error:
                    continue

    def evaluate(self, text: str) -> PatternEngineResult:
        if not text or not self._compiled:
            return PatternEngineResult()
        hits: list[PatternHit] = []
        best = 0.0
        top = ''
        for category, weight, severity, rx in self._compiled:
            if rx.search(text):
                score = min(1.0, weight * severity)
                hits.append(PatternHit(category=category, weight=weight, severity=severity))
                if score > best:
                    best = score
                    top = category
        return PatternEngineResult(pattern_score=best, hits=hits, top_category=top)

    @staticmethod
    def reason_for_category(pattern_category: str) -> str:
        return _CATEGORY_TO_REASON.get(pattern_category, pattern_category)
