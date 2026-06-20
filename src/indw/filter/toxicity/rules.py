from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

@dataclass
class RuleHit:
    category: str
    weight: float
    severity: float

@dataclass
class RuleEngineResult:
    rule_score: float = 0.0
    hits: list[RuleHit] = field(default_factory=list)
    top_category: str = ''

class RuleEngine:

    def __init__(self, policy_raw: dict[str, Any]):
        fast = policy_raw.get('fast_rules') or {}
        self._min_profanity_repeats = int((fast.get('profanity_spam') or {}).get('min_repeats', 5))
        self._min_token_len = int((fast.get('profanity_spam') or {}).get('min_token_len', 3))
        phrase = fast.get('repeated_phrase') or {}
        self._phrase_repeats = int(phrase.get('min_repeats', 4))
        self._phrase_window = int(phrase.get('window_tokens', 6))
        self._compiled: list[tuple[str, float, float, re.Pattern[str]]] = []
        for category, spec in (fast.get('rules') or {}).items():
            if not isinstance(spec, Mapping):
                continue
            weight = float(spec.get('weight', 0.9))
            severity = float(spec.get('severity', weight))
            for pat in spec.get('patterns') or []:
                try:
                    self._compiled.append((str(category), weight, severity, re.compile(str(pat))))
                except re.error:
                    continue

    def _profanity_spam_score(self, text: str) -> float:
        tokens = re.findall(r"\b[\w']+\b", text.lower())
        if len(tokens) < self._min_profanity_repeats:
            return 0.0
        counts = Counter(t for t in tokens if len(t) >= self._min_token_len)
        if not counts:
            return 0.0
        top = counts.most_common(1)[0][1]
        return min(1.0, top / max(len(tokens), 1) * 4.0) if top >= self._min_profanity_repeats else 0.0

    def _repeated_phrase_score(self, text: str) -> float:
        tokens = re.findall(r"\b[\w']+\b", text.lower())
        if len(tokens) < self._phrase_window * self._phrase_repeats:
            return 0.0
        for i in range(0, len(tokens) - self._phrase_window + 1):
            phrase = ' '.join(tokens[i : i + self._phrase_window])
            chunk = ' '.join(tokens)
            if chunk.count(phrase) >= self._phrase_repeats:
                return min(1.0, 0.55 + 0.1 * chunk.count(phrase))
        return 0.0

    def evaluate(self, text: str) -> RuleEngineResult:
        if not text:
            return RuleEngineResult()
        hits: list[RuleHit] = []
        best = 0.0
        top = ''
        spam = max(self._profanity_spam_score(text), self._repeated_phrase_score(text))
        if spam >= 0.5:
            hits.append(RuleHit(category='profanity_spam', weight=0.8, severity=spam))
            best = max(best, spam * 0.85)
            top = 'profanity_spam'
        for category, weight, severity, rx in self._compiled:
            if rx.search(text):
                score = min(1.0, weight * severity)
                hits.append(RuleHit(category=category, weight=weight, severity=severity))
                if score > best:
                    best = score
                    top = category
        if len(hits) > 1:
            best = min(1.0, best + 0.04 * (len(hits) - 1))
        return RuleEngineResult(rule_score=best, hits=hits, top_category=top)
