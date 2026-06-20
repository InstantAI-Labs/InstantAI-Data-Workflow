from __future__ import annotations

from dataclasses import dataclass

from indw.filter.toxicity.classifier_labels import CATEGORY_HYPOTHESES

TOXICITY_CATEGORIES = tuple(CATEGORY_HYPOTHESES.keys())

@dataclass
class CategoryScores:
    hate: float = 0.0
    harassment: float = 0.0
    violence: float = 0.0
    sexual: float = 0.0
    extremism: float = 0.0
    self_harm: float = 0.0
    toxicity_score: float = 0.0
    backend: str = 'rules'

    def to_dict(self) -> dict[str, float]:
        return {
            'hate': round(self.hate, 4),
            'harassment': round(self.harassment, 4),
            'violence': round(self.violence, 4),
            'sexual_abuse': round(self.sexual, 4),
            'extremism': round(self.extremism, 4),
            'self_harm': round(self.self_harm, 4),
            'toxicity_score': round(self.toxicity_score, 4),
        }

    def top_category(self) -> str:
        pairs = [
            ('hate', self.hate),
            ('harassment', self.harassment),
            ('violence', self.violence),
            ('sexual_abuse', self.sexual),
            ('extremism', self.extremism),
            ('self_harm', self.self_harm),
        ]
        return max(pairs, key=lambda x: x[1])[0]

    @classmethod
    def from_mapping(cls, scores: dict[str, float], *, backend: str = 'rules') -> CategoryScores:
        return cls(
            hate=float(scores.get('hate', 0.0)),
            harassment=float(scores.get('harassment', 0.0)),
            violence=float(scores.get('violence', 0.0)),
            sexual=float(scores.get('sexual_abuse', scores.get('sexual', 0.0))),
            extremism=float(scores.get('extremism', 0.0)),
            self_harm=float(scores.get('self_harm', 0.0)),
            toxicity_score=float(scores.get('toxicity_score', max(scores.values()) if scores else 0.0)),
            backend=backend,
        )

class RuleBasedToxicityScorer:
    _RULE_MAP = {
        'extremist_slogan': 'extremism',
        'harassment_spam': 'harassment',
        'harassment_direct': 'harassment',
        'hate_group': 'hate',
        'profanity_spam': 'harassment',
    }
    _PATTERN_MAP = {
        'direct_threat': 'violence',
        'self_harm_directive': 'self_harm',
        'recruitment': 'extremism',
        'harassment_insult': 'harassment',
        'hate_extermination': 'hate',
    }

    def predict(
        self,
        text: str,
        *,
        rule_hits: list[str],
        pattern_hits: list[str],
        rule_score: float,
        pattern_score: float,
    ) -> CategoryScores:
        del text
        scores = {c: 0.0 for c in TOXICITY_CATEGORIES}
        for cat in rule_hits:
            key = self._RULE_MAP.get(cat, 'harassment')
            scores[key] = max(scores.get(key, 0.0), rule_score * 0.9)
        for cat in pattern_hits:
            key = self._PATTERN_MAP.get(cat, 'harassment')
            scores[key] = max(scores.get(key, 0.0), pattern_score * 0.92)
        toxicity = max(scores.values()) if scores else max(rule_score, pattern_score) * 0.85
        return CategoryScores.from_mapping({**scores, 'toxicity_score': toxicity}, backend='rules')
