from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from indw.filter.language.confidence import ConfidenceEstimate, estimate_confidence, fragmentation_score
from indw.filter.language.config import LanguagePolicyConfig
from indw.filter.language.fast_detector import FastLanguageDetector
from indw.filter.language.mixed import MixedLanguageAnalyzer

def _locale_cap(locale: str, caps: dict[str, float]) -> float:
    if not caps:
        return 1.0
    return caps.get(locale, caps.get('other', 1.0))

@dataclass
class LanguageAssessment:
    primary_language: str
    confidence: float
    languages: dict[str, float]
    mixed_language: bool = False
    fragmentation: float = 0.0
    reject_reason: Optional[str] = None
    should_reject: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'primary_language': self.primary_language,
            'confidence': round(self.confidence, 4),
            'languages': {k: round(v, 4) for k, v in sorted(self.languages.items(), key=lambda kv: -kv[1])[:12]},
            'mixed_language': self.mixed_language,
            'fragmentation': round(self.fragmentation, 4),
            'reject_reason': self.reject_reason,
            'should_reject': self.should_reject,
        }

    def to_gate_dict(self) -> dict[str, Any]:
        return {
            'language': self.primary_language,
            'language_confidence': round(self.confidence, 4),
        }

class LanguageIdentifier:
    def __init__(
        self,
        policy: Optional[LanguagePolicyConfig] = None,
        *,
        locale_caps: Optional[dict[str, float]] = None,
    ):
        self.policy = policy or LanguagePolicyConfig.resolve()
        self.locale_caps = dict(locale_caps or {})
        self._detector = FastLanguageDetector(self.policy.detector)
        self._mixed = MixedLanguageAnalyzer(self._detector, self.policy.mixed)

    def assess(
        self,
        text: str,
        *,
        domain: str = '',
        script_segments: tuple[tuple[str, str], ...] | list[tuple[str, str]] | None = None,
    ) -> LanguageAssessment:
        if not self.policy.enabled or not text:
            return LanguageAssessment('und', 0.0, {}, reject_reason='empty')
        hints = self.policy.hints
        mixed = self._mixed.analyze(text, script_segments=script_segments)
        distribution = dict(mixed.languages)
        if not distribution:
            distribution = self._detector.predict_distribution(text)
        conf: ConfidenceEstimate = estimate_confidence(distribution)
        frag = fragmentation_score(distribution)
        primary = conf.primary_language
        if (
            conf.primary_probability < hints.und_primary_probability
            and conf.confidence < hints.und_confidence
            and frag >= hints.und_fragmentation
        ):
            primary = 'und'
            conf = ConfidenceEstimate('und', conf.confidence, conf.primary_probability, conf.margin)
        languages = {k: round(v, 4) for k, v in distribution.items()}
        if primary in languages:
            languages[primary] = max(languages.get(primary, 0.0), round(conf.primary_probability, 4))
        reject_reason = None
        should_reject = False
        gate = self.policy.gate
        technical_domains = set(hints.technical_domains)
        if gate.reject_unknown and primary in ('und', 'unknown'):
            reject_reason = 'unknown_language'
            should_reject = True
        elif conf.confidence < gate.min_confidence:
            reject_reason = 'low_language_confidence'
            should_reject = True
        elif conf.primary_probability < gate.min_primary_probability:
            if domain not in technical_domains:
                reject_reason = 'low_language_confidence'
                should_reject = True
        elif frag > gate.max_fragmentation and domain not in technical_domains:
            reject_reason = 'language_fragmentation'
            should_reject = True
        elif (
            gate.reject_zero_cap_locales
            and self.locale_caps
            and primary not in ('und', 'unknown')
            and _locale_cap(primary, self.locale_caps) <= 0.0
        ):
            reject_reason = 'language_cap'
            should_reject = True
        return LanguageAssessment(
            primary_language=primary,
            confidence=conf.confidence,
            languages=languages,
            mixed_language=mixed.mixed_language,
            fragmentation=frag,
            reject_reason=reject_reason,
            should_reject=should_reject,
        )

    def assess_english_fast(self, text: str, *, domain: str = '') -> LanguageAssessment:
        if not self.policy.enabled or not text:
            return LanguageAssessment('und', 0.0, {}, reject_reason='empty')
        distribution = self._detector.predict_distribution(text)
        conf: ConfidenceEstimate = estimate_confidence(distribution)
        frag = fragmentation_score(distribution)
        primary = conf.primary_language
        hints = self.policy.hints
        if (
            conf.primary_probability < hints.und_primary_probability
            and conf.confidence < hints.und_confidence
            and frag >= hints.und_fragmentation
        ):
            primary = 'und'
            conf = ConfidenceEstimate('und', conf.confidence, conf.primary_probability, conf.margin)
        languages = {k: round(v, 4) for k, v in distribution.items()}
        if primary in languages:
            languages[primary] = max(languages.get(primary, 0.0), round(conf.primary_probability, 4))
        reject_reason = None
        should_reject = False
        gate = self.policy.gate
        technical_domains = set(hints.technical_domains)
        if gate.reject_unknown and primary in ('und', 'unknown'):
            reject_reason = 'unknown_language'
            should_reject = True
        elif conf.confidence < gate.min_confidence:
            reject_reason = 'low_language_confidence'
            should_reject = True
        elif conf.primary_probability < gate.min_primary_probability:
            if domain not in technical_domains:
                reject_reason = 'low_language_confidence'
                should_reject = True
        elif frag > gate.max_fragmentation and domain not in technical_domains:
            reject_reason = 'language_fragmentation'
            should_reject = True
        elif (
            gate.reject_zero_cap_locales
            and self.locale_caps
            and primary not in ('und', 'unknown')
            and _locale_cap(primary, self.locale_caps) <= 0.0
        ):
            reject_reason = 'language_cap'
            should_reject = True
        return LanguageAssessment(
            primary_language=primary,
            confidence=conf.confidence,
            languages=languages,
            mixed_language=False,
            fragmentation=frag,
            reject_reason=reject_reason,
            should_reject=should_reject,
        )
