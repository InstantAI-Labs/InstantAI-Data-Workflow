from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from indw.config.defaults import LANGUAGE_MIN_CONFIDENCE
from indw.config.loader import ConfigRef, Resolver, thaw

DEFAULT_LANGUAGE_SPEC = 'language/identification'


@dataclass
class DetectorConfig:
    backend: str = 'langid'
    min_text_chars: int = 4
    max_chars: int = 12000


@dataclass
class MixedLanguageConfig:
    enabled: bool = True
    min_segment_chars: int = 3
    mixed_threshold: float = 0.28
    dominance_threshold: float = 0.85


@dataclass
class LanguageGateConfig:
    min_confidence: float = LANGUAGE_MIN_CONFIDENCE
    min_primary_probability: float = 0.35
    max_fragmentation: float = 0.72
    reject_unknown: bool = False
    reject_zero_cap_locales: bool = False


@dataclass
class LanguageHintConfig:
    und_primary_probability: float = 0.25
    und_confidence: float = 0.20
    und_fragmentation: float = 0.85
    technical_domains: tuple[str, ...] = ('code', 'docs', 'reasoning', 'qa', 'wiki')


@dataclass
class LanguagePolicyConfig:
    enabled: bool = True
    english_only: bool = False
    skip_post_clean_detection: bool = False
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    mixed: MixedLanguageConfig = field(default_factory=MixedLanguageConfig)
    hints: LanguageHintConfig = field(default_factory=LanguageHintConfig)
    gate: LanguageGateConfig = field(default_factory=LanguageGateConfig)
    validation_min_accuracy: float = 0.95
    validation_min_mixed_accuracy: float = 0.90
    validation_min_unknown_accuracy: float = 0.95
    reporting_output_dir: str = 'artifacts/data/language'

    @classmethod
    def resolve(cls, spec: str = DEFAULT_LANGUAGE_SPEC) -> LanguagePolicyConfig:
        if spec == DEFAULT_LANGUAGE_SPEC:
            cached = _resolved_language_policy()
            if cached is not None:
                return deepcopy(cached)
        resolved = Resolver.default().resolve(ConfigRef(kind='language', id=spec))
        cfg = cls.from_dict(thaw(resolved.raw))
        if spec == DEFAULT_LANGUAGE_SPEC:
            _set_resolved_language_policy(cfg)
        return deepcopy(cfg)

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> LanguagePolicyConfig:
        if not raw:
            return cls.resolve()
        det = raw.get('detector') or {}
        mixed = raw.get('mixed') or {}
        hints = raw.get('hints') or {}
        gate = raw.get('gate') or {}
        val = raw.get('validation') or {}
        reporting = raw.get('reporting') or {}
        cfg = cls(
            enabled=bool(raw.get('enabled', True)),
            english_only=bool(raw.get('english_only', False)),
            skip_post_clean_detection=bool(raw.get('skip_post_clean_detection', False)),
            detector=DetectorConfig(
                backend=str(det.get('backend', 'langid')),
                min_text_chars=int(det.get('min_text_chars', 4)),
                max_chars=int(det.get('max_chars', 12000)),
            ),
            hints=LanguageHintConfig(
                und_primary_probability=float(hints.get('und_primary_probability', 0.25)),
                und_confidence=float(hints.get('und_confidence', 0.20)),
                und_fragmentation=float(hints.get('und_fragmentation', 0.85)),
                technical_domains=tuple(
                    str(x) for x in (
                        hints.get('technical_domains') or LanguageHintConfig().technical_domains
                    )
                ),
            ),
            mixed=MixedLanguageConfig(
                enabled=bool(mixed.get('enabled', not bool(raw.get('english_only', False)))),
                min_segment_chars=int(mixed.get('min_segment_chars', 3)),
                mixed_threshold=float(mixed.get('mixed_threshold', 0.28)),
                dominance_threshold=float(mixed.get('dominance_threshold', 0.85)),
            ),
            gate=LanguageGateConfig(
                min_confidence=float(gate.get('min_confidence', 0.55)),
                min_primary_probability=float(gate.get('min_primary_probability', 0.35)),
                max_fragmentation=float(gate.get('max_fragmentation', 0.72)),
                reject_unknown=bool(gate.get('reject_unknown', False)),
                reject_zero_cap_locales=bool(gate.get('reject_zero_cap_locales', False)),
            ),
            validation_min_accuracy=float(val.get('min_accuracy', 0.95)),
            validation_min_mixed_accuracy=float(val.get('min_mixed_accuracy', 0.90)),
            validation_min_unknown_accuracy=float(val.get('min_unknown_accuracy', 0.95)),
            reporting_output_dir=str(reporting.get('output_dir', 'artifacts/data/language')),
        )
        if cfg.english_only and raw.get('skip_post_clean_detection') is not False:
            cfg.skip_post_clean_detection = True
        return cfg


_RESOLVED_POLICY: LanguagePolicyConfig | None = None


def _resolved_language_policy() -> LanguagePolicyConfig | None:
    return _RESOLVED_POLICY


def _set_resolved_language_policy(cfg: LanguagePolicyConfig) -> None:
    global _RESOLVED_POLICY
    _RESOLVED_POLICY = cfg
