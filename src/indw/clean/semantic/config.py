from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class EmbeddedHeuristics:
    url_line_ratio: float = 0.35
    contact_token_ratio: float = 0.12
    anchor_density: float = 0.06
    lead_noise_remove: float = 0.48
    lead_noise_rest_better: float = 0.38
    lead_utility_floor: float = 0.22
    lead_utility_short: float = 0.20
    lead_utility_keep: float = 0.16
    rest_utility_delta: float = 0.12
    prefix_noise_min: float = 0.38
    edge_line_remove: float = 0.55
    edge_line_early: float = 0.38
    suffix_threshold_fence: float = 0.72
    suffix_threshold_plain: float = 0.58
    suffix_utility_preserve: float = 0.15
    knowledge_dampen: float = 0.35
    line_nav_weight: float = 0.30
    line_meta_weight: float = 0.20
    line_boilerplate_weight: float = 0.25
    line_promo_weight: float = 0.15
    line_noise_weight: float = 0.10
    pipe_nav_boost: float = 0.35
    url_line_boost: float = 0.25
    knowledge_dampen_line: float = 0.50
    url_noise_boost: float = 0.20
    promo_boost: float = 0.35
    transactional_boost: float = 0.25
    no_preserve_boost: float = 0.22
    utility_floor: float = 0.22
    metadata_noise_floor: float = 0.40
    lead_short_noise: float = 0.40
    position_lead: float = 0.02
    position_mid: float = 0.5
    position_footer: float = 0.85
    lead_short_chars: int = 700
    edge_early_scan: int = 4
    paragraph_signal_divisor: float = 5.0
    prefix_min_text: int = 200
    prefix_min_sent_len: int = 24
    prefix_max_sentences: int = 8
    strip_prefix_min_len: int = 300
    edu_marker_min: int = 1
    lead_word_min: int = 15
    lead_avg_line_min: int = 35
    lead_keep_word_min: int = 12
    pipe_nav_max_len: int = 220
    url_line_max_len: int = 180
    struct_prefix_max_chars: int = 80

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> EmbeddedHeuristics:
        if not raw:
            return cls()
        defaults = cls()
        out: dict[str, Any] = {}
        for k in defaults.__dataclass_fields__:
            if k not in raw:
                out[k] = getattr(defaults, k)
            elif k in (
                'lead_short_chars', 'edge_early_scan', 'prefix_min_text', 'prefix_min_sent_len',
                'prefix_max_sentences', 'strip_prefix_min_len', 'edu_marker_min', 'lead_word_min',
                'lead_avg_line_min', 'lead_keep_word_min', 'pipe_nav_max_len', 'url_line_max_len',
                'struct_prefix_max_chars',
            ):
                out[k] = int(raw[k])
            else:
                out[k] = float(raw[k])
        return cls(**out)

@dataclass
class SemanticCleaningConfig:
    enabled: bool = True
    legacy_regex_fallback: bool = False
    remove_confidence: float = 0.0
    downweight_confidence: float = 0.0
    preserve_educational_floor: float = 0.22
    preserve_code: bool = True
    preserve_tables: bool = True
    max_remove_ratio: float = 0.45
    calibrate_thresholds: bool = True
    calibration_warmup: int = 200
    fingerprint_similarity_remove: float = 0.78
    record_samples: bool = True
    max_samples_per_doc: int = 4
    embedded: EmbeddedHeuristics = field(default_factory=EmbeddedHeuristics)

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> SemanticCleaningConfig:
        if not raw:
            return cls()
        emb = raw.get('embedded') or raw.get('embedded_heuristics') or {}
        return cls(
            enabled=bool(raw.get('enabled', True)),
            legacy_regex_fallback=bool(raw.get('legacy_regex_fallback', False)),
            remove_confidence=float(raw.get('remove_confidence', 0.0)),
            downweight_confidence=float(raw.get('downweight_confidence', 0.0)),
            preserve_educational_floor=float(raw.get('preserve_educational_floor', 0.22)),
            preserve_code=bool(raw.get('preserve_code', True)),
            preserve_tables=bool(raw.get('preserve_tables', True)),
            max_remove_ratio=float(raw.get('max_remove_ratio', 0.45)),
            calibrate_thresholds=bool(raw.get('calibrate_thresholds', True)),
            calibration_warmup=int(raw.get('calibration_warmup', 200)),
            fingerprint_similarity_remove=float(raw.get('fingerprint_similarity_remove', 0.78)),
            record_samples=bool(raw.get('record_samples', True)),
            max_samples_per_doc=int(raw.get('max_samples_per_doc', 4)),
            embedded=EmbeddedHeuristics.from_dict(emb if isinstance(emb, dict) else None),
        )
