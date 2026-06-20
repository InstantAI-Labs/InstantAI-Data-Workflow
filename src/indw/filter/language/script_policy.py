from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class MultilingualPolicyConfig:
    enabled: bool = True
    script_targets: dict[str, float] = field(default_factory=dict)
    locale_bucket_map: dict[str, str] = field(default_factory=dict)
    max_mixed_script_score: float = 0.72
    max_fragmentation_risk: float = 0.78
    max_unicode_instability: float = 0.12
    min_reasoning_stability: float = 0.18
    target_chars_per_token: float = 3.2
    max_token_inflation_risk: float = 0.65
    max_adapt_delta: float = 0.12
    starvation_floor: float = 0.03

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> MultilingualPolicyConfig:
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get('enabled', True)),
            script_targets=dict(
                raw.get('script_targets') or raw.get('language_targets') or {}
            ),
            locale_bucket_map=dict(raw.get('locale_bucket_map') or {}),
            max_mixed_script_score=float(raw.get('max_mixed_script_score', 0.72)),
            max_fragmentation_risk=float(raw.get('max_fragmentation_risk', 0.78)),
            max_unicode_instability=float(raw.get('max_unicode_instability', 0.12)),
            min_reasoning_stability=float(raw.get('min_reasoning_stability', 0.18)),
            target_chars_per_token=float(raw.get('target_chars_per_token', 3.2)),
            max_token_inflation_risk=float(
                raw.get('max_token_inflation_risk', 0.65)
            ),
            max_adapt_delta=float(raw.get('max_adapt_delta', 0.12)),
            starvation_floor=float(raw.get('starvation_floor', 0.03)),
        )
