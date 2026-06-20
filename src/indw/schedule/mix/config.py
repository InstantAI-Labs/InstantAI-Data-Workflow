from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class ContextStageConfig:
    name: str = 'short'
    max_chars: int = 4096
    min_chars: int = 100
    weight: float = 1.0

@dataclass
class MixtureOrchestrationConfig:
    enabled: bool = True
    replay_seed: int = 42
    max_adapt_delta: float = 0.12
    synthetic_cap: float = 0.05
    min_token_efficiency_chars: float = 2.5
    curriculum_start: str = 'simple'
    token_budget_per_stage: int = 50_000_000
    domain_targets: dict[str, float] = field(
        default_factory=lambda: {
            'web': 0.42,
            'wiki': 0.14,
            'docs': 0.12,
            'code': 0.16,
            'reasoning': 0.08,
            'conversation': 0.06,
            'qa': 0.06,
        }
    )
    language_targets: dict[str, float] = field(
        default_factory=lambda: {
            'en': 0.72,
            'hi': 0.08,
            'ar': 0.05,
            'zh': 0.08,
            'ja': 0.04,
            'ko': 0.04,
            'other': 0.08,
        }
    )
    context_stages: list[ContextStageConfig] = field(
        default_factory=lambda: [
            ContextStageConfig(name='short', max_chars=4096, weight=1.0),
            ContextStageConfig(name='medium', max_chars=16384, weight=0.85),
            ContextStageConfig(name='long', max_chars=65536, weight=0.65),
            ContextStageConfig(name='book', max_chars=100000, weight=0.45),
        ]
    )

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> MixtureOrchestrationConfig:
        if not raw:
            return cls()
        ctx_raw = raw.get('context_stages') or []
        stages: list[ContextStageConfig] = []
        for item in ctx_raw:
            if isinstance(item, dict):
                stages.append(
                    ContextStageConfig(
                        name=str(item.get('name', 'short')),
                        max_chars=int(item.get('max_chars', 4096)),
                        min_chars=int(item.get('min_chars', 100)),
                        weight=float(item.get('weight', 1.0)),
                    )
                )
        return cls(
            enabled=bool(raw.get('enabled', True)),
            replay_seed=int(raw.get('replay_seed', 42)),
            max_adapt_delta=float(raw.get('max_adapt_delta', 0.12)),
            synthetic_cap=float(raw.get('synthetic_cap', 0.05)),
            min_token_efficiency_chars=float(raw.get('min_token_efficiency_chars', 2.5)),
            curriculum_start=str(raw.get('curriculum_start', 'simple')),
            token_budget_per_stage=int(raw.get('token_budget_per_stage', 50_000_000)),
            domain_targets=dict(raw.get('domain_targets') or cls().domain_targets),
            language_targets=dict(raw.get('language_targets') or cls().language_targets),
            context_stages=stages or cls().context_stages,
        )
