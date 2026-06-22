from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from indw.util.stable_hash import stable_digest_hex

@dataclass
class CorpusMixturePlan:
    version: str = 'instant-mixture-v1'
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    curriculum_stage: str = 'simple'
    replay_seed: int = 42
    plan_digest: str = ''
    domain_weights: dict[str, float] = field(default_factory=dict)
    language_weights: dict[str, float] = field(default_factory=dict)
    context_stages: list[dict[str, Any]] = field(default_factory=list)
    epoch_schedule: list[dict[str, Any]] = field(default_factory=list)
    quality_multipliers: dict[str, float] = field(default_factory=dict)
    synthetic_cap: float = 0.05
    telemetry: dict[str, Any] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)

    def finalize_digest(self) -> None:
        payload = {
            'version': self.version,
            'curriculum_stage': self.curriculum_stage,
            'replay_seed': self.replay_seed,
            'domain_weights': self.domain_weights,
            'language_weights': self.language_weights,
            'context_stages': self.context_stages,
            'epoch_schedule': self.epoch_schedule,
            'quality_multipliers': self.quality_multipliers,
            'synthetic_cap': self.synthetic_cap,
        }
        self.plan_digest = stable_digest_hex(payload)[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            'version': self.version,
            'created_at': self.created_at,
            'curriculum_stage': self.curriculum_stage,
            'replay_seed': self.replay_seed,
            'plan_digest': self.plan_digest,
            'domain_weights': self.domain_weights,
            'language_weights': self.language_weights,
            'context_stages': self.context_stages,
            'epoch_schedule': self.epoch_schedule,
            'quality_multipliers': self.quality_multipliers,
            'synthetic_cap': self.synthetic_cap,
            'telemetry': self.telemetry,
            'observations': self.observations,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.plan_digest:
            self.finalize_digest()
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')
        return path

    @classmethod
    def load(cls, path: Path) -> CorpusMixturePlan:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(
            version=str(raw.get('version', 'instant-mixture-v1')),
            created_at=str(raw.get('created_at', '')),
            curriculum_stage=str(raw.get('curriculum_stage', 'simple')),
            replay_seed=int(raw.get('replay_seed', 42)),
            plan_digest=str(raw.get('plan_digest', '')),
            domain_weights=dict(raw.get('domain_weights') or {}),
            language_weights=dict(raw.get('language_weights') or {}),
            context_stages=list(raw.get('context_stages') or []),
            epoch_schedule=list(raw.get('epoch_schedule') or []),
            quality_multipliers=dict(raw.get('quality_multipliers') or {}),
            synthetic_cap=float(raw.get('synthetic_cap', 0.05)),
            telemetry=dict(raw.get('telemetry') or {}),
            observations=dict(raw.get('observations') or {}),
        )
