from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

from indw.config.loader import ConfigRef, Resolver, thaw

@dataclass(frozen=True)
class CurriculumStage:
    name: str
    domain_boost: dict[str, float]
    context_max_chars: int
    min_quality: float
    multilingual_sensitivity: float = 1.0
    reasoning_boost: float = 1.0
    code_boost: float = 1.0
    long_context_ratio: float = 0.0

def _stage_from_raw(raw: dict[str, Any]) -> CurriculumStage:
    name = str(raw.get('name') or '').strip()
    if not name:
        raise ValueError('curriculum stage requires name')
    return CurriculumStage(
        name=name,
        domain_boost=dict(raw.get('domain_boost') or {}),
        context_max_chars=int(raw.get('context_max_chars', 8192)),
        min_quality=float(raw.get('min_quality', 0.0)),
        multilingual_sensitivity=float(raw.get('multilingual_sensitivity', 1.0)),
        reasoning_boost=float(raw.get('reasoning_boost', 1.0)),
        code_boost=float(raw.get('code_boost', 1.0)),
        long_context_ratio=float(raw.get('long_context_ratio', 0.0)),
    )

def _load_manifest_stages() -> tuple[CurriculumStage, ...]:
    ref = os.environ.get('INSTANT_CURRICULUM_MANIFEST')
    if not ref:
        return ()
    raw = thaw(Resolver.default().resolve(ConfigRef(kind='curriculum_manifest', id=str(ref))).raw)
    stages_raw = raw.get('stages') or []
    if not isinstance(stages_raw, list):
        raise TypeError('curriculum manifest stages must be list')
    out = [_stage_from_raw(dict(x)) for x in stages_raw if isinstance(x, dict)]
    return tuple(out)

def _dynamic_fallback_stage(name: str) -> CurriculumStage:
    return CurriculumStage(
        name=str(name or 'dynamic'),
        domain_boost={},
        context_max_chars=8192,
        min_quality=0.0,
        multilingual_sensitivity=1.0,
    )

def _all_stages() -> tuple[CurriculumStage, ...]:
    st = _load_manifest_stages()
    if st:
        return st
    return (_dynamic_fallback_stage('dynamic'),)

CURRICULUM_STAGES: tuple[CurriculumStage, ...] = _all_stages()

def stage_by_name(name: str) -> CurriculumStage:
    for st in _all_stages():
        if st.name == name:
            return st
    return _dynamic_fallback_stage(name)

def build_epoch_schedule(
    *,
    start_stage: str,
    token_budget_per_stage: int,
    total_token_budget: int | None = None,
) -> list[dict[str, Any]]:
    stages = _all_stages()
    start_idx = 0
    for i, st in enumerate(stages):
        if st.name == start_stage:
            start_idx = i
            break
    budget = total_token_budget or token_budget_per_stage * max(1, len(stages) - start_idx)
    per = max(1, budget // max(1, len(stages) - start_idx))
    out: list[dict[str, Any]] = []
    cursor = 0
    for st in stages[start_idx:]:
        out.append(
            {
                'stage': st.name,
                'token_start': cursor,
                'token_end': cursor + per,
                'context_max_chars': st.context_max_chars,
                'min_quality': st.min_quality,
                'long_context_ratio': st.long_context_ratio,
            }
        )
        cursor += per
    return out
