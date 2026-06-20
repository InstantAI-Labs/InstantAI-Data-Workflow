from __future__ import annotations

from typing import Any

from indw.filter.language.script_opt import (
    MultilingualAdaptationState,
    optimize_multilingual_targets,
    quality_multipliers_from_telemetry,
)
from indw.filter.language.script_orch import adapt_targets_from_telemetry
from indw.filter.language.script_policy import MultilingualPolicyConfig
from indw.schedule.mix.config import MixtureOrchestrationConfig
from indw.schedule.mix.curriculum import CURRICULUM_STAGES, stage_by_name
from indw.schedule.mix.plan import CorpusMixturePlan

def _bounded_delta(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target

def _normalize(weights: dict[str, float]) -> dict[str, float]:
    pos = {k: max(0.0, float(v)) for k, v in weights.items()}
    s = sum(pos.values())
    if s <= 0:
        n = max(len(pos), 1)
        return {k: 1.0 / n for k in pos}
    return {k: v / s for k, v in pos.items()}

def adapt_mixture_from_telemetry(
    plan: CorpusMixturePlan,
    telemetry: dict[str, Any],
    *,
    cfg: MixtureOrchestrationConfig | None = None,
) -> CorpusMixturePlan:
    cfg = cfg or MixtureOrchestrationConfig()
    max_d = float(cfg.max_adapt_delta)
    domain = dict(plan.domain_weights)
    language = dict(plan.language_weights)
    quality = dict(plan.quality_multipliers)
    stage = plan.curriculum_stage

    loss = float(telemetry.get('loss', 0.0) or 0.0)
    reasoning_eval = float(telemetry.get('reasoning_eval', 0.0) or 0.0)
    code_eval = float(telemetry.get('code_eval', 0.0) or 0.0)
    hallucination = float(telemetry.get('hallucination_rate', 0.0) or 0.0)
    token_eff = float(telemetry.get('chars_per_token', 0.0) or 0.0)
    mpol = MultilingualPolicyConfig(max_adapt_delta=max_d)
    tok_telemetry = telemetry.get('tokenizer_telemetry') or telemetry.get('tokenizer_runtime') or {}
    if tok_telemetry:
        state = MultilingualAdaptationState()
        language, adapt_meta = optimize_multilingual_targets(
            language,
            tok_telemetry if isinstance(tok_telemetry, dict) else {},
            cfg=mpol,
            state=state,
        )
        tok_mult = quality_multipliers_from_telemetry(
            tok_telemetry.get('global', tok_telemetry),
            target_cpt=mpol.target_chars_per_token,
        )
        for k, v in tok_mult.items():
            quality[k] = _bounded_delta(float(quality.get(k, 1.0)), v, max_d)
    else:
        language = adapt_targets_from_telemetry(language, telemetry, cfg=mpol)
        adapt_meta = {'adapted': True, 'source': 'script_eval'}

    if reasoning_eval < 0.45:
        domain['reasoning'] = _bounded_delta(domain.get('reasoning', 0.05), domain.get('reasoning', 0.05) * 1.15, max_d)
        stage = 'reasoning'
    if code_eval < 0.4:
        domain['code'] = _bounded_delta(domain.get('code', 0.05), domain.get('code', 0.05) * 1.12, max_d)
    if hallucination > 0.2:
        domain['web'] = _bounded_delta(domain.get('web', 0.05), domain.get('web', 0.05) * 0.9, max_d)
        domain['wiki'] = _bounded_delta(domain.get('wiki', 0.05), domain.get('wiki', 0.05) * 1.08, max_d)
        quality['synthetic_penalty'] = _bounded_delta(float(quality.get('synthetic_penalty', 1.0)), 0.85, max_d)
    if loss > 3.0:
        domain['web'] = _bounded_delta(domain.get('web', 0.05), domain.get('web', 0.05) * 1.05, max_d)

    if token_eff > 0 and token_eff < cfg.min_token_efficiency_chars:
        quality['token_efficiency'] = _bounded_delta(float(quality.get('token_efficiency', 1.0)), 0.92, max_d)

    adapted = CorpusMixturePlan(
        version=plan.version,
        curriculum_stage=stage,
        replay_seed=plan.replay_seed,
        domain_weights=_normalize(domain),
        language_weights=_normalize(language),
        context_stages=list(plan.context_stages),
        epoch_schedule=list(plan.epoch_schedule),
        quality_multipliers=quality,
        synthetic_cap=plan.synthetic_cap,
        telemetry={'input': telemetry, 'adapted': True, 'multilingual_adapt': adapt_meta},
        observations=dict(plan.observations),
    )
    adapted.finalize_digest()
    return adapted

def stage_for_token_cursor(plan: CorpusMixturePlan, token_cursor: int) -> str:
    for block in plan.epoch_schedule:
        if token_cursor < int(block.get('token_end', 0)):
            return str(block.get('stage', plan.curriculum_stage))
    if plan.epoch_schedule:
        return str(plan.epoch_schedule[-1].get('stage', plan.curriculum_stage))
    return plan.curriculum_stage

def next_curriculum_stage(current: str) -> str:
    names = [s.name for s in CURRICULUM_STAGES]
    if current not in names:
        return CURRICULUM_STAGES[0].name
    idx = names.index(current)
    if idx + 1 >= len(names):
        return current
    return names[idx + 1]
