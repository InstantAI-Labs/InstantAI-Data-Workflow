from __future__ import annotations

from typing import Any, Optional

from indw.filter.language.script_opt import quality_multipliers_from_telemetry
from indw.filter.language.script_orch import gap_weights, stage_script_boost
from indw.filter.language.script_policy import MultilingualPolicyConfig
from indw.schedule.mix.config import MixtureOrchestrationConfig
from indw.schedule.mix.curriculum import build_epoch_schedule, stage_by_name
from indw.schedule.mix.plan import CorpusMixturePlan
from indw.filter.gate.quality import QualityGate

def _normalize(weights: dict[str, float]) -> dict[str, float]:
    pos = {k: max(0.0, float(v)) for k, v in weights.items()}
    s = sum(pos.values())
    if s <= 0:
        n = max(len(pos), 1)
        return {k: 1.0 / n for k in pos}
    return {k: v / s for k, v in pos.items()}

def _inverse_gap_weights(observed: dict[str, int], targets: dict[str, float]) -> dict[str, float]:
    total = max(sum(observed.values()), 1)
    obs_frac = {k: observed.get(k, 0) / total for k in targets}
    out: dict[str, float] = {}
    for k, tgt in targets.items():
        gap = max(0.0, tgt - obs_frac.get(k, 0.0))
        out[k] = 0.05 + gap * 2.0 + tgt
    return _normalize(out)

def document_sampling_weight(
    *,
    domain: str,
    language: str,
    score: float,
    synthetic_score: float,
    reasoning_density: float,
    factual_density: float,
    educational_value: float,
    token_spam_score: float,
    context_len: int,
    plan: CorpusMixturePlan,
    stage_name: str,
) -> float:
    st = stage_by_name(stage_name)
    dw = float(plan.domain_weights.get(domain, 0.05))
    lw = float(plan.language_weights.get(language, plan.language_weights.get('other', 0.05)))
    qm = float(plan.quality_multipliers.get('base', 1.0))
    quality = max(0.05, min(1.0, score)) * qm
    quality *= 1.0 + 0.15 * reasoning_density + 0.1 * factual_density + 0.08 * educational_value
    quality *= max(0.2, 1.0 - synthetic_score)
    quality *= max(0.3, 1.0 - token_spam_score)
    if context_len > st.context_max_chars:
        quality *= 0.35
    elif context_len > st.context_max_chars * 0.75:
        quality *= 0.75
    obs_lang = plan.observations.get('language_distribution') or {}
    lang_boost = stage_script_boost(
        language,
        float(obs_lang.get(language, 0.0)),
        plan.language_weights,
        sensitivity=st.multilingual_sensitivity,
    )
    boost = float(st.domain_boost.get(domain, 1.0)) * lang_boost
    if domain == 'reasoning':
        boost *= st.reasoning_boost
    if domain == 'code':
        boost *= st.code_boost
    return max(1e-6, dw * lw * quality * boost)

def build_corpus_mixture_plan(
    gate: QualityGate,
    *,
    cfg: MixtureOrchestrationConfig,
    tokenizer_stats: Optional[dict[str, Any]] = None,
    telemetry: Optional[dict[str, Any]] = None,
) -> CorpusMixturePlan:
    stats = gate.stats.to_dict()
    domain_obs = dict(stats.get('domain_kept') or {})
    lang_obs = dict(stats.get('language_kept') or {})
    stage = stage_by_name(cfg.curriculum_start)
    domain_weights = _inverse_gap_weights(domain_obs, cfg.domain_targets)
    mpol = MultilingualPolicyConfig.from_dict({'script_targets': cfg.language_targets})
    language_weights = gap_weights(lang_obs, mpol.script_targets or cfg.language_targets, floor=mpol.starvation_floor)
    for k, v in stage.domain_boost.items():
        domain_weights[k] = domain_weights.get(k, 0.05) * v
    domain_weights = _normalize(domain_weights)
    language_weights = _normalize(language_weights)
    quality_multipliers = {
        'base': 1.0,
        'reasoning_density': max(0.8, min(1.25, 0.9 + stats.get('reasoning_density_mean', 0.0))),
        'factual_density': max(0.8, min(1.25, 0.9 + stats.get('factual_density_mean', 0.0))),
        'token_efficiency': 1.0,
        'synthetic_penalty': max(0.5, 1.0 - stats.get('synthetic_score_mean', 0.0)),
    }
    if tokenizer_stats:
        tok_global = tokenizer_stats.get('global') or tokenizer_stats
        quality_multipliers.update(
            quality_multipliers_from_telemetry(
                tok_global,
                target_cpt=mpol.target_chars_per_token,
            )
        )
    plan = CorpusMixturePlan(
        curriculum_stage=cfg.curriculum_start,
        replay_seed=int(cfg.replay_seed),
        domain_weights=domain_weights,
        language_weights=language_weights,
        context_stages=[
            {'name': s.name, 'max_chars': s.max_chars, 'min_chars': s.min_chars, 'weight': s.weight}
            for s in cfg.context_stages
        ],
        epoch_schedule=build_epoch_schedule(
            start_stage=cfg.curriculum_start,
            token_budget_per_stage=cfg.token_budget_per_stage,
        ),
        quality_multipliers=quality_multipliers,
        synthetic_cap=float(cfg.synthetic_cap),
        telemetry=dict(telemetry or {}),
        observations={
            'domain_kept': domain_obs,
            'language_kept': lang_obs,
            'quality': stats,
            'domain_distribution': gate.domain_balancer.distribution(),
            'language_distribution': gate.lang_balancer.distribution(),
            'tokenizer_telemetry': stats.get('tokenizer_telemetry') or gate.stats.tokenizer_telemetry.to_dict(),
            'tokenizer_stats': tokenizer_stats or {},
        },
    )
    plan.finalize_digest()
    return plan
