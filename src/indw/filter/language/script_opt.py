from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indw.filter.language.script_orch import _bounded_delta, _normalize
from indw.filter.language.script_policy import MultilingualPolicyConfig

@dataclass
class MultilingualAdaptationState:
    step: int = 0
    ema_token_inflation: float = 0.0
    ema_fragmentation: float = 0.0
    ema_replay_stability: float = 1.0
    ema_chars_per_token: float = 0.0
    last_adapt_step: int = -1000
    min_steps_between_adapt: int = 50
    ema_alpha: float = 0.15

    def observe(self, telemetry: dict[str, Any]) -> None:
        self.step += 1
        a = self.ema_alpha

        def _ema(cur: float, val: float) -> float:
            return cur * (1.0 - a) + val * a

        self.ema_token_inflation = _ema(
            self.ema_token_inflation,
            float(telemetry.get('token_inflation_mean', 0.0) or 0.0),
        )
        self.ema_fragmentation = _ema(
            self.ema_fragmentation,
            float(telemetry.get('fragmentation_mean', 0.0) or 0.0),
        )
        self.ema_replay_stability = _ema(
            self.ema_replay_stability,
            float(telemetry.get('replay_stability_mean', 1.0) or 1.0),
        )
        self.ema_chars_per_token = _ema(
            self.ema_chars_per_token,
            float(
                telemetry.get('chars_per_token', telemetry.get('chars_per_token_mean', 0.0))
                or 0.0
            ),
        )

    def should_adapt(self) -> bool:
        return (self.step - self.last_adapt_step) >= self.min_steps_between_adapt

    def mark_adapted(self) -> None:
        self.last_adapt_step = self.step

def optimize_multilingual_targets(
    targets: dict[str, float],
    telemetry: dict[str, Any],
    *,
    cfg: MultilingualPolicyConfig,
    state: MultilingualAdaptationState | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    st = state or MultilingualAdaptationState()
    st.observe(telemetry)
    if not st.should_adapt():
        return _normalize(dict(targets)), {'adapted': False, 'step': st.step}

    out = dict(targets)
    max_d = float(cfg.max_adapt_delta)
    per_bucket = telemetry.get('per_bucket') or {}
    script_eval = telemetry.get('script_eval') or {}

    global_cpt = float(
        telemetry.get('chars_per_token_mean', st.ema_chars_per_token) or 0.0
    )
    target_cpt = float(cfg.target_chars_per_token)

    for key, slot in per_bucket.items():
        if not isinstance(slot, dict):
            continue
        bkey = str(key)
        cur = out.get(bkey, cfg.starvation_floor)
        cpt = float(slot.get('chars_per_token', 0.0) or 0.0)
        infl = float(slot.get('token_inflation_mean', 0.0) or 0.0)
        qual = float(
            slot.get('multilingual_quality_mean', script_eval.get(bkey, 0.5)) or 0.5
        )

        if cpt > 0 and cpt < target_cpt * 0.75:
            out[bkey] = _bounded_delta(
                cur, cur * 1.04 + cfg.starvation_floor * 0.5, max_d
            )
        if infl > 0.55:
            out[bkey] = _bounded_delta(cur, cur * 0.97, max_d)
        if qual < 0.35:
            out[bkey] = _bounded_delta(
                cur, cur * 1.06 + cfg.starvation_floor, max_d
            )
        elif qual > 0.88 and cur > cfg.starvation_floor * 2:
            out[bkey] = _bounded_delta(cur, cur * 0.96, max_d)

    if st.ema_fragmentation > 0.5:
        for k in list(out):
            out[k] = _bounded_delta(out[k], out[k] * 0.99, max_d)
    if st.ema_token_inflation > 0.45 and global_cpt > 0:
        for k in list(out):
            out[k] = _bounded_delta(out[k], out[k] * 1.01, max_d)
    if st.ema_replay_stability < 0.92:
        for k in list(out):
            out[k] = _bounded_delta(out[k], out[k] * 0.98, max_d)

    st.mark_adapted()
    return (
        _normalize(out),
        {
            'adapted': True,
            'step': st.step,
            'ema_token_inflation': st.ema_token_inflation,
            'ema_fragmentation': st.ema_fragmentation,
            'ema_replay_stability': st.ema_replay_stability,
            'ema_chars_per_token': st.ema_chars_per_token,
        },
    )

def quality_multipliers_from_telemetry(
    telemetry: dict[str, Any],
    *,
    target_cpt: float = 3.2,
) -> dict[str, float]:
    cpt = float(
        telemetry.get('chars_per_token_mean', telemetry.get('chars_per_token', 0.0))
        or 0.0
    )
    infl = float(telemetry.get('token_inflation_mean', 0.0) or 0.0)
    replay = float(telemetry.get('replay_stability_mean', 1.0) or 1.0)
    frag = float(telemetry.get('fragmentation_mean', 0.0) or 0.0)

    mult: dict[str, float] = {'base': 1.0}
    if cpt > 0:
        mult['token_efficiency'] = max(0.65, min(1.25, cpt / max(target_cpt, 0.5)))
    mult['token_inflation_penalty'] = max(0.5, 1.0 - infl * 0.35)
    mult['replay_stability'] = max(0.7, min(1.15, replay))
    mult['fragmentation_penalty'] = max(0.6, 1.0 - frag * 0.25)
    return mult
