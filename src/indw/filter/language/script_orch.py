from __future__ import annotations

from typing import Any

from indw.filter.language.script_metrics import aggregate_script_observations
from indw.filter.language.script_policy import MultilingualPolicyConfig

def _normalize(weights: dict[str, float]) -> dict[str, float]:
    pos = {k: max(0.0, float(v)) for k, v in weights.items()}
    s = sum(pos.values())
    if s <= 0:
        n = max(len(pos), 1)
        return {k: 1.0 / n for k, v in pos.items()}
    return {k: v / s for k, v in pos.items()}

def _bounded_delta(cur: float, target: float, max_delta: float) -> float:
    d = target - cur
    if d > max_delta:
        return cur + max_delta
    if d < -max_delta:
        return cur - max_delta
    return target

def gap_weights(
    observed: dict[str, int],
    targets: dict[str, float],
    *,
    floor: float = 0.03,
) -> dict[str, float]:
    obs = aggregate_script_observations(observed)
    out: dict[str, float] = {}
    keys = set(targets) | set(observed)
    for k in keys:
        tgt = float(targets.get(k, floor))
        seen = max(obs.get(k, 0.0), 1e-9)
        ratio = tgt / seen
        out[k] = max(floor, tgt * max(1.0, min(ratio, 50.0)))
    return _normalize(out)

def stage_script_boost(
    script_key: str,
    observed_frac: float,
    targets: dict[str, float],
    *,
    sensitivity: float = 1.0,
) -> float:
    tgt = float(targets.get(script_key, 0.05))
    gap = max(0.0, tgt - observed_frac)
    return 1.0 + gap * max(0.0, sensitivity)

def adapt_targets_from_telemetry(
    targets: dict[str, float],
    telemetry: dict[str, Any],
    *,
    cfg: MultilingualPolicyConfig,
) -> dict[str, float]:
    out = dict(targets)
    script_eval = (
        telemetry.get('script_eval')
        or telemetry.get('multilingual_eval')
        or {}
    )
    fragmentation = float(telemetry.get('fragmentation_mean', 0.0) or 0.0)
    token_inflation = float(telemetry.get('token_inflation_mean', 0.0) or 0.0)
    max_d = float(cfg.max_adapt_delta)

    for key, score in script_eval.items():
        s = float(score)
        cur = out.get(str(key), cfg.starvation_floor)
        if s < 0.35:
            out[str(key)] = _bounded_delta(
                cur, cur * 1.1 + cfg.starvation_floor, max_d
            )
        elif s > 0.85 and cur > cfg.starvation_floor * 2:
            out[str(key)] = _bounded_delta(cur, cur * 0.95, max_d)

    if fragmentation > 0.45:
        for k in list(out):
            out[k] = _bounded_delta(out[k], out[k] * 0.98, max_d)
    if token_inflation > 0.5:
        for k in list(out):
            out[k] = _bounded_delta(out[k], out[k] * 1.02, max_d)

    return _normalize(out)
