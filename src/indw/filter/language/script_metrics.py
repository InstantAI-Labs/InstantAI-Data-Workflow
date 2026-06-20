from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from indw.filter.language.script import ScriptProfile
from tokenizer import TokenizerRuntimeMetrics

@dataclass
class MultilingualMetrics:
    script_entropy: float = 0.0
    fragmentation: float = 0.0
    token_inflation_risk: float = 0.0
    reasoning_stability: float = 0.0
    semantic_diversity: float = 0.0
    structured_output_stability: float = 0.0
    multilingual_quality: float = 0.0
    chars_per_token: float = 0.0
    token_entropy: float = 0.0
    kv_efficiency: float = 0.0
    replay_stability: float = 1.0
    repeated_token_span_score: float = 0.0
    tokenizer_telemetry: Optional[TokenizerRuntimeMetrics] = None
    script_profile: ScriptProfile = field(default_factory=ScriptProfile)

    def to_dict(self) -> dict[str, Any]:
        out = {
            'script_entropy': self.script_entropy,
            'fragmentation': self.fragmentation,
            'token_inflation_risk': self.token_inflation_risk,
            'reasoning_stability': self.reasoning_stability,
            'semantic_diversity': self.semantic_diversity,
            'structured_output_stability': self.structured_output_stability,
            'multilingual_quality': self.multilingual_quality,
            'chars_per_token': self.chars_per_token,
            'token_entropy': self.token_entropy,
            'kv_efficiency': self.kv_efficiency,
            'replay_stability': self.replay_stability,
            'repeated_token_span_score': self.repeated_token_span_score,
            'dominant_script': self.script_profile.dominant_script,
            'mixed_script_score': self.script_profile.mixed_script_score,
        }
        if self.tokenizer_telemetry is not None:
            out['tokenizer_runtime'] = self.tokenizer_telemetry.to_dict()
        return out

def _entropy(fracs: dict[str, float]) -> float:
    h = 0.0
    for p in fracs.values():
        if p > 0:
            h -= p * math.log2(p)
    return h

def compute_multilingual_metrics(
    text: str,
    profile: ScriptProfile,
    *,
    reasoning_density: float = 0.0,
    structural_quality: float = 0.0,
    semantic_diversity: float = 0.0,
    tokenizer_ids: Optional[list[int]] = None,
    tokenizer_runtime: Optional[TokenizerRuntimeMetrics] = None,
    policy_target_cpt: float = 3.2,
) -> MultilingualMetrics:
    ent = _entropy(profile.script_fractions)
    frag = profile.fragmentation_risk
    cpt = 0.0
    token_inflation = 0.0
    tok_ent = 0.0
    kv_eff = 0.0
    replay = 1.0
    rep_span = 0.0
    runtime = tokenizer_runtime

    if runtime is None and tokenizer_ids:
        from tokenizer import measure_tokenizer_runtime

        runtime = measure_tokenizer_runtime(
            text,
            tokenizer_ids,
            target_chars_per_token=policy_target_cpt,
            text_reasoning_density=reasoning_density,
            structural_quality=structural_quality,
            unicode_instability=profile.unicode_instability,
            script_entropy=ent,
        )

    if runtime is not None:
        cpt = runtime.chars_per_token
        token_inflation = runtime.token_inflation
        tok_ent = runtime.token_entropy
        kv_eff = runtime.kv_efficiency_proxy
        replay = runtime.replay_stability
        rep_span = runtime.repeated_token_span_score
        struct_from_tok = runtime.structured_output_stability
    else:
        struct_from_tok = 0.0

    reasoning_stability = max(
        0.0,
        min(
            1.0,
            reasoning_density
            * (1.0 - frag)
            * (1.0 - profile.transliteration_score * 0.35),
        ),
    )
    if runtime is not None:
        reasoning_stability = max(
            reasoning_stability,
            min(1.0, runtime.reasoning_token_density * (1.0 - frag)),
        )

    struct_stab = max(
        0.0,
        min(
            1.0,
            max(structural_quality, struct_from_tok)
            * (1.0 - profile.punctuation_density * 0.5),
        ),
    )

    quality = (
        0.18 * min(1.0, ent / 2.5)
        + 0.16 * (1.0 - frag)
        + 0.16 * reasoning_stability
        + 0.12 * semantic_diversity
        + 0.14 * struct_stab
        + 0.1 * (1.0 - profile.unicode_instability)
        + 0.08 * kv_eff
        + 0.06 * replay
    )
    quality *= max(0.2, 1.0 - token_inflation)
    quality *= max(0.3, 1.0 - profile.mixed_script_score * 0.65)
    quality *= max(0.5, 1.0 - rep_span * 0.4)

    return MultilingualMetrics(
        script_entropy=ent,
        fragmentation=frag,
        token_inflation_risk=token_inflation,
        reasoning_stability=reasoning_stability,
        semantic_diversity=semantic_diversity,
        structured_output_stability=struct_stab,
        multilingual_quality=max(0.0, min(1.0, quality)),
        chars_per_token=cpt,
        token_entropy=tok_ent,
        kv_efficiency=kv_eff,
        replay_stability=replay,
        repeated_token_span_score=rep_span,
        tokenizer_telemetry=runtime,
        script_profile=profile,
    )

def aggregate_script_observations(bucket_counts: dict[str, int]) -> dict[str, float]:
    total = max(sum(bucket_counts.values()), 1)
    return {k: v / total for k, v in bucket_counts.items()}
