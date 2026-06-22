from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from indw.filter.language.script import ScriptProfile, analyze_script_profile
from indw.filter.language.token_metrics import TokenizerRuntimeMetrics, measure_tokenizer_runtime

@dataclass
class BucketTokenizerStats:
    samples: int = 0
    chars_per_token_sum: float = 0.0
    token_inflation_sum: float = 0.0
    fragmentation_sum: float = 0.0
    replay_stability_sum: float = 0.0
    kv_efficiency_sum: float = 0.0
    reasoning_density_sum: float = 0.0
    multilingual_quality_sum: float = 0.0

    def record(self, metrics: TokenizerRuntimeMetrics, *, profile: ScriptProfile) -> None:
        self.samples += 1
        self.chars_per_token_sum += metrics.chars_per_token
        self.token_inflation_sum += metrics.token_inflation
        self.fragmentation_sum += profile.fragmentation_risk
        self.replay_stability_sum += metrics.replay_stability
        self.kv_efficiency_sum += metrics.kv_efficiency_proxy
        self.reasoning_density_sum += metrics.reasoning_token_density
        self.multilingual_quality_sum += max(
            0.0,
            min(1.0, metrics.kv_efficiency_proxy * (1.0 - metrics.token_inflation)),
        )

    def to_dict(self) -> dict[str, Any]:
        n = max(self.samples, 1)
        return {
            'samples': self.samples,
            'chars_per_token': self.chars_per_token_sum / n,
            'token_inflation_mean': self.token_inflation_sum / n,
            'fragmentation_mean': self.fragmentation_sum / n,
            'replay_stability_mean': self.replay_stability_sum / n,
            'kv_efficiency_mean': self.kv_efficiency_sum / n,
            'reasoning_density_mean': self.reasoning_density_sum / n,
            'multilingual_quality_mean': self.multilingual_quality_sum / n,
        }

@dataclass
class MultilingualTokenizerTelemetry:
    global_stats: BucketTokenizerStats = field(default_factory=BucketTokenizerStats)
    by_bucket: dict[str, BucketTokenizerStats] = field(default_factory=dict)

    def record(
        self,
        text: str,
        token_ids: list[int],
        *,
        bucket: str,
        profile: Optional[ScriptProfile] = None,
        text_delimiter_density: float = 0.0,
        text_reasoning_density: float = 0.0,
        structural_quality: float = 0.0,
        replay_stability: float = 1.0,
        target_chars_per_token: float = 3.2,
    ) -> TokenizerRuntimeMetrics:
        prof = profile or analyze_script_profile(text)
        script_ent = 0.0
        for p in prof.script_fractions.values():
            if p > 0:
                script_ent -= p * math.log2(p)

        metrics = measure_tokenizer_runtime(
            text,
            token_ids,
            target_chars_per_token=target_chars_per_token,
            text_delimiter_density=text_delimiter_density,
            text_reasoning_density=text_reasoning_density,
            structural_quality=structural_quality,
            unicode_instability=prof.unicode_instability,
            script_entropy=script_ent,
            replay_stability=replay_stability,
        )
        self.global_stats.record(metrics, profile=prof)
        slot = self.by_bucket.setdefault(bucket, BucketTokenizerStats())
        slot.record(metrics, profile=prof)
        return metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            'global': self.global_stats.to_dict(),
            'per_bucket': {k: v.to_dict() for k, v in self.by_bucket.items()},
        }

    def merge_serving_summary(self, serving: dict[str, Any]) -> dict[str, Any]:
        out = self.to_dict()
        out['serving'] = serving
        return out

def encode_text_ids(tokenizer: Any, text: str) -> list[int]:
    enc = tokenizer.encode(text)
    ids = getattr(enc, 'ids', None)
    if ids is None:
        raise ValueError('tokenizer.encode() missing ids')
    return [int(x) for x in ids]

def sample_tokenizer_telemetry_by_bucket(
    jsonl_path: Path,
    tokenizer_path: Path,
    *,
    max_samples: int = 500,
    seed: int = 42,
    target_chars_per_token: float = 3.2,
    bucket_map: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    from tokenizers import Tokenizer

    from indw.filter.language.script_policy import MultilingualPolicyConfig

    tok = Tokenizer.from_file(str(tokenizer_path))
    mpol = MultilingualPolicyConfig(locale_bucket_map=bucket_map or {})
    telemetry = MultilingualTokenizerTelemetry()
    rng = random.Random(seed)
    lines: list[str] = []

    with jsonl_path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lines.append(line)

    if not lines:
        return {'samples': 0}

    sample = rng.sample(lines, min(max_samples, len(lines)))
    for line in sample:
        row = json.loads(line)
        text = row.get('text', '')
        if not text:
            continue
        ids = encode_text_ids(tok, text)
        profile = analyze_script_profile(text)
        bucket = profile.bucket_key(mpol.locale_bucket_map or None)
        telemetry.record(
            text,
            ids,
            bucket=bucket,
            profile=profile,
            target_chars_per_token=target_chars_per_token,
        )

    result = telemetry.to_dict()
    result['vocab_size'] = len(tok.get_vocab())
    result['samples'] = telemetry.global_stats.samples
    g = telemetry.global_stats
    if g.samples:
        result['chars_per_token_mean'] = g.chars_per_token_sum / g.samples
        result['token_inflation_mean'] = g.token_inflation_sum / g.samples
        result['fragmentation_mean'] = g.fragmentation_sum / g.samples
        result['replay_stability_mean'] = g.replay_stability_sum / g.samples
    return result
