from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

_REASONING_TOK = re.compile(
    r'\b(therefore|because|step|thus|hence|analysis|reasoning)\b',
    re.I,
)
_STRUCT = re.compile(r'[\{\}\[\]<>]|```|"[^"]*"\s*:|</?[\w-]+>')


@dataclass
class TokenizerRuntimeMetrics:
    chars_per_token: float = 0.0
    token_count: int = 0
    token_entropy: float = 0.0
    token_diversity: float = 0.0
    token_inflation: float = 0.0
    repeated_token_span_score: float = 0.0
    delimiter_density: float = 0.0
    reasoning_token_density: float = 0.0
    kv_efficiency_proxy: float = 0.0
    unicode_fragmentation: float = 0.0
    multilingual_entropy: float = 0.0
    structured_output_stability: float = 0.0
    replay_stability: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'chars_per_token': round(self.chars_per_token, 4),
            'token_count': self.token_count,
            'token_entropy': round(self.token_entropy, 4),
            'token_diversity': round(self.token_diversity, 4),
            'token_inflation': round(self.token_inflation, 4),
            'repeated_token_span_score': round(self.repeated_token_span_score, 4),
            'delimiter_density': round(self.delimiter_density, 4),
            'reasoning_token_density': round(self.reasoning_token_density, 4),
            'kv_efficiency_proxy': round(self.kv_efficiency_proxy, 4),
            'unicode_fragmentation': round(self.unicode_fragmentation, 4),
            'multilingual_entropy': round(self.multilingual_entropy, 4),
            'structured_output_stability': round(self.structured_output_stability, 4),
            'replay_stability': round(self.replay_stability, 4),
        }


def _shannon_entropy(ids: list[int]) -> float:
    if not ids:
        return 0.0
    counts = Counter(ids)
    n = len(ids)
    return -sum((c / n * math.log2(c / n) for c in counts.values()))


def _repeated_span_score(ids: list[int]) -> float:
    if len(ids) < 3:
        return 0.0
    max_run = 1
    cur = 1
    for i in range(1, len(ids)):
        if ids[i] == ids[i - 1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return min(1.0, max_run / max(len(ids) * 0.08, 3))


def measure_tokenizer_runtime(
    text: str,
    token_ids: list[int],
    *,
    target_chars_per_token: float = 3.2,
    text_delimiter_density: float = 0.0,
    text_reasoning_density: float = 0.0,
    structural_quality: float = 0.0,
    unicode_instability: float = 0.0,
    script_entropy: float = 0.0,
    replay_stability: float = 1.0,
) -> TokenizerRuntimeMetrics:
    n_tok = len(token_ids)
    if n_tok == 0:
        return TokenizerRuntimeMetrics(replay_stability=replay_stability)
    n_char = max(len(text), 1)
    cpt = n_char / n_tok
    diversity = len(set(token_ids)) / n_tok
    ent = _shannon_entropy(token_ids)
    inflation = max(
        0.0,
        min(
            1.0,
            (target_chars_per_token / max(cpt, 0.5) - 1.0) * 0.35 + (1.0 - diversity) * 0.4,
        ),
    )
    rep = _repeated_span_score(token_ids)
    delim = max(text_delimiter_density, len(_STRUCT.findall(text)) / n_char)
    reasoning = max(text_reasoning_density, len(_REASONING_TOK.findall(text)) / max(n_tok, 1))
    kv_proxy = max(0.0, min(1.0, diversity * (1.0 - min(1.0, n_tok / 16384.0))))
    mlang_ent = max(ent, script_entropy) / max(math.log2(max(n_tok, 2)), 1.0)
    struct_stab = max(
        0.0,
        min(1.0, structural_quality * (1.0 - delim * 0.45) * (1.0 - rep * 0.35)),
    )
    return TokenizerRuntimeMetrics(
        chars_per_token=cpt,
        token_count=n_tok,
        token_entropy=ent,
        token_diversity=diversity,
        token_inflation=inflation,
        repeated_token_span_score=rep,
        delimiter_density=delim,
        reasoning_token_density=reasoning,
        kv_efficiency_proxy=kv_proxy,
        unicode_fragmentation=unicode_instability,
        multilingual_entropy=mlang_ent,
        structured_output_stability=struct_stab,
        replay_stability=max(0.0, min(1.0, replay_stability)),
    )
