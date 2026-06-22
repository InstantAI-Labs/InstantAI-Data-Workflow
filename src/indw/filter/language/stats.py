from __future__ import annotations
import random
from pathlib import Path
from typing import Any, Optional

def sample_tokenizer_efficiency(jsonl_path: Path, tokenizer_path: Path, *, max_samples: int=500, seed: int=42) -> dict[str, Any]:
    from indw.util.hf_tokenizers import load_tokenizer_file
    import json
    tok = load_tokenizer_file(tokenizer_path)
    rng = random.Random(seed)
    ratios: list[float] = []
    frag: list[float] = []
    lines: list[str] = []
    with jsonl_path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lines.append(line)
    if not lines:
        return {'samples': 0}
    sample = rng.sample(lines, min(max_samples, len(lines)))
    for line in sample:
        text = json.loads(line).get('text', '')
        if not text:
            continue
        enc = tok.encode(text)
        n_tok = len(enc.ids)
        if n_tok == 0:
            continue
        ratios.append(len(text) / n_tok)
        frag.append(len(set(enc.ids)) / n_tok)
    if not ratios:
        return {'samples': 0}
    return     {
        'samples': len(ratios),
        'chars_per_token_mean': sum(ratios) / len(ratios),
        'chars_per_token_p10': sorted(ratios)[len(ratios) // 10],
        'token_diversity_mean': sum(frag) / len(frag),
        'vocab_size': len(tok.get_vocab())
    }
