from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from indw.filter.language.script import ScriptProfile, analyze_script_profile
from indw.filter.language.telemetry import encode_text_ids
from indw.filter.language.token_metrics import TokenizerRuntimeMetrics, measure_tokenizer_runtime

@dataclass
class LiveTokenizerEncoder:
    tokenizer_path: str
    target_chars_per_token: float = 3.2
    _tokenizer: Any = field(default=None, repr=False, compare=False)

    def _load(self) -> Any:
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(str(Path(self.tokenizer_path)))
        return self._tokenizer

    def encode_metrics(
        self,
        text: str,
        *,
        profile: Optional[ScriptProfile] = None,
        text_delimiter_density: float = 0.0,
        text_reasoning_density: float = 0.0,
        structural_quality: float = 0.0,
        replay_stability: float = 1.0,
    ) -> tuple[list[int], TokenizerRuntimeMetrics]:
        tok = self._load()
        ids = encode_text_ids(tok, text)
        prof = profile or analyze_script_profile(text)
        import math

        script_ent = 0.0
        for p in prof.script_fractions.values():
            if p > 0:
                script_ent -= p * math.log2(p)
        metrics = measure_tokenizer_runtime(
            text,
            ids,
            target_chars_per_token=self.target_chars_per_token,
            text_delimiter_density=text_delimiter_density,
            text_reasoning_density=text_reasoning_density,
            structural_quality=structural_quality,
            unicode_instability=prof.unicode_instability,
            script_entropy=script_ent,
            replay_stability=replay_stability,
        )
        return ids, metrics
