from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PackingConfig:
    enabled: bool = True
    mode: str = 'multi_sample'
    delimiter_token: Optional[int] = None
    pad_token_id: int = 0
    min_document_tokens: int = 32
    max_segments_per_seq: int = 64
    max_tokens_per_batch: Optional[int] = None
    allow_padding_tail: bool = False
    reset_positions: bool = True
    use_packed_attention_mask: bool = True

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> PackingConfig:
        if not raw:
            return cls(enabled=False)
        return cls(
            enabled=bool(raw.get('enabled', True)),
            mode=str(raw.get('mode', 'multi_sample')),
            delimiter_token=raw.get('delimiter_token'),
            pad_token_id=int(raw.get('pad_token_id', 0)),
            min_document_tokens=int(raw.get('min_document_tokens', 32)),
            max_segments_per_seq=int(raw.get('max_segments_per_seq', 64)),
            max_tokens_per_batch=raw.get('max_tokens_per_batch'),
            allow_padding_tail=bool(raw.get('allow_padding_tail', False)),
            reset_positions=bool(raw.get('reset_positions', True)),
            use_packed_attention_mask=bool(raw.get('use_packed_attention_mask', True)),
        )
