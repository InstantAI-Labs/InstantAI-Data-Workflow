from __future__ import annotations

from pathlib import Path
from typing import Any

_TOKENIZERS_MSG = (
    "This feature requires the optional dependency 'tokenizers'. "
    "Install it with: pip install tokenizers"
)


def tokenizer_class() -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ImportError(_TOKENIZERS_MSG) from exc
    return Tokenizer


def load_tokenizer_file(path: str | Path) -> Any:
    return tokenizer_class().from_file(str(path))
