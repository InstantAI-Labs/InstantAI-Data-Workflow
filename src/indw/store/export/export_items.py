from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ExportRecord:
    text: str
    split_key: str
    is_replay: bool = False
