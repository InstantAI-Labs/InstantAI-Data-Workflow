from __future__ import annotations

from dataclasses import dataclass

@dataclass
class SemanticSelectionConfig:
    enabled: bool = True
    section_mode: bool = False
