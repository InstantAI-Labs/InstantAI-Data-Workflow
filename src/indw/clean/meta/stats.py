from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable
from indw.clean.artifact.registry import line_is_artifact
from indw.clean.document.patterns import _CODE_FENCE, _METADATA_LINE, _UI_LINE, _WORD

@dataclass
class MetadataCleanStats:
    chars_before: int = 0
    chars_after: int = 0
    copyright_blocks_removed: int = 0
    comment_blocks_removed: int = 0
    metadata_lines_removed: int = 0
    boilerplate_lines_removed: int = 0
    repo_metadata_removed: int = 0
    header_metadata_removed: int = 0
    email_forum_removed: int = 0
    front_matter_removed: int = 0
    ai_prompt_lines_removed: int = 0
    instruction_wrappers_removed: int = 0
    cot_lines_removed: int = 0
    paragraphs_deduped: int = 0
    duplicate_lines_removed: int = 0
    format_repairs: int = 0
    license_detected: str = 'Unknown'
    license_confidence: float = 0.0
    license_tokens_removed: int = 0
    license_regions_removed: int = 0
    license_regions_flagged: int = 0

    def merge(self, other: MetadataCleanStats) -> None:
        for fld in self.__dataclass_fields__:
            setattr(self, fld, getattr(self, fld) + getattr(other, fld))

    @property
    def chars_removed(self) -> int:
        return max(0, self.chars_before - self.chars_after)

    @property
    def token_reduction_ratio(self) -> float:
        if self.chars_before <= 0:
            return 0.0
        return self.chars_removed / self.chars_before

    def to_dict(self) -> dict[str, int | float]:
        return {
            'chars_before': self.chars_before,
            'chars_after': self.chars_after,
            'chars_removed': self.chars_removed,
            'copyright_blocks_removed': self.copyright_blocks_removed,
            'comment_blocks_removed': self.comment_blocks_removed,
            'metadata_lines_removed': self.metadata_lines_removed,
            'boilerplate_lines_removed': self.boilerplate_lines_removed,
            'repo_metadata_removed': self.repo_metadata_removed,
            'header_metadata_removed': self.header_metadata_removed,
            'email_forum_removed': self.email_forum_removed,
            'front_matter_removed': self.front_matter_removed,
            'ai_prompt_lines_removed': self.ai_prompt_lines_removed,
            'instruction_wrappers_removed': self.instruction_wrappers_removed,
            'cot_lines_removed': self.cot_lines_removed,
            'paragraphs_deduped': self.paragraphs_deduped,
            'duplicate_lines_removed': self.duplicate_lines_removed,
            'format_repairs': self.format_repairs,
            'token_reduction_ratio': round(self.token_reduction_ratio, 4),
            'license_detected': self.license_detected,
            'license_confidence': round(self.license_confidence, 4),
            'license_tokens_removed': self.license_tokens_removed,
            'license_regions_removed': self.license_regions_removed,
            'license_regions_flagged': self.license_regions_flagged,
        }
