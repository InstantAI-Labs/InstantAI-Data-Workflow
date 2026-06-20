from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indw.extract.structure.recovery import RecoveredSection, recover_structure
from indw.extract.structure.aggregate import (
    expand_topic_sections,
    refine_aggregation_sections,
)


@dataclass(frozen=True)
class DocumentStructureSnapshot:
    sections: list[RecoveredSection]
    topic_split: bool


def _build_structure_snapshot(
    text: str,
    *,
    min_section_chars: int,
    pre_forum: bool,
) -> DocumentStructureSnapshot:
    sections = recover_structure(text, min_section_chars=min_section_chars)
    sections = refine_aggregation_sections(text, sections, min_section_chars=min_section_chars)
    topic_split = False
    if not pre_forum:
        pre_count = len(sections)
        sections = expand_topic_sections(text, sections, min_section_chars=min_section_chars)
        topic_split = len(sections) > pre_count
    return DocumentStructureSnapshot(sections=sections, topic_split=topic_split)


def recover_document_structure(
    text: str,
    *,
    min_section_chars: int,
    pre_forum: bool = False,
) -> DocumentStructureSnapshot:
    if not text or not text.strip():
        return DocumentStructureSnapshot(sections=[], topic_split=False)
    cache_key = (min_section_chars, pre_forum)
    try:
        from indw.extract.core.context import get_document_context
        dctx = get_document_context()
        if dctx is not None:
            return dctx.document_structure(
                cache_key,
                lambda: _build_structure_snapshot(
                    text, min_section_chars=min_section_chars, pre_forum=pre_forum,
                ),
            )
    except Exception:
        pass
    return _build_structure_snapshot(
        text, min_section_chars=min_section_chars, pre_forum=pre_forum,
    )


STRUCTURE_OWNERS = (
    'recover_structure',
    'refine_aggregation_sections',
    'expand_topic_sections',
)
