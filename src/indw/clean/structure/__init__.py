from indw.clean.document.config import CleaningConfig
from indw.clean.document.stats import StageStats
from indw.clean.structure.labeled_qa import (
    extract_labeled_qa,
    has_labeled_qa_markers,
    preprocess_labeled_qa,
)
from indw.clean.structure.reference_sections import (
    clean_reference_sections,
    has_reference_tail_markers,
)


def apply_structural_processors(
    text: str,
    *,
    cfg: CleaningConfig,
    stats: StageStats | None = None,
) -> str:
    if cfg.semantic_cleaning or not text:
        return text
    if has_labeled_qa_markers(text):
        return preprocess_labeled_qa(text, stats=stats, max_extra_answers=cfg.max_extra_answers)
    if has_reference_tail_markers(text):
        return clean_reference_sections(text, stats=stats)
    return text

__all__ = [
    'apply_structural_processors',
    'clean_reference_sections',
    'extract_labeled_qa',
    'has_labeled_qa_markers',
    'has_reference_tail_markers',
    'preprocess_labeled_qa',
]
