from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indw.extract.sections.classify import KnowledgeSectionClass

@dataclass
class KnowledgePageMetrics:
    original_tokens: int = 0
    knowledge_tokens: int = 0
    navigation_tokens: int = 0
    metadata_tokens: int = 0
    template_tokens: int = 0
    comment_tokens: int = 0
    forum_tokens: int = 0
    advertisement_tokens: int = 0
    sections_total: int = 0
    sections_kept: int = 0
    sections_dropped: int = 0
    extraction_efficiency: float = 0.0
    knowledge_retention: float = 0.0
    noise_removal: float = 0.0
    contamination_ratio: float = 0.0
    mixed: bool = False
    by_class: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'original_tokens': self.original_tokens,
            'knowledge_tokens': self.knowledge_tokens,
            'navigation_tokens': self.navigation_tokens,
            'metadata_tokens': self.metadata_tokens,
            'template_tokens': self.template_tokens,
            'comment_tokens': self.comment_tokens,
            'forum_tokens': self.forum_tokens,
            'advertisement_tokens': self.advertisement_tokens,
            'sections_total': self.sections_total,
            'sections_kept': self.sections_kept,
            'sections_dropped': self.sections_dropped,
            'extraction_efficiency': round(self.extraction_efficiency, 4),
            'knowledge_retention': round(self.knowledge_retention, 4),
            'noise_removal': round(self.noise_removal, 4),
            'contamination_ratio': round(self.contamination_ratio, 4),
            'mixed': self.mixed,
            'by_class': dict(self.by_class),
        }

def _tok(text: str) -> int:
    return max(1, len(text) // 4)

def _bucket_tokens(label: KnowledgeSectionClass, tokens: int, m: KnowledgePageMetrics) -> None:
    key = label.value
    m.by_class[key] = m.by_class.get(key, 0) + tokens
    if label in (
        KnowledgeSectionClass.NAVIGATION,
        KnowledgeSectionClass.FOOTER,
        KnowledgeSectionClass.RELATED,
    ):
        m.navigation_tokens += tokens
    elif label in (KnowledgeSectionClass.METADATA, KnowledgeSectionClass.AUTHOR_BIO):
        m.metadata_tokens += tokens
    elif label in (KnowledgeSectionClass.ADVERTISEMENT,):
        m.advertisement_tokens += tokens
    elif label in (KnowledgeSectionClass.COMMENT,):
        m.comment_tokens += tokens
    elif label in (KnowledgeSectionClass.FORUM, KnowledgeSectionClass.QUESTION, KnowledgeSectionClass.ANSWER):
        m.forum_tokens += tokens
    elif label in (
        KnowledgeSectionClass.ARCHIVE,
        KnowledgeSectionClass.NEWSLETTER,
        KnowledgeSectionClass.EVENT,
    ):
        m.template_tokens += tokens

def compute_page_metrics(
    original_text: str,
    *,
    sections: list[tuple[KnowledgeSectionClass, str, bool]],
    mixed: bool,
) -> KnowledgePageMetrics:
    m = KnowledgePageMetrics()
    m.original_tokens = _tok(original_text)
    m.mixed = mixed
    m.sections_total = len(sections)

    kept_tok = 0
    dropped_noise = 0
    for label, text, kept in sections:
        t = _tok(text)
        _bucket_tokens(label, t, m)
        if kept:
            m.sections_kept += 1
            kept_tok += t
            if label in (
                KnowledgeSectionClass.ARTICLE,
                KnowledgeSectionClass.EDUCATIONAL,
                KnowledgeSectionClass.SCIENTIFIC,
                KnowledgeSectionClass.MEDICAL,
                KnowledgeSectionClass.GOVERNMENT,
                KnowledgeSectionClass.REFERENCE,
                KnowledgeSectionClass.INSPECTION,
                KnowledgeSectionClass.ANSWER,
                KnowledgeSectionClass.QUESTION,
            ):
                m.knowledge_tokens += t
        else:
            m.sections_dropped += 1
            dropped_noise += t

    if m.original_tokens > 0:
        m.extraction_efficiency = kept_tok / m.original_tokens
        m.knowledge_retention = m.knowledge_tokens / m.original_tokens
        m.noise_removal = dropped_noise / m.original_tokens
        m.contamination_ratio = max(0.0, 1.0 - m.knowledge_retention - m.noise_removal)
    return m
