from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable
from indw.clean.artifact.registry import line_is_artifact
from indw.clean.document.patterns import _CODE_FENCE, _METADATA_LINE, _UI_LINE, _WORD
from indw.clean.meta.patterns import (
    _ADA_COMMENT_META,
    _ADA_LICENSE_BLOCK,
    _AI_PROMPT_LINE,
    _AI_TRAINING_BODY,
    _BOILERPLATE_LINE,
    _CODE_AUTHOR_META,
    _CODE_COPYRIGHT,
    _CODE_FENCE,
    _CODE_GENERATED,
    _CODE_LICENSE_BLOCK,
    _COLLAPSED_CODE_START,
    _COPYRIGHT_BLOCK,
    _COPYRIGHT_LINE,
    _COT_MARKERS,
    _EDITORIAL_LINE,
    _EMAIL_LINE,
    _ENCYCLOPEDIA_CHROME,
    _FORUM_ANSWER_TRANSITION,
    _FORUM_BULLET_COMMENT,
    _FORUM_LINE,
    _FORUM_QUOTE_PREFIX,
    _FORUM_STATS_LINE,
    _FORUM_TIMESTAMP,
    _FORUM_USERNAME_GREETING,
    _FRONT_MATTER,
    _HEADER_LINE,
    _INFORMATIVE_COMMENT,
    _INLINE_LICENSE_CHUNK,
    _INSTRUCTION_LABEL,
    _INSTRUCTION_SCAFFOLD,
    _KNOWLEDGE_LABELS,
    _LEGAL_FOOTER,
    _LICENSE_LINE,
    _METADATA_LINE,
    _META_LABELS,
    _README_EMPTY_HEADER,
    _REPO_LINE,
    _SOCIAL_PROMO_PREFIX,
    _TOC_ONLY_LINE,
    _TRAILING_FORUM_BLOCK,
    _UI_LINE,
    _VENDOR_NOTICE_LINE,
    _WIKI_NAV_INLINE,
    _WORD,
)
from indw.clean.meta.stats import MetadataCleanStats
from indw.clean.meta.strip import (
    _clean_code_fences,
    _drop_forum_metadata_lines,
    _drop_matching_lines,
    _protect_code_fences,
    _restore_code_fences,
    _strip_forum_inline_artifacts,
    _strip_forum_quote_blocks,
    _strip_inline_license_runs,
    _strip_leading_license_preamble,
    _strip_trailing_forum_footer,
    _strip_trailing_license_footer,
    apply_discovery_line_drops,
    clean_code_comments,
    forum_contamination_hits,
)

def clean_pretraining_metadata(
    text: str,
    *,
    preserve_code_fences: bool = True,
    strip_code_comments: bool = True,
    strip_license_blocks: bool = True,
    strip_copyright_lines: bool = True,
    license_remove_confidence: float = 0.82,
    license_review_confidence: float = 0.55,
    license_validate_syntax: bool = True,
) -> tuple[str, MetadataCleanStats]:
    stats = MetadataCleanStats(chars_before=len(text))
    if not text or not text.strip():
        stats.chars_after = 0
        return '', stats

    out = text.strip()

    if _FRONT_MATTER.match(out):
        out = _FRONT_MATTER.sub('', out, count=1).lstrip()
        stats.front_matter_removed += 1

    if preserve_code_fences:
        protected, placeholders = _protect_code_fences(out)
    else:
        protected, placeholders = out, {}

    if strip_license_blocks:
        from indw.clean.document.license import clean_license_content

        protected, lic_stats = clean_license_content(
            protected,
            enabled=True,
            remove_confidence=license_remove_confidence,
            review_confidence=license_review_confidence,
            validate_syntax=license_validate_syntax,
        )
        stats.copyright_blocks_removed += lic_stats.regions_removed
        stats.license_detected = lic_stats.license_detected
        stats.license_confidence = lic_stats.license_confidence
        stats.license_tokens_removed = lic_stats.tokens_removed
        stats.license_regions_removed = lic_stats.regions_removed
        stats.license_regions_flagged = lic_stats.regions_flagged_review

        ada_blocks = len(_ADA_LICENSE_BLOCK.findall(protected))
        if ada_blocks:
            protected = _ADA_LICENSE_BLOCK.sub('\n', protected, count=ada_blocks)
            stats.copyright_blocks_removed += ada_blocks

        protected, inline_removed = _strip_inline_license_runs(protected)
        stats.copyright_blocks_removed += inline_removed
    protected, forum_tail = _strip_trailing_forum_footer(protected)
    stats.email_forum_removed += forum_tail
    protected, forum_inline = _strip_forum_inline_artifacts(protected)
    stats.email_forum_removed += forum_inline

    if strip_copyright_lines:
        protected = _drop_matching_lines(
            protected,
            [_COPYRIGHT_LINE, _LEGAL_FOOTER, _ADA_COMMENT_META, _VENDOR_NOTICE_LINE],
            on_match=lambda: setattr(stats, 'copyright_blocks_removed', stats.copyright_blocks_removed + 1),
        )
    protected = _drop_matching_lines(
        protected,
        [_EDITORIAL_LINE, _METADATA_LINE],
        on_match=lambda: setattr(stats, 'metadata_lines_removed', stats.metadata_lines_removed + 1),
    )
    protected = _drop_matching_lines(
        protected,
        [_BOILERPLATE_LINE, _UI_LINE],
        on_match=lambda: setattr(stats, 'boilerplate_lines_removed', stats.boilerplate_lines_removed + 1),
    )
    protected = _drop_matching_lines(
        protected,
        [_REPO_LINE, _README_EMPTY_HEADER],
        on_match=lambda: setattr(stats, 'repo_metadata_removed', stats.repo_metadata_removed + 1),
    )
    protected = _drop_matching_lines(
        protected,
        [_HEADER_LINE, _TOC_ONLY_LINE],
        on_match=lambda: setattr(stats, 'header_metadata_removed', stats.header_metadata_removed + 1),
    )
    protected = _drop_forum_metadata_lines(protected, stats)
    protected = _drop_matching_lines(
        protected,
        [_EMAIL_LINE, _FORUM_LINE, _FORUM_STATS_LINE],
        on_match=lambda: setattr(stats, 'email_forum_removed', stats.email_forum_removed + 1),
    )
    protected = _drop_matching_lines(
        protected,
        [_AI_PROMPT_LINE],
        on_match=lambda: setattr(stats, 'ai_prompt_lines_removed', stats.ai_prompt_lines_removed + 1),
    )

    protected = _LEGAL_FOOTER.sub('', protected)

    out = _restore_code_fences(protected, placeholders)

    if strip_code_comments:
        out = _clean_code_fences(out, stats)

    out = re.sub(r'\n{3,}', '\n\n', out)
    out = re.sub(r'[ \t]+\n', '\n', out)
    out = out.strip()
    stats.chars_after = len(out)
    return out, stats

def _dedupe_consecutive_lines(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    if len(lines) < 2:
        return text, 0
    kept: list[str] = []
    removed = 0
    prev_norm = ''
    for line in lines:
        norm = re.sub(r'\s+', ' ', line.strip().lower())
        if norm and len(norm) > 24 and norm == prev_norm:
            removed += 1
            continue
        kept.append(line)
        if norm:
            prev_norm = norm
    return '\n'.join(kept), removed

def _dedupe_duplicate_headers(text: str) -> tuple[str, int]:
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        m = re.match(r'^\s{0,3}(#{1,6})\s+(.+?)\s*$', line)
        if m:
            key = re.sub(r'\s+', ' ', m.group(2).strip().lower())
            if key in seen and len(key) > 8:
                removed += 1
                continue
            seen.add(key)
        kept.append(line)
    return '\n'.join(kept), removed

def synthetic_prompt_hits(text: str) -> int:
    if not text:
        return 0
    return len(_AI_PROMPT_LINE.findall(text)) + len(_AI_TRAINING_BODY.findall(text))

def social_promo_hits(text: str) -> int:
    if not text:
        return 0
    sample = text[:2000]
    return len(_SOCIAL_PROMO_PREFIX.findall(sample))

def legal_boilerplate_hits(text: str) -> int:
    if not text:
        return 0
    return len(_LEGAL_FOOTER.findall(text)) + len(_LICENSE_LINE.findall(text))
