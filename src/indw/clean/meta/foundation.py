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
from indw.clean.meta.clean import (
    _dedupe_consecutive_lines,
    _dedupe_duplicate_headers,
    clean_pretraining_metadata,
)
from indw.clean.meta.strip import (
    _is_forum_metadata_line,
    _protect_code_fences,
    _restore_code_fences,
    apply_discovery_line_drops,
)

def _normalize_instruction_label(raw: str) -> str:
    key = raw.strip().lower().rstrip('.:')
    if key in ('q', 'question'):
        return 'question'
    if key in ('a', 'answer', 'response', 'output'):
        return 'answer'
    if key.startswith('choose') or key.startswith('select'):
        return 'choose'
    if key in ('explanation', 'justification', 'rationale'):
        return 'justification'
    return key

def _is_instruction_scaffold_text(text: str) -> bool:
    if not text or not text.strip():
        return True
    sample = text.strip()
    if _AI_PROMPT_LINE.match(sample):
        return True
    if _INSTRUCTION_SCAFFOLD.search(sample) and len(sample.split()) < 45:
        return True
    if _AI_TRAINING_BODY.search(sample) and len(sample.split()) < 60:
        return True
    low = sample.lower()
    if low in {
        'summarize this article.', 'summarize the following.', 'summarize the following article.',
        'provide a detailed answer.', 'provide a long answer.',
    }:
        return True
    return False

def _strip_cot_from_text(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    if not lines:
        return text, 0
    kept: list[str] = []
    removed = 0
    skip_until_blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            skip_until_blank = False
            kept.append(line)
            continue
        if _COT_MARKERS.search(stripped) and len(stripped) < 140:
            removed += 1
            skip_until_blank = True
            continue
        if skip_until_blank:
            removed += 1
            continue
        if _is_instruction_scaffold_text(stripped) and len(stripped) < 120:
            removed += 1
            continue
        kept.append(line)
    out = '\n'.join(kept)
    paras: list[str] = []
    for para in re.split(r'\n\s*\n', out):
        p = para.strip()
        if not p:
            continue
        inline = re.match(
            r'(?is)^(?:'
            r'let\'?s\s+think(?:\s+step\s+by\s+step)?[^.!?]*[.!?]\s*|'
            r'think\s+step[\s-]?by[\s-]?step[.!?]?\s*'
            r')+',
            p,
        )
        if inline:
            p = p[inline.end():].strip()
            removed += 1
        if p:
            paras.append(p)
    return '\n\n'.join(paras), removed

def _extract_conclusion_text(text: str) -> str:
    cleaned, _ = _strip_cot_from_text(text)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', cleaned) if p.strip()]
    if not paragraphs:
        return ''
    words = len(' '.join(paragraphs).split())
    if words > 120 and len(paragraphs) > 1:
        return paragraphs[-1]
    return '\n\n'.join(paragraphs)

def _split_instruction_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_label = 'body'
    current_lines: list[str] = []

    for line in text.splitlines():
        m = _INSTRUCTION_LABEL.match(line)
        if m:
            if current_lines or current_label != 'body':
                sections.append((current_label, '\n'.join(current_lines).strip()))
            current_label = _normalize_instruction_label(m.group('label'))
            tail = (m.group('tail') or '').strip()
            current_lines = [tail] if tail else []
            continue
        current_lines.append(line)

    if current_lines or current_label != 'body':
        sections.append((current_label, '\n'.join(current_lines).strip()))
    return sections

def _question_knowledge_parts(question: str) -> list[str]:
    parts: list[str] = []
    for para in re.split(r'\n\s*\n', question):
        para = para.strip()
        if not para:
            continue
        if _is_instruction_scaffold_text(para):
            continue
        parts.append(para)
    return parts

def _is_short_factual_question(text: str) -> bool:
    words = text.split()
    if len(words) >= 30:
        return False
    low = text.strip().lower()
    return (
        '?' in text
        or low.startswith(('what ', 'why ', 'how ', 'when ', 'where ', 'who ', 'which '))
    )

def _assemble_foundation_sections(sections: list[tuple[str, str]]) -> tuple[str, int, int]:
    parts: list[str] = []
    removed = 0
    cot_removed = 0
    i = 0
    while i < len(sections):
        label, content = sections[i]
        if label == 'choose' or (label in _META_LABELS and _is_instruction_scaffold_text(content)):
            removed += 1
            i += 1
            continue

        if label == 'question':
            q_parts = _question_knowledge_parts(content)
            if i + 1 < len(sections) and sections[i + 1][0] == 'answer':
                ans, ans_cot = _strip_cot_from_text(sections[i + 1][1])
                cot_removed += ans_cot
                removed += 1
                include_question = bool(
                    q_parts
                    and any(
                        len(p.split()) >= 35 or not _is_short_factual_question(p)
                        for p in q_parts
                    )
                )
                if include_question:
                    parts.extend(q_parts)
                if ans.strip():
                    parts.append(ans.strip())
                i += 2
                continue
            if q_parts and not _is_instruction_scaffold_text(content):
                parts.extend(q_parts)
            removed += 1
            i += 1
            continue

        if label == 'answer':
            cleaned, ans_cot = _strip_cot_from_text(content)
            cot_removed += ans_cot
            if cleaned.strip():
                parts.append(cleaned.strip())
            i += 1
            continue

        if label in ('justification', 'explanation', 'rationale'):
            stripped, just_cot = _strip_cot_from_text(content)
            cot_removed += just_cot
            conclusion = _extract_conclusion_text(stripped)
            removed += 1
            if conclusion:
                parts.append(conclusion)
            i += 1
            continue

        if label in _META_LABELS:
            sub_parts = _question_knowledge_parts(content)
            removed += 1
            if sub_parts:
                parts.extend(sub_parts)
            i += 1
            continue

        if label in _KNOWLEDGE_LABELS or label == 'body':
            cleaned, body_cot = _strip_cot_from_text(content)
            cot_removed += body_cot
            if cleaned.strip():
                parts.append(cleaned.strip())
        i += 1

    return '\n\n'.join(p for p in parts if p.strip()), removed, cot_removed

def instruction_wrapper_density(
    text: str,
    *,
    lines: list[str] | None = None,
    word_count: int | None = None,
) -> float:
    if not text or not text.strip():
        return 1.0
    if lines is None:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    wrapper = 0
    for ln in lines:
        m = _INSTRUCTION_LABEL.match(ln)
        if m and not (m.group('tail') or '').strip():
            wrapper += 1
        elif _AI_PROMPT_LINE.match(ln) or _is_instruction_scaffold_text(ln):
            wrapper += 1
        elif _COT_MARKERS.search(ln) and len(ln) < 140:
            wrapper += 1
    line_ratio = wrapper / len(lines)
    body_hits = len(_INSTRUCTION_SCAFFOLD.findall(text)) + len(_AI_TRAINING_BODY.findall(text))
    word_count = max(word_count if word_count is not None else len(_WORD.findall(text)), 1)
    body_ratio = min(1.0, body_hits / max(word_count / 50, 1))
    return min(1.0, line_ratio * 0.65 + body_ratio * 0.35)

def precurate_for_gate(text: str) -> tuple[str, int, int]:
    return unwrap_foundation_instructions(text)

def unwrap_foundation_instructions(text: str) -> tuple[str, int, int]:
    if not text or not text.strip():
        return text, 0, 0

    protected, placeholders = _protect_code_fences(text)
    sections = _split_instruction_sections(protected)
    has_labels = any(label != 'body' for label, _ in sections)
    removed = 0
    cot_removed = 0

    if has_labels:
        assembled, removed, cot_removed = _assemble_foundation_sections(sections)
    else:
        assembled, cot_removed = _strip_cot_from_text(protected)

    kept_lines: list[str] = []
    for line in assembled.splitlines():
        m = _INSTRUCTION_LABEL.match(line)
        if m and not (m.group('tail') or '').strip():
            removed += 1
            continue
        if _AI_PROMPT_LINE.match(line.strip()):
            removed += 1
            continue
        if _is_instruction_scaffold_text(line) and len(line.strip()) < 120:
            removed += 1
            continue
        kept_lines.append(line)

    out = _restore_code_fences('\n'.join(kept_lines), placeholders)
    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    return out, removed, cot_removed

def _strip_encyclopedia_chrome_lead(text: str) -> tuple[str, int]:
    if not text or not _ENCYCLOPEDIA_CHROME.search(text[:1200]):
        return text, 0
    lines = text.splitlines()
    idx = 0
    removed = 0
    while idx < min(len(lines), 12):
        ln = lines[idx].strip()
        if not ln:
            idx += 1
            continue
        if (
            _ENCYCLOPEDIA_CHROME.search(ln)
            or _HEADER_LINE.match(ln)
            or re.match(r'(?i)^#+\s*$', ln)
        ):
            removed += 1
            idx += 1
            continue
        break
    if removed == 0:
        return text, 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return '\n'.join(lines[idx:]), removed

def strip_social_promo_prefix(text: str) -> tuple[str, bool]:
    if not text:
        return text, False
    working, enc_removed = _strip_encyclopedia_chrome_lead(text)
    if enc_removed:
        return working, True
    if not _SOCIAL_PROMO_PREFIX.search(text[:400]):
        return text, False
    parts = re.split(r'(?=\n\s*#+\s+|\s#+\s+|\n\s*\$\$|\\begin\{)', text, maxsplit=1)
    if len(parts) == 2 and _SOCIAL_PROMO_PREFIX.search(parts[0]):
        cleaned = parts[1].lstrip()
        if len(cleaned) >= 80:
            return cleaned, True
    m = re.match(
        r'(?is)^\s*(?:'
        r'(?:our\s+discord|join\s+(?:our\s+)?discord|meet\s+students|network\s+with\s+us|'
        r'join\s+us\s+on\s+facebook|get\s+the\s+latest\s+news)'
        r'[^#]{0,400}?'
        r'(?:join\s+here!?|!|\.)\s*'
        r')',
        text,
    )
    if m and len(text) - m.end() >= 100:
        return text[m.end():].lstrip(), True
    return text, False

def _strip_social_promo_lead(text: str) -> tuple[str, int]:
    cleaned, changed = strip_social_promo_prefix(text)
    if changed:
        return cleaned, 1
    lines = text.splitlines()
    if not lines:
        return text, 0
    removed = 0
    idx = 0
    promo_line = re.compile(
        r'(?i)^\s*(?:'
        r'our\s+discord|join\s+(?:our\s+)?discord|discord\s+hit|'
        r'join\s+here|meet\s+students|network\s+with\s+us|'
        r'join\s+us\s+on\s+facebook|subscribe|sign\s+up'
        r')\b.*$'
    )
    while idx < min(len(lines), 8):
        if promo_line.match(lines[idx].strip()):
            removed += 1
            idx += 1
            continue
        break
    if removed == 0:
        return text, 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return '\n'.join(lines[idx:]), removed

def clean_foundation_document(
    text: str,
    *,
    preserve_code_fences: bool = True,
    strip_code_comments: bool = True,
    repair_formatting: bool = True,
    dedupe_paragraphs: bool = True,
    clean_html_tags: bool = True,
    discovery_trim: bool = False,
    discovery_shadow: bool = True,
    discovery_engine: Any | None = None,
) -> tuple[str, MetadataCleanStats]:
    stats = MetadataCleanStats(chars_before=len(text))
    if not text or not text.strip():
        stats.chars_after = 0
        return '', stats

    working = text.strip()
    if clean_html_tags and '<' in working:
        from indw.clean.document.html import clean_html
        working = clean_html(working)

    working, promo_removed = _strip_social_promo_lead(working)
    stats.boilerplate_lines_removed += promo_removed

    working, instr_removed, cot_removed = unwrap_foundation_instructions(working)
    stats.instruction_wrappers_removed += instr_removed
    stats.cot_lines_removed += cot_removed

    if repair_formatting:
        from indw.clean.document.normalize import normalize_text, repair_formatting as _repair
        working, repairs = _repair(working, preserve_code_fences=preserve_code_fences)
        stats.format_repairs += repairs
        working = normalize_text(working, preserve_code_fences=preserve_code_fences, repair=False)

    working, meta_stats = clean_pretraining_metadata(
        working,
        preserve_code_fences=preserve_code_fences,
        strip_code_comments=strip_code_comments,
    )
    stats.merge(meta_stats)
    stats.chars_before = len(text)

    if discovery_trim and not discovery_shadow:
        working, disc_removed = apply_discovery_line_drops(
            working,
            trim=True,
            shadow=False,
            discovery_engine=discovery_engine,
        )
        stats.metadata_lines_removed += disc_removed

    working, dup_lines = _dedupe_consecutive_lines(working)
    stats.duplicate_lines_removed += dup_lines
    working, dup_headers = _dedupe_duplicate_headers(working)
    stats.duplicate_lines_removed += dup_headers

    if dedupe_paragraphs:
        from indw.clean.document.dedup import dedupe_paragraphs as _dedupe_paragraphs
        before_blocks = len([b for b in re.split(r'\n\s*\n', working) if b.strip()])
        working, _ = _dedupe_paragraphs(working)
        after_blocks = len([b for b in re.split(r'\n\s*\n', working) if b.strip()])
        stats.paragraphs_deduped += max(0, before_blocks - after_blocks)

    working = working.strip()
    stats.chars_after = len(working)
    return working, stats

def metadata_noise_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    patterns = (
        _COPYRIGHT_LINE,
        _EDITORIAL_LINE,
        _BOILERPLATE_LINE,
        _REPO_LINE,
        _HEADER_LINE,
        _EMAIL_LINE,
        _FORUM_LINE,
        _FORUM_STATS_LINE,
        _AI_PROMPT_LINE,
        _UI_LINE,
        _METADATA_LINE,
    )
    hits = sum(
        1 for ln in lines
        if any(p.match(ln) for p in patterns)
        or _is_forum_metadata_line(ln)
        or _FORUM_BULLET_COMMENT.search(ln)
        or _FORUM_QUOTE_PREFIX.search(ln)
    )
    return hits / len(lines)

def is_metadata_only_document(text: str, *, min_words: int = 30) -> bool:
    from indw.clean.document.adaptive import metadata_only_drop

    noise = metadata_noise_ratio(text)
    return metadata_only_drop(text, noise, min_words=min_words)
