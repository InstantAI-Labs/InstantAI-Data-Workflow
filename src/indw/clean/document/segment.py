from __future__ import annotations

import re

from indw.clean.document.config import CleaningConfig
from indw.clean.document.stats import StageStats

_CODE_FENCE = re.compile(r'(```[\s\S]*?```)', re.M)
_HEADING = re.compile(
    r'(?m)^(?:#{1,6}\s+.+|={2,}\s*.+\s*={2,}|Title:|Question:|Answer:|Additional Answer:)\s*$'
)
_MATH_BLOCK = re.compile(r'(?m)^\$\$[\s\S]+?\$\$|^\\begin\{')
_LIST_LINE = re.compile(r'^\s*(?:[-*+•]|\d+[.)])\s+', re.M)


def _token_estimate(text: str, cfg: CleaningConfig) -> int:
    return max(1, int(len(text) / max(cfg.chars_per_token_estimate, 1.0)))


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text, flags=re.UNICODE))


def _is_list_block(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    list_lines = sum(1 for line in lines if _LIST_LINE.match(line))
    return list_lines >= max(2, int(len(lines) * 0.6))


def _is_atomic_block(block: str) -> bool:
    stripped = block.strip()
    if stripped.startswith('```') and stripped.endswith('```'):
        return True
    if _MATH_BLOCK.search(stripped):
        return True
    if _HEADING.match(stripped):
        return True
    if _is_list_block(stripped):
        return True
    return False


def _split_blocks(text: str, *, preserve_code: bool) -> list[str]:
    if not preserve_code or '```' not in text:
        return [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    blocks: list[str] = []
    pos = 0
    for match in _CODE_FENCE.finditer(text):
        if match.start() > pos:
            chunk = text[pos:match.start()]
            blocks.extend(b.strip() for b in re.split(r'\n\s*\n', chunk) if b.strip())
        blocks.append(match.group(1).strip())
        pos = match.end()
    if pos < len(text):
        chunk = text[pos:]
        blocks.extend(b.strip() for b in re.split(r'\n\s*\n', chunk) if b.strip())
    return blocks


def _split_by_sections(blocks: list[str]) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    for block in blocks:
        if _HEADING.match(block) and current:
            sections.append('\n\n'.join(current))
            current = [block]
        else:
            current.append(block)
    if current:
        sections.append('\n\n'.join(current))
    return sections if len(sections) > 1 else blocks


def _pack_sections(sections: list[str], cfg: CleaningConfig) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    min_tok = cfg.min_tokens
    max_tok = cfg.max_tokens

    for section in sections:
        sec_tokens = _token_estimate(section, cfg)
        if _is_atomic_block(section) and sec_tokens <= max_tok:
            if current:
                chunks.append('\n\n'.join(current))
                current = []
                current_tokens = 0
            chunks.append(section)
            continue
        if sec_tokens > max_tok:
            if current:
                chunks.append('\n\n'.join(current))
                current = []
                current_tokens = 0
            paras = [p.strip() for p in re.split(r'\n\s*\n', section) if p.strip()]
            buf: list[str] = []
            buf_tokens = 0
            for para in paras:
                pt = _token_estimate(para, cfg)
                if _is_atomic_block(para):
                    if buf:
                        chunks.append('\n\n'.join(buf))
                        buf = []
                        buf_tokens = 0
                    chunks.append(para)
                    continue
                if buf_tokens + pt > max_tok and buf:
                    chunks.append('\n\n'.join(buf))
                    buf = [para]
                    buf_tokens = pt
                else:
                    buf.append(para)
                    buf_tokens += pt
            if buf:
                chunks.append('\n\n'.join(buf))
            continue
        if current_tokens + sec_tokens > max_tok and current:
            chunks.append('\n\n'.join(current))
            current = [section]
            current_tokens = sec_tokens
            continue
        current.append(section)
        current_tokens += sec_tokens

    if current:
        chunks.append('\n\n'.join(current))

    merged: list[str] = []
    for chunk in chunks:
        if merged and _token_estimate(chunk, cfg) < min_tok // 2:
            prev = merged[-1]
            if _token_estimate(prev + '\n\n' + chunk, cfg) <= max_tok:
                merged[-1] = prev + '\n\n' + chunk
                continue
        merged.append(chunk)
    return merged


def _tail_overlap_text(prev: str, cfg: CleaningConfig) -> str:
    ratio = max(0.0, min(cfg.chunk_overlap_ratio, 0.2))
    if ratio <= 0:
        return ''
    target_tokens = max(1, int(_token_estimate(prev, cfg) * ratio))
    blocks = [block.strip() for block in re.split(r'\n\s*\n', prev) if block.strip()]
    picked: list[str] = []
    acc = 0
    for block in reversed(blocks):
        if _is_atomic_block(block):
            block_tokens = _token_estimate(block, cfg)
            if acc + block_tokens > target_tokens and picked:
                break
            picked.insert(0, block)
            acc += block_tokens
            continue
        sentences = re.split(r'(?<=[.!?])\s+', block)
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = _token_estimate(sentence, cfg)
            if acc + sentence_tokens > target_tokens and picked:
                return '\n\n'.join(picked)
            picked.insert(0, sentence)
            acc += sentence_tokens
            if acc >= target_tokens:
                return '\n\n'.join(picked)
    return '\n\n'.join(picked)


def _apply_overlap(chunks: list[str], cfg: CleaningConfig) -> list[str]:
    if len(chunks) <= 1 or cfg.chunk_overlap_ratio <= 0:
        return chunks
    out = [chunks[0]]
    for chunk in chunks[1:]:
        overlap = _tail_overlap_text(out[-1], cfg)
        if overlap:
            out.append(f'{overlap}\n\n{chunk}')
        else:
            out.append(chunk)
    return out


def segment_text(text: str, cfg: CleaningConfig, *, stats: StageStats | None = None) -> list[str]:
    if not text:
        return []
    if not cfg.split_long_documents:
        return [text] if len(text) <= cfg.hard_max_chars else []

    tokens = _token_estimate(text, cfg)
    if tokens <= cfg.max_tokens and len(text) <= cfg.hard_max_chars:
        if stats is not None:
            stats.in_docs += 1
            stats.out_docs += 1
        return [text]

    blocks = _split_blocks(text, preserve_code=cfg.preserve_code_fences)
    if _HEADING.search(text):
        blocks = _split_by_sections(blocks)
    chunks = _pack_sections(blocks, cfg)

    final: list[str] = []
    for chunk in chunks:
        if len(chunk) > cfg.hard_max_chars:
            sub_blocks = _split_blocks(chunk, preserve_code=cfg.preserve_code_fences)
            final.extend(_pack_sections(sub_blocks, cfg))
        else:
            final.append(chunk)

    final = _apply_overlap(final, cfg)

    if stats is not None:
        stats.in_docs += 1
        stats.out_docs = len(final)
        stats.dropped += max(0, 1 - len(final))
    return final
