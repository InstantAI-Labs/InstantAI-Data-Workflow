from __future__ import annotations

import re

from indw.clean.document.stats import StageStats

_CODE_FENCE = re.compile(r'(```[\s\S]*?```)', re.M)
_MATH_BLOCK = re.compile(r'(\$\$[\s\S]+?\$\$)', re.M)

_FILLER = re.compile(
    r'(?i)\b(?:'
    r'in this (?:article|blog\s+post|guide|section)|'
    r'without further ado|as we all know|'
    r'it goes without saying|needless to say|'
    r'at the end of the day|the fact of the matter is|'
    r'when all is said and done|'
    r'welcome to (?:our|the) (?:blog|website)|'
    r'let(?:\'s| us) (?:dive in|get started|begin)|'
    r'in today\'?s (?:article|post|world)|'
    r'before we (?:begin|start|dive)|'
    r'to summarize|in summary|in conclusion|'
    r'as mentioned (?:above|earlier|previously)|'
    r'as (?:discussed|noted) (?:above|earlier)'
    r')(?:\s+we\s+will\s+(?:discuss|explore|cover|learn about))?[,.]?\s*'
)

_SENSATIONAL = re.compile(
    r'(?i)\b(?:'
    r'absolutely (?:amazing|incredible|stunning)|'
    r'mind[\s-]blowing|jaw[\s-]dropping|'
    r'you won\'?t believe|shocking revelation|'
    r'game[\s-]changer|revolutionary breakthrough|'
    r'never seen before|unprecedented'
    r')\b'
)

_MARKETING = re.compile(
    r'(?i)\b(?:'
    r'world[\s-]class|best[\s-]in[\s-]class|industry[\s-]leading|'
    r'cutting[\s-]edge solution|unlock your potential|'
    r'transform your (?:life|business)|exclusive offer'
    r')\b'
)

_REDUNDANT_SUMMARY = re.compile(
    r'(?im)^(?:summary|conclusion|overview|key takeaways?|in (?:summary|conclusion|closing))'
    r'[:\s].{20,400}$'
)

def _protect_blocks(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}
    idx = 0

    def _stash(match: re.Match[str]) -> str:
        nonlocal idx
        key = f'\x00BLOCK{idx}\x00'
        placeholders[key] = match.group(0)
        idx += 1
        return key

    protected = _CODE_FENCE.sub(_stash, text)
    protected = _MATH_BLOCK.sub(_stash, protected)
    return protected, placeholders

def _restore_blocks(text: str, placeholders: dict[str, str]) -> str:
    out = text
    for key, val in placeholders.items():
        out = out.replace(key, val)
    return out

def _dedupe_paragraph_summaries(text: str) -> str:
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(paras) < 3:
        return text
    seen_openers: set[str] = set()
    kept: list[str] = []
    for para in paras:
        if _REDUNDANT_SUMMARY.match(para):
            opener = para[:80].lower()
            if opener in seen_openers:
                continue
            seen_openers.add(opener)
        words = para.split()
        if len(words) >= 12:
            opener = ' '.join(words[:8]).lower()
            if opener in seen_openers and _FILLER.search(para):
                continue
            seen_openers.add(opener)
        kept.append(para)
    return '\n\n'.join(kept) if kept else text

def compress_content(text: str, *, stats: StageStats | None = None) -> str:
    if not text or not text.strip():
        return text
    original_len = len(text)
    protected, placeholders = _protect_blocks(text)
    out = _FILLER.sub(' ', protected)
    out = _SENSATIONAL.sub(' ', out)
    out = _MARKETING.sub(' ', out)
    out = re.sub(r'[ \t]{2,}', ' ', out)
    out = re.sub(r' *\n *', '\n', out)
    out = _restore_blocks(out, placeholders)
    out = _dedupe_paragraph_summaries(out)
    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    if stats is not None:
        stats.in_docs += 1
        stats.out_docs += 1 if out else 0
        stats.chars_removed += max(0, original_len - len(out))
    return out
