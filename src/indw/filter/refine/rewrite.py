from __future__ import annotations

import re

from indw.clean.meta.foundation import clean_foundation_document, strip_social_promo_prefix
from indw.filter.score.signals import compute_signals
from indw.filter.spec.pipeline import PipelinePolicy, RewritePolicy
from indw.filter.spec.document import CorpusDocument

_QA_TAIL = re.compile(r'(?is)\s*question:\s*.+$')
_ARTIFACT_LINE = re.compile(
    r'(?i)^(?:table\s+of\s+contents|featured\s+mission|editor\'?s\s+note|'
    r'source|tags?|categories|author)\s*:',
)
_PADDING = re.compile(
    r'(?i)(?:\bfor\s+example\b|\bin\s+contrast\b|\bmoreover\b|\bfurthermore\b|'
    r'\badditionally\b|\boverall\b|\bin\s+summary\b|\bit\s+is\s+important\b|'
    r'\bkeep\s+in\s+mind\b)',
)

class RewriteEngine:
    def __init__(self, policy: PipelinePolicy | None = None) -> None:
        if policy is None:
            raise ValueError('RewriteEngine requires PipelinePolicy')
        self.policy = policy

    @property
    def rules(self) -> RewritePolicy:
        return self.policy.rewrite

    def apply(self, doc: CorpusDocument) -> CorpusDocument:
        if not doc.text:
            return doc
        working = doc.text.strip()
        flags = list(doc.flags)
        changed = False

        if self.rules.normalize_qa:
            rw, ok = _rewrite_qa_inline(working)
            if ok:
                working = rw
                changed = True
                flags.append('qa_normalized')

        if self.rules.strip_qa_tail:
            stripped, ch = _strip_qa_tail(working)
            if ch:
                working = stripped
                changed = True
                flags.append('qa_tail_removed')

        if self.rules.strip_artifact_lines:
            cleaned, ch = _strip_artifact_lines(working)
            if ch:
                working = cleaned
                changed = True

        working, _ = strip_social_promo_prefix(working)
        ft, _ = clean_foundation_document(working)
        if ft and ft != working:
            working = ft
            changed = True

        if self.rules.prefer_compression:
            compressed = _compress_whitespace(working)
            if len(compressed) < len(working):
                working = compressed
                changed = True

        sig = compute_signals(working)
        if sig.synthetic_score > self.rules.max_synthetic_score:
            flags.append('synthetic_high')
        if sig.seo_spam_score > self.rules.max_seo_score:
            flags.append('seo_high')
        if _padding_ratio(working) > self.rules.max_padding_ratio:
            flags.append('padding_high')

        return doc.with_text(working, modified=changed).with_flags(tuple(dict.fromkeys(flags)))

def _padding_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 1.0
    hits = sum(len(m.group().split()) for m in _PADDING.finditer(text))
    return hits / len(words)

def _strip_qa_tail(text: str) -> tuple[str, bool]:
    inline = re.match(r'(?is)^(?:question:\s*)?.+?\?\s*answer:\s*(.+)$', text.strip())
    if inline:
        body = inline.group(1).strip()
        if body and body != text.strip():
            return body, True
    m = _QA_TAIL.search(text)
    if m and m.start() > 40:
        return text[:m.start()].strip(), True
    lines: list[str] = []
    changed = False
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r'(?i)^question\s*:', s):
            ans = re.search(r'(?i)answer:\s*(.+)$', s)
            if ans and ans.group(1).strip():
                lines.append(ans.group(1).strip())
                changed = True
                continue
            if re.search(r'(?i)ai assistant|step-by-step', s):
                changed = True
                continue
        m_ans = re.match(r'(?i)^answer\s*:\s*(.+)$', s)
        if m_ans:
            lines.append(m_ans.group(1).strip())
            changed = True
            continue
        lines.append(ln)
    out = '\n'.join(lines).strip()
    return out, changed and out != text.strip()

def _rewrite_qa_inline(text: str) -> tuple[str, bool]:
    m = re.match(r'(?is)^(?:question:\s*)?(.+?\?)\s*answer:\s*(.+)$', text.strip())
    if not m:
        return text, False
    q, a = m.group(1).strip(), m.group(2).strip()
    if re.search(r"(?i)i'?m\s+sorry", a):
        return text, False
    if len(a.split()) < 30:
        return text, False
    topic = re.sub(r'(?i)^what\s+type\s+of\s+', '', q).rstrip('?').strip()
    return f'{topic}: {a}', True

def _strip_artifact_lines(text: str) -> tuple[str, bool]:
    changed = False
    lines: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if _ARTIFACT_LINE.match(s):
            changed = True
            continue
        lines.append(ln)
    out = '\n'.join(lines).strip()
    return out, changed and out != text.strip()

def _compress_whitespace(text: str) -> str:
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return '\n\n'.join(paras)
