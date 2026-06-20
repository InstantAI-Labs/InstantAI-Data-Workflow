from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
_TRUNC_MARKERS = re.compile(
    r'\[(?:truncated|continued|cut|excerpt|…+)\]|(?:\.\.\.|…)\s*$|<\s*/?\s*(?:page|div|article)\s*>',
    re.I,
)
_SENTENCE_END = re.compile(r'[.!?][\)"\'\]}>\s]*')
_DANGLING_END = frozenset({
    'and', 'or', 'because', 'if', 'while', 'the', 'a', 'to', 'of', 'in', 'on',
    'for', 'with', 'as', 'at', 'by', 'from', 'that', 'which', 'but', 'when',
    'where', 'although', 'though', 'since', 'until', 'unless', 'whether',
})
_TRUNC_MARKER = re.compile(
    r'\[(?:truncated|continued|cut|excerpt)\]|(?:\.\.\.|…)\s*$',
    re.I,
)
_MID_WORD_TAIL = re.compile(r'[a-zA-Z]{2,}$')

@dataclass
class TruncationResult:
    probability: float = 0.0
    severity: Literal['none', 'slight', 'heavy'] = 'none'
    trimmed: bool = False
    chars_removed: int = 0
    reason: str = ''

    @property
    def should_remove(self) -> bool:
        return self.severity == 'heavy'

    @property
    def should_trim(self) -> bool:
        return self.severity == 'slight'

def _last_sentence_boundary(text: str) -> int:
    from indw.extract.sections.boundaries import period_ends_sentence

    best = -1
    for m in _SENTENCE_END.finditer(text):
        punct = text[m.start()]
        if punct == '.' and not period_ends_sentence(text, m.start()):
            continue
        best = m.end()
    return best

def base_truncation_signal(text: str) -> float:
    t = text.strip()
    if len(t) < 80:
        return 0.0
    score = 0.0
    if _TRUNC_MARKERS.search(t):
        score = max(score, 0.95)
    if t.endswith('...') or t.endswith('…'):
        score = max(score, 0.85)
    tail = t[-120:]
    words = t.split()
    last_word = words[-1].strip('\'",;)') if words else ''
    ends_clean = bool(re.search(r'[.!?\)"\'\];}>\]]\s*$', tail))
    if not ends_clean:
        if t[-1] in '({[<':
            score = max(score, 0.9)
        elif t[-1] in ',-–—:':
            score = max(score, 0.55)
        elif last_word.isalpha() and len(last_word) <= 2 and len(t) > 250:
            score = max(score, 0.82)
        elif (
            last_word.isalpha()
            and len(last_word) >= 6
            and not re.search(r'[aeiouAEIOU]', last_word)
            and len(t) > 800
        ):
            score = max(score, 0.78)
    broken = t.count('\ufffd')
    if broken > 0:
        score = max(score, min(1.0, broken / max(len(t) / 200, 1)))
    return min(1.0, score)

def _ending_penalty(text: str) -> float:
    from indw.extract.sections.semantic import analyze_completion

    t = text.strip()
    if len(t) < 40:
        return 0.0
    score = 0.0
    if _TRUNC_MARKER.search(t[-80:]):
        score = max(score, 0.90)
    comp = analyze_completion(t)
    score = max(score, comp.incomplete_probability * 0.95)
    words = _WORD.findall(t)
    if not words:
        return score
    last = words[-1].lower().strip('\'",;:)')
    if last in _DANGLING_END:
        score = max(score, 0.78)
    tail = t[-1]
    if tail in '({[<':
        score = max(score, 0.88)
    elif tail in ',-–—:':
        score = max(score, 0.62)
    elif tail.isalpha():
        if not re.search(r'[.!?\)"\'\];}>\]]\s*$', t[-30:]):
            if _MID_WORD_TAIL.search(t) and len(last) >= 4:
                vowels = sum(1 for c in last if c in 'aeiou')
                if vowels == 0 and len(t) > 200:
                    score = max(score, 0.80)
                else:
                    score = max(score, 0.55)
            else:
                score = max(score, 0.50)
    return score

def analyze_truncation(text: str) -> TruncationResult:
    if not text or len(text.strip()) < 30:
        return TruncationResult()
    base = base_truncation_signal(text)
    ending = _ending_penalty(text)
    prob = min(1.0, max(base, ending) * 0.55 + ending * 0.45)
    if _TRUNC_MARKER.search(text[-120:]):
        prob = max(prob, 0.92)
    if prob < 0.30:
        return TruncationResult(probability=prob, severity='none')
    if prob >= 0.72:
        return TruncationResult(
            probability=prob,
            severity='heavy',
            reason='heavily_truncated',
        )
    return TruncationResult(
        probability=prob,
        severity='slight',
        reason='slightly_truncated',
    )

def _apply_truncation_trim(t: str) -> tuple[str, int]:
    original_len = len(t)

    t = re.sub(r'(?:\.\.\.|…)\s*$', '', t).strip()
    t = _TRUNC_MARKER.sub('', t).strip()

    boundary = _last_sentence_boundary(t)
    words = _WORD.findall(t)
    last_word = words[-1].lower().strip('\'",;:)') if words else ''

    if last_word in _DANGLING_END:
        if boundary > max(60, len(t) * 0.20):
            t = t[:boundary].strip()
        else:
            sentences = re.split(r'(?<=[.!?])\s+', t)
            if len(sentences) > 1:
                t = ' '.join(sentences[:-1]).strip()
            elif boundary > 0:
                t = t[:boundary].strip()
    elif boundary > max(80, len(t) * 0.25):
        t = t[:boundary].strip()
    elif last_word in _DANGLING_END:
        paras = [p.strip() for p in t.split('\n\n') if p.strip()]
        if len(paras) > 1:
            t = '\n\n'.join(paras[:-1])
        elif boundary > 0:
            t = t[:boundary].strip()

    return t, max(0, original_len - len(t))

def repair_truncation(text: str) -> tuple[str, TruncationResult]:
    result = analyze_truncation(text)
    if result.severity == 'none':
        return text, result

    t = text.strip()
    trimmed, chars_removed = _apply_truncation_trim(t)

    if chars_removed > 0 and len(trimmed) >= max(40, len(t) * 0.45):
        return trimmed, TruncationResult(
            probability=result.probability,
            severity='slight',
            trimmed=True,
            chars_removed=chars_removed,
            reason='trimmed_incomplete_ending',
        )

    if result.severity == 'heavy':
        return text, result

    return text, result
