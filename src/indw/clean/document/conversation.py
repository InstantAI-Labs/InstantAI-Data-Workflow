from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from indw.clean.document.patterns import _QA_ANSWER, _QA_QUESTION, _THREAD_MARKER
from indw.clean.document.stats import StageStats

_COMMENT_BLOCK = re.compile(
    r'(?im)^\s*(?:comment|reply)\s*:\s*.*?(?=^\s*(?:comment|reply|answer|question)\s*:|$)',
    re.S,
)
_ANSWER_BLOCK = re.compile(r'(?im)^\s*(?:answer|accepted\s+answer|best\s+answer|a)\s*:\s*(.+?)(?=^\s*(?:answer|comment|reply|question)\s*:|$)', re.S)
_REPLY_HEADER = re.compile(
    r'(?i)^\s*(?:'
    r'(?:reply|nested\s+reply)\s+by\s+\w+|'
    r'posted\s+by\b|signature\s*:'
    r')',
)
_ACCEPTED_INLINE = re.compile(r'(?i)accepted\s+answer')
_TECH_ANSWER = re.compile(
    r'(?:'
    r'[A-Z]{2,}_[A-Z0-9_]{2,}|'
    r'(?:fp16|bf16|int8|checkpointing|backpropagation|cross-validation)|'
    r'expandable_segments|PYTORCH_|CUDA_'
    r')',
    re.I,
)

@dataclass
class QAPair:
    question: str
    answers: list[str]
    format: str = 'text'

    def to_text(self) -> str:
        parts = [f'Question: {self.question.strip()}']
        if self.answers:
            parts.append('')
            parts.append(f'Answer: {self.answers[0].strip()}')
            for extra in self.answers[1:]:
                parts.append('')
                parts.append(f'Additional Answer: {extra.strip()}')
        return '\n'.join(parts)

    def to_messages(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [{'role': 'user', 'content': self.question.strip()}]
        for ans in self.answers:
            msgs.append({'role': 'assistant', 'content': ans.strip()})
        return msgs

def _score_answer(text: str) -> float:
    words = text.split()
    tech = bool(_TECH_ANSWER.search(text)) or bool(re.search(r'[a-z]+_[a-z]+', text))
    min_words = 5 if tech else 8
    if len(words) < min_words:
        return 0.18 if tech and len(words) >= 3 else 0.0
    score = min(1.0, len(words) / 120.0)
    if tech:
        score += 0.22
    if '```' in text or '    ' in text:
        score += 0.25
    if re.search(r'\$\$?[^$]+\$?\$?|\\begin\{|\\frac', text):
        score += 0.2
    return min(1.0, score)

def _extract_from_row(row: dict[str, Any], max_extra: int) -> Optional[QAPair]:
    question = (row.get('question') or row.get('title') or row.get('prompt') or '').strip()
    if not question and row.get('messages'):
        messages = row.get('messages') or []
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                question = (first.get('content') or first.get('text') or '').strip()
    answers_raw = row.get('answers') or row.get('answer') or row.get('response') or []
    answers: list[str] = []
    if isinstance(answers_raw, str):
        answers = [answers_raw.strip()] if answers_raw.strip() else []
    elif isinstance(answers_raw, list):
        for item in answers_raw:
            if isinstance(item, str) and item.strip():
                answers.append(item.strip())
            elif isinstance(item, dict):
                body = (item.get('text') or item.get('answer') or item.get('content') or '').strip()
                if body:
                    answers.append(body)
    if not question:
        return None
    ranked = sorted(answers, key=_score_answer, reverse=True)
    keep = ranked[: 1 + max(0, max_extra)] if ranked else []
    return QAPair(question=question, answers=keep)

def _extract_reddit_style(text: str) -> Optional[QAPair]:
    from indw.clean.document.patterns import _METADATA_LINE, _UI_LINE

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    question = ''
    for ln in lines:
        if _UI_LINE.search(ln) or _METADATA_LINE.search(ln):
            continue
        if '?' in ln and len(ln) > 16 and not _THREAD_MARKER.match(ln):
            question = ln
            break
    answer_match = re.search(
        r'(?is)accepted\s+answer\s*:\s*(.+?)(?=\n\s*(?:comment|reply|answer)\s*:|$)',
        text,
    )
    if question and answer_match:
        answer = answer_match.group(1).strip()
        if answer:
            return QAPair(question=question, answers=[answer])
    block_match = re.search(
        r'(?is)(?:accepted\s+answer|best\s+answer)(?:\s*:\s*|\s*$)\s*\n(.+?)(?=\n\s*(?:nested\s+)?reply\b|\n\s*user\d|\n\s*signature\b|$)',
        text,
    )
    if question and block_match:
        answer = block_match.group(1).strip()
        if answer and _score_answer(answer) > 0:
            return QAPair(question=question, answers=[answer])
    return None

def _extract_from_text(text: str, max_extra: int) -> Optional[QAPair]:
    from indw.clean.document.patterns import _METADATA_LINE, _UI_LINE

    lines = text.splitlines()
    question = ''
    answers: list[str] = []
    current_answer: list[str] = []
    mode = ''
    for line in lines:
        qm = _QA_QUESTION.match(line)
        am = _QA_ANSWER.match(line)
        if qm:
            if current_answer and mode == 'answer':
                answers.append('\n'.join(current_answer).strip())
                current_answer = []
            question = qm.group(1).strip()
            mode = 'question'
            continue
        if am:
            if current_answer:
                answers.append('\n'.join(current_answer).strip())
            current_answer = [am.group(1).strip()]
            mode = 'answer'
            continue
        if mode == 'question' and line.strip():
            if _UI_LINE.search(line) or _METADATA_LINE.search(line):
                continue
            if re.search(r'(?i)(?:posted\s+by|score:\s*-?\d+|views:\s*\d+)', line):
                continue
            if _REPLY_HEADER.match(line) or _THREAD_MARKER.match(line):
                if _ACCEPTED_INLINE.search(line):
                    mode = 'answer'
                    inline = re.sub(r'(?i)^.*accepted\s+answer\s*:?\s*', '', line).strip()
                    if inline:
                        current_answer = [inline]
                else:
                    mode = ''
                continue
            question = f'{question}\n{line.strip()}'.strip()
            continue
        if _REPLY_HEADER.match(line) or _THREAD_MARKER.match(line):
            if current_answer and mode == 'answer':
                answers.append('\n'.join(current_answer).strip())
                current_answer = []
            if _ACCEPTED_INLINE.search(line):
                mode = 'answer'
                inline = re.sub(r'(?i)^.*accepted\s+answer\s*:?\s*', '', line).strip()
                if inline:
                    current_answer = [inline]
            else:
                mode = ''
            continue
        if mode == 'answer' and line.strip():
            current_answer.append(line)
    if current_answer:
        answers.append('\n'.join(current_answer).strip())
    if question and answers:
        ranked = sorted(answers, key=_score_answer, reverse=True)
        return QAPair(question=question, answers=ranked[: 1 + max(0, max_extra)])
    blocks = list(_ANSWER_BLOCK.finditer(text))
    if blocks:
        q_match = re.search(r'(?im)^\s*(?:question|q)\s*:\s*(.+)$', text)
        question = q_match.group(1).strip() if q_match else text.split('\n\n', 1)[0].strip()
        extracted = [m.group(1).strip() for m in blocks if m.group(1).strip()]
        ranked = sorted(extracted, key=_score_answer, reverse=True)
        if question and ranked:
            return QAPair(question=question, answers=ranked[: 1 + max(0, max_extra)])
    return None

def extract_conversation(
    text: str,
    *,
    row: Optional[dict[str, Any]] = None,
    max_extra_answers: int = 1,
    stats: StageStats | None = None,
) -> Optional[QAPair]:
    if stats is not None:
        stats.in_docs += 1
    pair = None
    if row:
        pair = _extract_from_row(row, max_extra_answers)
    if pair is None:
        pair = _extract_from_text(text, max_extra_answers)
    if pair is None:
        pair = _extract_reddit_style(text)
    if pair is None:
        stripped = _COMMENT_BLOCK.sub('', text)
        stripped = re.sub(r'(?im)^\s*(?:comment|reply)\s*:\s*.*$', '', stripped)
        stripped = re.sub(r'\n{3,}', '\n\n', stripped).strip()
        if stripped != text.strip() and stats is not None:
            stats.lines_removed += text.count('\n') - stripped.count('\n')
        if stats is not None:
            stats.out_docs += 1 if stripped else 0
        return None
    if stats is not None:
        stats.out_docs += 1
    return pair
