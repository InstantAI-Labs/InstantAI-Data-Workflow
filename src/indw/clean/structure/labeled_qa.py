from __future__ import annotations

import re

from indw.clean.document.conversation import QAPair, _score_answer
from indw.clean.document.patterns import _ACK_LINE, _METADATA_LINE, _UI_LINE
from indw.clean.document.stats import StageStats

_ANSWER_HDR = re.compile(r'(?im)^\s*(?:accepted\s+answer|best\s+answer|answer)\s*:\s*(.*)$')
_QUESTION_HDR = re.compile(r'(?im)^\s*(?:question|q)\s*:\s*(.*)$')
_COMMENT_HDR = re.compile(r'(?im)^\s*(?:comment|reply|meta)\s*:\s*')

def has_labeled_qa_markers(text: str) -> bool:
    if not text:
        return False
    return bool(_QUESTION_HDR.search(text) and _ANSWER_HDR.search(text))

def _is_low_value_answer(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 24:
        return True
    if _ACK_LINE.match(stripped):
        return True
    words = stripped.split()
    if len(words) < 12 and '```' not in stripped and '$' not in stripped:
        return True
    return False

def extract_labeled_qa(text: str, *, max_extra_answers: int = 1) -> str | None:
    question = ''
    answers: list[tuple[float, str, bool]] = []
    current_answer: list[str] = []
    accepted = False
    mode = ''

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _UI_LINE.search(stripped) or _METADATA_LINE.search(stripped) or _COMMENT_HDR.match(stripped):
            if current_answer and mode == 'answer':
                body = '\n'.join(current_answer).strip()
                if body and not _is_low_value_answer(body):
                    answers.append((_score_answer(body) + (0.5 if accepted else 0.0), body, accepted))
                current_answer = []
                accepted = False
            mode = ''
            continue
        qm = _QUESTION_HDR.match(stripped)
        if qm:
            question = (qm.group(1) or '').strip() or question
            mode = 'question'
            continue
        am = _ANSWER_HDR.match(stripped)
        if am:
            if current_answer:
                body = '\n'.join(current_answer).strip()
                if body and not _is_low_value_answer(body):
                    answers.append((_score_answer(body) + (0.5 if accepted else 0.0), body, accepted))
            accepted = 'accepted' in stripped.lower() or 'best' in stripped.lower()
            inline = (am.group(1) or '').strip()
            current_answer = [inline] if inline else []
            mode = 'answer'
            continue
        if '?' in stripped and len(stripped) > 20 and not question and mode != 'answer':
            question = stripped
            mode = 'question'
            continue
        if mode == 'answer':
            current_answer.append(line)

    if current_answer:
        body = '\n'.join(current_answer).strip()
        if body and not _is_low_value_answer(body):
            answers.append((_score_answer(body) + (0.5 if accepted else 0.0), body, accepted))

    if not question or not answers:
        return None

    answers.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    kept: list[str] = []
    for _, body, _ in answers:
        key = body[:200].lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(body)
        if len(kept) >= 1 + max(0, max_extra_answers):
            break

    return QAPair(question=question, answers=kept).to_text()

def preprocess_labeled_qa(text: str, *, stats: StageStats | None = None, max_extra_answers: int = 1) -> str:
    extracted = extract_labeled_qa(text, max_extra_answers=max_extra_answers)
    if extracted:
        if stats is not None:
            stats.in_docs += 1
            stats.out_docs += 1
        return extracted
    if stats is not None:
        stats.in_docs += 1
        stats.out_docs += 1
    return text
