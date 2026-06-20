from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from indw.clean.document.conversation import QAPair, extract_conversation
from indw.extract.roles.forum import (
    ConversationRole,
    DISCARD_ROLES,
    KNOWLEDGE_ROLES,
    decompose_conversation_spans,
    detect_forum_document,
    dominant_role,
    is_low_value_span,
    recover_forum_structure,
    score_answer_substance,
    score_conversation_roles,
    score_forum_answer_rank,
    strip_discard_spans,
)
from indw.extract.structure.recovery import RecoveredSection
from indw.extract.sections.classify import (
    DISCARD_CLASSES,
    KnowledgeSectionClass,
    SectionClassification,
)
from indw.extract.sections.quality import SectionQualityScore, assess_section_quality

@dataclass
class ForumUnit:
    kind: str
    text: str
    score: float = 0.0

def _best_answer_sections(
    sections: list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]],
) -> list[ForumUnit]:
    answers: list[tuple[float, str]] = []
    for sec, cls, qual in sections:
        if cls.label == KnowledgeSectionClass.ANSWER and qual.keep:
            answers.append((qual.retention_score, sec.text))
        elif cls.label == KnowledgeSectionClass.ARTICLE and qual.keep and '?' not in sec.text[:40]:
            answers.append((qual.retention_score * 0.9, sec.text))
    if not answers:
        for sec, cls, qual in sections:
            if cls.label not in DISCARD_CLASSES:
                substance = score_answer_substance(sec.text)
                if qual.retention_score > 0.12 or substance > 0:
                    answers.append((max(qual.retention_score, substance), sec.text))
    answers.sort(key=lambda x: -x[0])
    out: list[ForumUnit] = []
    for score, text in answers[:3]:
        cleaned = strip_discard_spans(text)
        if not cleaned and score_answer_substance(text) > 0.22:
            cleaned = text.strip()
        if not cleaned:
            continue
        substance = score_answer_substance(cleaned)
        rank = max(score, substance)
        if substance > 0.08 or rank > 0.12:
            if out:
                primary = out[0].text
                if (
                    text.lstrip()[:1].islower()
                    and score_answer_substance(text) < score_answer_substance(primary) * 0.92
                ):
                    continue
            out.append(ForumUnit(kind='answer', text=cleaned, score=rank))
    return out

def _span_forum_units(text: str, *, max_extra_answers: int = 1) -> list[ForumUnit]:
    structure = recover_forum_structure(text)
    if not structure.spans:
        return []

    questions: list[str] = []
    answers: list[tuple[float, str]] = []
    explanations: list[tuple[float, str]] = []

    prev_role: ConversationRole | None = None
    depth = 0
    for span in structure.spans:
        if is_low_value_span(span.role, span.text, confidence=span.confidence):
            prev_role = span.role
            if span.role in (ConversationRole.COMMENT, ConversationRole.CONVERSATION):
                depth += 1
            continue
        cleaned = span.text.strip()
        if not cleaned:
            prev_role = span.role
            continue
        rank = score_forum_answer_rank(cleaned, prev_role=prev_role, thread_depth=depth)
        if span.role == ConversationRole.QUESTION:
            questions.append(cleaned)
        elif span.role == ConversationRole.ANSWER:
            conv_scores = score_conversation_roles(
                cleaned, position_ratio=0.5, prev_role=prev_role, thread_depth=depth,
            )
            conv_role, conv_conf = dominant_role(conv_scores)
            answer_mass = conv_scores.get(ConversationRole.ANSWER, 0.0)
            conv_mass = max(
                conv_scores.get(ConversationRole.CONVERSATION, 0.0),
                conv_scores.get(ConversationRole.COMMENT, 0.0),
            )
            followup_turn = prev_role in (ConversationRole.ANSWER, ConversationRole.EXPLANATION)
            if (
                conv_role == ConversationRole.CONVERSATION
                and conv_conf > 0.40
                and rank < 0.38
            ):
                depth += 1
                prev_role = ConversationRole.CONVERSATION
                continue
            if (
                followup_turn
                and conv_mass > answer_mass + 0.02
                and rank < 0.42
                and score_answer_substance(cleaned) < 0.55
            ):
                depth += 1
                prev_role = ConversationRole.CONVERSATION
                continue
            answers.append((rank, cleaned))
        elif span.role == ConversationRole.EXPLANATION:
            explanations.append((rank, cleaned))
        elif span.role == ConversationRole.UNKNOWN:
            substance = score_answer_substance(cleaned)
            scores = score_conversation_roles(
                cleaned, prev_role=prev_role, thread_depth=depth,
            )
            role, conf = dominant_role(scores)
            if role == ConversationRole.QUESTION and conf > 0.30:
                questions.append(cleaned)
            elif substance > 0.15:
                answers.append((rank, cleaned))
        if span.role in (ConversationRole.COMMENT, ConversationRole.CONVERSATION):
            depth += 1
        prev_role = span.role

    out: list[ForumUnit] = []
    if questions:
        out.append(ForumUnit(kind='question', text='\n\n'.join(questions), score=1.0))

    answers.sort(key=lambda x: -x[0])
    for i, (score, ans) in enumerate(answers[: 1 + max_extra_answers]):
        kind = 'answer' if i == 0 else 'answer_extra'
        out.append(ForumUnit(kind=kind, text=ans, score=score))

    if not out and explanations:
        explanations.sort(key=lambda x: -x[0])
        out.append(ForumUnit(kind='answer', text=explanations[0][1], score=explanations[0][0]))

    return out

def _section_forum_units(
    sections: list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]],
    *,
    max_extra_answers: int = 1,
) -> list[ForumUnit]:
    units: list[ForumUnit] = []
    question = ''
    for sec, cls, qual in sections:
        if cls.label == KnowledgeSectionClass.QUESTION and qual.keep:
            question = strip_discard_spans(sec.text)
            break
        if not question and '?' in sec.text and qual.retention_score > 0.15:
            spans = decompose_conversation_spans(sec.text)
            q_parts = [s.text for s in spans if s.role == ConversationRole.QUESTION]
            if q_parts:
                question = '\n\n'.join(q_parts)
            elif cls.label not in DISCARD_CLASSES:
                question = strip_discard_spans(sec.text)
            break

    if question:
        units.append(ForumUnit(kind='question', text=question, score=1.0))

    units.extend(_best_answer_sections(sections))
    if len(units) > 1 + max_extra_answers:
        primary = [units[0]] + [u for u in units[1:] if u.kind == 'answer'][:1]
        extras = [u for u in units if u.kind == 'answer_extra'][:max_extra_answers]
        units = primary + extras
    return units

def extract_forum_units(
    text: str,
    sections: list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]],
    *,
    row: dict | None = None,
    max_extra_answers: int = 1,
) -> list[ForumUnit]:
    knowledge_text = '\n\n'.join(
        strip_discard_spans(sec.text.strip())
        for sec, cls, qual in sections
        if len(sec.text.strip()) > 20 and (
            cls.label not in DISCARD_CLASSES
            or (
                cls.label == KnowledgeSectionClass.RELATED
                and score_answer_substance(sec.text) >= 0.35
            )
        )
    ).strip() or text

    cleaned_knowledge = strip_discard_spans(knowledge_text)
    if not cleaned_knowledge:
        cleaned_knowledge = knowledge_text

    pair: QAPair | None = None
    if row:
        pair = extract_conversation(cleaned_knowledge, row=row, max_extra_answers=max_extra_answers)
    if pair is None:
        pair = extract_conversation(cleaned_knowledge, max_extra_answers=max_extra_answers)

    units: list[ForumUnit] = []
    if pair is not None:
        q_clean = strip_discard_spans(pair.question)
        if q_clean:
            units.append(ForumUnit(kind='question', text=q_clean, score=1.0))
        for i, ans in enumerate(pair.answers):
            ans_clean = strip_discard_spans(ans)
            if not ans_clean:
                continue
            substance = score_answer_substance(ans_clean)
            if substance < 0.08:
                continue
            kind = 'answer' if i == 0 or not any(u.kind == 'answer' for u in units) else 'answer_extra'
            units.append(ForumUnit(kind=kind, text=ans_clean, score=substance))
        if any(u.kind == 'answer' for u in units):
            return units

    units = _span_forum_units(cleaned_knowledge, max_extra_answers=max_extra_answers)
    section_units = _section_forum_units(sections, max_extra_answers=max_extra_answers)
    if section_units and units:
        span_ans = next((u for u in units if u.kind == 'answer'), None)
        sec_ans = next((u for u in section_units if u.kind == 'answer'), None)
        if (
            span_ans
            and sec_ans
            and sec_ans.score > span_ans.score + 0.04
            and span_ans.text.lstrip()[:1].islower()
        ):
            return section_units
    if units and any(u.kind == 'answer' for u in units):
        return units

    return section_units
