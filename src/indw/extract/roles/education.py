from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from indw.extract.structure.analyze import analyze_structure
from indw.clean.artifact.evidence_engine import compute_semantic_evidence
from indw.clean.artifact.evidence_features import DocumentFeatureExtractor
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator
from indw.clean.document.value import compute_structure_profile

class EducationalRole(str, Enum):
    KNOWLEDGE = 'knowledge'
    EXPLANATION = 'explanation'
    ARTICLE = 'article'
    REFERENCE = 'reference'
    PRIMARY_SOURCE = 'primary_source'
    QUESTION_PROMPT = 'question_prompt'
    INSTRUCTION = 'instruction'
    ASSIGNMENT = 'assignment'
    EXERCISE = 'exercise'
    DISCUSSION_STARTER = 'discussion_starter'
    LEARNING_OBJECTIVE = 'learning_objective'
    MIXED = 'mixed'
    UNKNOWN = 'unknown'

DISCARD_EDUCATIONAL_ROLES = frozenset({
    EducationalRole.QUESTION_PROMPT,
    EducationalRole.INSTRUCTION,
    EducationalRole.ASSIGNMENT,
    EducationalRole.EXERCISE,
    EducationalRole.DISCUSSION_STARTER,
    EducationalRole.LEARNING_OBJECTIVE,
})

KEEP_EDUCATIONAL_ROLES = frozenset({
    EducationalRole.KNOWLEDGE,
    EducationalRole.EXPLANATION,
    EducationalRole.ARTICLE,
    EducationalRole.REFERENCE,
    EducationalRole.PRIMARY_SOURCE,
})

@dataclass
class EducationalSpan:
    text: str
    start: int
    end: int
    role: EducationalRole
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)

@dataclass
class EducationalRoleScore:
    knowledge: float = 0.0
    explanation: float = 0.0
    article: float = 0.0
    reference: float = 0.0
    primary_source: float = 0.0
    question_prompt: float = 0.0
    instruction: float = 0.0
    assignment: float = 0.0
    exercise: float = 0.0
    discussion_starter: float = 0.0
    learning_objective: float = 0.0

    def instruction_mass(self) -> float:
        return max(
            self.question_prompt,
            self.instruction,
            self.assignment,
            self.exercise,
            self.discussion_starter,
            self.learning_objective,
        )

    def knowledge_mass(self) -> float:
        return max(self.knowledge, self.explanation, self.article, self.reference, self.primary_source)

    def dominant(self) -> tuple[EducationalRole, float]:
        items = (
            (EducationalRole.KNOWLEDGE, self.knowledge),
            (EducationalRole.EXPLANATION, self.explanation),
            (EducationalRole.ARTICLE, self.article),
            (EducationalRole.REFERENCE, self.reference),
            (EducationalRole.PRIMARY_SOURCE, self.primary_source),
            (EducationalRole.QUESTION_PROMPT, self.question_prompt),
            (EducationalRole.INSTRUCTION, self.instruction),
            (EducationalRole.ASSIGNMENT, self.assignment),
            (EducationalRole.EXERCISE, self.exercise),
            (EducationalRole.DISCUSSION_STARTER, self.discussion_starter),
            (EducationalRole.LEARNING_OBJECTIVE, self.learning_objective),
        )
        role, conf = max(items, key=lambda x: x[1])
        if conf < 0.12:
            return EducationalRole.UNKNOWN, conf
        return role, conf

def _sentence_units(text: str) -> list[tuple[str, int, int]]:
    blob = text.strip()
    if not blob:
        return []
    units: list[tuple[str, int, int]] = []
    cursor = 0
    buf: list[str] = []
    buf_start = 0
    for i, ch in enumerate(blob):
        if ch not in '.!?':
            if not buf:
                buf_start = i
            buf.append(ch)
            continue
        buf.append(ch)
        chunk = ''.join(buf).strip()
        if chunk:
            units.append((chunk, buf_start, i + 1))
        buf = []
        cursor = i + 1
    if buf:
        chunk = ''.join(buf).strip()
        if chunk:
            units.append((chunk, buf_start, len(blob)))
    if not units and blob:
        units.append((blob, 0, len(blob)))
    return units

def _interrogative_density(text: str, raw) -> float:
    if not text.strip():
        return 0.0
    q_marks = text.count('?')
    if q_marks <= 0:
        return 0.0
    units = _sentence_units(text)
    if not units:
        return min(1.0, q_marks / max(raw.sentence_count, 1))
    q_sents = sum(1 for s, _, _ in units if s.rstrip().endswith('?'))
    ratio = q_sents / len(units)
    density = q_marks / max(len(units), 1)
    return min(1.0, ratio * 0.65 + density * 0.45)

def _declarative_density(text: str) -> float:
    units = _sentence_units(text)
    if not units:
        return 0.0
    decl = sum(1 for s, _, _ in units if s.rstrip().endswith('.'))
    return decl / len(units)

def _question_prompt_signal(text: str, raw, profile, ev, structural, *, interrogative: float) -> float:
    if interrogative < 0.28:
        return 0.0
    rep = ev.representation
    util = ev.utility
    explain = profile.explanation_ratio
    factual = profile.fact_ratio
    referential = rep.referential if rep else 0.0
    score = interrogative * 0.55
    if factual < 0.22:
        score += 0.18
    if explain < 0.38 and util < 0.18:
        score += 0.15
    if referential > 0.12 and factual < 0.30:
        score += 0.12
    if raw.fact_relation_hits <= 1 and interrogative > 0.45:
        score += 0.10
    if structural.sentence_completeness_mean > 0.40 and text.rstrip().endswith('?'):
        score += 0.08
    if util > 0.24 and factual > 0.35 and explain > 0.35:
        score *= 0.35
    return min(1.0, score)

def _instruction_signal(text: str, raw, profile, ev, *, interrogative: float) -> float:
    instr = profile.instruction_ratio
    rep = ev.representation
    procedural = rep.procedural if rep else 0.0
    if instr < 0.10 and procedural < 0.12:
        return 0.0
    score = instr * 0.50 + procedural * 0.35
    if raw.step_line_hits > 0:
        score += min(0.25, raw.step_line_hits / max(raw.line_count, 1) * 0.8)
    if interrogative > 0.20:
        score += 0.08
    if ev.utility > 0.22 and profile.fact_ratio > 0.28:
        score *= 0.40
    return min(1.0, score)

def _assignment_signal(text: str, raw, profile, ev, structural, *, interrogative: float) -> float:
    score = 0.0
    if profile.instruction_ratio > 0.14:
        score += profile.instruction_ratio * 0.45
    if interrogative > 0.35:
        score += interrogative * 0.30
    if structural.sentence_completeness_mean < 0.55 and ev.utility < 0.16:
        score += 0.12
    if raw.word_count <= 28 and interrogative > 0.40:
        score += 0.15
    if (
        raw.fact_relation_hits == 0
        and raw.copula_def_hits == 0
        and profile.explanation_ratio < 0.06
    ):
        if raw.numeric_token_ratio > 0.15 and ev.utility < 0.16 and raw.year_hits < 2:
            if (
                ev.quality.educational < 0.12
                and profile.explanation_ratio < 0.06
                and raw.word_count <= 16
            ):
                score += 0.38
        elif (
            raw.word_count <= 10
            and ev.utility < 0.18
            and profile.fact_ratio < 0.12
            and ev.quality.technical < 0.55
        ):
            score += 0.35
    if ev.utility > 0.20 and profile.explanation_ratio > 0.30:
        score *= 0.35
    return min(1.0, score)

def _exercise_signal(text: str, raw, profile, ev) -> float:
    if raw.step_line_hits < 1 and profile.instruction_ratio < 0.18:
        return 0.0
    score = min(1.0, raw.step_line_hits / max(raw.line_count, 1) * 0.9)
    score = max(score, profile.instruction_ratio * 0.55)
    if profile.listing_ratio > 0.15:
        score += 0.12
    if ev.utility > 0.22 and profile.explanation_ratio > 0.28:
        score *= 0.40
    return min(1.0, score)

def _discussion_starter_signal(text: str, raw, profile, ev, *, interrogative: float) -> float:
    if interrogative < 0.35:
        return 0.0
    units = _sentence_units(text)
    if len(units) < 2:
        if interrogative < 0.55:
            return 0.0
    q_units = sum(1 for s, _, _ in units if s.rstrip().endswith('?'))
    if q_units < 2 and interrogative < 0.60:
        return 0.0
    score = interrogative * 0.45 + min(1.0, q_units / max(len(units), 1)) * 0.35
    if profile.explanation_ratio < 0.20 and ev.utility < 0.15:
        score += 0.15
    if profile.fact_ratio > 0.30:
        score *= 0.35
    return min(1.0, score)

def _learning_objective_signal(text: str, raw, profile, ev, structural) -> float:
    if raw.word_count > 40:
        return 0.0
    if structural.sentence_completeness_mean < 0.40:
        return 0.0
    score = profile.instruction_ratio * 0.40
    if profile.explanation_ratio < 0.14 and ev.utility < 0.14:
        score += 0.20
    if raw.copula_def_hits > 0 and profile.fact_ratio > 0.20:
        score *= 0.30
    return min(1.0, score)

def _knowledge_signal(text: str, raw, profile, ev, structural, *, interrogative: float) -> float:
    util = ev.utility
    factual = profile.fact_ratio
    explain = profile.explanation_ratio
    score = util * 0.45 + factual * 0.25 + explain * 0.20
    if raw.fact_relation_hits > 0:
        score += min(0.18, raw.fact_relation_hits / max(raw.sentence_count, 1) * 0.35)
    if raw.copula_def_hits > 0:
        score += min(0.12, raw.copula_def_hits / max(raw.sentence_count, 1) * 0.25)
    if _declarative_density(text) > 0.45:
        score += 0.10
    if interrogative > 0.40:
        score *= max(0.25, 1.0 - interrogative * 0.85)
    if (
        raw.fact_relation_hits == 0
        and raw.copula_def_hits == 0
        and profile.explanation_ratio < 0.06
        and ev.utility < 0.16
        and raw.year_hits < 2
        and ev.quality.technical < 0.50
    ):
        score *= 0.42
    if ev.quality.technical > 0.58 and _declarative_density(text) > 0.80:
        score += 0.22
    if structural.sentence_completeness_mean > 0.50:
        score += 0.08
    return min(1.0, score)

def _explanation_signal(text: str, raw, profile, ev, structural, *, interrogative: float) -> float:
    explain = profile.explanation_ratio
    if explain < 0.12 and ev.quality.educational < 0.12:
        return 0.0
    score = explain * 0.55 + ev.quality.educational * 0.35
    if raw.copula_def_hits > 0:
        score += 0.10
    if interrogative > 0.35:
        score *= max(0.30, 1.0 - interrogative * 0.70)
    if structural.sentence_completeness_mean > 0.48:
        score += 0.08
    return min(1.0, score)

def _article_signal(text: str, raw, profile, ev, structural, *, interrogative: float) -> float:
    util = ev.utility
    if util < 0.10 and profile.explanation_ratio < 0.14:
        return 0.0
    score = util * 0.50 + profile.explanation_ratio * 0.25 + structural.information_density * 0.15
    if interrogative > 0.35:
        score *= max(0.25, 1.0 - interrogative * 0.80)
    if len(text.split()) >= 18 and _declarative_density(text) > 0.35:
        score += 0.10
    return min(1.0, score)

def _reference_signal(text: str, raw, profile, ev) -> float:
    rep = ev.representation
    if not rep:
        return 0.0
    score = ev.quality.reference * 0.55 + rep.referential * 0.30
    if raw.citation_hits > 0:
        score += min(0.20, raw.citation_hits / max(raw.sentence_count, 1) * 0.35)
    if profile.instruction_ratio > 0.20 and text.count('?') > 0:
        score *= 0.35
    if _declarative_density(text) > 0.35 and profile.fact_ratio > 0.14:
        score *= 0.40
    return min(1.0, score)

def score_educational_roles(
    text: str,
    *,
    position_ratio: float = 0.5,
) -> EducationalRoleScore:
    if not text or not text.strip():
        return EducationalRoleScore()

    raw = DocumentFeatureExtractor().extract(text)
    ev = compute_semantic_evidence(text)
    profile = compute_structure_profile(text, evidence=ev)
    structural = analyze_structure(text)
    interrogative = _interrogative_density(text, raw)

    know = _knowledge_signal(text, raw, profile, ev, structural, interrogative=interrogative)
    explain = _explanation_signal(text, raw, profile, ev, structural, interrogative=interrogative)
    article = _article_signal(text, raw, profile, ev, structural, interrogative=interrogative)
    reference = _reference_signal(text, raw, profile, ev)
    q_prompt = _question_prompt_signal(text, raw, profile, ev, structural, interrogative=interrogative)
    instruction = _instruction_signal(text, raw, profile, ev, interrogative=interrogative)
    assignment = _assignment_signal(text, raw, profile, ev, structural, interrogative=interrogative)
    exercise = _exercise_signal(text, raw, profile, ev)
    discussion = _discussion_starter_signal(text, raw, profile, ev, interrogative=interrogative)
    objective = _learning_objective_signal(text, raw, profile, ev, structural)

    if position_ratio > 0.55 and q_prompt > 0.22:
        q_prompt = min(1.0, q_prompt + 0.08)
        assignment = min(1.0, assignment + 0.06)
    if position_ratio < 0.35 and know > 0.18:
        know = min(1.0, know + 0.06)
        article = min(1.0, article + 0.04)

    return EducationalRoleScore(
        knowledge=know,
        explanation=explain,
        article=article,
        reference=reference,
        primary_source=min(1.0, reference * 0.85 + profile.fact_ratio * 0.20),
        question_prompt=q_prompt,
        instruction=instruction,
        assignment=assignment,
        exercise=exercise,
        discussion_starter=discussion,
        learning_objective=objective,
    )

def dominant_educational_role(scores: EducationalRoleScore) -> tuple[EducationalRole, float]:
    return scores.dominant()

def is_instructional_span(role: EducationalRole, text: str, *, confidence: float = 0.0) -> bool:
    if role not in DISCARD_EDUCATIONAL_ROLES:
        return False
    if confidence < 0.30:
        return False
    ev = compute_semantic_evidence(text)
    profile = compute_structure_profile(text, evidence=ev)
    if ev.utility > 0.28 and profile.explanation_ratio > 0.32 and profile.fact_ratio > 0.28:
        if confidence < 0.55:
            return False
    if len(text.split()) > 60 and ev.utility > 0.18 and profile.explanation_ratio > 0.25:
        return False
    return True

def educational_role_boundary(left: str, right: str, *, left_pos: float, right_pos: float) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    left_s = score_educational_roles(left, position_ratio=left_pos)
    right_s = score_educational_roles(right, position_ratio=right_pos)
    left_role, left_conf = left_s.dominant()
    right_role, right_conf = right_s.dominant()

    know_left = left_role in KEEP_EDUCATIONAL_ROLES and left_conf > 0.25
    instr_right = right_role in DISCARD_EDUCATIONAL_ROLES and right_conf > 0.28
    if know_left and instr_right:
        return min(1.0, left_conf * 0.40 + right_conf * 0.45 + 0.15)

    util_l = compute_semantic_evidence(left).utility
    util_r = compute_semantic_evidence(right).utility
    if left_s.knowledge_mass() > 0.22 and right_s.instruction_mass() > 0.30:
        return min(1.0, left_s.knowledge_mass() * 0.35 + right_s.instruction_mass() * 0.45 + 0.12)
    if util_l > 0.12 and right_s.instruction_mass() > 0.35:
        return min(1.0, right_s.instruction_mass() * 0.50 + (util_l - util_r) * 0.35 + 0.10)
    if left_role != right_role and left_conf > 0.28 and right_conf > 0.28:
        if left_role in KEEP_EDUCATIONAL_ROLES or right_role in KEEP_EDUCATIONAL_ROLES:
            return min(1.0, abs(left_conf - right_conf) * 0.40 + 0.22)
    return 0.0

def decompose_educational_spans(text: str) -> list[EducationalSpan]:
    if not text or not text.strip():
        return []

    blob = text.strip()
    base = text.find(blob)
    spans: list[EducationalSpan] = []

    blocks: list[tuple[str, int]] = []
    if '\n\n' in blob:
        cursor = 0
        for para in blob.split('\n\n'):
            stripped = para.strip()
            if not stripped:
                cursor += len(para) + 2
                continue
            idx = blob.find(stripped, cursor)
            if idx < 0:
                idx = cursor
            blocks.append((stripped, base + idx))
            cursor = idx + len(stripped)
    else:
        blocks = [(blob, base)]

    for block, block_start in blocks:
        units = _sentence_units(block)
        if len(units) <= 1:
            scores = score_educational_roles(block, position_ratio=0.5)
            role, conf = scores.dominant()
            spans.append(EducationalSpan(
                text=block,
                start=block_start,
                end=block_start + len(block),
                role=role,
                confidence=conf,
                scores=_score_dict(scores),
            ))
            continue

        groups: list[tuple[EducationalRole, float, list[tuple[str, int, int]]]] = []
        for sent, rel_start, rel_end in units:
            abs_pos = (block_start - base + rel_start) / max(len(blob), 1)
            scores = score_educational_roles(sent, position_ratio=abs_pos)
            role, conf = scores.dominant()
            if groups and groups[-1][0] == role:
                groups[-1][2].append((sent, rel_start, rel_end))
            else:
                groups.append((role, conf, [(sent, rel_start, rel_end)]))

        for role, conf, parts in groups:
            chunk = ' '.join(s for s, _, _ in parts).strip()
            if not chunk:
                continue
            start = block_start + parts[0][1]
            end = block_start + parts[-1][2]
            spans.append(EducationalSpan(
                text=chunk,
                start=start,
                end=end,
                role=role,
                confidence=conf,
                scores={},
            ))

    return spans

def _score_dict(scores: EducationalRoleScore) -> dict[str, float]:
    return {k: round(v, 4) for k, v in {
        'knowledge': scores.knowledge,
        'explanation': scores.explanation,
        'article': scores.article,
        'question_prompt': scores.question_prompt,
        'instruction': scores.instruction,
        'assignment': scores.assignment,
    }.items()}

def strip_instructional_scaffolding(text: str) -> str:
    spans = decompose_educational_spans(text)
    if not spans:
        return text.strip()

    if len(spans) == 1:
        span = spans[0]
        if is_instructional_span(span.role, span.text, confidence=span.confidence):
            return ''
        if span.role in KEEP_EDUCATIONAL_ROLES or span.role == EducationalRole.UNKNOWN:
            return span.text.strip()
        return span.text.strip()

    kept: list[str] = []
    for span in spans:
        if is_instructional_span(span.role, span.text, confidence=span.confidence):
            continue
        if span.role in KEEP_EDUCATIONAL_ROLES:
            kept.append(span.text.strip())
        elif span.role == EducationalRole.UNKNOWN:
            ev = compute_semantic_evidence(span.text)
            if ev.utility >= 0.10 or len(span.text.split()) >= 12:
                kept.append(span.text.strip())
    return '\n\n'.join(kept) if kept else ''

@dataclass
class EducationalLearner:
    _leakage: list[tuple[float, ...]] = field(default_factory=list)
    _boost: dict[str, float] = field(default_factory=dict)

    def record_surviving_instruction(self, text: str, *, position_ratio: float = 0.5) -> None:
        scores = score_educational_roles(text, position_ratio=position_ratio)
        if scores.instruction_mass() < 0.28:
            return
        role, conf = scores.dominant()
        if role not in DISCARD_EDUCATIONAL_ROLES:
            return
        vec = (
            scores.question_prompt,
            scores.instruction,
            scores.assignment,
            scores.exercise,
            scores.discussion_starter,
            conf,
            position_ratio,
        )
        self._leakage.append(vec)
        if len(self._leakage) > 400:
            self._leakage.pop(0)

    def role_boost(self, role: str) -> float:
        return self._boost.get(role, 0.0)

    def cluster_report(self) -> dict[str, Any]:
        if not self._leakage:
            return {'samples': 0}
        n = len(self._leakage)
        avg_q = sum(v[0] for v in self._leakage) / n
        avg_i = sum(v[1] for v in self._leakage) / n
        return {
            'samples': n,
            'avg_question_prompt': round(avg_q, 4),
            'avg_instruction': round(avg_i, 4),
        }
