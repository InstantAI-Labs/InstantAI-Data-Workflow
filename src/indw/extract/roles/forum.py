from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from indw.clean.artifact.decompose import compute_layout
from indw.extract.structure.analyze import analyze_structure
from indw.clean.artifact.evidence_engine import compute_semantic_evidence
from indw.clean.artifact.evidence_features import DocumentFeatureExtractor
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator
from indw.clean.document.value import compute_structure_profile


class ConversationRole(str, Enum):
    QUESTION = 'question'
    ANSWER = 'answer'
    EXPLANATION = 'explanation'
    COMMENT = 'comment'
    CONVERSATION = 'conversation'
    FORUM_UI = 'forum_ui'
    COMMUNITY_WRAPPER = 'community_wrapper'
    METADATA = 'metadata'
    SIGNATURE = 'signature'
    NAVIGATION = 'navigation'
    ADVERTISEMENT = 'advertisement'
    UNKNOWN = 'unknown'


DISCARD_ROLES = frozenset({
    ConversationRole.FORUM_UI,
    ConversationRole.COMMUNITY_WRAPPER,
    ConversationRole.METADATA,
    ConversationRole.SIGNATURE,
    ConversationRole.NAVIGATION,
    ConversationRole.ADVERTISEMENT,
    ConversationRole.CONVERSATION,
    ConversationRole.COMMENT,
})

FORUM_INFRA_ROLES = frozenset({
    ConversationRole.FORUM_UI,
    ConversationRole.METADATA,
    ConversationRole.COMMUNITY_WRAPPER,
    ConversationRole.NAVIGATION,
    ConversationRole.SIGNATURE,
    ConversationRole.ADVERTISEMENT,
})

KNOWLEDGE_ROLES = frozenset({
    ConversationRole.QUESTION,
    ConversationRole.ANSWER,
    ConversationRole.EXPLANATION,
})


@dataclass
class ConversationSpan:
    text: str
    start: int
    end: int
    role: ConversationRole
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ForumStructure:
    spans: list[ConversationSpan]
    is_forum: bool = False
    has_question: bool = False
    has_answer: bool = False
    wrapper_mass: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'is_forum': self.is_forum,
            'has_question': self.has_question,
            'has_answer': self.has_answer,
            'wrapper_mass': round(self.wrapper_mass, 4),
            'spans': [
                {'role': s.role.value, 'confidence': round(s.confidence, 4), 'preview': s.text[:120]}
                for s in self.spans[:24]
            ],
        }


def _contact_handle_density(text: str, raw) -> float:
    if not raw.words:
        return 0.0
    handles = sum(1 for w in raw.words if w.startswith('@') and len(w) > 1)
    return handles / max(raw.word_count, 1)


def _imperative_density(profile, ev) -> float:
    return max(
        profile.instruction_ratio,
        ev.negative.get('transactional', 0.0) * 0.65,
        ev.representation.procedural if ev.representation else 0.0,
    )


def _narrative_social(text: str, ev, raw) -> float:
    rep = ev.representation
    if rep is None:
        return 0.0
    narrative = rep.narrative
    noise = ev.negative.get('noise', 0.0)
    fp = raw.first_person_ratio
    short = 1.0 if raw.word_count <= 18 else max(0.0, 1.0 - raw.word_count / 80.0)
    return max(0.0, min(1.0, narrative * 0.45 + noise * 0.35 + fp * 0.25 + short * 0.2))


def _metadata_line_signal(text: str, raw, profile, ev, *, position_ratio: float) -> float:
    contact = _contact_handle_density(text, raw) + profile.contact_ratio
    temporal = profile.date_ratio + raw.schedule_token_ratio
    util = ev.utility
    explain = profile.explanation_ratio
    dash_lead = 1.0 if text.lstrip()[:1] in '–—-' else 0.0
    short = 1.0 if raw.word_count <= 22 else max(0.0, 1.0 - raw.word_count / 60.0)
    score = contact * 0.35 + temporal * 0.30 + dash_lead * 0.20 + short * 0.15
    if util > 0.22 and explain > 0.20:
        score *= 0.35
    if position_ratio > 0.55 and contact > 0.08:
        score += 0.12
    return min(1.0, score)


def _structural_span_noise_ratio(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    from indw.extract.structure.inline import _collect_structural_spans
    spans = _collect_structural_spans(text)
    if not spans:
        return 0.0
    noise = sum(e - s for s, e in spans)
    return min(1.0, noise / max(len(text), 1))


def score_conversation_roles(
    text: str,
    *,
    position_ratio: float = 0.5,
    prev_role: ConversationRole | None = None,
    thread_depth: int = 0,
) -> dict[ConversationRole, float]:
    if not text or not text.strip():
        return {ConversationRole.UNKNOWN: 1.0}

    raw = DocumentFeatureExtractor().extract(text)
    ev = compute_semantic_evidence(text)
    profile = compute_structure_profile(text, evidence=ev)
    structural = analyze_structure(text)
    layout = compute_layout(text)
    baseline = AdaptiveBaselineEstimator()

    util = ev.utility
    tech = ev.quality.technical
    edu = ev.quality.educational
    ref = ev.quality.reference
    noise = ev.negative.get('noise', 0.0)
    nav_neg = ev.negative.get('navigational', 0.0)
    promo = ev.negative.get('promotional', 0.0)
    trans = ev.negative.get('transactional', 0.0)
    admin = ev.negative.get('administrative', 0.0)
    dash_lead = 1.0 if text.lstrip()[:1] in '–—-' else 0.0

    scores: dict[ConversationRole, float] = {}

    imperative = _imperative_density(profile, ev)
    ui_score = min(1.0, imperative * 0.55 + trans * 0.30 + (
        0.25 if raw.word_count <= 8 and util < 0.14 else 0.0
    ))
    if raw.word_count <= 6 and util < 0.10 and profile.explanation_ratio < 0.08:
        ui_score = max(ui_score, 0.55)
    if raw.word_count <= 5 and raw.uppercase_token_ratio > 0.45 and util < 0.14:
        ui_score = max(ui_score, 0.72)
    if raw.word_count <= 10 and imperative > 0.28 and tech < 0.12 and edu < 0.14:
        ui_score = max(ui_score, 0.65)
    if dash_lead > 0 and raw.word_count <= 22 and util < 0.16:
        ui_score = max(ui_score, 0.60)
    if layout.line_count <= 2 and raw.uniform_line_ratio > 0.7 and util < 0.12 and profile.explanation_ratio < 0.10:
        ui_score = max(ui_score, 0.55)
    span_noise = _structural_span_noise_ratio(text)
    if span_noise > 0.08:
        ui_score = max(ui_score, min(1.0, span_noise * 1.35 + imperative * 0.20))
    if util > 0.14 or profile.explanation_ratio > 0.14 or tech > 0.16 or edu > 0.14:
        ui_score *= 0.35
    if structural.sentence_completeness_mean > 0.52 and span_noise < 0.06:
        ui_score *= 0.45
    scores[ConversationRole.FORUM_UI] = ui_score

    scores[ConversationRole.COMMUNITY_WRAPPER] = min(1.0, promo * 0.40 + trans * 0.35 + profile.listing_ratio * 0.30)
    if promo > 0.22 and trans > 0.22:
        scores[ConversationRole.COMMUNITY_WRAPPER] = max(
            scores[ConversationRole.COMMUNITY_WRAPPER], min(1.0, promo * 0.45 + trans * 0.40 + 0.15),
        )
    if profile.explanation_ratio < 0.10 and imperative > 0.20:
        scores[ConversationRole.COMMUNITY_WRAPPER] = max(
            scores[ConversationRole.COMMUNITY_WRAPPER],
            min(1.0, imperative * 0.50 + promo * 0.30 + profile.listing_ratio * 0.25),
        )
    if raw.word_count > 25 and profile.listing_ratio > 0.15 and promo > 0.20:
        scores[ConversationRole.COMMUNITY_WRAPPER] = max(
            scores[ConversationRole.COMMUNITY_WRAPPER], 0.55,
        )

    meta_sig = _metadata_line_signal(text, raw, profile, ev, position_ratio=position_ratio)
    scores[ConversationRole.METADATA] = meta_sig
    if _contact_handle_density(text, raw) > 0.06 and dash_lead > 0:
        scores[ConversationRole.METADATA] = max(meta_sig, 0.58)

    sig_score = 0.0
    if position_ratio > 0.78 and raw.word_count <= 30:
        sig_score = profile.contact_ratio * 0.45 + (1.0 - util) * 0.30
    if text.lstrip()[:2] == '--' and raw.word_count <= 40:
        sig_score = max(sig_score, 0.62)
    scores[ConversationRole.SIGNATURE] = min(1.0, sig_score)

    scores[ConversationRole.NAVIGATION] = min(1.0, nav_neg * 0.55 + profile.navigation_ratio * 0.40 + raw.nav_line_ratio * 0.35)
    scores[ConversationRole.ADVERTISEMENT] = min(1.0, promo * 0.60 + profile.listing_ratio * 0.25)

    social = _narrative_social(text, ev, raw)
    conv_score = min(1.0, social * 0.70 + noise * 0.25)
    if util > 0.14 or profile.explanation_ratio > 0.16:
        conv_score *= 0.35
    scores[ConversationRole.CONVERSATION] = conv_score
    if prev_role in (ConversationRole.ANSWER, ConversationRole.QUESTION) and raw.word_count <= 42:
        if raw.first_person_ratio > 0.02 or social > 0.28:
            scores[ConversationRole.CONVERSATION] = max(
                scores[ConversationRole.CONVERSATION],
                min(1.0, social * 0.85 + raw.first_person_ratio * 0.55 + thread_depth * 0.10),
            )
    if prev_role in (ConversationRole.ANSWER, ConversationRole.QUESTION) and raw.word_count <= 16 and util < 0.14:
        scores[ConversationRole.CONVERSATION] = max(scores[ConversationRole.CONVERSATION], social * 0.85)

    comment_score = min(1.0, noise * 0.45 + (1.0 - util) * 0.30 + thread_depth * 0.08)
    if util > 0.16 or tech > 0.18 or edu > 0.16:
        comment_score *= 0.30
    if '?' in text:
        comment_score *= 0.25
    scores[ConversationRole.COMMENT] = comment_score

    q_signal = ref * 0.35 + profile.instruction_ratio * 0.30
    if '?' in text:
        q_signal += 0.45
    if ':' in text and profile.instruction_ratio > 0.10 and util < 0.35:
        q_signal += 0.28
    if prev_role is None and ':' in text and profile.instruction_ratio > 0.14:
        q_signal += 0.22
    colon_idx = text.find(':')
    if colon_idx > 0 and colon_idx < 90 and '@' not in text[:colon_idx]:
        head_words = text[:colon_idx].split()
        if 2 <= len(head_words) <= 14 and meta_sig < 0.32:
            q_signal += 0.38
            if prev_role is None:
                q_signal += 0.12
    if prev_role is None and position_ratio < 0.35:
        q_signal += 0.10
    if util < 0.12 and '?' in text:
        q_signal += 0.15
    scores[ConversationRole.QUESTION] = min(1.0, q_signal)

    a_signal = util * 0.40 + tech * 0.30 + edu * 0.20 + structural.information_density * 0.15
    if prev_role == ConversationRole.QUESTION:
        a_signal += 0.20
    if '?' not in text and profile.explanation_ratio > 0.14:
        a_signal += 0.18
    if util > 0.14 and '?' not in text:
        a_signal += 0.12
    if raw.word_count <= 14 and util < 0.17 and tech < 0.14:
        a_signal *= 0.55
        scores[ConversationRole.CONVERSATION] = max(scores[ConversationRole.CONVERSATION], 0.52)
    if raw.word_count <= 6 and (imperative > 0.30 or raw.uppercase_token_ratio > 0.45):
        a_signal *= 0.20
    if ':' in text and profile.instruction_ratio > 0.16 and prev_role is None:
        a_signal *= 0.45
    colon_idx = text.find(':')
    if colon_idx > 0 and colon_idx < 90 and prev_role is None:
        head_words = text[:colon_idx].split()
        if 2 <= len(head_words) <= 14:
            a_signal *= 0.42
    if trans > 0.28 and promo > 0.22 and profile.explanation_ratio < 0.12:
        a_signal *= 0.15
    if prev_role == ConversationRole.ANSWER and tech < 0.68 and raw.first_person_ratio > 0.02:
        a_signal *= 0.68
    scores[ConversationRole.ANSWER] = min(1.0, a_signal)

    expl = edu * 0.40 + tech * 0.35 + profile.explanation_ratio * 0.30
    if structural.sentence_completeness_mean > 0.55 and util > 0.20:
        scores[ConversationRole.EXPLANATION] = min(1.0, expl)

    dom_know = max(
        scores.get(ConversationRole.QUESTION, 0.0),
        scores.get(ConversationRole.ANSWER, 0.0),
        scores.get(ConversationRole.EXPLANATION, 0.0),
    )
    dom_infra = max(
        scores.get(ConversationRole.FORUM_UI, 0.0),
        scores.get(ConversationRole.METADATA, 0.0),
        scores.get(ConversationRole.COMMUNITY_WRAPPER, 0.0),
        scores.get(ConversationRole.CONVERSATION, 0.0),
    )
    if dom_know < 0.22 and dom_infra < 0.30:
        scores[ConversationRole.UNKNOWN] = baseline.baseline([util, 1.0 - dom_infra])

    return scores


def dominant_role(scores: dict[ConversationRole, float]) -> tuple[ConversationRole, float]:
    if not scores:
        return ConversationRole.UNKNOWN, 0.0
    role, conf = max(scores.items(), key=lambda x: x[1])
    return role, conf


def score_answer_substance(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    raw = DocumentFeatureExtractor().extract(text)
    ev = compute_semantic_evidence(text)
    tech = ev.quality.technical
    edu = ev.quality.educational
    util = ev.utility
    wc = raw.word_count
    min_w = 5 if tech > 0.18 else 8
    if wc < min_w:
        return 0.18 if tech > 0.22 and wc >= 3 else 0.0
    score = min(1.0, wc / 120.0)
    score += tech * 0.25
    score += edu * 0.15
    score += util * 0.12
    if raw.fence_char_ratio > 0.03:
        score += 0.25
    return min(1.0, score)


def score_forum_answer_rank(
    text: str,
    *,
    prev_role: ConversationRole | None = None,
    thread_depth: int = 0,
) -> float:
    if not text or not text.strip():
        return 0.0
    raw = DocumentFeatureExtractor().extract(text)
    ev = compute_semantic_evidence(text)
    profile = compute_structure_profile(text, evidence=ev)
    rank = score_answer_substance(text)
    rank += ev.quality.technical * 0.22
    rank += profile.explanation_ratio * 0.14
    rank += ev.quality.educational * 0.08
    if '?' in text:
        rank *= 0.82
    social = _narrative_social(text, ev, raw)
    if thread_depth > 0 or prev_role in (ConversationRole.ANSWER, ConversationRole.QUESTION):
        rank -= social * 0.42
    if raw.first_person_ratio > 0.03 and ev.quality.technical < 0.68:
        rank -= raw.first_person_ratio * 0.35
    if prev_role == ConversationRole.ANSWER:
        rank -= 0.14
    return max(0.0, min(1.0, rank))


def is_low_value_span(role: ConversationRole, text: str, *, confidence: float) -> bool:
    span_noise = _structural_span_noise_ratio(text)
    from indw.clean.artifact.engine import get_artifact_engine
    ui_ratio = get_artifact_engine().ui_noise_ratio(text)
    if role in DISCARD_ROLES and (span_noise > 0.10 or ui_ratio > 0.14):
        return True
    if role in DISCARD_ROLES:
        if role in (ConversationRole.COMMENT, ConversationRole.CONVERSATION):
            if score_answer_substance(text) > 0.28:
                return False
        if confidence < 0.42 and score_answer_substance(text) > 0.22 and span_noise < 0.06:
            return False
        return True
    if role == ConversationRole.UNKNOWN:
        substance = score_answer_substance(text)
        ev = compute_semantic_evidence(text)
        if substance > 0.12 or ev.utility > 0.14:
            return False
        return True
    if role == ConversationRole.ANSWER and score_answer_substance(text) < 0.08:
        return True
    if role == ConversationRole.ANSWER:
        raw = DocumentFeatureExtractor().extract(text)
        if raw.word_count <= 6 and raw.uppercase_token_ratio > 0.40:
            return True
    return False


def _split_question_answer_inline(
    chunk: str,
    base_start: int,
    text: str,
) -> list[tuple[str, int, int]]:
    q_idx = chunk.find('?')
    if q_idx < 0 or q_idx >= len(chunk) - 12:
        return [(chunk, base_start, base_start + len(chunk))]
    head = chunk[:q_idx + 1].strip()
    tail_start = base_start + q_idx + 1
    while tail_start < base_start + len(chunk) and text[tail_start].isspace():
        tail_start += 1
    tail = text[tail_start:base_start + len(chunk)].strip()
    if not tail:
        return [(chunk, base_start, base_start + len(chunk))]
    tail_sub = score_answer_substance(tail)
    if tail_sub < 0.18:
        return [(chunk, base_start, base_start + len(chunk))]
    head_end = base_start + len(head)
    while head_end < tail_start and text[head_end].isspace():
        head_end += 1
    return [
        (head, base_start, head_end),
        (tail, tail_start, base_start + len(chunk)),
    ]


def _refine_inline_question_splits(
    units: list[tuple[str, int, int]],
    text: str,
) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for chunk, start, end in units:
        out.extend(_split_question_answer_inline(chunk, start, text))
    return out if out else units


def _split_inline_units(text: str) -> list[tuple[str, int, int]]:
    if not text.strip():
        return []
    total = max(len(text), 1)
    units: list[tuple[str, int, int]] = []

    if '\n' in text:
        cursor = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                cursor += len(line) + 1
                continue
            idx = text.find(stripped, cursor)
            if idx < 0:
                idx = cursor
            units.append((stripped, idx, idx + len(stripped)))
            cursor = idx + len(stripped)
        return _refine_inline_question_splits(units, text)

    if '?' in text:
        parts: list[tuple[str, int, int]] = []
        cursor = 0
        for chunk in text.replace('?', '?\n').splitlines():
            chunk = chunk.strip()
            if not chunk:
                continue
            idx = text.find(chunk, cursor)
            if idx < 0:
                idx = cursor
            parts.append((chunk, idx, idx + len(chunk)))
            cursor = idx + len(chunk)
        if len(parts) > 1:
            return parts

    dash_positions = [i for i, ch in enumerate(text) if ch in '–—' and i > 0]
    if dash_positions:
        bounds = [0, *dash_positions, len(text)]
        units = []
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            chunk = text[start:end].strip()
            if chunk:
                units.append((chunk, start, end))
        if len(units) > 1:
            split_score = 0.0
            for i in range(len(units) - 1):
                left, right = units[i][0], units[i + 1][0]
                left_pos = units[i][1] / total
                right_pos = units[i + 1][1] / total
                split_score = max(
                    split_score,
                    conversation_role_boundary(
                        left, right, left_pos=left_pos, right_pos=right_pos,
                    ),
                )
            if split_score >= 0.32:
                return units

    s = text.strip()
    return [(s, text.find(s), text.find(s) + len(s))]


def decompose_conversation_spans(text: str) -> list[ConversationSpan]:
    if not text or not text.strip():
        return []

    units = _split_inline_units(text)
    if not units:
        return []

    total = max(len(text), 1)
    spans: list[ConversationSpan] = []
    prev_role: ConversationRole | None = None
    depth = 0

    for chunk, start, end in units:
        pos = (start + end) / (2 * total)
        scores = score_conversation_roles(
            chunk, position_ratio=pos, prev_role=prev_role, thread_depth=depth,
        )
        role, conf = dominant_role(scores)

        if role in (ConversationRole.COMMENT, ConversationRole.CONVERSATION):
            depth += 1
        elif role in KNOWLEDGE_ROLES:
            depth = max(0, depth - 1)

        spans.append(ConversationSpan(
            text=chunk,
            start=start,
            end=end,
            role=role,
            confidence=conf,
            scores={k.value: round(v, 4) for k, v in scores.items()},
        ))
        prev_role = role

    return spans


def conversation_role_boundary(left: str, right: str, *, left_pos: float, right_pos: float) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    left_scores = score_conversation_roles(left, position_ratio=left_pos)
    right_scores = score_conversation_roles(right, position_ratio=right_pos, prev_role=dominant_role(left_scores)[0])
    left_role, left_conf = dominant_role(left_scores)
    right_role, right_conf = dominant_role(right_scores)

    know_left = left_role in KNOWLEDGE_ROLES and left_conf > 0.28
    infra_right = right_role in DISCARD_ROLES and right_conf > 0.32
    if know_left and infra_right:
        return min(1.0, left_conf * 0.40 + right_conf * 0.45 + 0.15)

    util_l = compute_semantic_evidence(left).utility
    util_r = compute_semantic_evidence(right).utility
    meta_r = right_scores.get(ConversationRole.METADATA, 0.0)
    ui_r = right_scores.get(ConversationRole.FORUM_UI, 0.0)
    if util_l > 0.15 and (meta_r > 0.38 or ui_r > 0.42):
        return min(1.0, meta_r * 0.45 + ui_r * 0.40 + (util_l - util_r) * 0.35 + 0.12)

    if left_role != right_role and left_conf > 0.30 and right_conf > 0.30:
        if left_role in KNOWLEDGE_ROLES or right_role in KNOWLEDGE_ROLES:
            return min(1.0, abs(left_conf - right_conf) * 0.5 + 0.28)
    return 0.0


def recover_forum_structure(text: str) -> ForumStructure:
    from indw.extract.core.context import get_document_context
    dctx = get_document_context()
    if dctx is not None:
        return dctx.forum_structure(text, lambda: _recover_forum_structure_impl(text))
    return _recover_forum_structure_impl(text)


def _recover_forum_structure_impl(text: str) -> ForumStructure:
    spans = decompose_conversation_spans(text)
    if not spans:
        return ForumStructure(spans=[])

    roles = {s.role for s in spans}
    wrapper_mass = sum(
        s.confidence for s in spans if s.role in DISCARD_ROLES
    ) / max(len(spans), 1)

    has_q = ConversationRole.QUESTION in roles
    has_a = ConversationRole.ANSWER in roles or ConversationRole.EXPLANATION in roles
    forum_infra = roles & FORUM_INFRA_ROLES
    is_forum = bool(
        (has_q and has_a)
        or (has_q and forum_infra)
        or (has_a and forum_infra and len(spans) >= 2)
        or (
            forum_infra
            and KNOWLEDGE_ROLES & roles
            and len(spans) >= 2
            and wrapper_mass > 0.20
        )
    )

    return ForumStructure(
        spans=spans,
        is_forum=is_forum,
        has_question=has_q,
        has_answer=has_a,
        wrapper_mass=wrapper_mass,
    )


def detect_forum_document(
    text: str,
    *,
    section_labels: set[str] | None = None,
) -> bool:
    if section_labels:
        forum_labels = {'forum', 'question', 'answer'}
        if section_labels & forum_labels:
            return True
        edu_only = section_labels <= {
            'discussion_prompt', 'instruction', 'assignment',
        }
        if edu_only and section_labels:
            return False

    structure = recover_forum_structure(text)
    if structure.is_forum and structure.has_question and structure.has_answer:
        return True

    ev = compute_semantic_evidence(text)
    profile = compute_structure_profile(text, evidence=ev)
    if '?' in text and profile.instruction_ratio > 0.12:
        if ev.quality.reference > 0.14 and structure.has_answer:
            return True
    return False


def _block_is_knowledge_prose(block: str) -> bool:
    if len(block.split()) < 8:
        return False
    substance = score_answer_substance(block)
    if substance < 0.24:
        return False
    ev = compute_semantic_evidence(block)
    rep = ev.representation
    role, conf = dominant_role(score_conversation_roles(block))
    if role in DISCARD_ROLES and conf > 0.42:
        if not (ev.quality.technical > 0.22 and rep and rep.factual > 0.22):
            return False
    if rep is None:
        return substance > 0.40
    know = rep.factual + rep.educational + ev.quality.technical * 0.5
    noise = rep.transactional + rep.narrative * 0.6 + ev.negative.get('promotional', 0.0)
    return know > noise and substance > 0.24


def _document_has_substantive_knowledge(text: str) -> bool:
    blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
    if not blocks:
        blocks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = list(blocks)
    if len(blocks) == 1 and len(blocks[0].split()) > 40:
        from indw.extract.structure.aggregate import segment_topics
        candidates.extend(s.text for s in segment_topics(blocks[0]))
    for block in candidates:
        if _block_is_knowledge_prose(block):
            return True
    return False


def _substantive_knowledge_span(span: ConversationSpan) -> bool:
    substance = score_answer_substance(span.text)
    if substance <= 0.22 or len(span.text.split()) <= 8:
        return False
    if span.role in DISCARD_ROLES and span.role not in (ConversationRole.CONVERSATION,):
        return False
    ev = compute_semantic_evidence(span.text)
    promo = ev.negative.get('promotional', 0.0)
    trans = ev.negative.get('transactional', 0.0)
    if promo + trans > 0.65:
        return False
    profile = compute_structure_profile(span.text, evidence=ev)
    if profile.explanation_ratio < 0.08 and promo > 0.30:
        return False
    return substance > 0.30


def is_community_wrapper_document(text: str) -> bool:
    if _document_has_substantive_knowledge(text):
        return False
    from indw.extract.structure.aggregate import segment_topics
    topics = segment_topics(text.strip())
    if len(topics) >= 4:
        know_topics = 0
        for topic in topics:
            role, conf = dominant_role(score_conversation_roles(topic.text))
            if role == ConversationRole.COMMUNITY_WRAPPER and conf > 0.55:
                continue
            if score_answer_substance(topic.text) > 0.25:
                know_topics += 1
        if know_topics >= 3:
            return False
    structure = recover_forum_structure(text)
    ev = compute_semantic_evidence(text)
    profile = compute_structure_profile(text, evidence=ev)
    promo = ev.negative.get('promotional', 0.0)
    trans = ev.negative.get('transactional', 0.0)
    return (
        structure.wrapper_mass > 0.48
        and profile.explanation_ratio < 0.14
        and (
            promo + trans > 0.85
            or (promo > 0.45 and trans > 0.45)
        )
    )


def strip_discard_spans(text: str) -> str:
    structure = recover_forum_structure(text)
    if not structure.spans:
        return text.strip()

    kept: list[str] = []
    for span in structure.spans:
        if is_low_value_span(span.role, span.text, confidence=span.confidence):
            continue
        if span.role in KNOWLEDGE_ROLES:
            kept.append(span.text.strip())
        elif span.role in (ConversationRole.CONVERSATION, ConversationRole.COMMENT):
            if score_answer_substance(span.text) > 0.28:
                kept.append(span.text.strip())
        elif span.role == ConversationRole.UNKNOWN:
            if _structural_span_noise_ratio(span.text) > 0.10:
                continue
            raw = DocumentFeatureExtractor().extract(span.text)
            if raw.word_count <= 6 and raw.uppercase_token_ratio > 0.40:
                continue
            substance = score_answer_substance(span.text)
            ev = compute_semantic_evidence(span.text)
            if substance >= 0.10 or ev.utility >= 0.14:
                kept.append(span.text.strip())

    if not kept:
        return ''
    return '\n\n'.join(kept)
