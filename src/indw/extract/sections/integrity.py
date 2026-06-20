from __future__ import annotations

from dataclasses import dataclass, field

from indw.extract.roles.forum import (
    ConversationRole,
    DISCARD_ROLES,
    dominant_role,
    score_conversation_roles,
)
from indw.extract.sections.semantic import (
    analyze_completion_cached,
    boundary_before_orphan_quote,
    boundary_confidence_from_completion,
    continuation_incomplete,
    last_complete_boundary,
    lookahead_complete_within,
    repair_chunk_start,
    trim_to_complete_boundary,
)
from indw.extract.structure.analyze import analyze_structure
from indw.filter.refine.truncation import analyze_truncation

@dataclass
class ChunkIntegrityResult:
    completeness: float = 0.0
    boundary_confidence: float = 0.0
    quote_balanced: bool = True
    trimmed: bool = False
    rejected: bool = False
    chars_removed: int = 0
    reason: str = ''
    stage: str = 'finalize'

    def to_dict(self) -> dict[str, object]:
        return {
            'completeness': round(self.completeness, 4),
            'boundary_confidence': round(self.boundary_confidence, 4),
            'quote_balanced': self.quote_balanced,
            'trimmed': self.trimmed,
            'rejected': self.rejected,
            'chars_removed': self.chars_removed,
            'reason': self.reason,
            'stage': self.stage,
        }

def _quote_balance(text: str) -> tuple[bool, float]:
    score = 1.0
    balanced = True
    if text.count('"') % 2 != 0:
        balanced = False
        score = min(score, 0.35)
    pairs = (
        ('\u201c', '\u201d'),
        ('\u2018', '\u2019'),
        ('«', '»'),
    )
    for open_q, close_q in pairs:
        opens = text.count(open_q)
        closes = text.count(close_q)
        if opens != closes:
            balanced = False
            delta = abs(opens - closes)
            score = min(score, max(0.0, 1.0 - delta * 0.35))
    return balanced, score

def _terminal_boundary_score_impl(text: str) -> float:
    comp = analyze_completion_cached(text)
    return min(1.0, comp.overall * 0.60 + (1.0 - comp.incomplete_probability) * 0.40)


def _terminal_boundary_score(text: str) -> float:
    from indw.extract.core.context import get_document_context
    dctx = get_document_context()
    if dctx is not None:
        return dctx.terminal_boundary(text, lambda: _terminal_boundary_score_impl(text))
    return _terminal_boundary_score_impl(text)

def conversation_contamination_score(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    from indw.clean.artifact.engine import get_artifact_engine
    from indw.extract.roles.forum import (
        _structural_span_noise_ratio,
        recover_forum_structure,
    )
    engine = get_artifact_engine()
    ui_ratio = engine.ui_noise_ratio(text)
    span_noise = _structural_span_noise_ratio(text)
    forum = recover_forum_structure(text)
    scores = score_conversation_roles(text, position_ratio=0.5)
    role, conf = dominant_role(scores)
    ui_mass = sum(scores.get(r, 0.0) for r in (
        ConversationRole.FORUM_UI,
        ConversationRole.METADATA,
        ConversationRole.NAVIGATION,
        ConversationRole.CONVERSATION,
    ))
    discard_mass = sum(scores.get(r, 0.0) for r in DISCARD_ROLES)
    ev_role = 0.0
    if role in DISCARD_ROLES:
        ev_role = conf * 0.55
    base = ui_ratio * 0.55 + span_noise * 0.35
    if not forum.is_forum and span_noise < 0.06 and ui_ratio < 0.08:
        return min(1.0, base + ev_role * 0.20)
    return min(1.0, base + ui_mass * 0.22 + discard_mass * 0.10 + ev_role * 0.35)

def _strip_contamination(text: str) -> tuple[str, int]:
    from indw.extract.structure.inline import strip_inline_structural
    from indw.extract.roles.forum import strip_discard_spans

    original = len(text)
    t = strip_discard_spans(text) or text
    t, _ = strip_inline_structural(t, preserve_code_fences=True)
    t = t.strip()
    return t, max(0, original - len(t))

def score_chunk_integrity(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    structural = analyze_structure(text)
    trunc = analyze_truncation(text)
    comp = analyze_completion_cached(text)
    _, quote_score = _quote_balance(text)
    contam = conversation_contamination_score(text)
    inc_pen = max(0.0, comp.incomplete_probability - 0.35) * 0.25
    return min(1.0, (
        structural.sentence_completeness_mean * 0.20
        + comp.overall * 0.34
        + (1.0 - trunc.probability) * 0.16
        + (1.0 - comp.incomplete_probability) * 0.16
        + quote_score * 0.08
        + (1.0 - contam) * 0.06
        - inc_pen
    ))

def boundary_confidence(text: str) -> float:
    return boundary_confidence_from_completion(text)

def _trim_incomplete_paren_tail(text: str, *, min_chars: int) -> tuple[str, int]:
    t = text.strip()
    if not t:
        return t, 0
    last_paren = t.rfind('(')
    if last_paren < max(min_chars, int(len(t) * 0.25)):
        return t, 0
    tail = t[last_paren:]
    if ')' in tail:
        return t, 0
    head = t[:last_paren].strip()
    if len(head) < min_chars:
        return t, 0
    head_comp = analyze_completion_cached(head)
    tail_comp = analyze_completion_cached(tail)
    if (
        tail_comp.incomplete_probability >= 0.42
        and head_comp.incomplete_probability < tail_comp.incomplete_probability - 0.10
    ):
        return head, len(t) - len(head)
    return t, 0

def _trim_incomplete_tail(text: str) -> tuple[str, int]:
    t = text.strip()
    if not t:
        return t, 0
    original = len(t)
    comp = analyze_completion_cached(t)
    if comp.incomplete_probability < 0.38 and comp.overall >= 0.58:
        return t, 0
    trimmed, removed = _trim_incomplete_paren_tail(t, min_chars=max(30, int(original * 0.35)))
    if removed > 0:
        return trimmed, removed
    trimmed, removed = trim_to_complete_boundary(
        t,
        min_chars=max(30, int(original * 0.35)),
        min_retain_ratio=0.35,
        min_completion=0.56,
    )
    if removed > 0:
        return trimmed, removed
    return t, 0

def _retain_trimmed_tail(trimmed: str) -> str:
    t = trimmed.strip()
    if not t.endswith(','):
        return t
    stripped = t.rstrip(',').strip()
    if not stripped:
        return t
    if analyze_completion_cached(stripped).incomplete_probability > analyze_completion_cached(t).incomplete_probability + 0.04:
        return t
    return stripped

def finalize_semantic_unit(
    text: str,
    *,
    min_chars: int = 40,
    min_completeness: float = 0.58,
) -> tuple[str | None, ChunkIntegrityResult]:
    if not text or not text.strip():
        return None, ChunkIntegrityResult(rejected=True, reason='empty')

    t, head_removed = repair_chunk_start(text.strip())
    t = t.strip()
    if not t:
        return None, ChunkIntegrityResult(rejected=True, reason='empty_head')

    contam = conversation_contamination_score(t)
    if contam >= 0.42:
        stripped, stripped_removed = _strip_contamination(t)
        if stripped and len(stripped) >= min_chars and conversation_contamination_score(stripped) < contam:
            t = stripped
            head_removed += stripped_removed
            contam = conversation_contamination_score(t)
        elif contam >= 0.55:
            return None, ChunkIntegrityResult(
                rejected=True,
                reason='forum_ui_contamination',
                chars_removed=head_removed,
            )

    quote_balanced, _ = _quote_balance(t)
    before = score_chunk_integrity(t)
    comp = analyze_completion_cached(t)
    trunc = analyze_truncation(t)

    if not quote_balanced:
        boundary = boundary_before_orphan_quote(t)
        if boundary > max(40, int(len(t) * 0.20)) and boundary < len(t):
            head = t[:boundary].strip()
            if len(head) >= min_chars and score_chunk_integrity(head) >= min_completeness - 0.05:
                return head, ChunkIntegrityResult(
                    completeness=score_chunk_integrity(head),
                    boundary_confidence=boundary_confidence(head),
                    quote_balanced=True,
                    trimmed=True,
                    chars_removed=head_removed + (len(t) - len(head)),
                    reason='quote_head_kept',
                )

    from indw.filter.refine.truncation import _DANGLING_END
    last_w = t.split()[-1].strip('.,;:!?)]}').lower() if t.split() else ''

    if comp.incomplete_probability >= 0.38 or before < min_completeness:
        trimmed, removed = _trim_incomplete_tail(t)
        if removed > 0:
            after = score_chunk_integrity(trimmed)
            after_comp = analyze_completion_cached(trimmed)
            total_removed = head_removed + removed
            if after_comp.incomplete_probability >= 0.45 or after < min_completeness - 0.10:
                if (
                    removed >= 20
                    and len(trimmed) >= min_chars
                    and after >= min_completeness - 0.14
                    and after_comp.incomplete_probability <= comp.incomplete_probability + 0.02
                ):
                    t = _retain_trimmed_tail(trimmed)
                    before = score_chunk_integrity(t)
                    comp = analyze_completion_cached(t)
                    head_removed = total_removed
                else:
                    return None, ChunkIntegrityResult(
                        completeness=after,
                        boundary_confidence=boundary_confidence(trimmed),
                        quote_balanced=quote_balanced,
                        rejected=True,
                        reason='incomplete_semantic_unit',
                        chars_removed=total_removed,
                    )
            if len(trimmed) < min_chars:
                return None, ChunkIntegrityResult(
                    completeness=after,
                    rejected=True,
                    reason='trim_too_short',
                    chars_removed=total_removed,
                )
            t = _retain_trimmed_tail(trimmed)
            before = score_chunk_integrity(t)
            comp = analyze_completion_cached(t)
            head_removed = total_removed
        elif comp.incomplete_probability >= 0.38 or before < min_completeness:
            return None, ChunkIntegrityResult(
                completeness=before,
                boundary_confidence=boundary_confidence(t),
                quote_balanced=quote_balanced,
                rejected=True,
                reason='incomplete_terminal',
            )
    elif trunc.severity != 'none' and comp.incomplete_probability >= 0.30:
        trimmed, removed = _trim_incomplete_tail(t)
        if removed > 0:
            after = score_chunk_integrity(trimmed)
            after_comp = analyze_completion_cached(trimmed)
            total_removed = head_removed + removed
            if after_comp.incomplete_probability >= 0.38 or len(trimmed) < min_chars:
                return None, ChunkIntegrityResult(
                    completeness=after,
                    boundary_confidence=boundary_confidence(trimmed),
                    quote_balanced=quote_balanced,
                    rejected=True,
                    reason='incomplete_semantic_unit',
                    chars_removed=total_removed,
                )
            t = _retain_trimmed_tail(trimmed)
            before = score_chunk_integrity(t)
            comp = analyze_completion_cached(t)
            head_removed = total_removed
        elif comp.incomplete_probability >= 0.30:
            return None, ChunkIntegrityResult(
                completeness=before,
                boundary_confidence=boundary_confidence(t),
                quote_balanced=quote_balanced,
                rejected=True,
                reason='incomplete_terminal',
            )

    if not lookahead_complete_within(t):
        boundary = last_complete_boundary(t, min_chars=min_chars, min_completion=min_completeness - 0.04)
        if boundary > 0:
            candidate = t[:boundary].strip()
            if len(candidate) >= min_chars and continuation_incomplete(candidate) < comp.incomplete_probability:
                t = candidate
                before = score_chunk_integrity(t)
                comp = analyze_completion_cached(t)

    if trunc.severity == 'heavy' and comp.incomplete_probability >= 0.42:
        trimmed, removed = _trim_incomplete_tail(t)
        if removed > 0:
            after = score_chunk_integrity(trimmed)
            after_comp = analyze_completion_cached(trimmed)
            total_removed = head_removed + removed
            if after_comp.incomplete_probability >= 0.42 or len(trimmed) < min_chars or after < min_completeness:
                return None, ChunkIntegrityResult(
                    completeness=after,
                    boundary_confidence=boundary_confidence(trimmed),
                    quote_balanced=quote_balanced,
                    rejected=True,
                    reason='heavily_truncated',
                    chars_removed=total_removed,
                )
            t = _retain_trimmed_tail(trimmed)
            before = score_chunk_integrity(t)
            comp = analyze_completion_cached(t)
            head_removed = total_removed
        elif comp.incomplete_probability >= 0.42:
            return None, ChunkIntegrityResult(
                completeness=before,
                boundary_confidence=boundary_confidence(t),
                quote_balanced=quote_balanced,
                rejected=True,
                reason='heavily_truncated',
            )

    if last_w in _DANGLING_END:
        trimmed, removed = _trim_incomplete_tail(t)
        if removed > 0 and len(trimmed) >= min_chars:
            t = _retain_trimmed_tail(trimmed)
            before = score_chunk_integrity(t)
            comp = analyze_completion_cached(t)
            head_removed += removed
            last_w = t.split()[-1].strip('.,;:!?)]}').lower() if t.split() else ''

    incomplete_limit = 0.62 if head_removed > 0 else 0.38
    if comp.incomplete_probability >= incomplete_limit or conversation_contamination_score(t) >= 0.48:
        return None, ChunkIntegrityResult(
            completeness=before,
            boundary_confidence=boundary_confidence(t),
            quote_balanced=quote_balanced,
            rejected=True,
            reason='incomplete_terminal' if comp.incomplete_probability >= 0.38 else 'forum_ui_contamination',
        )

    if (
        trunc.severity != 'none'
        and comp.incomplete_probability >= 0.30
        and head_removed == 0
        and not t.rstrip().endswith(('.', '!', '?', '"', '\u201d', ')', ']', '}'))
    ):
        return None, ChunkIntegrityResult(
            completeness=before,
            boundary_confidence=boundary_confidence(t),
            quote_balanced=quote_balanced,
            rejected=True,
            reason='incomplete_terminal',
        )

    if len(t) < min_chars and before < 0.65:
        return None, ChunkIntegrityResult(
            completeness=before,
            rejected=True,
            reason='short_incomplete',
        )

    if not quote_balanced:
        trimmed, removed = _trim_incomplete_tail(t)
        if _quote_balance(trimmed)[0] and score_chunk_integrity(trimmed) >= min_completeness:
            return trimmed, ChunkIntegrityResult(
                completeness=score_chunk_integrity(trimmed),
                boundary_confidence=boundary_confidence(trimmed),
                quote_balanced=True,
                trimmed=True,
                chars_removed=head_removed + removed,
                reason='quote_repaired',
            )
        boundary = boundary_before_orphan_quote(t)
        if boundary > max(40, int(len(t) * 0.20)) and boundary < len(t):
            head = t[:boundary].strip()
            if len(head) >= min_chars and score_chunk_integrity(head) >= min_completeness - 0.05:
                return head, ChunkIntegrityResult(
                    completeness=score_chunk_integrity(head),
                    boundary_confidence=boundary_confidence(head),
                    quote_balanced=True,
                    trimmed=True,
                    chars_removed=head_removed + (len(t) - len(head)),
                    reason='quote_head_kept',
                )
        return None, ChunkIntegrityResult(
            completeness=before,
            quote_balanced=False,
            rejected=True,
            reason='orphaned_quote',
        )

    return t, ChunkIntegrityResult(
        completeness=before,
        boundary_confidence=boundary_confidence(t),
        quote_balanced=quote_balanced,
        trimmed=head_removed > 0,
        chars_removed=head_removed,
        reason='head_repaired' if head_removed else '',
    )

def integrity_trace(
    text: str,
    *,
    emitted: str | None = None,
    stage: str = '',
    recovered_boundary: int = -1,
) -> dict[str, object]:
    payload = emitted or text or ''
    comp = analyze_completion_cached(payload)
    trunc = analyze_truncation(payload)
    root_cause = 'none'
    if comp.incomplete_probability >= 0.38:
        root_cause = 'source_truncated' if stage in ('recovery', 'topic_split') else 'finalize_reject'
    elif conversation_contamination_score(payload) >= 0.45:
        root_cause = 'forum_ui'
    elif stage == 'post_mutate':
        root_cause = 'post_mutate'
    return {
        'original_chars': len(text or ''),
        'emitted_chars': len(emitted or ''),
        'integrity': score_chunk_integrity(payload),
        'boundary': boundary_confidence(payload),
        'completion': comp.to_dict(),
        'truncation': trunc.__dict__,
        'contamination': round(conversation_contamination_score(payload), 4),
        'recovered_boundary': recovered_boundary,
        'emitted_boundary': len(emitted or ''),
        'root_cause': root_cause,
        'stage': stage,
    }
