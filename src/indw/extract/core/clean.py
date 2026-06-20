from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from indw.clean.semantic.clean import clean_section_text
from indw.extract.sections.integrity import _quote_balance, _terminal_boundary_score_impl
from indw.extract.core.context import get_document_context
from indw.extract.roles.education import (
    educational_role_boundary,
    strip_instructional_scaffolding,
)
from indw.extract.roles.forum import (
    conversation_role_boundary,
    score_answer_substance,
    strip_discard_spans,
)
from indw.extract.structure.inline import strip_inline_structural
from indw.extract.roles.publication import (
    KNOWLEDGE_PUBLICATION_ROLES,
    decompose_publication_spans,
    is_scaffold_span,
    publication_role_boundary,
    score_publication_roles,
    strip_leading_publication_wrapper,
    strip_publication_scaffolding,
    strip_trailing_inline_scaffold,
)
from indw.extract.structure.analyze import analyze_structure
from indw.clean.semantic.section_artifacts import score_section_artifact
from indw.clean.artifact.evidence_cache import text_fingerprint
from indw.clean.artifact.evidence_engine import resolve_semantic_evidence


@dataclass
class UnitCleanScope:
    source_text: str
    role: str
    forum_strip: bool
    scaffold_stripped: str
    dctx: Any = field(default=None)
    _evidence: dict[tuple[int, bytes], Any] = field(default_factory=dict)
    _completion: dict[tuple[int, bytes], Any] = field(default_factory=dict)
    _terminal: dict[tuple[int, bytes], float] = field(default_factory=dict)
    _structure: dict[tuple[int, bytes], Any] = field(default_factory=dict)

    def evidence(self, text: str) -> Any:
        from indw.extract.core.profile import ke_record, ke_timed

        fp = text_fingerprint(text)
        nbytes = len(text.encode('utf-8', 'surrogatepass'))
        with ke_timed('clean_evidence', payload_bytes=nbytes):
            if fp is not None and fp in self._evidence:
                ke_record('clean_evidence', cache_hit=True, dedupe_key=fp)
                return self._evidence[fp]
            if self.dctx is not None:
                result = self.dctx.section_evidence(text, lambda: resolve_semantic_evidence(text))
            else:
                result = resolve_semantic_evidence(text)
            if fp is not None:
                self._evidence[fp] = result
                ke_record('clean_evidence', cache_hit=False, dedupe_key=fp)
            return result

    def completion(self, text: str) -> Any:
        from indw.extract.sections.semantic import analyze_completion_cached
        from indw.extract.core.profile import ke_record, ke_timed

        fp = text_fingerprint(text)
        nbytes = len(text.encode('utf-8', 'surrogatepass'))
        with ke_timed('clean_completion', payload_bytes=nbytes):
            if fp is not None and fp in self._completion:
                ke_record('clean_completion', cache_hit=True, dedupe_key=fp)
                return self._completion[fp]
            result = analyze_completion_cached(text)
            if fp is not None:
                self._completion[fp] = result
                ke_record('clean_completion', cache_hit=False, dedupe_key=fp)
            return result

    def terminal(self, text: str) -> float:
        from indw.extract.core.profile import ke_record, ke_timed

        fp = text_fingerprint(text)
        with ke_timed('clean_terminal'):
            if fp is not None and fp in self._terminal:
                ke_record('clean_terminal', cache_hit=True, dedupe_key=fp)
                return self._terminal[fp]
            if self.dctx is not None:
                score = self.dctx.terminal_boundary(text, lambda: _terminal_boundary_score_impl(text))
            else:
                score = _terminal_boundary_score_impl(text)
            if fp is not None:
                self._terminal[fp] = score
                ke_record('clean_terminal', cache_hit=False, dedupe_key=fp)
            return score

    def structure(self, text: str) -> Any:
        from indw.extract.core.profile import ke_record, ke_timed

        fp = text_fingerprint(text)
        with ke_timed('clean_structure'):
            if fp is not None and fp in self._structure:
                ke_record('clean_structure', cache_hit=True, dedupe_key=fp)
                return self._structure[fp]
            result = analyze_structure(text)
            if fp is not None:
                self._structure[fp] = result
                ke_record('clean_structure', cache_hit=False, dedupe_key=fp)
            return result


def _repair_attribution_breaks(text: str) -> str:
    t = text
    while '\n\n,' in t:
        t = t.replace('\n\n,', ',')
    while '\n\n;' in t:
        t = t.replace('\n\n;', ';')
    return t


def _is_code_payload(text: str, *, role: str = 'body') -> bool:
    if role == 'code':
        return True
    t = text.strip()
    if not t:
        return False
    lines = [ln for ln in t.splitlines() if ln.strip()]
    code_lines = sum(
        1 for ln in lines
        if ln.lstrip().startswith(('import ', 'from ', 'def ', 'class ', 'print ', 'bus ='))
        or '=' in ln and not ln.strip().endswith('?')
    )
    return code_lines >= 2 and code_lines / max(len(lines), 1) >= 0.45


def run_clean_unit(
    text: str,
    *,
    role: str = 'body',
    forum_strip: bool = False,
    scaffold_stripped: str = '',
) -> str:
    from indw.extract.core.profile import (
        active_unit_assembly_profile,
        ke_timed,
        record_clean_unit_section,
    )
    from indw.schedule.monitor.budget import doc_budget_exceeded

    if doc_budget_exceeded():
        return text.strip()

    fp = text_fingerprint(text)
    nbytes = len(text.encode('utf-8', 'surrogatepass'))
    t0 = time.perf_counter()
    scope = UnitCleanScope(
        source_text=text,
        role=role,
        forum_strip=forum_strip,
        scaffold_stripped=scaffold_stripped,
        dctx=get_document_context(),
    )

    with ke_timed('clean_unit_total', payload_bytes=nbytes):
        result = _clean_unit_impl(scope, text, role=role, forum_strip=forum_strip, scaffold_stripped=scaffold_stripped)

    record_clean_unit_section(
        wall_sec=time.perf_counter() - t0,
        fingerprint=fp,
        chars=nbytes,
    )
    return result


def _clean_unit_impl(
    scope: UnitCleanScope,
    text: str,
    *,
    role: str,
    forum_strip: bool,
    scaffold_stripped: str,
) -> str:
    from indw.extract.core.profile import ke_timed
    from indw.schedule.monitor.budget import doc_budget_exceeded

    _ev = scope.evidence
    _completion = scope.completion
    _terminal = scope.terminal
    _structure = scope.structure

    t = _repair_attribution_breaks(text.lstrip())
    while t and t[0] in '×✕✖':
        t = t[1:].lstrip()
    if forum_strip:
        with ke_timed('clean_forum_strip'):
            t = strip_discard_spans(t) or t
    elif _is_code_payload(t, role=role):
        with ke_timed('clean_code_path'):
            t, _ = strip_inline_structural(t, preserve_code_fences=True)
            out, _ = clean_section_text(t, role='code', preserve_educational=True)
        return (out or t).strip()

    with ke_timed('clean_publication_lead'):
        head_probe = t[:min(len(t), 240)]
        pub_head = score_publication_roles(head_probe, position_ratio=0.05)
        if pub_head.scaffold_mass() >= 0.12 or pub_head.author_block > 0.10:
            lead = strip_leading_publication_wrapper(t)
        else:
            lead = ''
        if lead and lead != t and len(lead.split()) >= 6:
            removed = t[:t.find(lead)].strip() if lead in t else ''
            apply_lead = False
            if removed:
                left_pub = score_publication_roles(removed, position_ratio=0.05)
                left_ev = _ev(removed)
                right_ev = _ev(lead)
                if (
                    ' said:' in removed.lower()
                    and lead.lstrip()[:1] in ('"', '\u201c', "'")
                    and left_pub.author_block + left_pub.metadata >= 0.30
                ):
                    apply_lead = False
                elif (
                    left_pub.scaffold_mass() >= 0.48
                    and score_answer_substance(removed) < 0.22
                    and left_ev.utility < max(0.16, right_ev.utility * 0.85)
                    and left_ev.quality.educational < 0.08
                ):
                    term_lead = _terminal(lead)
                    term_t = _terminal(t)
                    apply_lead = term_lead >= max(0.52, term_t * 0.85)
                elif (
                    left_pub.scaffold_mass() >= 0.42
                    and right_ev.utility > left_ev.utility * 1.05
                    and right_ev.quality.educational >= left_ev.quality.educational * 0.80
                    and _terminal(lead) >= max(0.52, _terminal(t) * 0.82)
                ):
                    apply_lead = True
            if apply_lead:
                t = lead

    dctx = scope.dctx
    same_span = t.strip() == text.strip()
    cached_scaffold = ''
    if same_span:
        cached_scaffold = scaffold_stripped or (dctx.scaffold_stripped_for(text) if dctx else '')
    with ke_timed('clean_scaffold_strip'):
        pub_candidate = cached_scaffold or strip_publication_scaffolding(t)
    if dctx and pub_candidate and same_span and not cached_scaffold:
        dctx.remember_scaffold_stripped(text, pub_candidate)
    if pub_candidate and pub_candidate != t:
        pub_span = score_publication_roles(t, position_ratio=0.5)
        with ke_timed('clean_pub_decompose'):
            if dctx is not None:
                spans = dctx.publication_spans(t, lambda: decompose_publication_spans(t))
            else:
                spans = decompose_publication_spans(t)
        has_scaffold_split = (
            len(spans) >= 2
            and any(is_scaffold_span(s.role, s.text, confidence=s.confidence) for s in spans)
            and any(
                s.role in KNOWLEDGE_PUBLICATION_ROLES or s.scores.get('knowledge', 0) > 0.15
                for s in spans
            )
        )
        cand_words = set(pub_candidate.lower().split())
        orig_words = set(t.lower().split())
        overlap = len(cand_words & orig_words) / max(len(orig_words), 1)
        cand_recall = len(cand_words & orig_words) / max(len(cand_words), 1)
        keep_pub = False
        term_t = _terminal(t)
        term_pub = _terminal(pub_candidate)
        if has_scaffold_split:
            keep_pub = (
                cand_recall >= 0.92
                and len(pub_candidate.split()) >= max(8, int(len(t.split()) * 0.25))
                and term_pub >= max(0.55, term_t * 0.90)
            )
        elif pub_span.scaffold_mass() > 0.52:
            keep_pub = (
                overlap >= 0.55
                and len(pub_candidate.split()) >= max(8, int(len(t.split()) * 0.55))
                and term_pub >= max(0.55, term_t * 0.90)
            )
        if pub_candidate.split()[:1] != t.split()[:1]:
            keep_pub = keep_pub and (
                has_scaffold_split or pub_span.scaffold_mass() > 0.64
            )
        orig_comp = None
        cand_comp = None
        if not keep_pub and pub_candidate and pub_candidate != t:
            orig_comp = _completion(t)
            cand_comp = _completion(pub_candidate)
            if (
                pub_candidate.count(':') < t.count(':') - 1
                and len(pub_candidate.split()) >= 8
                and term_pub >= term_t * 0.82
                and cand_comp.incomplete_probability <= orig_comp.incomplete_probability + 0.05
            ):
                keep_pub = True
        if keep_pub:
            if orig_comp is None:
                orig_comp = _completion(t)
                cand_comp = _completion(pub_candidate)
            scaffold_trim = pub_candidate.count(':') < t.count(':') - 1
            if not scaffold_trim and (
                cand_comp.incomplete_probability > orig_comp.incomplete_probability + 0.01
                or cand_comp.overall < orig_comp.overall
                or (
                    t.rstrip().endswith(('.', '!', '?'))
                    and not pub_candidate.rstrip().endswith(('.', '!', '?'))
                )
            ):
                keep_pub = False
        if keep_pub:
            t = pub_candidate
    if '\n\n' not in t:
        with ke_timed('clean_inline_scaffold'):
            inline = strip_trailing_inline_scaffold(t)
        if inline and inline != t and len(inline.split()) >= 6:
            orig_comp = _completion(t)
            inline_comp = _completion(inline)
            term_inline = _terminal(inline)
            if (
                inline_comp.incomplete_probability <= orig_comp.incomplete_probability + 0.01
                and inline_comp.overall >= orig_comp.overall
                and (not t.rstrip().endswith(('.', '!', '?')) or inline.rstrip().endswith(('.', '!', '?')))
                and term_inline >= max(0.55, _terminal(t) * 0.90)
            ):
                t = inline
    with ke_timed('clean_edu_scaffold'):
        edu_candidate = strip_instructional_scaffolding(t)
    if edu_candidate and edu_candidate != t:
        pre_bal, _ = _quote_balance(t)
        post_bal, _ = _quote_balance(edu_candidate)
        pre_paren = t.count('(') == t.count(')')
        post_paren = edu_candidate.count('(') == edu_candidate.count(')')
        if pre_bal and not post_bal:
            edu_candidate = t
        elif pre_paren and not post_paren:
            edu_candidate = t
        elif _ev(edu_candidate).utility < _ev(t).utility * 0.75:
            edu_candidate = t
    t = edu_candidate or t

    with ke_timed('clean_inline_structural'):
        t, _ = strip_inline_structural(t, preserve_code_fences=True)
    if forum_strip:
        t = strip_discard_spans(t) or t
    with ke_timed('clean_section_text'):
        out, _ = clean_section_text(t, role=role, preserve_educational=True)
    cleaned = (out or t).strip()
    if role in ('body', 'introduction', 'title') and '\n\n' in cleaned:
        with ke_timed('clean_para_trim'):
            paras = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
            while len(paras) > 1:
                if doc_budget_exceeded():
                    break
                head = '\n\n'.join(paras[:-1])
                tail = paras[-1]
                ev_head = _ev(head)
                ev_tail = _ev(tail)
                role_cut = max(
                    educational_role_boundary(head, tail, left_pos=0.32, right_pos=0.82),
                    publication_role_boundary(head, tail, left_pos=0.32, right_pos=0.82),
                    conversation_role_boundary(head, tail, left_pos=0.32, right_pos=0.82),
                )
                if role_cut >= 0.30 and ev_tail.utility < max(0.12, ev_head.utility * 0.82):
                    paras.pop()
                    continue
                art_tail = score_section_artifact(tail, position_ratio=0.85, section_role='promotional')
                if ev_tail.utility >= 0.10 and len(tail.split()) >= 10:
                    break
                tail_struct = _structure(tail)
                if (
                    tail.strip()
                    and tail.strip()[-1] in '.!?)"\'»]})'
                    and tail_struct.sentence_completeness_mean >= 0.80
                    and len(tail.split()) >= 5
                ):
                    break
                if ev_tail.utility < 0.14 and (art_tail.promotional > 0.26 or art_tail.navigation > 0.35):
                    paras.pop()
                    continue
                break
            cleaned = '\n\n'.join(paras)
    return _repair_attribution_breaks(cleaned.strip())
