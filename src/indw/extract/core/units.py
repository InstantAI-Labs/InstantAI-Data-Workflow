from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from indw.clean.document.config import CleaningConfig
from indw.clean.document.stats import KnowledgeExtractionStats
from indw.clean.semantic.clean import clean_section_text
from indw.extract.roles.units import extract_forum_units
from indw.extract.roles.forum import (
    ConversationRole,
    conversation_role_boundary,
    detect_forum_document,
    dominant_role,
    is_community_wrapper_document,
    recover_forum_structure,
    score_answer_substance,
    score_conversation_roles,
    strip_discard_spans,
)
from indw.extract.roles.publication import (
    is_pagination_footer_line,
    publication_role_boundary,
    score_publication_roles,
    strip_publication_scaffolding,
    decompose_publication_spans,
    is_scaffold_span,
    KNOWLEDGE_PUBLICATION_ROLES,
    strip_trailing_inline_scaffold,
    strip_leading_publication_wrapper,
)
from indw.clean.document.normalize import normalize_text
from indw.extract.sections.integrity import finalize_semantic_unit, _terminal_boundary_score, _quote_balance
from indw.extract.structure.inline import strip_inline_structural
from indw.extract.roles.education import (
    educational_role_boundary,
    score_educational_roles,
    strip_instructional_scaffolding,
)
from indw.extract.assess.metrics import KnowledgePageMetrics, compute_page_metrics
from indw.extract.structure.document import recover_document_structure
from indw.extract.structure.recovery import RecoveredSection
from indw.extract.sections.classify import (
    DISCARD_CLASSES,
    KnowledgeSectionClass,
    PRIMARY_CLASSES,
    SectionClassification,
    _WRAPPER_PREV,
    classify_section,
    document_is_mixed,
)
from indw.extract.nav.context import (
    NavigationContext,
    get_navigation_context,
    set_navigation_context,
)
from indw.extract.structure.aggregate import (
    AggregationContext,
    analyze_aggregation,
    get_aggregation_context,
    segment_topics,
    set_aggregation_context,
    trim_structural_tail,
)
from indw.extract.sections.boundaries import _decompose_spans
from indw.extract.sections.quality import SectionQualityScore, assess_section_quality
from indw.extract.structure.analyze import analyze_structure
from indw.clean.semantic.section_artifacts import score_section_artifact
from indw.clean.artifact.evidence_engine import resolve_semantic_evidence


@dataclass
class KnowledgeUnit:
    text: str
    section_class: str = 'unknown'
    retention_score: float = 0.0
    artifact_score: float = 0.0
    chunk_index: int = 0
    source_kind: str = 'section'
    start: int = 0
    end: int = 0
    confidence: float = 0.0
    coherence_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'section_class': self.section_class,
            'retention_score': round(self.retention_score, 4),
            'artifact_score': round(self.artifact_score, 4),
            'source_kind': self.source_kind,
            'chars': len(self.text),
            'start': self.start,
            'end': self.end,
            'confidence': round(self.confidence, 4),
            'coherence_score': round(self.coherence_score, 4),
        }


@dataclass
class KnowledgeExtractionResult:
    units: list[KnowledgeUnit] = field(default_factory=list)
    metrics: KnowledgePageMetrics = field(default_factory=KnowledgePageMetrics)
    mixed: bool = False
    dropped_all: bool = False
    drop_reason: str = ''
    _scored: list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]] | None = field(
        default=None, repr=False, compare=False,
    )
    _sections_cache: list[dict[str, Any]] | None = field(default=None, repr=False, compare=False)
    ke_profile: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def _build_sections(self) -> list[dict[str, Any]]:
        if self._sections_cache is not None:
            return self._sections_cache
        if not self._scored:
            self._sections_cache = []
        else:
            from indw.extract.core.profile import ke_record, ke_timed
            with ke_timed('sections_dict_build', object_count=len(self._scored)):
                self._sections_cache = [
                    {**s.to_dict(), **c.to_dict(), **q.to_dict()}
                    for s, c, q in self._scored
                ]
            ke_record('sections_dict_build', object_count=len(self._sections_cache or []))
        return self._sections_cache or []

    @property
    def sections(self) -> list[dict[str, Any]]:
        return self._build_sections()

    def to_dict(self) -> dict[str, Any]:
        return {
            'units': [u.to_dict() for u in self.units],
            'metrics': self.metrics.to_dict(),
            'mixed': self.mixed,
            'dropped_all': self.dropped_all,
            'drop_reason': self.drop_reason,
            'sections': self._build_sections(),
            'ke_profile': self.ke_profile,
        }


def _code_tail(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    start = None
    for marker in ('import ', 'from ', 'def ', 'class '):
        idx = t.find(marker)
        if idx >= 0 and (start is None or idx < start):
            start = idx
    if start is not None and start > 0:
        return t[start:].strip()
    return t


def _merge_code_sections(
    scored: list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]],
) -> str:
    parts: list[str] = []
    for sec, _, _ in scored:
        t = _code_tail(sec.text)
        if (
            sec.in_fence
            or sec.structural_role == 'code'
            or 'import ' in t
            or 'def ' in t
            or 'print ' in t
        ):
            t = t.strip()
            if t and t not in parts:
                parts.append(t)
    return '\n'.join(parts)


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


def _analyze_completion_cached(text: str):
    from indw.extract.sections.semantic import analyze_completion_cached
    return analyze_completion_cached(text)


def _clean_unit(
    text: str,
    *,
    role: str = 'body',
    forum_strip: bool = False,
    scaffold_stripped: str = '',
) -> str:
    from indw.extract.core.context import get_document_context
    from indw.extract.core.clean import run_clean_unit

    dctx = get_document_context()
    if dctx is not None:
        return dctx.clean_unit(
            text,
            role=role,
            forum_strip=forum_strip,
            scaffold_stripped=scaffold_stripped,
            compute=lambda: run_clean_unit(
                text, role=role, forum_strip=forum_strip, scaffold_stripped=scaffold_stripped,
            ),
        )
    return run_clean_unit(
        text, role=role, forum_strip=forum_strip, scaffold_stripped=scaffold_stripped,
    )


def _emit_unit_text(
    text: str,
    *,
    role: str = 'body',
    forum_strip: bool = False,
    min_chars: int = 40,
    preserve_code_fences: bool = True,
    scaffold_stripped: str = '',
) -> str | None:
    from indw.extract.core.context import get_document_context
    from indw.extract.core.profile import ke_timed

    def _build() -> str | None:
        with ke_timed('emit_clean_unit', payload_bytes=len(text.encode('utf-8', 'surrogatepass'))):
            cleaned = _clean_unit(
                text, role=role, forum_strip=forum_strip, scaffold_stripped=scaffold_stripped,
            )
        if not cleaned:
            return None
        with ke_timed('emit_normalize'):
            cleaned = normalize_text(cleaned, preserve_code_fences=preserve_code_fences)
        if not cleaned:
            return None
        with ke_timed('emit_finalize'):
            finalized, result = finalize_semantic_unit(cleaned, min_chars=min_chars)
        if result.rejected or not finalized:
            return None
        return finalized

    dctx = get_document_context()
    if dctx is not None:
        return dctx.emit_unit(
            text,
            role=role,
            forum_strip=forum_strip,
            min_chars=min_chars,
            preserve_code_fences=preserve_code_fences,
            scaffold_stripped=scaffold_stripped,
            compute=_build,
        )
    return _build()


def _forum_to_units(
    forum_units,
    *,
    start_index: int,
) -> list[KnowledgeUnit]:
    out: list[KnowledgeUnit] = []
    pending: list[tuple[ForumUnit, str | None]] = []
    for i, fu in enumerate(forum_units):
        if fu.kind == 'answer':
            primary_text = fu.text
        elif fu.kind == 'answer_extra' and primary_text:
            frag = fu.text.strip()
            if (
                frag.split()[0][:1].islower()
                and score_answer_substance(frag) < score_answer_substance(primary_text) * 0.92
            ):
                continue
        cleaned = _emit_unit_text(fu.text, role='body', forum_strip=True)
        if not cleaned:
            continue
        pending.append((fu, cleaned))

    if (
        len(pending) >= 2
        and pending[0][0].kind == 'question'
        and pending[1][0].kind == 'answer'
        and pending[0][1].strip().lower() == pending[1][1].strip().lower()
    ):
        merged = pending[0][1]
        parts = [p.strip() for p in merged.split('\n') if p.strip()]
        if len(parts) <= 1:
            parts = [p.strip() for p in merged.replace('?', '?\n').split('\n') if p.strip()]
        best = max(parts, key=score_answer_substance, default=merged)
        cleaned = _emit_unit_text(best, role='body', forum_strip=True)
        if cleaned:
            out.append(KnowledgeUnit(
                text=cleaned,
                section_class='answer',
                retention_score=score_answer_substance(cleaned),
                chunk_index=start_index,
                source_kind='forum',
            ))
        return out

    for i, (fu, cleaned) in enumerate(pending):
        out.append(KnowledgeUnit(
            text=cleaned,
            section_class=fu.kind,
            retention_score=fu.score,
            chunk_index=start_index + i,
            source_kind='forum',
        ))
    return out


def _section_to_unit(
    sec: RecoveredSection,
    cls: SectionClassification,
    qual: SectionQualityScore,
    *,
    chunk_index: int,
    agg_ctx: AggregationContext | None = None,
    forum_strip: bool = False,
    preserve_code_fences: bool = True,
) -> KnowledgeUnit | None:
    role = sec.structural_role if sec.structural_role in ('body', 'introduction', 'title', 'code', 'table') else 'body'
    raw_text = sec.text
    agg_ctx = agg_ctx or get_aggregation_context()
    if agg_ctx.is_aggregated():
        raw_text = trim_structural_tail(raw_text)
    cleaned = _emit_unit_text(
        raw_text, role=role, min_chars=40, forum_strip=forum_strip,
        preserve_code_fences=preserve_code_fences,
        scaffold_stripped=qual.scaffold_stripped,
    )
    if not cleaned:
        return None
    return KnowledgeUnit(
        text=cleaned,
        section_class=cls.label.value,
        retention_score=qual.retention_score,
        artifact_score=qual.artifact_score,
        chunk_index=chunk_index,
        source_kind='section',
        start=sec.start,
        end=sec.end,
        confidence=cls.confidence,
        coherence_score=qual.coherence,
    )


def extract_knowledge(
    text: str,
    *,
    cfg: CleaningConfig,
    row: Optional[dict[str, Any]] = None,
    source: str = '',
    nav_ctx: NavigationContext | None = None,
    agg_ctx: AggregationContext | None = None,
    ke_stats: KnowledgeExtractionStats | None = None,
) -> KnowledgeExtractionResult:
    nav_token: NavigationContext | None = None
    agg_token: AggregationContext | None = None
    if nav_ctx is not None:
        nav_token = get_navigation_context()
        set_navigation_context(nav_ctx)
    if agg_ctx is not None:
        agg_token = get_aggregation_context()
        set_aggregation_context(agg_ctx)
    try:
        from indw.extract.core.profile import ke_profile_enabled, ke_profile_session

        if ke_profile_enabled():
            from indw.extract.core.profile import (
                active_unit_assembly_profile,
                ke_profile_session,
            )
            with ke_profile_session() as prof:
                result = _extract_knowledge_impl(
                    text,
                    cfg=cfg,
                    row=row,
                    nav_ctx=nav_ctx or get_navigation_context(),
                    agg_ctx=agg_ctx or get_aggregation_context(),
                    ke_stats=ke_stats,
                )
                payload = prof.to_dict()
                asm = active_unit_assembly_profile()
                if asm is not None:
                    payload['unit_assembly'] = asm.to_dict()
                result.ke_profile = payload
                return result
        return _extract_knowledge_impl(
            text,
            cfg=cfg,
            row=row,
            nav_ctx=nav_ctx or get_navigation_context(),
            agg_ctx=agg_ctx or get_aggregation_context(),
            ke_stats=ke_stats,
        )
    finally:
        if nav_ctx is not None:
            set_navigation_context(nav_token)
        if agg_ctx is not None:
            set_aggregation_context(agg_token)


_PUBLICATION_SALVAGE_CLASSES = frozenset({
    KnowledgeSectionClass.ARCHIVE,
    KnowledgeSectionClass.METADATA,
    KnowledgeSectionClass.NAVIGATION,
    KnowledgeSectionClass.NEWSLETTER,
    KnowledgeSectionClass.EVENT,
})


def _is_educational_worksheet(
    scored: list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]],
    *,
    forum_structure,
) -> bool:
    prompts = sum(
        1 for _, c, _ in scored
        if c.label == KnowledgeSectionClass.DISCUSSION_PROMPT
    )
    tasks = sum(
        1 for _, c, _ in scored
        if c.label in (KnowledgeSectionClass.INSTRUCTION, KnowledgeSectionClass.ASSIGNMENT)
    )
    know = sum(
        1 for _, c, q in scored
        if c.label in (
            KnowledgeSectionClass.EDUCATIONAL,
            KnowledgeSectionClass.ARTICLE,
            KnowledgeSectionClass.SCIENTIFIC,
            KnowledgeSectionClass.MEDICAL,
            KnowledgeSectionClass.GOVERNMENT,
        )
        and q.retention_score > 0.08
    )
    if know < 1:
        if prompts >= 2:
            return True
        if prompts >= 1 and tasks >= 1:
            return True
        if prompts >= 1 and all(
            c.label in (
                KnowledgeSectionClass.DISCUSSION_PROMPT,
                KnowledgeSectionClass.INSTRUCTION,
                KnowledgeSectionClass.ASSIGNMENT,
            )
            for _, c, _ in scored
        ):
            return True
        return False
    if forum_structure.wrapper_mass >= 0.20:
        return False
    if prompts >= 2:
        return True
    if prompts >= 1 and tasks >= 1:
        return True
    return False


def _join_merged_section_text(left: str, right: str) -> str:
    lt = left.rstrip()
    rt = right.lstrip()
    if rt[:1] in ',;':
        return f'{lt}{rt}'
    if rt[:2] in ("''", '""'):
        return f'{lt} {rt}'
    if rt[:1] in ('"', '\u201c', "'") and lt.endswith((':', '.', '!', '?', '"', '\u201d', ')', ']')):
        return f'{lt} {rt}'
    return f'{lt}\n\n{rt}'


def _repair_attribution_breaks(text: str) -> str:
    t = text
    while '\n\n,' in t:
        t = t.replace('\n\n,', ',')
    while '\n\n;' in t:
        t = t.replace('\n\n;', ';')
    return t


def _merge_attribution_quotes(
    scored: list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]],
) -> list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]]:
    if len(scored) < 2:
        return scored
    merged: list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]] = []
    for sec, cls, qual in scored:
        if merged:
            prev_sec, prev_cls, prev_qual = merged[-1]
            pt = prev_sec.text.rstrip()
            st = sec.text.lstrip()
            st_low = st.lower()
            attr_cont = (
                len(pt.split()) >= 4
                and (
                    st_low.startswith(', said')
                    or st_low.startswith(' said ')
                    or st_low.startswith('; said')
                )
                and pt[-1] in '.)]}"\u201d\''
            )
            if (
                ' said:' in pt.lower()
                and pt.endswith(':')
                and st[:1] in ('"', '\u201c', "'")
            ) or attr_cont:
                joiner = '' if st[:1] == ',' else ' '
                combined = RecoveredSection(
                    text=f'{pt}{joiner}{st}' if attr_cont else f'{pt} {st}',
                    start=prev_sec.start,
                    end=sec.end,
                    position_ratio=(prev_sec.position_ratio + sec.position_ratio) / 2,
                    structural_role=prev_sec.structural_role,
                    layout_kind=sec.layout_kind,
                    in_fence=sec.in_fence,
                )
                keep_cls = prev_cls if prev_qual.retention_score >= qual.retention_score else cls
                keep_qual = prev_qual if prev_qual.retention_score >= qual.retention_score else qual
                if not keep_qual.keep:
                    keep_qual = SectionQualityScore(
                        **{
                            **keep_qual.__dict__,
                            'keep': True,
                            'retention_score': max(prev_qual.retention_score, qual.retention_score, 0.12),
                        }
                    )
                merged[-1] = (combined, keep_cls, keep_qual)
                continue
        merged.append((sec, cls, qual))
    return merged


def _merge_adjacent_primary(
    scored: list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]],
    *,
    agg_ctx: AggregationContext | None = None,
) -> list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]]:
    if not scored:
        return scored
    agg_ctx = agg_ctx or get_aggregation_context()
    if agg_ctx.is_aggregated() or (agg_ctx.profile and agg_ctx.profile.is_aggregated):
        return scored
    if len(scored) > 4:
        return scored
    merged: list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]] = []
    for sec, cls, qual in scored:
        if (
            merged
            and cls.label in PRIMARY_CLASSES
            and merged[-1][1].label in PRIMARY_CLASSES
            and cls.label not in (KnowledgeSectionClass.QUESTION, KnowledgeSectionClass.ANSWER)
            and merged[-1][1].label not in (KnowledgeSectionClass.QUESTION, KnowledgeSectionClass.ANSWER)
        ):
            prev_sec, prev_cls, prev_qual = merged[-1]
            if prev_cls.label in DISCARD_CLASSES:
                merged.append((sec, cls, qual))
                continue
            if publication_role_boundary(
                prev_sec.text, sec.text,
                left_pos=prev_sec.position_ratio,
                right_pos=sec.position_ratio,
            ) >= 0.38:
                merged.append((sec, cls, qual))
                continue
            if educational_role_boundary(
                prev_sec.text, sec.text,
                left_pos=prev_sec.position_ratio,
                right_pos=sec.position_ratio,
            ) >= 0.32:
                merged.append((sec, cls, qual))
                continue
            if conversation_role_boundary(
                prev_sec.text, sec.text,
                left_pos=prev_sec.position_ratio,
                right_pos=sec.position_ratio,
            ) >= 0.35:
                merged.append((sec, cls, qual))
                continue
            pair_keep = (qual.keep and prev_qual.keep) or (
                qual.keep and prev_qual.retention_score > 0.08
            ) or (
                prev_qual.keep and qual.retention_score > 0.08
            )
            if not pair_keep:
                merged.append((sec, cls, qual))
                continue
            combined = RecoveredSection(
                text=_join_merged_section_text(prev_sec.text, sec.text),
                start=prev_sec.start,
                end=sec.end,
                position_ratio=(prev_sec.position_ratio + sec.position_ratio) / 2,
                structural_role=prev_sec.structural_role,
                layout_kind=prev_sec.layout_kind,
                in_fence=prev_sec.in_fence,
            )
            keep_cls = prev_cls if prev_qual.retention_score >= qual.retention_score else cls
            keep_qual = prev_qual if prev_qual.retention_score >= qual.retention_score else qual
            if not keep_qual.keep and (prev_qual.retention_score > 0.08 or qual.retention_score > 0.08):
                keep_qual = SectionQualityScore(
                    **{**keep_qual.__dict__, 'keep': True, 'retention_score': max(prev_qual.retention_score, qual.retention_score)}
                )
            merged[-1] = (combined, keep_cls, keep_qual)
        else:
            merged.append((sec, cls, qual))
    return merged


def _extract_knowledge_impl(
    text: str,
    *,
    cfg: CleaningConfig,
    row: Optional[dict[str, Any]],
    nav_ctx: NavigationContext,
    agg_ctx: AggregationContext,
    ke_stats: KnowledgeExtractionStats | None = None,
) -> KnowledgeExtractionResult:
    ks = ke_stats

    def _timed(stage: str):
        from contextlib import contextmanager
        from indw.schedule.monitor.doc import monitored_stage

        if ks is None:
            @contextmanager
            def _mon_only() -> Iterator[None]:
                with monitored_stage(f'ke_{stage}'):
                    yield
            return _mon_only()

        stage_stat = getattr(ks, stage)

        @contextmanager
        def _both() -> Iterator[None]:
            with stage_stat.timed():
                with monitored_stage(f'ke_{stage}'):
                    yield
        return _both()

    from indw.schedule.monitor.budget import doc_budget_exceeded

    def _budget_drop() -> KnowledgeExtractionResult | None:
        if doc_budget_exceeded():
            return KnowledgeExtractionResult(
                dropped_all=True, drop_reason='document_budget_exceeded',
            )
        return None

    if not text or not text.strip():
        return KnowledgeExtractionResult(dropped_all=True, drop_reason='empty')

    if is_community_wrapper_document(text):
        return KnowledgeExtractionResult(dropped_all=True, drop_reason='community_wrapper')

    min_chars = max(40, cfg.min_chars_after_clean // 4)
    pre_forum = detect_forum_document(text)
    forum_structure = recover_forum_structure(text)
    with _timed('structure_recovery'):
        snap = recover_document_structure(
            text, min_section_chars=min_chars, pre_forum=pre_forum,
        )
        if drop := _budget_drop():
            return drop
        sections_raw = snap.sections
        topic_split = snap.topic_split
    if not sections_raw:
        return KnowledgeExtractionResult(dropped_all=True, drop_reason='no_structure')

    with _timed('aggregation'):
        agg_profile = analyze_aggregation(
            text, sections_raw, ctx=agg_ctx, topic_split=topic_split,
        )
        if pre_forum or forum_structure.is_forum:
            agg_profile.is_aggregated = False
            agg_profile.is_headline_index = False
        agg_ctx.profile = agg_profile

    classified: list[tuple[RecoveredSection, SectionClassification, bool, Any]] = []
    prev_label: KnowledgeSectionClass | None = None
    seen_wrapper_prefix = False
    seen_primary = False
    with _timed('section_classify'):
        from indw.extract.sections.scratch import build_section_analysis
        for s in sections_raw:
            if drop := _budget_drop():
                return drop
            post_wrapper = seen_wrapper_prefix and not seen_primary
            analysis = build_section_analysis(s.text)
            cls = classify_section(
                s, nav_ctx=nav_ctx, agg_ctx=agg_ctx, prev_label=prev_label,
                wrapper_seen=post_wrapper, analysis=analysis,
            )
            classified.append((s, cls, post_wrapper, analysis))
            if cls.label in PRIMARY_CLASSES and s.structural_role == 'body':
                seen_primary = True
            elif cls.label in _WRAPPER_PREV or (
                cls.label == KnowledgeSectionClass.NAVIGATION and s.position_ratio < 0.32
            ):
                seen_wrapper_prefix = True
            prev_label = cls.label
    mixed = document_is_mixed([(s, c) for s, c, _, _ in classified]) or len(classified) >= 3 or agg_profile.is_aggregated
    if not mixed and len(classified) >= 2:
        labels = {c.label for _, c, _, _ in classified}
        if labels & set(PRIMARY_CLASSES) and labels & set(DISCARD_CLASSES):
            mixed = True
        elif any(pw for _, _, pw, _ in classified) and len(classified) >= 2:
            mixed = True

    scored: list[tuple[RecoveredSection, SectionClassification, SectionQualityScore]] = []
    with _timed('section_quality'):
        for sec, cls, post_wrapper, analysis in classified:
            if drop := _budget_drop():
                return drop
            qual = assess_section_quality(
                sec, cls, mixed_document=mixed, nav_ctx=nav_ctx, agg_ctx=agg_ctx,
                post_wrapper=post_wrapper, analysis=analysis,
            )
            scored.append((sec, cls, qual))

    with _timed('boundary_role'):
        scored = _merge_attribution_quotes(_merge_adjacent_primary(scored, agg_ctx=agg_ctx))

    has_code_body = any(
        sec.in_fence
        or sec.structural_role == 'code'
        or 'import ' in sec.text
        or 'def ' in sec.text
        for sec, _, _ in scored
    )

    section_labels = {c.label.value for _, c, _ in scored}
    forum_signal = (
        cfg.extract_qa
        and detect_forum_document(text, section_labels=section_labels)
        and forum_structure.has_question
        and (forum_structure.has_answer or forum_structure.is_forum)
    )
    if _is_educational_worksheet(scored, forum_structure=forum_structure):
        forum_signal = False

    units: list[KnowledgeUnit] = []
    forum_units_added = False
    use_forum = cfg.extract_qa and forum_signal and not agg_profile.is_aggregated
    if use_forum:
        forum = extract_forum_units(
            text,
            scored,
            row=row,
            max_extra_answers=cfg.max_extra_answers,
        )
        fu = _forum_to_units(forum, start_index=0)
        if fu:
            units.extend(fu)
            forum_units_added = any(
                u.section_class in ('question', 'answer', 'answer_extra') for u in fu
            )

    chunk_idx = len(units)
    metric_sections: list[tuple[KnowledgeSectionClass, str, bool]] = []

    with _timed('unit_assembly'):
        if forum_units_added:
            for sec, cls, qual in scored:
                metric_sections.append((cls.label, sec.text, qual.keep))
        elif use_forum and not agg_profile.is_aggregated:
            for sec, cls, qual in scored:
                metric_sections.append((cls.label, sec.text, qual.keep))
        elif not use_forum or mixed or agg_profile.is_aggregated:
            last_primary_ref: tuple[str, float] | None = None
            for sec, cls, qual in scored:
                if drop := _budget_drop():
                    return drop
                if cls.label in (
                    KnowledgeSectionClass.INSTRUCTION,
                    KnowledgeSectionClass.ASSIGNMENT,
                    KnowledgeSectionClass.DISCUSSION_PROMPT,
                ):
                    metric_sections.append((cls.label, sec.text, qual.keep))
                    continue
                if cls.label in _PUBLICATION_SALVAGE_CLASSES:
                    recovered = qual.scaffold_stripped or strip_publication_scaffolding(sec.text)
                    if recovered and recovered != sec.text and len(recovered.split()) >= 10:
                        ev_rec = resolve_semantic_evidence(recovered)
                        if ev_rec.utility >= 0.08:
                            sec = RecoveredSection(
                                text=recovered,
                                start=sec.start,
                                end=sec.end,
                                position_ratio=sec.position_ratio,
                                structural_role='body',
                                layout_kind=sec.layout_kind,
                                in_fence=sec.in_fence,
                            )
                            cls = SectionClassification(
                                KnowledgeSectionClass.ARTICLE,
                                confidence=max(ev_rec.utility, 0.32),
                            )
                            qual = assess_section_quality(
                                sec, cls, mixed_document=mixed, nav_ctx=nav_ctx, agg_ctx=agg_ctx,
                                post_wrapper=False,
                                analysis=build_section_analysis(sec.text),
                            )
                metric_sections.append((cls.label, sec.text, qual.keep))
                if not qual.keep:
                    continue
                if cls.label in DISCARD_CLASSES:
                    continue
                sec_ev = None

                def _sec_evidence():
                    nonlocal sec_ev
                    if sec_ev is None:
                        sec_ev = resolve_semantic_evidence(sec.text)
                    return sec_ev

                if last_primary_ref and cls.label in PRIMARY_CLASSES:
                    prev_text, prev_pos = last_primary_ref
                    role_cut = max(
                        educational_role_boundary(
                            prev_text, sec.text, left_pos=prev_pos, right_pos=sec.position_ratio,
                        ),
                        publication_role_boundary(
                            prev_text, sec.text, left_pos=prev_pos, right_pos=sec.position_ratio,
                        ),
                        conversation_role_boundary(
                            prev_text, sec.text, left_pos=prev_pos, right_pos=sec.position_ratio,
                        ),
                    )
                    if role_cut >= 0.32:
                        ev_sec = _sec_evidence()
                        ev_prev = resolve_semantic_evidence(prev_text)
                        conv_role, conv_conf = dominant_role(
                            score_conversation_roles(sec.text, position_ratio=sec.position_ratio),
                        )
                        if ev_sec.utility < max(0.13, ev_prev.utility * 0.80):
                            continue
                        if (
                            role_cut >= 0.30
                            and conv_role == ConversationRole.ANSWER
                            and conv_conf > 0.40
                            and not forum_structure.has_question
                        ):
                            continue
                if (
                    cls.label == KnowledgeSectionClass.EDUCATIONAL
                    and sec.position_ratio > 0.70
                    and score_section_artifact(sec.text, position_ratio=sec.position_ratio).promotional > 0.18
                    and _sec_evidence().utility < 0.13
                    and score_educational_roles(sec.text, position_ratio=sec.position_ratio).instruction_mass() < 0.22
                ):
                    continue
                if (
                    cls.label == KnowledgeSectionClass.EDUCATIONAL
                    and sec.position_ratio > 0.55
                    and score_section_artifact(sec.text, position_ratio=sec.position_ratio).promotional > 0.28
                    and _sec_evidence().utility < 0.14
                    and score_educational_roles(sec.text, position_ratio=sec.position_ratio).knowledge_mass() < 0.38
                ):
                    continue
                if forum_signal and cls.label == KnowledgeSectionClass.COMMENT:
                    continue
                if (
                    has_code_body
                    and cls.label == KnowledgeSectionClass.EDUCATIONAL
                    and sec.position_ratio < 0.55
                ):
                    conv_role, conv_conf = dominant_role(
                        score_conversation_roles(sec.text, position_ratio=sec.position_ratio),
                    )
                    if (
                        conv_conf > 0.48
                        and conv_role in (ConversationRole.ANSWER, ConversationRole.CONVERSATION)
                        and (
                            score_answer_substance(sec.text) < 0.52
                            or 'import ' not in sec.text
                        )
                    ):
                        continue
                pub_span = score_publication_roles(sec.text, position_ratio=sec.position_ratio)
                if is_pagination_footer_line(sec.text, position_ratio=sec.position_ratio):
                    continue
                if (
                    len(sec.text.split()) <= 8
                    and pub_span.footer_block > 0.38
                    and pub_span.knowledge < 0.30
                ):
                    continue
                unit = _section_to_unit(
                    sec, cls, qual, chunk_index=chunk_idx, agg_ctx=agg_ctx,
                    forum_strip=forum_signal or pre_forum,
                    preserve_code_fences=cfg.preserve_code_fences,
                )
                if unit is None:
                    continue
                if len(unit.text) < cfg.min_chars_after_clean and cls.label not in PRIMARY_CLASSES:
                    continue
                units.append(unit)
                if cls.label in PRIMARY_CLASSES:
                    last_primary_ref = (unit.text, sec.position_ratio)
                chunk_idx += 1
        else:
            for sec, cls, qual in scored:
                metric_sections.append((cls.label, sec.text, qual.keep))

        if has_code_body:
            code_blob = _merge_code_sections(scored)
            if code_blob:
                code_text = _emit_unit_text(
                    code_blob,
                    role='code',
                    min_chars=20,
                    forum_strip=forum_signal or pre_forum,
                    preserve_code_fences=cfg.preserve_code_fences,
                )
                if code_text:
                    units = [
                        u for u in units
                        if not (
                            u.section_class == 'educational'
                            and has_code_body
                            and (
                                'import ' not in u.text
                                or _code_tail(u.text) != u.text.strip()
                            )
                            and dominant_role(score_conversation_roles(u.text))[0]
                            in (ConversationRole.ANSWER, ConversationRole.CONVERSATION)
                        )
                    ]
                    if not any('import ' in u.text or 'def ' in u.text for u in units):
                        units.append(KnowledgeUnit(
                            text=code_text,
                            section_class='scientific',
                            retention_score=0.32,
                            chunk_index=len(units),
                            source_kind='code',
                        ))
        if not units and mixed:
            best: tuple[RecoveredSection, SectionClassification, SectionQualityScore] | None = None
            for item in scored:
                sec, cls, qual = item
                if not qual.keep:
                    continue
                if cls.label not in PRIMARY_CLASSES or cls.label == KnowledgeSectionClass.ARCHIVE:
                    continue
                if cls.label in (KnowledgeSectionClass.NEWSLETTER, KnowledgeSectionClass.EVENT):
                    continue
                if qual.retention_score < 0.10:
                    continue
                if best is None or qual.retention_score > best[2].retention_score:
                    best = item
            if best and best[2].retention_score > 0.10 and best[1].label not in DISCARD_CLASSES:
                unit = _section_to_unit(
                    best[0], best[1], best[2], chunk_index=0, agg_ctx=agg_ctx,
                    forum_strip=forum_signal or pre_forum,
                    preserve_code_fences=cfg.preserve_code_fences,
                )
                if unit:
                    units.append(unit)
                    metric_sections = [(best[1].label, best[0].text, True)]

        has_discard = any(
            cls.label in DISCARD_CLASSES or not qual.keep
            for _, cls, qual in scored
        )
        span_units = len(_decompose_spans(text))
        if not units and not mixed and len(scored) <= 1 and not has_discard and span_units <= 1:
            best: tuple[RecoveredSection, SectionClassification, SectionQualityScore] | None = None
            for item in scored:
                sec, cls, qual = item
                if not qual.keep:
                    continue
                if cls.label == KnowledgeSectionClass.ARCHIVE:
                    continue
                if best is None or qual.retention_score > best[2].retention_score:
                    best = item
            if best and best[2].retention_score > 0.12 and best[1].label not in DISCARD_CLASSES:
                unit = _section_to_unit(
                    best[0], best[1], best[2], chunk_index=0, agg_ctx=agg_ctx,
                    forum_strip=forum_signal or pre_forum,
                    preserve_code_fences=cfg.preserve_code_fences,
                )
                if unit and len(unit.text) >= cfg.min_chars_after_clean:
                    units.append(unit)
                    metric_sections = [(best[1].label, best[0].text, True)]

    with _timed('serialization'):
        metrics = compute_page_metrics(text, sections=metric_sections, mixed=mixed)

    from indw.extract.core.profile import active_profile, active_unit_assembly_profile
    prof = active_profile()
    if prof is not None and ks is not None:
        payload = prof.to_dict()
        asm = active_unit_assembly_profile()
        if asm is not None:
            payload['unit_assembly'] = asm.to_dict()
        ks.ke_ops = payload

    if not units:
        return KnowledgeExtractionResult(
            metrics=metrics,
            mixed=mixed,
            dropped_all=True,
            drop_reason='no_knowledge_units',
            _scored=scored,
        )

    return KnowledgeExtractionResult(
        units=units,
        metrics=metrics,
        mixed=mixed,
        _scored=scored,
    )
