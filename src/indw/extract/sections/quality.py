from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from indw.extract.structure.recovery import RecoveredSection
from indw.extract.sections.classify import (
    DISCARD_CLASSES,
    KnowledgeSectionClass,
    PRIMARY_CLASSES,
    SectionClassification,
    _section_prose_like,
)
from indw.extract.structure.analyze import analyze_structure
from indw.extract.nav.context import (
    NavigationContext,
    get_navigation_context,
    score_navigation_role,
)
from indw.extract.structure.aggregate import (
    AggregationContext,
    effective_position,
    get_aggregation_context,
)
from indw.extract.roles.publication import (
    score_publication_roles,
    strip_publication_scaffolding,
)
from indw.extract.nav.template import TemplateMiner
from indw.clean.artifact.evidence_engine import resolve_semantic_evidence
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator
from indw.clean.document.value import analyze_content_value, compute_structure_profile, resolve_analysis_bundle


@dataclass
class SectionQualityScore:
    knowledge_density: float = 0.0
    information_density: float = 0.0
    template_density: float = 0.0
    metadata_density: float = 0.0
    navigation_density: float = 0.0
    educational_value: float = 0.0
    coherence: float = 0.0
    redundancy: float = 0.0
    novelty: float = 0.0
    retention_score: float = 0.0
    artifact_score: float = 0.0
    keep: bool = False
    drop_reason: str = ''
    scaffold_stripped: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


_FAST_DISCARD = frozenset({
    KnowledgeSectionClass.NAVIGATION,
    KnowledgeSectionClass.FOOTER,
    KnowledgeSectionClass.RELATED,
    KnowledgeSectionClass.METADATA,
    KnowledgeSectionClass.ADVERTISEMENT,
})

_HARD_DROP_DISCARD = frozenset({
    KnowledgeSectionClass.INSTRUCTION,
    KnowledgeSectionClass.ASSIGNMENT,
    KnowledgeSectionClass.DISCUSSION_PROMPT,
})


def _assess_hard_drop_section(
    text: str,
    classification: SectionClassification,
    *,
    section: RecoveredSection,
    nav_ctx: NavigationContext,
    agg_ctx: AggregationContext,
) -> SectionQualityScore:
    from indw.clean.document.value import resolve_analysis_bundle

    pos = effective_position(section, agg_ctx=agg_ctx)
    bundle = resolve_analysis_bundle(text)
    ev = bundle.evidence(text)
    cv = analyze_content_value(text, bundle=bundle)
    structural = analyze_structure(text)
    template = nav_ctx.miner().analyze(text)
    profile = compute_structure_profile(text, evidence=ev)
    baseline = AdaptiveBaselineEstimator()
    noise = baseline.baseline(list(ev.negative.values()) or [0.0])
    know = ev.utility
    substance = baseline.baseline([
        cv.educational_score, cv.technical_score, cv.reference_score, know,
        structural.information_density,
    ])
    artifact = baseline.baseline([
        noise, structural.navigation_density, structural.boilerplate_density,
        template.template_density, profile.navigation_ratio,
    ])
    coherence = baseline.baseline([
        structural.sentence_completeness_mean, ev.coherence, cv.overall_value_score,
    ])
    redundancy = ev.redundancy
    novelty = max(0.0, 1.0 - redundancy)
    retention = substance * (1.0 - artifact * 0.85) * coherence * novelty
    return SectionQualityScore(
        knowledge_density=know,
        information_density=structural.information_density,
        template_density=template.template_density,
        metadata_density=profile.contact_ratio + profile.date_ratio * 0.5,
        navigation_density=profile.navigation_ratio,
        educational_value=cv.educational_score,
        coherence=coherence,
        redundancy=redundancy,
        novelty=novelty,
        retention_score=retention,
        artifact_score=artifact,
        keep=False,
        drop_reason=f'low_value_{classification.label.value}',
    )


def _assess_scaffold_discard_section(
    text: str,
    classification: SectionClassification,
    *,
    section: RecoveredSection,
    nav_ctx: NavigationContext,
    agg_ctx: AggregationContext,
    post_wrapper: bool,
) -> SectionQualityScore:
    from indw.clean.document.value import resolve_analysis_bundle

    pos = effective_position(section, agg_ctx=agg_ctx)
    bundle = resolve_analysis_bundle(text)
    ev = bundle.evidence(text)
    cv = analyze_content_value(text, bundle=bundle)
    structural = analyze_structure(text)
    template = nav_ctx.miner().analyze(text)
    nav_score = score_navigation_role(text, position_ratio=pos, ctx=nav_ctx)
    profile = compute_structure_profile(text, evidence=ev)
    baseline = AdaptiveBaselineEstimator()
    noise = baseline.baseline(list(ev.negative.values()) or [0.0])
    know = ev.utility
    substance = baseline.baseline([
        cv.educational_score, cv.technical_score, cv.reference_score, know,
        structural.information_density,
    ])
    artifact = baseline.baseline([
        noise, structural.navigation_density, structural.boilerplate_density,
        template.template_density, profile.navigation_ratio, profile.listing_ratio,
    ])
    coherence = baseline.baseline([
        structural.sentence_completeness_mean, ev.coherence, cv.overall_value_score,
    ])
    redundancy = ev.redundancy
    novelty = max(0.0, 1.0 - redundancy)
    retention = substance * (1.0 - artifact * 0.85) * coherence * novelty
    thr = baseline.baseline([0.08, artifact * 0.4 + 0.05])
    keep = retention >= baseline.baseline([thr, 0.28]) and substance > baseline.baseline([noise, 0.35])
    if nav_score.is_navigation(threshold=0.48):
        keep = False
    drop_reason = ''
    if not keep:
        drop_reason = f'low_value_{classification.label.value}'
    return SectionQualityScore(
        knowledge_density=know,
        information_density=structural.information_density,
        template_density=template.template_density,
        metadata_density=profile.contact_ratio + profile.date_ratio * 0.5,
        navigation_density=profile.navigation_ratio,
        educational_value=cv.educational_score,
        coherence=coherence,
        redundancy=redundancy,
        novelty=novelty,
        retention_score=retention,
        artifact_score=artifact,
        keep=keep,
        drop_reason=drop_reason,
    )


def assess_section_quality(
    section: RecoveredSection,
    classification: SectionClassification,
    *,
    mixed_document: bool = False,
    min_retention: float = 0.0,
    nav_ctx: NavigationContext | None = None,
    agg_ctx: AggregationContext | None = None,
    post_wrapper: bool = False,
    analysis: Any | None = None,
) -> SectionQualityScore:
    from indw.extract.core.context import get_document_context

    text = section.text
    dctx = get_document_context()
    if not text.strip():
        return SectionQualityScore(drop_reason='empty')

    nav_ctx = nav_ctx or get_navigation_context()
    agg_ctx = agg_ctx or get_aggregation_context()

    if not mixed_document and classification.confidence >= 0.58:
        if classification.label in _HARD_DROP_DISCARD:
            return _assess_hard_drop_section(
                text, classification, section=section, nav_ctx=nav_ctx, agg_ctx=agg_ctx,
            )
        if classification.label in _FAST_DISCARD:
            return _assess_scaffold_discard_section(
                text, classification, section=section, nav_ctx=nav_ctx, agg_ctx=agg_ctx,
                post_wrapper=post_wrapper,
            )

    pos = effective_position(section, agg_ctx=agg_ctx)

    if analysis is not None:
        bundle = analysis.bundle
        ev = analysis.evidence
        cv = analysis.content_value
        structural = analysis.structural
        profile = analysis.profile
    else:
        bundle = resolve_analysis_bundle(text)
        ev = bundle.evidence(text)
        cv = analyze_content_value(text, bundle=bundle)
        structural = analyze_structure(text)
        profile = compute_structure_profile(text, evidence=ev)
    template = nav_ctx.miner().analyze(text)
    nav_score = score_navigation_role(text, position_ratio=pos, ctx=nav_ctx)
    baseline = AdaptiveBaselineEstimator()

    noise = baseline.baseline(list(ev.negative.values()) or [0.0])
    know = ev.utility
    substance = baseline.baseline([
        cv.educational_score,
        cv.technical_score,
        cv.reference_score,
        know,
        structural.information_density,
    ])

    artifact = baseline.baseline([
        noise,
        structural.navigation_density,
        structural.boilerplate_density,
        template.template_density,
        profile.navigation_ratio,
        profile.listing_ratio if classification.label in DISCARD_CLASSES else 0.0,
    ])

    coherence = baseline.baseline([
        structural.sentence_completeness_mean,
        ev.coherence,
        cv.overall_value_score,
    ])
    redundancy = ev.redundancy
    novelty = max(0.0, 1.0 - redundancy)

    retention = substance * (1.0 - artifact * 0.85) * coherence * novelty
    if classification.label in PRIMARY_CLASSES:
        retention *= 1.15

    thr = baseline.baseline([min_retention, 0.08, artifact * 0.4 + 0.05])
    keep = retention >= thr and substance >= noise * 0.8

    if classification.label in PRIMARY_CLASSES:
        keep = retention >= thr * 0.5 or (substance > noise * 0.85 and know >= 0.10)

    if classification.label in DISCARD_CLASSES:
        keep = retention >= baseline.baseline([thr, 0.28]) and substance > baseline.baseline([noise, 0.35])
    if classification.label in (KnowledgeSectionClass.ARCHIVE, KnowledgeSectionClass.NEWSLETTER):
        keep = profile.explanation_ratio > 0.28 and retention >= baseline.baseline([thr, 0.35])
    if classification.label == KnowledgeSectionClass.ARCHIVE:
        keep = False
    if classification.label in (
        KnowledgeSectionClass.INSTRUCTION,
        KnowledgeSectionClass.ASSIGNMENT,
        KnowledgeSectionClass.DISCUSSION_PROMPT,
    ):
        keep = False
    if classification.label == KnowledgeSectionClass.COMMENT:
        keep = retention >= baseline.baseline([thr, 0.30]) and (
            cv.technical_score > 0.2 or (mixed_document and know >= 0.10 and structural.sentence_completeness_mean > 0.5)
        )
    if classification.label == KnowledgeSectionClass.QUESTION:
        keep = retention >= thr * 0.85 or cv.educational_score > 0.15
    if classification.label == KnowledgeSectionClass.ANSWER:
        keep = retention >= thr * 0.8 or cv.technical_score > 0.18

    pub_tail = score_publication_roles(text, position_ratio=pos)
    if pub_tail.footer_block > 0.45 and len(text.split()) <= 10 and ev.utility < 0.14:
        keep = False

    if (
        post_wrapper
        and classification.label in PRIMARY_CLASSES
        and retention >= 0.10
        and structural.sentence_completeness_mean > 0.45
        and len(text.split()) >= 12
    ):
        keep = True

    if (
        classification.label in (
            KnowledgeSectionClass.SCIENTIFIC,
            KnowledgeSectionClass.EDUCATIONAL,
            KnowledgeSectionClass.ARTICLE,
        )
        and _section_prose_like(text, ev, profile, structural)
        and retention >= 0.08
        and substance > noise * 0.65
        and (
            score_publication_roles(text[:min(len(text), 200)], position_ratio=pos).scaffold_mass() < 0.55
            or ev.quality.educational > 0.17
        )
    ):
        keep = True

    ends_complete = bool(text.strip()) and text.strip()[-1] in '.!?)"\'»]})'
    if (
        ends_complete
        and structural.sentence_completeness_mean >= 0.82
        and len(text.split()) >= 5
        and classification.label in (
            KnowledgeSectionClass.SCIENTIFIC,
            KnowledgeSectionClass.EDUCATIONAL,
            KnowledgeSectionClass.ARTICLE,
            KnowledgeSectionClass.MEDICAL,
            KnowledgeSectionClass.GOVERNMENT,
        )
        and retention >= 0.08
        and substance > noise * 0.55
    ):
        keep = True

    if mixed_document and classification.label in (
        KnowledgeSectionClass.SCIENTIFIC,
        KnowledgeSectionClass.MEDICAL,
        KnowledgeSectionClass.EDUCATIONAL,
        KnowledgeSectionClass.ARTICLE,
    ):
        head = text[:min(len(text), 240)]
        pub_head = score_publication_roles(head, position_ratio=min(pos, 0.12))
        meta_boiler = (
            pub_head.scaffold_mass() > 0.42
            and pub_head.knowledge < 0.22
            and profile.explanation_ratio < 0.20
        )
        junk_tail = (
            profile.listing_ratio > 0.18
            or (ev.negative.get('noise', 0.0) > 0.38 and profile.explanation_ratio < 0.16)
            or structural.repeated_line_ratio > 0.22
            or score_publication_roles(
                text[max(0, len(text) - 240):],
                position_ratio=max(pos, 0.75),
            ).scaffold_mass() > 0.40
        )
        if not meta_boiler and not junk_tail and know >= 0.10 and substance > noise * 0.80 and retention >= 0.10:
            keep = True
        if (
            post_wrapper
            and section.structural_role in ('body', 'introduction', 'title')
            and not junk_tail
            and know >= 0.08
            and retention >= 0.08
        ):
            keep = True

    pub_full = score_publication_roles(text, position_ratio=pos)
    if pub_full.scaffold_mass() > 0.55 and pub_full.knowledge < 0.25:
        keep = profile.explanation_ratio > 0.5 and len(text.split()) > 40
    scaffold_stripped = ''
    if pub_full.scaffold_mass() > 0.48:
        stripped = strip_publication_scaffolding(text)
        scaffold_stripped = stripped
        if dctx is not None and stripped:
            dctx.remember_scaffold_stripped(text, stripped)
        if not stripped or len(stripped.split()) < 8:
            if not (
                ev.quality.educational > 0.17
                and _section_prose_like(text, ev, profile, structural)
            ):
                keep = False
        elif classification.label in PRIMARY_CLASSES:
            ev_stripped = (
                dctx.section_evidence(stripped, lambda: resolve_semantic_evidence(stripped))
                if dctx is not None
                else resolve_semantic_evidence(stripped)
            )
            if ev_stripped.utility >= 0.08 and len(stripped.split()) >= 12:
                keep = True

    if mixed_document and classification.label in DISCARD_CLASSES:
        keep = False
    elif mixed_document and classification.label == KnowledgeSectionClass.COMMENT:
        keep = False

    agg_unit = agg_ctx.profile.unit_for(section) if agg_ctx.profile else None
    if agg_ctx.is_aggregated() and agg_unit is not None:
        if agg_unit.role in ('headline', 'wrapper', 'subscription'):
            keep = False
        elif agg_unit.role == 'article' and agg_unit.independence >= 0.28:
            if ev.utility >= 0.08 and structural.sentence_completeness_mean >= 0.45:
                keep = True
        if agg_ctx.profile and agg_ctx.profile.is_headline_index:
            keep = False

    if nav_score.is_navigation(threshold=0.48) and classification.label in PRIMARY_CLASSES:
        if not (agg_ctx.is_aggregated() and agg_unit and agg_unit.role == 'article'):
            if not (
                _section_prose_like(text, ev, profile, structural)
                and ev.quality.educational > 0.17
            ):
                keep = False

    drop_reason = ''
    if not keep:
        if classification.label in DISCARD_CLASSES:
            drop_reason = f'low_value_{classification.label.value}'
        elif substance <= noise:
            drop_reason = 'noise_over_substance'
        else:
            drop_reason = 'low_retention'

    return SectionQualityScore(
        knowledge_density=know,
        information_density=structural.information_density,
        template_density=template.template_density,
        metadata_density=profile.contact_ratio + profile.date_ratio * 0.5,
        navigation_density=profile.navigation_ratio,
        educational_value=cv.educational_score,
        coherence=coherence,
        redundancy=redundancy,
        novelty=novelty,
        retention_score=retention,
        artifact_score=artifact,
        keep=keep,
        drop_reason=drop_reason,
        scaffold_stripped=scaffold_stripped,
    )
