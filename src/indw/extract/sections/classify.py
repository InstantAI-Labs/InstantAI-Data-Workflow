from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from indw.extract.structure.recovery import RecoveredSection
from indw.extract.nav.context import (
    NavigationContext,
    extract_navigation_features,
    get_navigation_context,
    score_navigation_role,
    structural_listing_score,
)
from indw.extract.structure.aggregate import (
    AggregationContext,
    effective_position,
    get_aggregation_context,
)
from indw.extract.roles.publication import (
    PublicationRole,
    score_publication_roles,
)
from indw.extract.roles.education import (
    DISCARD_EDUCATIONAL_ROLES,
    EducationalRole,
    score_educational_roles,
)
from indw.clean.semantic.section_artifacts import score_section_artifact
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator
from indw.extract.roles.forum import (
    ConversationRole,
    dominant_role,
    recover_forum_structure,
    score_conversation_roles,
)

class KnowledgeSectionClass(str, Enum):
    ARTICLE = 'article'
    FORUM = 'forum'
    QUESTION = 'question'
    ANSWER = 'answer'
    COMMENT = 'comment'
    NAVIGATION = 'navigation'
    ARCHIVE = 'archive'
    NEWSLETTER = 'newsletter'
    EVENT = 'event'
    ADVERTISEMENT = 'advertisement'
    REFERENCE = 'reference'
    INSPECTION = 'inspection'
    GOVERNMENT = 'government'
    EDUCATIONAL = 'educational'
    INSTRUCTION = 'instruction'
    ASSIGNMENT = 'assignment'
    DISCUSSION_PROMPT = 'discussion_prompt'
    MEDICAL = 'medical'
    SCIENTIFIC = 'scientific'
    AUTHOR_BIO = 'author_bio'
    FOOTER = 'footer'
    RELATED = 'related'
    METADATA = 'metadata'
    MIXED = 'mixed'
    UNKNOWN = 'unknown'

PRIMARY_CLASSES = frozenset({
    KnowledgeSectionClass.ARTICLE,
    KnowledgeSectionClass.EDUCATIONAL,
    KnowledgeSectionClass.SCIENTIFIC,
    KnowledgeSectionClass.MEDICAL,
    KnowledgeSectionClass.GOVERNMENT,
    KnowledgeSectionClass.REFERENCE,
    KnowledgeSectionClass.QUESTION,
    KnowledgeSectionClass.ANSWER,
    KnowledgeSectionClass.INSPECTION,
    KnowledgeSectionClass.FORUM,
})

DISCARD_CLASSES = frozenset({
    KnowledgeSectionClass.NAVIGATION,
    KnowledgeSectionClass.ADVERTISEMENT,
    KnowledgeSectionClass.FOOTER,
    KnowledgeSectionClass.RELATED,
    KnowledgeSectionClass.METADATA,
    KnowledgeSectionClass.AUTHOR_BIO,
    KnowledgeSectionClass.ARCHIVE,
    KnowledgeSectionClass.NEWSLETTER,
    KnowledgeSectionClass.EVENT,
    KnowledgeSectionClass.COMMENT,
    KnowledgeSectionClass.INSTRUCTION,
    KnowledgeSectionClass.ASSIGNMENT,
    KnowledgeSectionClass.DISCUSSION_PROMPT,
})

@dataclass
class SectionClassification:
    label: KnowledgeSectionClass = KnowledgeSectionClass.UNKNOWN
    confidence: float = 0.0
    discovered: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'label': self.label.value,
            'confidence': round(self.confidence, 4),
            'discovered': [{'label': l, 'weight': round(w, 4)} for l, w in self.discovered[:8]],
        }

_WRAPPER_PREV = frozenset({
    KnowledgeSectionClass.NAVIGATION,
    KnowledgeSectionClass.ARCHIVE,
    KnowledgeSectionClass.FOOTER,
    KnowledgeSectionClass.RELATED,
    KnowledgeSectionClass.ADVERTISEMENT,
    KnowledgeSectionClass.NEWSLETTER,
    KnowledgeSectionClass.EVENT,
    KnowledgeSectionClass.METADATA,
    KnowledgeSectionClass.INSTRUCTION,
    KnowledgeSectionClass.ASSIGNMENT,
    KnowledgeSectionClass.DISCUSSION_PROMPT,
})

def _educational_role_to_section(role: EducationalRole, confidence: float) -> SectionClassification:
    mapping = {
        EducationalRole.QUESTION_PROMPT: KnowledgeSectionClass.DISCUSSION_PROMPT,
        EducationalRole.INSTRUCTION: KnowledgeSectionClass.INSTRUCTION,
        EducationalRole.ASSIGNMENT: KnowledgeSectionClass.ASSIGNMENT,
        EducationalRole.EXERCISE: KnowledgeSectionClass.ASSIGNMENT,
        EducationalRole.DISCUSSION_STARTER: KnowledgeSectionClass.DISCUSSION_PROMPT,
        EducationalRole.LEARNING_OBJECTIVE: KnowledgeSectionClass.INSTRUCTION,
        EducationalRole.KNOWLEDGE: KnowledgeSectionClass.EDUCATIONAL,
        EducationalRole.EXPLANATION: KnowledgeSectionClass.EDUCATIONAL,
        EducationalRole.ARTICLE: KnowledgeSectionClass.ARTICLE,
        EducationalRole.REFERENCE: KnowledgeSectionClass.REFERENCE,
        EducationalRole.PRIMARY_SOURCE: KnowledgeSectionClass.REFERENCE,
    }
    label = mapping.get(role, KnowledgeSectionClass.UNKNOWN)
    return SectionClassification(label, confidence)

def _feat_digits_high(text: str) -> bool:
    feat = extract_navigation_features(text, position_ratio=0.05)
    return feat.digit_token_ratio > 0.06 or feat.caps_token_ratio > 0.18

def _section_prose_like(text: str, ev, profile, structural) -> bool:
    head = score_publication_roles(text[:min(len(text), 240)], position_ratio=0.08)
    if head.scaffold_mass() > 0.55 and ev.quality.educational < 0.17:
        return False
    if profile.listing_ratio > 0.24 and profile.explanation_ratio < 0.10:
        return False
    return (
        structural.sentence_completeness_mean > 0.46
        and len(text.split()) > 8
        and (
            ev.utility >= 0.08
            or ev.quality.educational > 0.06
            or ev.quality.technical > 0.08
        )
    )

def classify_section(
    section: RecoveredSection,
    *,
    nav_ctx: NavigationContext | None = None,
    agg_ctx: AggregationContext | None = None,
    prev_label: KnowledgeSectionClass | None = None,
    wrapper_seen: bool = False,
    analysis: Any | None = None,
) -> SectionClassification:
    text = section.text
    if not text.strip():
        return SectionClassification()

    if analysis is None:
        from indw.extract.sections.scratch import build_section_analysis
        analysis = build_section_analysis(text)
    ev = analysis.evidence
    profile = analysis.profile
    structural = analysis.structural
    baseline = AdaptiveBaselineEstimator()
    discovered: list[tuple[str, float]] = []

    agg_ctx = agg_ctx or get_aggregation_context()
    pos = effective_position(section, agg_ctx=agg_ctx)
    role = section.structural_role
    raw = ev.representation

    forum_struct = recover_forum_structure(text)
    if forum_struct.is_forum or (
        forum_struct.has_question
        and forum_struct.has_answer
        and forum_struct.wrapper_mass > 0.22
    ):
        conv_scores = score_conversation_roles(text, position_ratio=pos)
        conv_role, conv_conf = dominant_role(conv_scores)
        prose_like = _section_prose_like(text, ev, profile, structural)
        if conv_role == ConversationRole.FORUM_UI and conv_conf >= 0.40:
            return SectionClassification(KnowledgeSectionClass.METADATA, conv_conf)
        if conv_role == ConversationRole.METADATA and conv_conf >= 0.42:
            return SectionClassification(KnowledgeSectionClass.METADATA, conv_conf)
        if conv_role == ConversationRole.NAVIGATION and conv_conf >= 0.40:
            return SectionClassification(KnowledgeSectionClass.NAVIGATION, conv_conf)
        if conv_role == ConversationRole.CONVERSATION and conv_conf >= 0.48 and not prose_like:
            return SectionClassification(KnowledgeSectionClass.COMMENT, conv_conf)
        if conv_role == ConversationRole.QUESTION and conv_conf >= 0.38:
            return SectionClassification(KnowledgeSectionClass.QUESTION, conv_conf)
        if conv_role in (ConversationRole.ANSWER, ConversationRole.EXPLANATION) and conv_conf >= 0.36:
            return SectionClassification(KnowledgeSectionClass.ANSWER, conv_conf)

    agg_unit = agg_ctx.profile.unit_for(section) if agg_ctx.profile else None
    if agg_unit is not None and agg_ctx.is_aggregated():
        if agg_unit.role == 'headline':
            return SectionClassification(KnowledgeSectionClass.NEWSLETTER, max(0.72, agg_unit.headline_score))
        if agg_unit.role == 'subscription':
            return SectionClassification(KnowledgeSectionClass.METADATA, max(0.70, agg_unit.wrapper_score))
        if agg_unit.role == 'wrapper':
            return SectionClassification(KnowledgeSectionClass.METADATA, max(0.68, agg_unit.wrapper_score))
        if agg_unit.role == 'article' and agg_unit.independence >= 0.30:
            discovered.append(('article', agg_unit.independence))
            if ev.quality.technical > 0.2:
                discovered.append(('scientific', ev.quality.technical))
            if ev.quality.educational > baseline.baseline([ev.quality.technical, ev.quality.educational]):
                discovered.append(('educational', ev.quality.educational))

    agg_article = (
        agg_unit is not None and agg_ctx.is_aggregated()
        and agg_unit.role == 'article' and agg_unit.independence >= 0.30
    )

    nav_ctx = nav_ctx or get_navigation_context()
    neighbor_know = None
    if prev_label in DISCARD_CLASSES:
        neighbor_know = max(ev.utility, profile.explanation_ratio, structural.sentence_completeness_mean)
    pub = score_publication_roles(text, position_ratio=pos)
    nav = score_navigation_role(
        text,
        position_ratio=pos,
        ctx=nav_ctx,
        neighbor_knowledge=neighbor_know,
    )
    nav_role, nav_conf = nav.dominant()
    post_wrapper = prev_label in _WRAPPER_PREV or wrapper_seen

    art = score_section_artifact(text, position_ratio=pos)
    edu = score_educational_roles(text, position_ratio=pos)
    edu_role, edu_conf = edu.dominant()
    if edu_role in DISCARD_EDUCATIONAL_ROLES and edu.instruction_mass() > 0.26:
        if edu_conf >= 0.28 and edu.instruction_mass() >= edu.knowledge_mass() * 0.62:
            return _educational_role_to_section(edu_role, max(edu_conf, edu.instruction_mass()))

    pub_role, pub_conf = pub.dominant()
    if (
        pos < 0.50
        and pub.scaffold_mass() > 0.40
        and ev.quality.educational < 0.06
        and profile.explanation_ratio < 0.08
        and len(text.split()) > 14
        and art.promotional > 0.20
    ):
        return SectionClassification(KnowledgeSectionClass.EVENT, max(0.68, art.promotional))

    if (
        pos < 0.35
        and pub.scaffold_mass() > 0.55
        and profile.explanation_ratio < 0.10
        and pub_role in (
            PublicationRole.DATE_BLOCK,
            PublicationRole.ARCHIVE_REF,
            PublicationRole.ISSUE_INFO,
            PublicationRole.VOLUME_INFO,
        )
    ):
        if pub_role in (PublicationRole.DATE_BLOCK, PublicationRole.ARCHIVE_REF):
            return SectionClassification(KnowledgeSectionClass.ARCHIVE, max(0.72, pub_conf))
        return SectionClassification(KnowledgeSectionClass.METADATA, max(0.70, pub_conf))

    if not agg_article and pos > 0.48 and art.promotional > 0.26 and ev.utility < 0.14:
        if (
            art.promotional >= ev.quality.educational * 0.45
            and edu.instruction_mass() < 0.35
            and edu.knowledge_mass() < 0.35
        ):
            return SectionClassification(KnowledgeSectionClass.RELATED, max(0.72, art.promotional))

    if (
        pos < 0.25
        and pub.archive_ref > 0.42
        and pub.scaffold_mass() > 0.45
        and pub.knowledge < 0.28
        and profile.explanation_ratio < 0.20
    ):
        return SectionClassification(KnowledgeSectionClass.ARCHIVE, max(0.72, pub.archive_ref))

    if (
        pos < 0.30
        and pub.archive_ref > 0.38
        and pub.scaffold_mass() > 0.42
        and ev.quality.educational < 0.18
        and profile.explanation_ratio < 0.08
    ):
        return SectionClassification(KnowledgeSectionClass.ARCHIVE, max(0.72, pub.archive_ref))

    prose_like = _section_prose_like(text, ev, profile, structural)
    if prose_like and structural.sentence_completeness_mean > 0.45:
        if ev.quality.educational > 0.04:
            return SectionClassification(KnowledgeSectionClass.EDUCATIONAL, max(ev.quality.educational, 0.35))
        if ev.utility >= 0.08 and profile.explanation_ratio < 0.14:
            return SectionClassification(KnowledgeSectionClass.ARTICLE, max(ev.utility, 0.32))

    feat = extract_navigation_features(text, position_ratio=pos)
    scaffold_mass = pub.scaffold_mass()
    if (
        scaffold_mass > 0.48
        and ev.utility < 0.18
        and profile.explanation_ratio < 0.16
        and len(text.split()) <= 40
        and (
            feat.separator_density > 0.06
            or feat.template_density > 0.15
            or feat.listing_ratio > 0.22
        )
    ):
        pub_role, pub_conf = pub.dominant()
        if pub_role in (
            PublicationRole.MASTHEAD,
            PublicationRole.PUBLICATION_HEADER,
            PublicationRole.ISSUE_INFO,
            PublicationRole.VOLUME_INFO,
        ):
            return SectionClassification(KnowledgeSectionClass.METADATA, max(0.72, pub_conf))
        if pub_role in (PublicationRole.ARCHIVE_REF, PublicationRole.HOMEPAGE_LINK):
            return SectionClassification(KnowledgeSectionClass.ARCHIVE, max(0.70, pub_conf))
        if pub_role in (PublicationRole.AUTHOR_BLOCK, PublicationRole.DATE_BLOCK):
            return SectionClassification(KnowledgeSectionClass.METADATA, max(0.68, pub_conf))
        if pub_role in (PublicationRole.NAVIGATION, PublicationRole.FOOTER_BLOCK):
            return SectionClassification(KnowledgeSectionClass.NAVIGATION, max(0.70, pub_conf))
    if post_wrapper and len(text.split()) <= 14 and text.strip().endswith(':'):
        return SectionClassification(KnowledgeSectionClass.QUESTION, 0.72)
    if (
        post_wrapper
        and section.structural_role == 'introduction'
        and nav.nav_mass() > 0.38
        and ev.utility < 0.15
    ):
        return SectionClassification(KnowledgeSectionClass.NAVIGATION, max(0.72, nav.nav_mass()))
    if (
        post_wrapper
        and (
            section.structural_role == 'body'
            or (
                section.structural_role in ('introduction', 'title')
                and not nav.is_navigation()
            )
        )
        and len(text.split()) >= 5
        and structural.sentence_completeness_mean > 0.40
        and (ev.utility >= 0.08 or not nav.is_navigation())
        and (
            text.rstrip().endswith(('.', '!', '?'))
            or ev.utility >= 0.08
        )
    ):
        return SectionClassification(
            KnowledgeSectionClass.SCIENTIFIC,
            max(ev.utility, ev.quality.technical, 0.55),
        )

    if (
        not agg_article
        and prev_label is None
        and len(text.split()) <= 8
        and feat.explanation_ratio < 0.12
        and ev.utility < 0.15
        and feat.knowledge_density < 0.13
        and (
            feat.digit_token_ratio > 0.04
            or feat.template_density > 0.18
            or pos < 0.15
        )
    ):
        return SectionClassification(KnowledgeSectionClass.NAVIGATION, 0.78)

    if (
        not agg_article
        and len(text.split()) <= 14
        and text.strip().endswith(':')
        and profile.instruction_ratio > 0.08
    ):
        return SectionClassification(KnowledgeSectionClass.QUESTION, max(0.72, profile.instruction_ratio))

    neg_noise = ev.negative.get('noise', 0.0)
    if not agg_article:
        if profile.listing_ratio > 0.22 and profile.explanation_ratio < 0.16:
            return SectionClassification(KnowledgeSectionClass.RELATED, profile.listing_ratio)
        if (
            structural.layout.line_count <= 2
            and feat.uniform_line_ratio > 0.85
            and 5 <= len(text.split()) <= 14
            and profile.explanation_ratio < 0.12
            and ev.utility < 0.16
            and nav.article < 0.45
            and pub.knowledge < 0.35
        ):
            return SectionClassification(KnowledgeSectionClass.NAVIGATION, 0.76)
        if (
            neg_noise > 0.38
            and profile.explanation_ratio < 0.12
            and pos > 0.38
            and not prose_like
            and edu.knowledge_mass() < 0.38
        ):
            return SectionClassification(KnowledgeSectionClass.COMMENT, neg_noise)

    neg_nav = ev.negative.get('navigational', 0.0)
    neg_promo = ev.negative.get('promotional', 0.0)
    neg_admin = ev.negative.get('administrative', 0.0)
    neg_trans = ev.negative.get('transactional', 0.0)

    if not agg_article:
        if (
            neg_promo > baseline.baseline([neg_promo, 0.28])
            and neg_trans > baseline.baseline([neg_trans, 0.25])
            and profile.instruction_ratio > 0.22
            and profile.explanation_ratio < 0.12
            and ev.utility < 0.18
        ):
            return SectionClassification(KnowledgeSectionClass.ADVERTISEMENT, max(neg_promo, neg_trans))

    if not agg_article:
        if nav.archive > 0.32 or nav.collection > 0.32:
            prose_like = (
                structural.sentence_completeness_mean > 0.48
                and len(text.split()) > 8
                and (ev.utility >= 0.10 or ev.quality.educational > 0.08)
                and ev.quality.educational >= 0.18
            )
            if not prose_like:
                return SectionClassification(KnowledgeSectionClass.ARCHIVE, max(nav.archive, nav.collection))
        if structural_listing_score(text, position_ratio=pos) > 0.35 and profile.explanation_ratio < 0.15:
            prose_like = (
                structural.sentence_completeness_mean > 0.48
                and len(text.split()) > 12
                and ev.utility >= 0.10
            )
            if not prose_like:
                return SectionClassification(KnowledgeSectionClass.ARCHIVE, 0.72)
        if pos < 0.2 and profile.explanation_ratio < 0.06 and _feat_digits_high(text):
            if ev.utility < 0.12 and structural.sentence_completeness_mean < 0.45:
                return SectionClassification(KnowledgeSectionClass.ARCHIVE, 0.68)

    prose_block = (
        '\n\n' in text
        or (ev.utility > 0.16 and profile.explanation_ratio > 0.18)
        or (
            structural.sentence_completeness_mean > 0.52
            and text.rstrip().endswith(('.', '!', '?'))
            and (post_wrapper or nav.nav_mass() < 0.22)
        )
    )
    promo_tail = pos > 0.62 and art.promotional > 0.28
    if not agg_article:
        if prose_block:
            pass
        elif pub.footer_block > 0.42 and len(text.split()) <= 10:
            return SectionClassification(KnowledgeSectionClass.FOOTER, max(0.75, pub.footer_block))
        elif role in ('navigation',) or (nav.is_navigation() and nav_role in (
            'navigation', 'breadcrumb', 'menu', 'pagination', 'sidebar', 'sitemap',
        ) and nav.article < 0.38):
            return SectionClassification(KnowledgeSectionClass.NAVIGATION, max(0.82, nav_conf))
        if pos > 0.62 and art.promotional > 0.28 and ev.utility < 0.16:
            if edu.knowledge_mass() < 0.35:
                return SectionClassification(KnowledgeSectionClass.RELATED, max(0.72, art.promotional))
    if not agg_article:
        if role in ('footer',) or (nav_role == 'footer' and nav.footer > 0.45):
            return SectionClassification(KnowledgeSectionClass.FOOTER, max(0.80, nav.footer))
        if role in ('promotional', 'related_content'):
            if edu.knowledge_mass() < 0.35:
                return SectionClassification(KnowledgeSectionClass.RELATED, 0.78)
        if role in ('contact', 'author_info'):
            return SectionClassification(KnowledgeSectionClass.AUTHOR_BIO, 0.72)
        if role in ('metadata', 'legal'):
            return SectionClassification(KnowledgeSectionClass.METADATA, 0.75)
        if role in ('references',):
            return SectionClassification(KnowledgeSectionClass.REFERENCE, 0.70)

    if neg_nav > baseline.baseline([neg_nav, 0.25]) and profile.navigation_ratio > 0.2:
        discovered.append(('navigation', max(neg_nav, nav.navigation)))
    if nav.breadcrumb > 0.4:
        discovered.append(('navigation', nav.breadcrumb))
    prose_like = _section_prose_like(text, ev, profile, structural)
    if nav.archive > 0.42 and not prose_like:
        discovered.append(('archive', nav.archive))
    if nav.collection > 0.38 and not prose_like:
        discovered.append(('archive', nav.collection))
    if neg_promo > baseline.baseline([neg_promo, 0.22]):
        discovered.append(('advertisement', neg_promo))
    if profile.listing_ratio > 0.22 and profile.explanation_ratio < 0.14 and 0.15 < pos < 0.88:
        discovered.append(('related', profile.listing_ratio))
    if neg_admin > baseline.baseline([neg_admin, 0.2]):
        discovered.append(('government', neg_admin))

    qa_signal = baseline.baseline([
        profile.instruction_ratio,
        ev.quality.reference,
        structural.layout.list_ratio,
    ])
    raw_feats = structural.layout
    if ev.quality.reference > 0.15 and '?' in text[:200]:
        discovered.append(('question', ev.quality.reference + qa_signal))
    if profile.explanation_ratio > 0.2 and ev.utility > 0.18 and '?' not in text[:80]:
        discovered.append(('answer', ev.utility))

    if profile.listing_ratio > 0.35 and profile.explanation_ratio < 0.15:
        if structural.repeated_line_ratio > 0.25:
            discovered.append(('newsletter', profile.listing_ratio))
        elif profile.date_ratio > 0.12:
            discovered.append(('event', profile.date_ratio))
        else:
            listing = structural_listing_score(text, position_ratio=pos)
            if listing > 0.35:
                discovered.append(('archive', listing))

    if structural.table_density > 0.4 and neg_admin > 0.15:
        discovered.append(('inspection', structural.table_density))

    if ev.quality.educational > baseline.baseline([ev.quality.technical, ev.quality.educational]) and not promo_tail:
        discovered.append(('educational', ev.quality.educational))
    elif prose_like and ev.quality.educational > 0.04:
        discovered.append(('educational', ev.quality.educational + 0.32))
    if ev.quality.technical > 0.2 and not promo_tail:
        discovered.append(('scientific', ev.quality.technical))
    if profile.contact_ratio > 0.08 and profile.explanation_ratio > 0.12:
        discovered.append(('medical', profile.explanation_ratio))

    if not agg_article:
        if neg_noise > 0.3 and ev.utility < 0.15 and ev.representation and ev.representation.narrative > ev.representation.factual:
            if not post_wrapper:
                w = neg_noise + (pos * 0.35 if pos > 0.45 else 0.0)
                discovered.append(('comment', w))
        if neg_noise > 0.22 and ev.utility < 0.17 and profile.explanation_ratio < 0.14:
            if not (post_wrapper and structural.sentence_completeness_mean > 0.5):
                if prose_like or ev.quality.educational > 0.12 or profile.instruction_ratio > 0.08:
                    discovered.append(('educational', max(ev.quality.educational, 0.12) + 0.18))
                else:
                    w = neg_noise + (pos * 0.35 if pos > 0.45 else 0.0)
                    discovered.append(('comment', w))

    if not discovered:
        if ev.quality.educational > baseline.baseline([ev.quality.educational, 0.12]):
            return SectionClassification(KnowledgeSectionClass.EDUCATIONAL, ev.quality.educational)
        if prose_like and structural.sentence_completeness_mean > 0.45:
            return SectionClassification(KnowledgeSectionClass.ARTICLE, max(ev.utility, 0.30))
        if ev.utility > baseline.baseline([ev.utility, 0.18]):
            label = KnowledgeSectionClass.ARTICLE
            conf = ev.utility
        else:
            label = KnowledgeSectionClass.UNKNOWN
            conf = baseline.baseline([neg_noise, neg_nav])
        return SectionClassification(label, conf)

    discovered.sort(key=lambda x: -x[1])
    top_label, top_weight = discovered[0]
    try:
        label = KnowledgeSectionClass(top_label)
    except ValueError:
        label = KnowledgeSectionClass.UNKNOWN

    if len(discovered) > 2:
        spread = baseline.spread([w for _, w in discovered[:4]])
        if spread > baseline.baseline([spread, 0.12]):
            label = KnowledgeSectionClass.MIXED

    if label == KnowledgeSectionClass.UNKNOWN and ev.utility > 0.2:
        label = KnowledgeSectionClass.ARTICLE

    if agg_article and label in DISCARD_CLASSES:
        for lbl, w in discovered:
            try:
                primary = KnowledgeSectionClass(lbl)
            except ValueError:
                continue
            if primary in PRIMARY_CLASSES:
                label = primary
                top_weight = w
                break

    return SectionClassification(label, top_weight, discovered=discovered)

def document_is_mixed(sections: list[tuple[RecoveredSection, SectionClassification]]) -> bool:
    if len(sections) < 2:
        return False
    labels = {c.label for _, c in sections if c.label != KnowledgeSectionClass.UNKNOWN}
    if len(labels) <= 1:
        return False
    primary = labels & PRIMARY_CLASSES
    noise = labels & DISCARD_CLASSES
    return bool(primary and noise)
