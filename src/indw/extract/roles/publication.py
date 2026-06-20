from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from indw.clean.artifact.decompose import compute_layout
from indw.extract.nav.context import (
    extract_navigation_features,
    score_navigation_role,
    structural_listing_score,
)
from indw.extract.structure.analyze import analyze_structure
from indw.clean.artifact.evidence_engine import resolve_semantic_evidence
from indw.clean.artifact.evidence_features import shared_feature_extractor
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator
from indw.extract.sections.integrity import _terminal_boundary_score
from indw.extract.sections.semantic import analyze_completion_cached as analyze_completion
from indw.clean.document.value import compute_structure_profile

_SEP_CHARS = '|>»·•→←/\\'
_PIPE = '|'


def _scaffold_budget_exit() -> bool:
    from indw.schedule.monitor.budget import doc_budget_exceeded
    return doc_budget_exceeded()


def _bounded_reverse_indices(high: int, low: int) -> list[int]:
    from indw.config.defaults import MERGE_SCAFFOLD_PROBE_CAP
    if high <= low:
        return []
    span = high - low
    if span <= MERGE_SCAFFOLD_PROBE_CAP:
        return list(range(high, low, -1))
    step = max(1, span // MERGE_SCAFFOLD_PROBE_CAP)
    return list(range(high, low, -step))[:MERGE_SCAFFOLD_PROBE_CAP]


def _bounded_forward_indices(low: int, high: int) -> list[int]:
    from indw.config.defaults import MERGE_SCAFFOLD_PROBE_CAP
    if high <= low:
        return []
    span = high - low
    if span <= MERGE_SCAFFOLD_PROBE_CAP:
        return list(range(low, high))
    step = max(1, span // MERGE_SCAFFOLD_PROBE_CAP)
    return list(range(low, high, step))[:MERGE_SCAFFOLD_PROBE_CAP]


def _scaffold_probe_indices(word_count: int, *, reverse: bool, low: int, high: int) -> list[int]:
    from indw.config.defaults import MERGE_SCAFFOLD_PROBE_WORD_FLOOR
    if word_count <= MERGE_SCAFFOLD_PROBE_WORD_FLOOR:
        if reverse:
            return list(range(high, low, -1))
        return list(range(low, high))
    if reverse:
        return _bounded_reverse_indices(high, low)
    return _bounded_forward_indices(low, high)


class PublicationRole(str, Enum):
    PUBLICATION_HEADER = 'publication_header'
    ISSUE_INFO = 'issue_info'
    VOLUME_INFO = 'volume_info'
    EDITION_INFO = 'edition_info'
    ARCHIVE_REF = 'archive_ref'
    HOMEPAGE_LINK = 'homepage_link'
    AUTHOR_BLOCK = 'author_block'
    DATE_BLOCK = 'date_block'
    ARTICLE_TITLE = 'article_title'
    SUBTITLE = 'subtitle'
    MASTHEAD = 'masthead'
    FOOTER_BLOCK = 'footer_block'
    CITATION = 'citation'
    NAVIGATION = 'navigation'
    ADVERTISEMENT = 'advertisement'
    KNOWLEDGE = 'knowledge'
    UNKNOWN = 'unknown'


DISCARD_PUBLICATION_ROLES = frozenset({
    PublicationRole.PUBLICATION_HEADER,
    PublicationRole.ISSUE_INFO,
    PublicationRole.VOLUME_INFO,
    PublicationRole.EDITION_INFO,
    PublicationRole.ARCHIVE_REF,
    PublicationRole.HOMEPAGE_LINK,
    PublicationRole.AUTHOR_BLOCK,
    PublicationRole.DATE_BLOCK,
    PublicationRole.MASTHEAD,
    PublicationRole.FOOTER_BLOCK,
    PublicationRole.NAVIGATION,
    PublicationRole.ADVERTISEMENT,
})

KNOWLEDGE_PUBLICATION_ROLES = frozenset({
    PublicationRole.KNOWLEDGE,
    PublicationRole.ARTICLE_TITLE,
    PublicationRole.SUBTITLE,
    PublicationRole.CITATION,
})


@dataclass
class PublicationSpan:
    text: str
    start: int
    end: int
    role: PublicationRole
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class PublicationRoleScore:
    publication_header: float = 0.0
    issue_info: float = 0.0
    volume_info: float = 0.0
    edition_info: float = 0.0
    archive_ref: float = 0.0
    homepage_link: float = 0.0
    author_block: float = 0.0
    date_block: float = 0.0
    article_title: float = 0.0
    subtitle: float = 0.0
    masthead: float = 0.0
    footer_block: float = 0.0
    citation: float = 0.0
    navigation: float = 0.0
    advertisement: float = 0.0
    knowledge: float = 0.0

    def scaffold_mass(self) -> float:
        return max(
            self.publication_header, self.issue_info, self.volume_info,
            self.edition_info, self.archive_ref, self.homepage_link,
            self.author_block, self.date_block, self.masthead,
            self.footer_block, self.navigation, self.advertisement,
        )

    def dominant(self) -> tuple[PublicationRole, float]:
        items = (
            (PublicationRole.PUBLICATION_HEADER, self.publication_header),
            (PublicationRole.ISSUE_INFO, self.issue_info),
            (PublicationRole.VOLUME_INFO, self.volume_info),
            (PublicationRole.EDITION_INFO, self.edition_info),
            (PublicationRole.ARCHIVE_REF, self.archive_ref),
            (PublicationRole.HOMEPAGE_LINK, self.homepage_link),
            (PublicationRole.AUTHOR_BLOCK, self.author_block),
            (PublicationRole.DATE_BLOCK, self.date_block),
            (PublicationRole.ARTICLE_TITLE, self.article_title),
            (PublicationRole.SUBTITLE, self.subtitle),
            (PublicationRole.MASTHEAD, self.masthead),
            (PublicationRole.FOOTER_BLOCK, self.footer_block),
            (PublicationRole.CITATION, self.citation),
            (PublicationRole.NAVIGATION, self.navigation),
            (PublicationRole.ADVERTISEMENT, self.advertisement),
            (PublicationRole.KNOWLEDGE, self.knowledge),
        )
        return max(items, key=lambda x: x[1])


def _separator_density(text: str) -> float:
    if not text:
        return 0.0
    tokens = max(len(text.split()), 1)
    sep = sum(text.count(c) for c in _SEP_CHARS)
    return min(1.0, sep / tokens)


def _pipe_part_count(text: str) -> int:
    if _PIPE not in text:
        return 0
    return len([p for p in text.split(_PIPE) if p.strip()])


def _prose_guard(raw, structural, ev, profile) -> float:
    if raw.word_count > 35 and structural.sentence_completeness_mean > 0.42:
        return 0.15
    if ev.utility > 0.14 and profile.explanation_ratio > 0.12:
        return 0.20
    if raw.word_count > 50:
        return 0.10
    return 1.0


def _venue_contact_tail_signal(text: str, raw, profile, ev) -> float:
    at_idx = text.find('@')
    if at_idx < 12:
        return 0.0
    head = text[:at_idx].strip()
    tail = text[at_idx:].strip()
    if len(head.split()) < 8:
        return 0.0
    head_ev = resolve_semantic_evidence(head)
    if head_ev.utility < 0.08:
        return 0.0
    tail_raw = shared_feature_extractor().extract(tail)
    contact = profile.contact_ratio + tail_raw.contact_token_ratio
    temporal = profile.date_ratio
    score = contact * 0.45 + temporal * 0.20 + tail_raw.numeric_token_ratio * 0.35
    if tail_raw.word_count > 28:
        score *= 0.55
    if ev.utility > 0.18:
        score *= 0.40
    return min(1.0, score)


def _pagination_footer_signal(
    text: str,
    raw,
    feat,
    ev,
    *,
    position_ratio: float,
) -> float:
    if raw.word_count > 10 or position_ratio < 0.55:
        return 0.0
    score = feat.digit_token_ratio * 0.55 + feat.uniform_line_ratio * 0.15
    if feat.digit_token_ratio > 0.20 and raw.word_count <= 8:
        score += 0.42
    if raw.word_count <= 6 and feat.digit_token_ratio > 0.12:
        score += 0.28
    if position_ratio > 0.75:
        score += 0.15
    if score >= 0.48:
        return min(1.0, score)
    if ev.utility > 0.22:
        return 0.0
    return min(1.0, score)


def _caps_run_signal(text: str, raw, feat, ev, *, position_ratio: float) -> float:
    if raw.word_count > 14 or raw.word_count < 3:
        return 0.0
    if feat.sentence_completeness > 0.45:
        return 0.0
    if ev.utility > 0.16:
        return 0.0
    score = feat.caps_token_ratio * 0.55 + (1.0 - feat.sentence_completeness) * 0.35
    if position_ratio < 0.12:
        score += 0.22
    if raw.copula_def_hits == 0 and raw.fact_relation_hits == 0:
        score += 0.15
    return min(1.0, score)


def _structural_kv_signal(text: str, raw, profile) -> float:
    stripped = text.strip()
    if len(stripped) < 6 or len(stripped) > 160:
        return 0.0
    words = stripped.split()
    if len(words) > 12:
        return 0.0
    colon_idx = stripped.find(':')
    if colon_idx < 1 or colon_idx > 32:
        return 0.0
    head = stripped[:colon_idx + 1]
    tail = stripped[colon_idx + 1:].strip()
    if not tail or len(head.split()) > 6:
        return 0.0
    if _separator_density(head) > 0.15:
        return 0.0
    score = 0.30
    if profile.date_ratio > 0.08 or raw.schedule_token_ratio > 0.06:
        score += 0.28
    if raw.numeric_token_ratio > 0.06:
        score += 0.15
    if raw.copula_def_hits == 0 and raw.fact_relation_hits == 0:
        score += 0.12
    return min(1.0, score)


def _pipe_chain_signal(text: str, feat, profile, ev) -> float:
    parts = _pipe_part_count(text)
    if parts < 2:
        return 0.0
    sep = _separator_density(text)
    util = ev.utility
    explain = profile.explanation_ratio
    score = min(1.0, sep * 0.55 + (parts / 8.0) * 0.35 + feat.uniform_line_ratio * 0.25)
    if util < 0.14:
        score += 0.18
    if explain < 0.12:
        score += 0.15
    if feat.caps_token_ratio > 0.12:
        score += 0.10
    if feat.digit_token_ratio > 0.05:
        score += 0.08
    if util > 0.22 and explain > 0.22:
        score *= 0.35
    return min(1.0, score)


def _masthead_signal(text: str, feat, profile, ev, *, position_ratio: float) -> float:
    pipe = _pipe_chain_signal(text, feat, profile, ev)
    listing = structural_listing_score(text, position_ratio=position_ratio)
    nav = score_navigation_role(text, position_ratio=position_ratio)
    util = ev.utility
    explain = profile.explanation_ratio
    score = max(pipe, listing * 0.85, nav.nav_mass() * 0.70)
    if feat.digit_token_ratio > 0.04 and explain < 0.14:
        score = max(score, feat.digit_token_ratio * 0.55 + listing * 0.30)
    if feat.caps_token_ratio > 0.10 and util < 0.16:
        score = max(score, feat.caps_token_ratio * 0.45 + pipe * 0.35)
    if position_ratio > 0.55 and pipe > 0.28:
        score += 0.12
    if util > 0.20 and explain > 0.20 and profile.instruction_ratio < 0.10:
        score *= 0.30
    return min(1.0, score)


def score_publication_roles(
    text: str,
    *,
    position_ratio: float = 0.5,
) -> PublicationRoleScore:
    if not text or not text.strip():
        return PublicationRoleScore(knowledge=1.0)

    from indw.clean.artifact.evidence_cache import get_publication_role_cache, publication_role_cache_key

    key = publication_role_cache_key(text, position_ratio)
    if key is not None:
        cache = get_publication_role_cache()
        hit = cache.get(key)
        if hit is not None:
            return hit
        result = _score_publication_roles_impl(text, position_ratio=position_ratio)
        cache.put(key, result)
        return result
    return _score_publication_roles_impl(text, position_ratio=position_ratio)


def _score_publication_roles_impl(
    text: str,
    *,
    position_ratio: float = 0.5,
) -> PublicationRoleScore:
    ev = resolve_semantic_evidence(text)
    from indw.clean.artifact.evidence_engine import evidence_raw_features
    raw = evidence_raw_features(ev, text)
    profile = compute_structure_profile(text, evidence=ev)
    structural = analyze_structure(text)
    feat = extract_navigation_features(text, position_ratio=position_ratio)
    nav = score_navigation_role(text, position_ratio=position_ratio, feat=feat)
    baseline = AdaptiveBaselineEstimator()

    util = ev.utility
    explain = profile.explanation_ratio
    prose_scale = _prose_guard(raw, structural, ev, profile)
    pipe = _pipe_chain_signal(text, feat, profile, ev)
    kv = _structural_kv_signal(text, raw, profile)
    masthead = _masthead_signal(text, feat, profile, ev, position_ratio=position_ratio)
    listing = structural_listing_score(text, position_ratio=position_ratio)
    temporal = profile.date_ratio + raw.schedule_token_ratio
    promo = ev.negative.get('promotional', 0.0)
    trans = ev.negative.get('transactional', 0.0)
    admin = ev.negative.get('administrative', 0.0)
    venue_tail = _venue_contact_tail_signal(text, raw, profile, ev)

    out = PublicationRoleScore()

    out.masthead = masthead
    caps_run = _caps_run_signal(text, raw, feat, ev, position_ratio=position_ratio)
    out.publication_header = min(1.0, max(
        masthead * 0.70 + listing * 0.35,
        caps_run * 0.85,
    ) + (0.15 if position_ratio < 0.18 and feat.explanation_ratio < 0.12 else 0.0))
    out.issue_info = min(1.0, (
        feat.digit_token_ratio * 0.55 + pipe * 0.40 + listing * 0.25
    ) * (1.0 - min(1.0, explain * 2.5)) * prose_scale)
    out.volume_info = min(1.0, (
        feat.caps_token_ratio * 0.50 + feat.digit_token_ratio * 0.35 + pipe * 0.30
    ) * (1.0 - min(1.0, explain * 2.2)) * prose_scale)
    out.edition_info = min(1.0, (temporal * 0.45 + feat.digit_token_ratio * 0.35 + pipe * 0.25) * prose_scale)
    out.archive_ref = min(1.0, nav.archive * 0.65 + nav.collection * 0.55 + listing * 0.40)
    out.homepage_link = min(1.0, nav.menu * 0.50 + feat.link_density * 0.55 + pipe * 0.30)
    out.author_block = min(1.0, kv * 0.55 + profile.contact_ratio * 0.40 + (
        0.25 if raw.word_count <= 10 and util < 0.14 and explain < 0.10 else 0.0
    ))
    out.date_block = 0.0
    if raw.word_count <= 22:
        out.date_block = min(1.0, temporal * 0.65 + kv * 0.35 + feat.digit_token_ratio * 0.20)
    out.navigation = min(1.0, nav.nav_mass() * 0.75 + pipe * 0.35 + venue_tail * 0.30)
    out.footer_block = min(1.0, max(
        nav.footer * 0.70 + (0.20 if position_ratio > 0.78 and util < 0.14 else 0.0),
        _pagination_footer_signal(text, raw, feat, ev, position_ratio=position_ratio) * 0.90,
    ))
    out.advertisement = min(1.0, promo * 0.55 + trans * 0.45 + feat.link_density * 0.25)

    out.article_title = baseline.baseline([
        util * 0.35,
        explain * 0.25,
        1.0 - masthead,
        structural.sentence_completeness_mean * 0.30,
    ]) if position_ratio < 0.22 and util > 0.08 and caps_run < 0.45 else 0.0
    out.subtitle = baseline.baseline([
        util * 0.30,
        explain * 0.20,
    ]) if 0.15 < position_ratio < 0.35 and util > 0.10 and masthead < 0.35 else 0.0
    out.citation = min(1.0, ev.quality.reference * 0.55 + admin * 0.25) if raw.citation_hits > 0 else 0.0

    know = baseline.baseline([
        util,
        explain,
        structural.sentence_completeness_mean,
        1.0 - out.scaffold_mass(),
    ])
    if util > 0.18 and explain > 0.18 and masthead < 0.38:
        know = max(know, util * 0.85 + explain * 0.45)
    if structural.sentence_completeness_mean > 0.52 and text.rstrip().endswith(('.', '!', '?')):
        know = max(know, util + explain * 0.35)
    page_footer = _pagination_footer_signal(text, raw, feat, ev, position_ratio=position_ratio)
    if page_footer > 0.42:
        know *= max(0.05, 1.0 - page_footer * 0.85)
    out.knowledge = min(1.0, know)

    scaffold = out.scaffold_mass()
    if scaffold > 0.42:
        out.knowledge = max(0.0, out.knowledge * (1.0 - scaffold * 0.75))

    return out


def dominant_publication_role(scores: PublicationRoleScore) -> tuple[PublicationRole, float]:
    return scores.dominant()


def is_scaffold_span(
    role: PublicationRole,
    text: str,
    *,
    confidence: float,
    position_ratio: float = 0.5,
) -> bool:
    if role == PublicationRole.ARTICLE_TITLE:
        scores = score_publication_roles(text, position_ratio=position_ratio)
        ev = resolve_semantic_evidence(text)
        memo_routing = text.count(':') >= 2 and scores.masthead + scores.publication_header > 0.22
        press_boiler = (
            scores.publication_header > 0.30
            or scores.masthead > 0.35
            or (memo_routing and scores.scaffold_mass() > scores.knowledge * 0.80)
        )
        if press_boiler and ev.utility < 0.20:
            return True
        if memo_routing and len(text.split()) <= 48:
            return True
    if role in KNOWLEDGE_PUBLICATION_ROLES:
        return False
    if role not in DISCARD_PUBLICATION_ROLES:
        return False
    if confidence < 0.32:
        return False
    ev = resolve_semantic_evidence(text)
    profile = compute_structure_profile(text, evidence=ev)
    if ev.utility > 0.28 and profile.explanation_ratio > 0.28:
        if confidence < 0.55:
            return False
    if len(text.split()) > 80 and ev.utility > 0.20 and profile.explanation_ratio > 0.22:
        return False
    return True


def publication_pipe_boundary(left: str, right: str, *, left_pos: float, right_pos: float) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    left_scores = score_publication_roles(left, position_ratio=left_pos)
    right_scores = score_publication_roles(right, position_ratio=right_pos)
    ev_l = resolve_semantic_evidence(left)
    ev_r = resolve_semantic_evidence(right)
    util_shift = max(0.0, ev_l.utility - ev_r.utility)
    scaffold_r = right_scores.scaffold_mass()
    know_l = left_scores.knowledge
    pipe_r = _pipe_chain_signal(
        right,
        extract_navigation_features(right, position_ratio=right_pos),
        compute_structure_profile(right, evidence=ev_r),
        ev_r,
    )
    if know_l > 0.22 and scaffold_r > 0.32:
        return min(1.0, know_l * 0.35 + scaffold_r * 0.45 + util_shift * 0.35 + 0.12)
    if pipe_r > 0.38 and ev_l.utility > 0.12:
        return min(1.0, pipe_r * 0.55 + util_shift * 0.40 + 0.15)
    if left_scores.knowledge > 0.18 and right_scores.masthead > 0.35:
        return min(1.0, left_scores.knowledge * 0.30 + right_scores.masthead * 0.50 + 0.12)
    return 0.0


def publication_role_boundary(left: str, right: str, *, left_pos: float, right_pos: float) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    left_scores = score_publication_roles(left, position_ratio=left_pos)
    right_scores = score_publication_roles(right, position_ratio=right_pos)
    left_role, left_conf = dominant_publication_role(left_scores)
    right_role, right_conf = dominant_publication_role(right_scores)

    know_left = left_role in KNOWLEDGE_PUBLICATION_ROLES and left_conf > 0.25
    scaffold_right = right_role in DISCARD_PUBLICATION_ROLES and right_conf > 0.30
    if know_left and scaffold_right:
        return min(1.0, left_conf * 0.40 + right_conf * 0.45 + 0.15)

    util_l = resolve_semantic_evidence(left).utility
    util_r = resolve_semantic_evidence(right).utility
    if util_l > 0.15 and right_scores.scaffold_mass() > 0.38:
        return min(1.0, right_scores.scaffold_mass() * 0.50 + (util_l - util_r) * 0.40 + 0.12)

    pipe_cut = publication_pipe_boundary(left, right, left_pos=left_pos, right_pos=right_pos)
    if pipe_cut > 0.0:
        return pipe_cut

    if left_role != right_role and left_conf > 0.28 and right_conf > 0.28:
        if left_role in KNOWLEDGE_PUBLICATION_ROLES or right_role in KNOWLEDGE_PUBLICATION_ROLES:
            return min(1.0, abs(left_conf - right_conf) * 0.45 + 0.25)
    return 0.0


def pipe_split_offsets(text: str) -> list[int]:
    if _PIPE not in text:
        return []
    cuts: list[int] = []
    baseline = AdaptiveBaselineEstimator()
    total = max(len(text), 1)
    for i, ch in enumerate(text):
        if ch != _PIPE:
            continue
        left = text[:i].rstrip()
        right = text[i + 1:].lstrip()
        if not left or not right:
            continue
        left_pos = len(left) / total
        right_pos = (i + 1) / total
        score = publication_pipe_boundary(left, right, left_pos=left_pos, right_pos=right_pos)
        if score >= baseline.baseline([0.38, 0.42]):
            cuts.append(i)
            if right.count(_PIPE) >= 1:
                break
    return cuts


def decompose_publication_spans(text: str) -> list[PublicationSpan]:
    from indw.clean.artifact.evidence_cache import cached_scaffold
    return cached_scaffold(text, 'decompose_spans', lambda: _decompose_publication_spans_impl(text))


def _decompose_publication_spans_impl(text: str) -> list[PublicationSpan]:
    if not text or not text.strip():
        return []

    blob = text.strip()
    base = text.find(blob)
    units: list[tuple[str, int, int]] = []

    blocks: list[tuple[str, int]] = []
    if '\n' in blob:
        cursor = 0
        for para in blob.split('\n'):
            stripped = para.strip()
            if not stripped:
                cursor += len(para) + 1
                continue
            idx = blob.find(stripped, cursor)
            if idx < 0:
                idx = cursor
            blocks.append((stripped, base + idx))
            cursor = idx + len(stripped)
    else:
        blocks = [(blob, base)]

    for block, block_start in blocks:
        pipe_cuts = pipe_split_offsets(block)
        if pipe_cuts:
            bounds = [0, *pipe_cuts, len(block)]
            for j in range(len(bounds) - 1):
                start, end = bounds[j], bounds[j + 1]
                chunk = block[start:end].strip()
                while chunk and chunk[0] == _PIPE:
                    chunk = chunk[1:].strip()
                    start += 1
                if not chunk:
                    continue
                units.append((chunk, block_start + start, block_start + start + len(chunk)))
        else:
            at_idx = block.find('@')
            if at_idx > 20:
                head = block[:at_idx].strip()
                tail = block[at_idx:].strip()
                tail_scores = score_publication_roles(tail, position_ratio=0.82)
                head_scores = score_publication_roles(head, position_ratio=0.25)
                if (
                    head_scores.knowledge > 0.18
                    and tail_scores.scaffold_mass() > 0.28
                ):
                    units.append((head, block_start, block_start + len(head)))
                    units.append((tail, block_start + at_idx, block_start + at_idx + len(tail)))
                    continue
            lead_stripped = strip_leading_publication_wrapper(block)
            latin_chars = sum(1 for c in block if 'a' <= c.lower() <= 'z')
            if (
                latin_chars >= max(12, len(block) * 0.12)
                and lead_stripped != block.strip()
                and len(lead_stripped.split()) >= 8
            ):
                idx = block.find(lead_stripped)
                if idx > 12:
                    open_word = lead_stripped.split()[0].lower()
                    if open_word in {'feeling', 'the', 'and', 'with', 'that', 'this', 'which', 'where', 'when'}:
                        pass
                    elif not lead_stripped[0].isupper():
                        pass
                    else:
                        head = block[:idx].strip()
                        head_scores = score_publication_roles(head, position_ratio=0.08)
                        if head_scores.scaffold_mass() > 0.40:
                            units.append((head, block_start, block_start + len(head)))
                            units.append((
                                lead_stripped,
                                block_start + idx,
                                block_start + idx + len(lead_stripped),
                            ))
                            continue
            units.append((block, block_start, block_start + len(block)))

    total = max(len(blob), 1)
    spans: list[PublicationSpan] = []
    for chunk, start, end in units:
        pos = (start - base) / total
        scores = score_publication_roles(chunk, position_ratio=pos)
        role, conf = dominant_publication_role(scores)
        spans.append(PublicationSpan(
            text=chunk,
            start=start,
            end=end,
            role=role,
            confidence=conf,
            scores={k: round(v, 4) for k, v in {
                'publication_header': scores.publication_header,
                'masthead': scores.masthead,
                'issue_info': scores.issue_info,
                'volume_info': scores.volume_info,
                'archive_ref': scores.archive_ref,
                'author_block': scores.author_block,
                'date_block': scores.date_block,
                'navigation': scores.navigation,
                'knowledge': scores.knowledge,
            }.items()},
        ))
    return spans


def _listing_tail_opening(words: list[str]) -> bool:
    if not words:
        return True
    w = words[0].lower().rstrip(':.,')
    if w in {'til', 'until', 'to', 'at', 'am', 'pm', '&', 'and', 'the', 'on'}:
        return True
    if w and w[0].isdigit():
        return True
    return False


def strip_trailing_inline_scaffold(text: str) -> str:
    from indw.clean.artifact.evidence_cache import cached_scaffold
    return cached_scaffold(text, 'trail_inline', lambda: _strip_trailing_inline_scaffold_impl(text))


def _strip_trailing_inline_scaffold_impl(text: str) -> str:
    blob = text.strip()
    if not blob:
        return blob
    latin = sum(1 for c in blob if 'a' <= c.lower() <= 'z')
    if latin < max(8, len(blob) * 0.08):
        return blob
    words = blob.split()
    if len(words) < 8:
        return blob
    full_comp = analyze_completion(blob)
    probe = _scaffold_probe_indices(len(words), reverse=True, low=4, high=len(words) - 1)
    left_cache: dict[int, tuple[Any, Any, Any]] = {}
    right_cache: dict[int, tuple[Any, Any, Any, Any]] = {}
    for i in probe:
        if _scaffold_budget_exit():
            return blob
        if i not in left_cache:
            left = ' '.join(words[:i])
            left_cache[i] = (
                left,
                score_publication_roles(left, position_ratio=0.35),
                resolve_semantic_evidence(left),
            )
        left, left_s, ev_l = left_cache[i]
        right = ' '.join(words[i:])
        if i not in right_cache:
            right_cache[i] = (
                right,
                score_publication_roles(right, position_ratio=0.78),
                resolve_semantic_evidence(right),
                extract_navigation_features(right, position_ratio=0.78),
            )
        right, right_s, ev_r, feat_r = right_cache[i]
        open_word = right.split()[0].lower() if right.split() else ''
        if open_word in {'with', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'the'}:
            continue
        tail_words = len(right.split())
        if tail_words > 6 and (right_s.knowledge > 0.22 or ev_r.utility > 0.14):
            continue
        if tail_words <= 6 and ev_r.utility > 0.16 and right_s.scaffold_mass() < 0.22:
            continue
        if right_s.scaffold_mass() < 0.24 and feat_r.caps_token_ratio < 0.38 and tail_words > 4:
            continue
        if left_s.knowledge < 0.14 and ev_l.utility < 0.10:
            continue
        cut = publication_role_boundary(left, right, left_pos=0.35, right_pos=0.78)
        if cut >= 0.18 and ev_l.utility >= ev_r.utility * 1.2:
            candidate = left.strip()
            comp = analyze_completion(candidate)
            if (
                comp.overall >= 0.55
                and comp.incomplete_probability < 0.42
                and comp.incomplete_probability <= full_comp.incomplete_probability + 0.01
                and comp.overall >= full_comp.overall
                and (not blob.rstrip().endswith(('.', '!', '?')) or candidate.rstrip().endswith(('.', '!', '?')))
            ):
                return candidate
        if len(words) - i <= 5 and ev_r.utility < 0.16 and right_s.scaffold_mass() > 0.20:
            if ev_l.utility >= 0.12 and ev_l.utility >= ev_r.utility:
                if (
                    feat_r.caps_token_ratio > 0.22
                    or right_s.publication_header > 0.18
                    or right_s.footer_block > 0.26
                ):
                    candidate = left.strip()
                    if candidate:
                        comp = analyze_completion(candidate)
                        if (
                            comp.overall >= 0.55
                            and comp.incomplete_probability < 0.42
                            and comp.incomplete_probability <= full_comp.incomplete_probability + 0.01
                            and comp.overall >= full_comp.overall
                            and (not blob.rstrip().endswith(('.', '!', '?')) or candidate.rstrip().endswith(('.', '!', '?')))
                        ):
                            return candidate
    return blob


def strip_leading_publication_wrapper(text: str) -> str:
    from indw.clean.artifact.evidence_cache import cached_scaffold
    return cached_scaffold(text, 'lead_wrapper', lambda: _strip_leading_publication_wrapper_impl(text))


def _strip_leading_publication_wrapper_impl(text: str) -> str:
    blob = text.strip()
    words = blob.split()
    min_words = 10 if len(words) < 20 else 16
    if len(words) < min_words:
        return blob
    search_end = min(len(words) - 6, max(14, int(len(words) * 0.65)))
    best_right = ''
    best_score = 0.0
    for i in _scaffold_probe_indices(len(words), reverse=False, low=5, high=search_end):
        if _scaffold_budget_exit():
            break
        left = ' '.join(words[:i])
        right = ' '.join(words[i:])
        if _listing_tail_opening(right.split()):
            continue
        colon_i = left.rfind(':')
        if colon_i > 8 and colon_i < 80:
            after_colon = left[colon_i + 1:].lstrip()
            before_colon = left[colon_i - 1] if colon_i > 0 else ''
            if not (after_colon[:1].isdigit() if after_colon else False) and not before_colon.isdigit():
                head_ev = resolve_semantic_evidence(left[:colon_i + 1])
                prof_h = compute_structure_profile(left[:colon_i + 1], evidence=head_ev)
                if prof_h.instruction_ratio > 0.06 or head_ev.quality.reference > 0.08:
                    continue
            if ' said' in left.lower()[-24:] and right.lstrip()[:1] in ('"', '\u201c', "'"):
                continue
        left_s = score_publication_roles(left, position_ratio=0.05)
        right_s = score_publication_roles(right, position_ratio=0.22)
        ev_l = resolve_semantic_evidence(left)
        ev_r = resolve_semantic_evidence(right)
        if ev_l.utility >= 0.14 and ev_l.utility >= ev_r.utility * 0.72:
            continue
        if left_s.scaffold_mass() < 0.48:
            continue
        if right_s.knowledge < 0.22 or ev_r.utility < 0.10:
            continue
        cut = publication_role_boundary(left, right, left_pos=0.05, right_pos=0.22)
        if cut >= 0.28 and left_s.scaffold_mass() >= 0.48:
            score = cut * left_s.scaffold_mass() * right_s.knowledge
            if score > best_score:
                best_score = score
                best_right = right.strip()
    return best_right if best_right else blob


def _strip_leading_masthead_body(text: str) -> str:
    blob = text.strip()
    words = blob.split()
    if len(words) < 14:
        return blob
    for i in range(6, min(12, len(words) - 8)):
        left = ' '.join(words[:i])
        right = ' '.join(words[i:])
        ls = score_publication_roles(left, position_ratio=0.12)
        rs = score_publication_roles(right, position_ratio=0.42)
        ev_r = resolve_semantic_evidence(right)
        cut = publication_role_boundary(left, right, left_pos=0.12, right_pos=0.42)
        if (
            cut >= 0.26
            and ls.article_title > 0.70
            and 6 <= len(left.split()) <= 11
            and rs.knowledge > 0.30
            and ev_r.utility >= 0.08
            and ls.scaffold_mass() >= 0.30
        ):
            return right.strip()
    return blob


def _strip_trailing_scaffold_tail(text: str) -> str:
    t = text.strip()
    if not t or len(t.split()) < 12:
        return t
    best = t
    for colon in (i for i, ch in enumerate(t) if ch == ':'):
        if colon < 8 or colon > len(t) - 12:
            continue
        head = t[:colon + 1].strip()
        tail = t[colon + 1:].strip()
        tail_block = tail.split('\n\n')[0].strip()
        if not tail_block or len(tail_block.split()) > 18:
            continue
        tail_s = score_publication_roles(tail_block, position_ratio=0.14)
        head_s = score_publication_roles(head, position_ratio=0.08)
        memo_routing = tail_block.count(':') >= 2 and len(tail_block.split()) <= 16
        if (
            tail_s.scaffold_mass() > 0.52
            and (
                tail_s.knowledge < 0.28
                or (memo_routing and tail_s.scaffold_mass() > tail_s.knowledge * 0.90)
            )
            and head_s.knowledge >= max(0.12, tail_s.knowledge * 0.85)
        ) and len(head.split()) >= 8:
            best = head
    return best


def strip_trailing_publication_footer(text: str) -> str:
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paras:
        return text.strip()
    while paras:
        tail = paras[-1]
        raw = shared_feature_extractor().extract(tail)
        feat = extract_navigation_features(tail, position_ratio=0.88)
        ev = resolve_semantic_evidence(tail)
        page = _pagination_footer_signal(tail, raw, feat, ev, position_ratio=0.88)
        tail_s = score_publication_roles(tail, position_ratio=0.88)
        if (
            page > 0.42
            or (tail_s.footer_block > 0.38 and len(tail.split()) <= 10)
            or (tail_s.scaffold_mass() > 0.45 and tail_s.knowledge < 0.25)
        ):
            paras.pop()
            continue
        break
    return '\n\n'.join(paras) if paras else ''


def is_pagination_footer_line(text: str, *, position_ratio: float = 0.85) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped.split()) > 10:
        return False
    raw = shared_feature_extractor().extract(stripped)
    feat = extract_navigation_features(stripped, position_ratio=position_ratio)
    ev = resolve_semantic_evidence(stripped)
    return _pagination_footer_signal(stripped, raw, feat, ev, position_ratio=position_ratio) > 0.42


def _strip_venue_tail(text: str) -> str:
    at_idx = text.find('@')
    if at_idx < 12:
        return text.strip()
    head = text[:at_idx].strip()
    tail = text[at_idx:].strip()
    if len(head.split()) < 8:
        return text.strip()
    tail_scores = score_publication_roles(tail, position_ratio=0.82)
    head_scores = score_publication_roles(head, position_ratio=0.30)
    if head_scores.knowledge > 0.18 and tail_scores.scaffold_mass() > 0.25:
        return head
    return text.strip()


def strip_publication_scaffolding(text: str) -> str:
    from indw.clean.artifact.evidence_cache import cached_scaffold
    return cached_scaffold(text, 'pub_scaffold', lambda: _strip_publication_scaffolding_impl(text))


def _strip_publication_scaffolding_impl(text: str) -> str:
    spans = decompose_publication_spans(text)
    if len(spans) > 1:
        kept: list[str] = []
        for span in spans:
            if is_scaffold_span(span.role, span.text, confidence=span.confidence):
                continue
            if span.role in KNOWLEDGE_PUBLICATION_ROLES or span.scores.get('knowledge', 0) > 0.20:
                cleaned = _strip_leading_masthead_body(
                    _strip_trailing_scaffold_tail(_strip_venue_tail(span.text.strip())),
                )
                if cleaned:
                    kept.append(cleaned)
            elif span.role == PublicationRole.UNKNOWN:
                ev = resolve_semantic_evidence(span.text)
                if ev.utility >= 0.12 or len(span.text.split()) >= 12:
                    cleaned = _strip_venue_tail(span.text.strip())
                    if cleaned:
                        kept.append(cleaned)
        if kept:
            return '\n\n'.join(kept)

    text = strip_trailing_publication_footer(text)
    if not text.strip():
        return ''
    spans = decompose_publication_spans(text)
    if not spans:
        return text.strip()

    if len(spans) == 1:
        span = spans[0]
        if is_scaffold_span(span.role, span.text, confidence=span.confidence):
            return ''
        body = _strip_trailing_scaffold_tail(
            _strip_venue_tail(span.text),
        )
        inline = strip_trailing_inline_scaffold(
            _strip_inline_scaffold(body),
        )
        return inline if inline else body.strip()

    kept: list[str] = []
    for span in spans:
        if is_scaffold_span(span.role, span.text, confidence=span.confidence):
            continue
        if span.role in KNOWLEDGE_PUBLICATION_ROLES:
            cleaned = _strip_inline_scaffold(_strip_venue_tail(span.text))
            if cleaned:
                kept.append(cleaned)
        elif span.role == PublicationRole.UNKNOWN:
            ev = resolve_semantic_evidence(span.text)
            if ev.utility >= 0.12 or len(span.text.split()) >= 12:
                cleaned = _strip_inline_scaffold(_strip_venue_tail(span.text))
                if cleaned:
                    kept.append(cleaned)

    if not kept:
        for span in spans:
            if span.role in KNOWLEDGE_PUBLICATION_ROLES or span.scores.get('knowledge', 0) > 0.25:
                cleaned = _strip_inline_scaffold(span.text)
                if cleaned:
                    kept.append(cleaned)
                    break
    return '\n\n'.join(kept) if kept else ''


def preprocess_publication_document(text: str) -> str:
    stripped = strip_publication_scaffolding(text)
    if stripped:
        return stripped
    spans = decompose_publication_spans(text)
    if not spans:
        return text.strip()
    know = [
        s.text.strip() for s in spans
        if s.role in KNOWLEDGE_PUBLICATION_ROLES
        or s.scores.get('knowledge', 0) > 0.22
        or not is_scaffold_span(s.role, s.text, confidence=s.confidence)
    ]
    if know:
        return '\n\n'.join(know)
    return text.strip()


def _strip_inline_scaffold(text: str) -> str:
    if _PIPE not in text:
        return text.strip()
    cuts = pipe_split_offsets(text)
    if not cuts:
        return text.strip()
    left = text[:cuts[0]].rstrip()
    if left and resolve_semantic_evidence(left).utility >= 0.08:
        return left
    spans = decompose_publication_spans(text)
    kept = [
        s.text.strip() for s in spans
        if not is_scaffold_span(s.role, s.text, confidence=s.confidence)
        and (s.role in KNOWLEDGE_PUBLICATION_ROLES or s.scores.get('knowledge', 0) > 0.22)
    ]
    return '\n\n'.join(kept) if kept else text.strip()


@dataclass
class PublicationLearner:
    _leakage: list[tuple[float, ...]] = field(default_factory=list)
    _boost: dict[str, float] = field(default_factory=dict)

    def record_surviving_scaffold(self, text: str, *, position_ratio: float = 0.5) -> None:
        scores = score_publication_roles(text, position_ratio=position_ratio)
        feat = extract_navigation_features(text, position_ratio=position_ratio)
        sig = (
            round(_separator_density(text), 3),
            round(scores.scaffold_mass(), 3),
            round(feat.digit_token_ratio, 3),
            round(feat.caps_token_ratio, 3),
            round(position_ratio, 3),
        )
        self._leakage.append(sig)
        if len(self._leakage) > 400:
            self._leakage.pop(0)

    def scaffold_boost(self) -> float:
        if len(self._leakage) < 6:
            return 0.0
        recent = self._leakage[-80:]
        mean_sep = sum(s[0] for s in recent) / len(recent)
        mean_scaffold = sum(s[1] for s in recent) / len(recent)
        boost = 0.0
        if mean_sep > 0.12:
            boost += 0.04
        if mean_scaffold > 0.35:
            boost += 0.05
        return min(0.12, boost)

    def cluster_report(self) -> list[dict[str, Any]]:
        if not self._leakage:
            return []
        buckets: dict[str, int] = {}
        for sep, scaffold, digit, caps, pos in self._leakage:
            if sep > 0.18:
                buckets['pipe_chain'] = buckets.get('pipe_chain', 0) + 1
            if scaffold > 0.40:
                buckets['masthead'] = buckets.get('masthead', 0) + 1
            if digit > 0.08 and pos < 0.20:
                buckets['header_digits'] = buckets.get('header_digits', 0) + 1
            if caps > 0.15 and pos < 0.25:
                buckets['header_caps'] = buckets.get('header_caps', 0) + 1
            if pos > 0.70:
                buckets['tail_scaffold'] = buckets.get('tail_scaffold', 0) + 1
        return [{'family': k, 'count': v} for k, v in sorted(buckets.items(), key=lambda x: -x[1])]
