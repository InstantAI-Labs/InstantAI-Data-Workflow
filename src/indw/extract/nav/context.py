from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from indw.clean.artifact.discovery_corpus import CorpusStatsAccumulator
from indw.clean.artifact.decompose import LayoutVector, compute_layout
from indw.clean.artifact.positional import position_confidence
from indw.extract.nav.template import TemplateMiner
from indw.filter.score.signals import shannon_entropy
from indw.clean.artifact.evidence_engine import compute_semantic_evidence
from indw.clean.artifact.evidence_features import shared_feature_extractor
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator
from indw.clean.document.value import compute_structure_profile

_NAV_ROLES = (
    'navigation', 'breadcrumb', 'menu', 'pagination',
    'archive', 'collection', 'footer', 'sidebar', 'sitemap',
)
_SEP_CHARS = '|>»/\\·•–—:'

_DEFAULT_WEIGHTS = {
    'nav_line': 0.22,
    'separator': 0.18,
    'link': 0.14,
    'low_knowledge': 0.12,
    'corpus': 0.16,
    'edge_position': 0.10,
    'uniform': 0.08,
    'incomplete': 0.06,
    'template': 0.10,
    'listing': 0.08,
}


@dataclass
class NavigationFeatures:
    position_ratio: float = 0.0
    char_entropy: float = 0.0
    knowledge_density: float = 0.0
    link_density: float = 0.0
    separator_density: float = 0.0
    nav_line_ratio: float = 0.0
    uniform_line_ratio: float = 0.0
    template_density: float = 0.0
    corpus_doc_rate: float = 0.0
    corpus_position_conf: float = 0.0
    sentence_completeness: float = 0.0
    caps_token_ratio: float = 0.0
    line_count: int = 0
    avg_line_len: float = 0.0
    digit_token_ratio: float = 0.0
    listing_ratio: float = 0.0
    explanation_ratio: float = 0.0
    neg_navigational: float = 0.0

    def layout_signature(self) -> tuple[float, ...]:
        return (
            round(self.separator_density, 3),
            round(self.nav_line_ratio, 3),
            round(self.uniform_line_ratio, 3),
            round(self.position_ratio, 3),
            round(self.digit_token_ratio, 3),
        )


@dataclass
class NavigationRoleScore:
    navigation: float = 0.0
    breadcrumb: float = 0.0
    menu: float = 0.0
    pagination: float = 0.0
    archive: float = 0.0
    collection: float = 0.0
    footer: float = 0.0
    sidebar: float = 0.0
    sitemap: float = 0.0
    article: float = 0.0

    def nav_mass(self) -> float:
        return max(
            self.navigation, self.breadcrumb, self.menu, self.pagination,
            self.archive, self.collection, self.footer, self.sidebar, self.sitemap,
        )

    def is_navigation(self, *, threshold: float = 0.42) -> bool:
        return self.nav_mass() >= threshold and self.article < 0.38

    def dominant(self) -> tuple[str, float]:
        items = (
            ('navigation', self.navigation),
            ('breadcrumb', self.breadcrumb),
            ('menu', self.menu),
            ('pagination', self.pagination),
            ('archive', self.archive),
            ('collection', self.collection),
            ('footer', self.footer),
            ('sidebar', self.sidebar),
            ('sitemap', self.sitemap),
            ('article', self.article),
        )
        return max(items, key=lambda x: x[1])

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in {
            'navigation': self.navigation,
            'breadcrumb': self.breadcrumb,
            'menu': self.menu,
            'pagination': self.pagination,
            'archive': self.archive,
            'collection': self.collection,
            'footer': self.footer,
            'sidebar': self.sidebar,
            'sitemap': self.sitemap,
            'article': self.article,
        }.items()}


@dataclass
class NavigationContext:
    accumulator: CorpusStatsAccumulator | None = None
    learner: NavigationLearner | None = None
    template_miner: TemplateMiner | None = None

    def miner(self) -> TemplateMiner:
        if self.template_miner is None:
            self.template_miner = TemplateMiner(self.accumulator)
        return self.template_miner


_CTX = NavigationContext()


def set_navigation_context(ctx: NavigationContext | None) -> None:
    global _CTX
    _CTX = ctx or NavigationContext()


def get_navigation_context() -> NavigationContext:
    return _CTX


def _sentence_completeness(line: str) -> float:
    s = line.strip()
    if not s:
        return 0.0
    if s[-1] in '.!?':
        return 1.0
    words = s.split()
    if len(words) >= 6:
        return 0.65
    if len(words) <= 2:
        return 0.2
    return 0.45


def _separator_density(text: str) -> float:
    if not text:
        return 0.0
    tokens = max(len(text.split()), 1)
    sep = sum(text.count(c) for c in _SEP_CHARS)
    return min(1.0, sep / tokens)


def _caps_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)


def extract_navigation_features(
    text: str,
    *,
    position_ratio: float = 0.5,
    layout: LayoutVector | None = None,
    ctx: NavigationContext | None = None,
) -> NavigationFeatures:
    ctx = ctx or get_navigation_context()
    corpus_active = (
        ctx.accumulator is not None and ctx.accumulator.docs_seen > 0
    )
    from indw.clean.artifact.evidence_cache import get_nav_feature_cache, nav_feature_cache_key

    key = nav_feature_cache_key(text, position_ratio, corpus_active=corpus_active)
    if key is not None:
        cache = get_nav_feature_cache()
        hit = cache.get(key)
        if hit is not None:
            return hit
        result = _extract_navigation_features_impl(
            text, position_ratio=position_ratio, layout=layout, ctx=ctx,
        )
        cache.put(key, result)
        return result
    return _extract_navigation_features_impl(
        text, position_ratio=position_ratio, layout=layout, ctx=ctx,
    )


def _extract_navigation_features_impl(
    text: str,
    *,
    position_ratio: float = 0.5,
    layout: LayoutVector | None = None,
    ctx: NavigationContext,
) -> NavigationFeatures:
    lay = layout or compute_layout(text)
    raw = shared_feature_extractor().extract(text)
    from indw.clean.artifact.evidence_engine import resolve_semantic_evidence
    ev = resolve_semantic_evidence(text)
    profile = compute_structure_profile(text, evidence=ev)
    lines = [ln for ln in raw.lines if ln.strip()]
    completeness = (
        sum(_sentence_completeness(ln) for ln in lines) / max(len(lines), 1)
        if lines else 0.0
    )
    tmpl = ctx.miner().analyze(text)

    feat = NavigationFeatures(
        position_ratio=position_ratio,
        char_entropy=shannon_entropy(text) / 8.0,
        knowledge_density=ev.utility,
        link_density=min(1.0, raw.url_char_ratio + raw.anchor_density),
        separator_density=_separator_density(text),
        nav_line_ratio=raw.nav_line_ratio,
        uniform_line_ratio=raw.uniform_line_ratio,
        template_density=tmpl.template_density,
        sentence_completeness=completeness,
        caps_token_ratio=_caps_ratio(raw.words),
        line_count=lay.line_count,
        avg_line_len=lay.avg_len,
        digit_token_ratio=raw.numeric_token_ratio,
        listing_ratio=profile.listing_ratio,
        explanation_ratio=profile.explanation_ratio,
        neg_navigational=ev.negative.get('navigational', 0.0),
    )

    if ctx.accumulator is not None and ctx.accumulator.docs_seen > 0:
        frag = ctx.accumulator.fragment_for_text(text, lay)
        if frag is not None:
            feat.corpus_doc_rate = frag.doc_rate(ctx.accumulator.docs_seen)
            feat.corpus_position_conf = position_confidence(
                frag.position_histogram, frag.doc_frequency, ctx.accumulator.docs_seen,
            )
    return feat


def _edge_position(pos: float) -> float:
    if pos < 0.12:
        return 1.0 - pos / 0.12
    if pos > 0.88:
        return (pos - 0.88) / 0.12
    return 0.0


def score_navigation_role(
    text: str,
    *,
    position_ratio: float = 0.5,
    layout: LayoutVector | None = None,
    ctx: NavigationContext | None = None,
    neighbor_knowledge: float | None = None,
    feat: NavigationFeatures | None = None,
) -> NavigationRoleScore:
    if not text or not text.strip():
        return NavigationRoleScore(article=1.0)

    ctx = ctx or get_navigation_context()
    if feat is None:
        feat = extract_navigation_features(
            text, position_ratio=position_ratio, layout=layout, ctx=ctx,
        )
    lay = layout or compute_layout(text)
    baseline = AdaptiveBaselineEstimator()
    weights = dict(_DEFAULT_WEIGHTS)
    if ctx.learner is not None:
        weights.update(ctx.learner.weight_adjustments())

    edge = _edge_position(position_ratio)
    corpus_sig = feat.corpus_doc_rate * feat.corpus_position_conf
    low_know = 1.0 - min(1.0, feat.knowledge_density)
    incomplete = 1.0 - feat.sentence_completeness

    nav_core = baseline.baseline([
        feat.nav_line_ratio * weights['nav_line'],
        feat.separator_density * weights['separator'],
        feat.link_density * weights['link'],
        low_know * weights['low_knowledge'],
        corpus_sig * weights['corpus'],
        edge * weights['edge_position'],
        feat.uniform_line_ratio * weights['uniform'],
        incomplete * weights['incomplete'],
        feat.template_density * weights['template'],
        feat.listing_ratio * weights['listing'],
        feat.neg_navigational * 0.12,
    ])

    if feat.knowledge_density > 0.14 and feat.explanation_ratio > 0.18:
        nav_core *= max(0.15, 1.0 - feat.knowledge_density * 1.4)
    if feat.sentence_completeness > 0.55 and feat.avg_line_len > 80:
        nav_core *= 0.45
    if feat.knowledge_density > 0.20 and feat.explanation_ratio > 0.25:
        nav_core = min(nav_core, 0.28)

    structural_nav = baseline.baseline([
        feat.nav_line_ratio * 2.2,
        feat.separator_density * 2.0,
        feat.link_density * 1.8,
        feat.neg_navigational,
    ])
    if (
        feat.sentence_completeness > 0.52
        and structural_nav < 0.14
        and feat.nav_line_ratio < 0.08
        and feat.separator_density < 0.08
    ):
        nav_core = min(nav_core, 0.16)
        if feat.knowledge_density > 0.08:
            nav_core = min(nav_core, 0.12)

    if neighbor_knowledge is not None and neighbor_knowledge > 0.18 and nav_core > 0.15:
        nav_core = min(nav_core, max(0.08, nav_core * 0.45))

    article = baseline.baseline([
        feat.knowledge_density,
        feat.explanation_ratio,
        feat.sentence_completeness,
        1.0 - nav_core,
    ])

    out = NavigationRoleScore(article=article)
    out.navigation = nav_core

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chain_lines = sum(1 for ln in lines if sum(ln.count(c) for c in _SEP_CHARS) >= 1)
    chain_ratio = chain_lines / max(len(lines), 1)

    if position_ratio < 0.18 and chain_ratio > 0.5 and feat.avg_line_len < 120:
        out.breadcrumb = min(1.0, nav_core * 0.7 + chain_ratio * 0.35 + edge * 0.2)
    if lay.list_ratio > 0.35 and feat.avg_line_len < 90 and feat.line_count <= 8:
        out.menu = min(1.0, nav_core * 0.65 + feat.uniform_line_ratio * 0.35)
    if feat.digit_token_ratio > 0.12 and feat.avg_line_len < 80 and edge > 0.3:
        out.pagination = min(1.0, nav_core * 0.5 + feat.digit_token_ratio * 0.45 + edge * 0.2)
    if feat.listing_ratio > 0.28 and feat.explanation_ratio < 0.22:
        out.archive = min(1.0, feat.listing_ratio * 0.55 + feat.template_density * 0.35 + low_know * 0.2)
        out.collection = min(1.0, feat.listing_ratio * 0.45 + corpus_sig * 0.4)
    if position_ratio < 0.15 and feat.digit_token_ratio > 0.08 and feat.explanation_ratio < 0.18:
        out.archive = max(out.archive, min(1.0, feat.digit_token_ratio * 0.65 + feat.listing_ratio * 0.35 + edge * 0.1))
    if position_ratio < 0.25 and feat.explanation_ratio < 0.08 and feat.digit_token_ratio > 0.06:
        out.archive = max(out.archive, 0.52)
    if position_ratio > 0.82 and (edge > 0.4 or feat.template_density > 0.25):
        out.footer = min(1.0, nav_core * 0.75 + edge * 0.25)
    if 0.05 < position_ratio < 0.25 and lay.line_count <= 12 and feat.uniform_line_ratio > 0.4:
        out.sidebar = min(1.0, nav_core * 0.55 + feat.uniform_line_ratio * 0.3)
    if feat.corpus_doc_rate > 0.15 and feat.template_density > 0.3 and feat.line_count >= 4:
        out.sitemap = min(1.0, corpus_sig * 0.6 + feat.template_density * 0.4)

    nav_peak = out.nav_mass()
    if nav_peak > 0.45:
        out.article = max(0.0, article * (1.0 - nav_peak * 0.65))
    else:
        out.navigation = max(0.0, out.navigation - article * 0.25)

    return out


def structural_listing_score(text: str, *, position_ratio: float = 0.0) -> float:
    feat = extract_navigation_features(text, position_ratio=position_ratio)
    base = min(1.0, feat.listing_ratio * 0.5 + feat.template_density * 0.35 + (1.0 - feat.explanation_ratio) * 0.2)
    if position_ratio < 0.2 and feat.digit_token_ratio > 0.10:
        base = max(base, min(1.0, feat.digit_token_ratio * 0.55 + feat.listing_ratio * 0.3))
    return base


def nav_transition_score(left_text: str, right_text: str, *, left_pos: float, right_pos: float) -> float:
    left = score_navigation_role(left_text, position_ratio=left_pos)
    right = score_navigation_role(right_text, position_ratio=right_pos)
    ev_l = compute_semantic_evidence(left_text)
    ev_r = compute_semantic_evidence(right_text)
    util_shift = max(0.0, ev_r.utility - ev_l.utility)
    left_list = structural_listing_score(left_text, position_ratio=left_pos)
    if left.nav_mass() < 0.35 and right.article > 0.28:
        return min(1.0, right.article - left.nav_mass() + util_shift * 0.35 + 0.12)
    if left_list > 0.32 and ev_r.utility > 0.35 and util_shift > 0.1:
        return min(1.0, left_list * 0.45 + util_shift * 0.5 + 0.1)
    if left.article > 0.4 and right.nav_mass() > 0.35:
        return min(1.0, right.nav_mass() - left.article + 0.15)
    if util_shift > 0.25 and right.article > 0.22 and left_pos < 0.35:
        return min(1.0, util_shift * 0.6 + right.article * 0.25)
    return 0.0


@dataclass
class NavigationLearner:
    _leakage: list[NavigationFeatures] = field(default_factory=list)
    _adjust: dict[str, float] = field(default_factory=dict)

    def record_surviving_nav(self, text: str, *, position_ratio: float = 0.5) -> None:
        feat = extract_navigation_features(text, position_ratio=position_ratio)
        self._leakage.append(feat)
        if len(self._leakage) > 400:
            self._leakage.pop(0)

    def weight_adjustments(self) -> dict[str, float]:
        if len(self._leakage) < 8:
            return self._adjust
        clusters: dict[tuple[float, ...], list[NavigationFeatures]] = {}
        for f in self._leakage[-120:]:
            sig = f.layout_signature()
            key = tuple(round(v, 2) for v in sig)
            clusters.setdefault(key, []).append(f)
        top = max(clusters.values(), key=len)
        if len(top) < 3:
            return self._adjust
        mean_sep = sum(f.separator_density for f in top) / len(top)
        mean_nav = sum(f.nav_line_ratio for f in top) / len(top)
        mean_corpus = sum(f.corpus_doc_rate for f in top) / len(top)
        adj = dict(self._adjust)
        if mean_sep > 0.15:
            adj['separator'] = min(0.28, _DEFAULT_WEIGHTS['separator'] + 0.04)
        if mean_nav > 0.2:
            adj['nav_line'] = min(0.30, _DEFAULT_WEIGHTS['nav_line'] + 0.04)
        if mean_corpus > 0.08:
            adj['corpus'] = min(0.22, _DEFAULT_WEIGHTS['corpus'] + 0.03)
        self._adjust = adj
        return adj

    def cluster_report(self) -> list[dict[str, Any]]:
        if not self._leakage:
            return []
        buckets: dict[str, int] = {}
        for f in self._leakage:
            if f.separator_density > 0.2:
                buckets['separator_chain'] = buckets.get('separator_chain', 0) + 1
            if f.corpus_doc_rate > 0.1:
                buckets['corpus_template'] = buckets.get('corpus_template', 0) + 1
            if f.position_ratio < 0.15:
                buckets['head_position'] = buckets.get('head_position', 0) + 1
            if f.position_ratio > 0.85:
                buckets['tail_position'] = buckets.get('tail_position', 0) + 1
        return [{'family': k, 'count': v} for k, v in sorted(buckets.items(), key=lambda x: -x[1])]
