from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from indw.clean.document.adaptive import aggregate_component_noise
from indw.clean.document.normalize import meaningful_char_count
from indw.clean.document.patterns import _CONTROL, _HTML_TAG, _METADATA_LINE, _UI_LINE, _WORD
from indw.clean.document.value import (
    ContentValueSignals,
    DocumentStructureProfile,
    analyze_content_value,
    build_analysis_bundle,
)
from indw.clean.artifact.evidence import (
    shared_baseline_estimator,
    PopulationAdaptiveScaler,
    structural_integrity,
)
from indw.clean.artifact.evidence_features import shared_feature_extractor
from indw.clean.gate.policy import DocumentGatePolicy, resolve_document_gate_policy
from indw.filter.score.signals import QualitySignals

_HTML_DOM = re.compile(
    r'(?i)(?:'
    r'<\s*(?:html|body|div|nav|footer|header|script|style|iframe|form)\b|'
    r'(?:function\s*\(|document\.|window\.|onclick\s*=|onload\s*=)|'
    r'(?:\{[^}]{0,80}(?:color|font-size|margin|padding|display)\s*:)|'
    r'(?:var\s+\w+\s*=|const\s+\w+\s*=.*=>)'
    r')'
)

_DISAMBIG = re.compile(
    r'(?i)(?:'
    r'\bmay\s+refer\s+to\s*:|\bcan\s+refer\s+to\s*:|\bmight\s+refer\s+to\s*:|\bdisambiguation\b|'
    r'this\s+(?:term|page|article)\s+may\s+refer\s+to'
    r')'
)

_DISAMBIG_LIST = re.compile(r'(?im)^\s*(?:\d+\.\s+)?[\w\s\-]{2,60}\s*(?:\(\s*disambiguation\s*\))?\s*$')
_URL = re.compile(r'https?://\S+|www\.\S+')

_KEYBOARD_SMASH = re.compile(
    r'(?i)\b(?:'
    r'asdfgh(?:jkl)?|qwertyuiop|zxcvbnm|qazwsx|'
    r'[bcdfghjklmnpqrstvwxyz]{6,}|'
    r'(?:[a-z]\s+){4,}[a-z]\s*$'
    r')\b'
)

REPLACEMENT_CHAR = '\ufffd'
_REPLACEMENT_CHAR = REPLACEMENT_CHAR

@dataclass
class DocumentGateResult:
    keep: bool = True
    reason: str = ''
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {'keep': self.keep, 'reason': self.reason, 'scores': self.scores}

def resolve_gate_policy() -> DocumentGatePolicy:
    return resolve_document_gate_policy()


def document_gate_raw(text: str, raw: Any | None = None) -> Any:
    if raw is not None:
        return raw
    return shared_feature_extractor().extract(text)


def html_tag_count(text: str) -> int:
    return len(_HTML_TAG.findall(text))


def html_dom_pattern_count(text: str) -> int:
    return len(_HTML_DOM.findall(text))


def keyboard_smash_hits(text: str) -> int:
    return len(_KEYBOARD_SMASH.findall(text))


def disambiguation_match(text: str) -> bool:
    return bool(_DISAMBIG.search(text))


def disambig_list_line_count(text: str) -> int:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return sum(
        1 for ln in lines
        if _DISAMBIG_LIST.match(ln) or re.match(r'^\s*\d+\.\s+\S', ln)
    )


def _line_nav_ratio(text: str, raw: Any | None = None) -> float:
    raw = document_gate_raw(text, raw=raw)
    return raw.nav_line_ratio


def line_nav_ratio(text: str, raw: Any | None = None) -> float:
    return _line_nav_ratio(text, raw=raw)

def _evergreen_score(
    text: str,
    *,
    word_count: int,
    meaningful: int,
    profile: DocumentStructureProfile | None = None,
    raw: Any | None = None,
) -> float:
    raw = raw or document_gate_raw(text)
    baseline = shared_baseline_estimator()
    anchor = PopulationAdaptiveScaler.rate(raw.anchor_density, raw.word_count, raw.line_count)
    if profile is not None:
        base = baseline.baseline([profile.explanation_ratio, profile.fact_ratio, anchor])
    else:
        base = anchor
    if raw.fence_char_ratio > 0 or raw.structured_line_ratio > 0:
        base = max(base, structural_integrity(raw))
    if profile is not None:
        depth = PopulationAdaptiveScaler.rate(meaningful, word_count, raw.char_count)
        base = max(base, profile.explanation_ratio * depth)
    return min(1.0, base)

def _has_substantive_education(
    text: str,
    *,
    word_count: int,
    meaningful: int,
    profile: DocumentStructureProfile | None = None,
    raw: Any | None = None,
) -> bool:
    raw = raw or document_gate_raw(text)
    baseline = shared_baseline_estimator()
    anchor = PopulationAdaptiveScaler.rate(raw.anchor_density, raw.word_count, raw.line_count)
    if profile is not None:
        if profile.explanation_ratio > baseline.baseline([profile.fact_ratio, anchor]) and anchor > 0:
            return True
        if profile.fact_ratio > baseline.baseline([profile.explanation_ratio, anchor]):
            return True
    if raw.fence_char_ratio > 0 or raw.table_line_ratio > 0:
        return True
    evergreen = _evergreen_score(text, word_count=word_count, meaningful=meaningful, profile=profile, raw=raw)
    return evergreen > baseline.baseline([evergreen, anchor, profile.explanation_ratio if profile else 0.0])

def _has_evergreen_value(
    text: str,
    *,
    word_count: int,
    meaningful: int,
    profile: DocumentStructureProfile | None = None,
) -> bool:
    if _has_substantive_education(
        text, word_count=word_count, meaningful=meaningful, profile=profile,
    ):
        return True
    raw = document_gate_raw(text)
    evg = _evergreen_score(text, word_count=word_count, meaningful=meaningful, profile=profile, raw=raw)
    anchor = PopulationAdaptiveScaler.rate(raw.anchor_density, raw.word_count, raw.line_count)
    baseline = shared_baseline_estimator()
    evg_floor = baseline.baseline([
        anchor,
        profile.explanation_ratio if profile else 0.0,
        profile.fact_ratio if profile else 0.0,
    ])
    return evg > baseline.spread([evg_floor, anchor])

def _knowledge_salvage(
    *,
    evidence: Any,
    intent: Any,
    profile: DocumentStructureProfile,
    evergreen_score: float,
    nav_ratio: float = 0.0,
    html_ratio: float = 0.0,
    seo_density: float = 0.0,
) -> bool:
    baseline = shared_baseline_estimator()
    neg = evidence.negative or {}
    promo = max(
        neg.get('promotional', 0.0),
        neg.get('transactional', 0.0),
        intent.promotional,
        intent.transactional,
        seo_density,
    )
    structural = baseline.baseline([
        nav_ratio,
        html_ratio,
        seo_density,
        neg.get('navigational', 0.0),
        neg.get('corruption', 0.0),
        neg.get('synthetic', 0.0),
    ])
    substance = baseline.baseline([
        evidence.utility,
        evidence.semantic_strength,
        intent.knowledge,
        evergreen_score,
        profile.explanation_ratio,
        profile.fact_ratio,
    ])
    nav_neg = neg.get('navigational', 0.0)
    if evidence.preserve:
        return True
    edu_floor = baseline.baseline([structural, promo, evidence.uncertainty])
    if profile.explanation_ratio > edu_floor:
        if nav_neg <= baseline.baseline([nav_neg, profile.explanation_ratio]):
            return True
    if profile.fact_ratio > edu_floor:
        if nav_neg <= baseline.baseline([nav_neg, profile.fact_ratio]):
            return True
    if structural > baseline.spread([structural, substance]):
        return False
    if promo >= baseline.spread([promo, intent.knowledge, evidence.semantic_strength]):
        if intent.knowledge <= baseline.baseline([promo, intent.entertainment]):
            return False
    return substance > baseline.spread([substance, structural, promo])

def _ai_filler_without_substance(
    text: str,
    *,
    word_count: int,
    meaningful: int,
    bundle: Any,
    policy: DocumentGatePolicy,
) -> float:
    filt = bundle.filters
    profile = bundle.profile
    filler_density = max(
        filt.ai_verbosity_score,
        filt.discourse_template_score * policy.ai_discourse_template_multiplier,
        filt.low_information_score * policy.ai_low_information_multiplier,
    )
    evergreen = _evergreen_score(
        text, word_count=word_count, meaningful=meaningful, profile=profile,
    )
    if filler_density < policy.ai_min_filler_density:
        return 0.0
    raw = document_gate_raw(text)
    anchor = PopulationAdaptiveScaler.rate(raw.anchor_density, raw.word_count, raw.line_count)
    if evergreen > filler_density or anchor > filler_density:
        return 0.0
    unique_ratio = len(set(_WORD.findall(text.lower()))) / max(word_count, 1)
    if unique_ratio > filler_density and filler_density < evergreen:
        return 0.0
    damp = shared_baseline_estimator().baseline([evergreen, unique_ratio, filler_density])
    return min(1.0, filler_density * (1.0 - damp))

def evaluate_document_gate(
    text: str,
    *,
    bundle: Any | None = None,
    policy: DocumentGatePolicy | None = None,
    raw: Any | None = None,
) -> DocumentGateResult:
    pol = policy or resolve_document_gate_policy()
    if not text or not text.strip():
        return DocumentGateResult(keep=False, reason='empty')

    t = text.strip()
    n = len(t)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    words = _WORD.findall(t.lower())
    word_count = max(len(words), 1)
    meaningful = meaningful_char_count(t)
    ctx = bundle or build_analysis_bundle(t)
    profile = ctx.profile
    raw = raw or document_gate_raw(t)
    evidence = ctx.evidence(t)
    evergreen_score = _evergreen_score(
        t, word_count=word_count, meaningful=meaningful, profile=profile, raw=raw,
    )
    substantive = _has_substantive_education(
        t, word_count=word_count, meaningful=meaningful, profile=profile, raw=raw,
    )

    baseline = shared_baseline_estimator()
    short_bound = PopulationAdaptiveScaler.short_doc_boundary(raw)

    scores: dict[str, float] = {'evergreen': round(evergreen_score, 4)}

    repl = t.count(_REPLACEMENT_CHAR)
    ctrl = len(_CONTROL.findall(t))
    alpha = sum(c.isalpha() for c in t) / max(n, 1)
    smash_hits = len(_KEYBOARD_SMASH.findall(t))
    corruption = min(
        1.0,
        repl / max(n / pol.repl_char_divisor, 1)
        + ctrl / max(n / pol.ctrl_char_divisor, 1)
        + smash_hits * pol.smash_hit_weight,
    )
    scores['corruption'] = round(corruption, 4)
    corr_peer = baseline.baseline([
        corruption, 1.0 - alpha,
        smash_hits / max(word_count / pol.word_count_smash_divisor, 1),
    ])
    alpha_floor = baseline.baseline([alpha, pol.alpha_floor])
    if (
        repl > pol.max_replacement_chars
        or alpha < alpha_floor
        or smash_hits >= pol.min_keyboard_smash_hits
    ) and not evidence.preserve:
        if corruption > corr_peer or (
            alpha < baseline.spread([alpha, corruption]) and meaningful < short_bound
        ):
            return DocumentGateResult(keep=False, reason='extreme_corruption', scores=scores)

    nav_ratio = _line_nav_ratio(t, raw)
    scores['nav_ratio'] = round(nav_ratio, 4)
    html_tags = len(_HTML_TAG.findall(t))
    dom_hits = len(_HTML_DOM.findall(t))
    html_ratio = min(1.0, (html_tags * pol.html_tag_weight + dom_hits * pol.html_dom_weight) / max(n, 1))
    scores['html_dump'] = round(html_ratio, 4)
    intent = ctx.intent(text=t)
    salvage = _knowledge_salvage(
        evidence=evidence,
        intent=intent,
        profile=profile,
        evergreen_score=evergreen_score,
        nav_ratio=nav_ratio,
        html_ratio=html_ratio,
    )
    html_junk = baseline.baseline([
        html_ratio, nav_ratio, (evidence.negative or {}).get('corruption', 0.0),
        (evidence.negative or {}).get('navigational', 0.0),
    ])
    if (
        html_junk > baseline.spread([html_junk, evidence.utility, evidence.semantic_strength])
        and html_tags >= max(pol.html_min_tags_floor, int(short_bound))
    ):
        return DocumentGateResult(keep=False, reason='html_dump', scores=scores)

    nav_junk = baseline.baseline([nav_ratio, html_ratio, (evidence.negative or {}).get('navigational', 0.0)])
    if (
        nav_junk > baseline.spread([nav_junk, evidence.utility, evidence.semantic_strength])
        and meaningful < short_bound * pol.nav_meaningful_short_multiplier
    ):
        return DocumentGateResult(keep=False, reason='navigation_boilerplate', scores=scores)

    if _DISAMBIG.search(t):
        list_lines = sum(1 for ln in lines if _DISAMBIG_LIST.match(ln) or re.match(r'^\s*\d+\.\s+\S', ln))
        nav_neg = (evidence.negative or {}).get('navigational', 0.0)
        disambig_bound = short_bound * max(pol.disambig_short_bound_line_multiplier, list_lines)
        nav_dom = nav_neg > baseline.baseline([intent.knowledge, evidence.semantic_strength])
        if list_lines >= pol.disambig_min_list_lines and (meaningful < disambig_bound or nav_dom):
            scores['disambig_list_ratio'] = list_lines / max(len(lines), 1)
            return DocumentGateResult(keep=False, reason='disambiguation_page', scores=scores)

    seo_density = max(
        ctx.filters.seo_spam_score,
        ctx.filters.keyword_stuffing_score * pol.seo_keyword_stuffing_multiplier,
        profile.transaction_ratio * pol.seo_transaction_ratio_multiplier,
    )
    scores['seo_spam'] = round(min(1.0, seo_density), 4)
    link_density = len(_URL.findall(t)) / max(word_count / pol.seo_link_density_word_divisor, 1)
    knowledge_floor = baseline.baseline([intent.knowledge, evidence.semantic_strength, scores['evergreen']])
    seo_peer = baseline.baseline([seo_density, intent.entertainment, profile.transaction_ratio])
    commercial = max(
        seo_density,
        intent.promotional,
        intent.transactional,
        profile.transaction_ratio,
        ctx.filters.commercial_score,
    )
    comm_dom = baseline.spread([commercial, intent.knowledge, evidence.semantic_strength, scores['evergreen']])
    salvage = _knowledge_salvage(
        evidence=evidence,
        intent=intent,
        profile=profile,
        evergreen_score=evergreen_score,
        nav_ratio=nav_ratio,
        html_ratio=html_ratio,
        seo_density=seo_density,
    )
    blatant_seo = (
        commercial > comm_dom
        and intent.knowledge < knowledge_floor
        and not salvage
    )
    if blatant_seo:
        return DocumentGateResult(keep=False, reason='seo_spam', scores=scores)
    if (
        seo_density > baseline.spread([seo_density, nav_ratio, html_ratio])
        and meaningful < short_bound * pol.seo_meaningful_short_multiplier
        and not salvage
    ):
        return DocumentGateResult(keep=False, reason='seo_spam', scores=scores)

    scores['entertainment'] = round(min(1.0, intent.entertainment), 4)
    transient_density = shared_baseline_estimator().baseline([
        intent.transience, profile.date_ratio, raw.schedule_token_ratio,
    ])
    scores['transient_news'] = round(min(1.0, transient_density), 4)
    ent_dom = baseline.spread([intent.entertainment, intent.knowledge, evidence.semantic_strength])
    if intent.entertainment > ent_dom and intent.knowledge < knowledge_floor and not salvage:
        return DocumentGateResult(
            keep=False,
            reason=evidence.discard_reason or 'entertainment_clickbait',
            scores=scores,
        )
    ai_filler = _ai_filler_without_substance(
        t, word_count=word_count, meaningful=meaningful, bundle=ctx, policy=pol,
    )
    scores['ai_filler'] = round(ai_filler, 4)
    filler_peer = baseline.baseline([ai_filler, seo_density, nav_ratio])
    synth_neg = (evidence.negative or {}).get('synthetic', 0.0)
    if (
        ai_filler > filler_peer
        and synth_neg > baseline.baseline([synth_neg, evidence.semantic_strength])
        and meaningful < short_bound * pol.ai_meaningful_short_multiplier
        and not salvage
    ):
        return DocumentGateResult(keep=False, reason='ai_filler', scores=scores)

    if not evidence.preserve and not salvage:
        return DocumentGateResult(keep=False, reason=evidence.discard_reason or 'low_utility', scores=scores)

    if meaningful < short_bound and not salvage:
        junk_vals = [nav_ratio, seo_density, html_ratio, intent.entertainment]
        junk_peer = baseline.baseline(junk_vals)
        junk_spread = baseline.spread(junk_vals)
        if junk_peer > junk_spread or (
            len(lines) <= pol.short_junk_max_lines
            and nav_ratio > baseline.baseline([nav_ratio, junk_peer])
        ):
            return DocumentGateResult(keep=False, reason='short_junk', scores=scores)

    return DocumentGateResult(keep=True, scores=scores)

def _line_pattern_ratio(text: str, patterns: tuple[re.Pattern[str], ...]) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    hits = sum(1 for ln in lines if any(p.match(ln) for p in patterns))
    return hits / len(lines)

def _url_contact_ratio(text: str, *, word_count: int) -> float:
    raw = document_gate_raw(text)
    return shared_baseline_estimator().baseline([
        raw.url_char_ratio, raw.contact_token_ratio,
        PopulationAdaptiveScaler.rate(raw.contact_token_ratio, word_count),
    ])

def compute_artifact_ratio(
    text: str,
    *,
    signals: QualitySignals | None = None,
    content_value: ContentValueSignals | None = None,
    bundle: Any | None = None,
    metadata_noise: float | None = None,
    instruction_wrappers: float | None = None,
    include_discovery: bool = True,
    discovery_engine: Any | None = None,
    policy: DocumentGatePolicy | None = None,
) -> tuple[float, dict[str, float]]:
    pol = policy or resolve_document_gate_policy()
    ctx = bundle or build_analysis_bundle(text)
    sig = signals or ctx.signals(text)
    cv = content_value or analyze_content_value(text, bundle=ctx)
    words = _WORD.findall(text)
    word_count = max(len(words), 1)
    meta = metadata_noise if metadata_noise is not None else ctx.metadata_noise(text)
    instr = instruction_wrappers if instruction_wrappers is not None else ctx.instruction_density(text)

    intent = ctx.intent(cv, text=text)
    components = {
        'copyright_license': meta,
        'navigation': _line_pattern_ratio(text, (_UI_LINE,)),
        'forum_metadata': _line_pattern_ratio(text, (_METADATA_LINE,)),
        'boilerplate': sig.boilerplate_score,
        'commercial': max(sig.commercial_score, cv.commercial_score),
        'transactional': intent.transactional,
        'administrative': intent.administrative,
        'advertisement': min(
            1.0,
            max(sig.seo_spam_score, sig.commercial_score * pol.artifact_advertisement_commercial_multiplier),
        ),
        'instruction_wrappers': instr,
        'urls_contact': _url_contact_ratio(text, word_count=word_count),
        'low_information': sig.low_information_score,
        'seo_spam': sig.seo_spam_score,
    }

    if include_discovery:
        disc = 0.0
        try:
            eng = discovery_engine
            if eng is None:
                from indw.clean.artifact.discovery_engine import get_discovery_engine
                eng = get_discovery_engine()
            disc = eng.document_artifact_ratio(text)
        except Exception:
            disc = 0.0
        components['discovery_artifact'] = disc

    evidence = cv.evidence or ctx.evidence(text)
    ratio = aggregate_component_noise(components, evidence=evidence)
    return min(1.0, ratio), components
