from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from indw.dedup.embed.providers import HashEmbeddingProvider
from indw.clean.artifact.discovery_corpus import CorpusStatsAccumulator
from indw.clean.artifact.decompose import compute_layout
from indw.extract.structure.recovery import RecoveredSection
from indw.extract.sections.boundaries import period_ends_sentence
from indw.extract.structure.analyze import analyze_structure
from indw.extract.nav.template import TemplateMiner
from indw.filter.score.signals import shannon_entropy
from indw.clean.artifact.evidence_engine import resolve_semantic_evidence
from indw.clean.artifact.evidence_features import shared_feature_extractor
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator
from indw.clean.document.value import compute_structure_profile

_EMBED = HashEmbeddingProvider(dimension=96)
_SEP_CHARS = '|>»/\\·•–—:'

@dataclass
class AggregationUnit:
    text: str
    start: int
    end: int
    local_index: int = 0
    local_position: float = 0.0
    independence: float = 0.0
    headline_score: float = 0.0
    wrapper_score: float = 0.0
    role: str = 'unknown'

    def to_dict(self) -> dict[str, Any]:
        return {
            'local_index': self.local_index,
            'local_position': round(self.local_position, 4),
            'independence': round(self.independence, 4),
            'headline_score': round(self.headline_score, 4),
            'wrapper_score': round(self.wrapper_score, 4),
            'role': self.role,
            'chars': len(self.text),
            'preview': self.text[:120],
        }

@dataclass
class AggregationProfile:
    is_aggregated: bool = False
    confidence: float = 0.0
    is_headline_index: bool = False
    units: list[AggregationUnit] = field(default_factory=list)
    mean_independence: float = 0.0
    inter_unit_distance: float = 0.0
    topic_split: bool = False
    signals: dict[str, float] = field(default_factory=dict)

    def unit_for(self, section: RecoveredSection) -> AggregationUnit | None:
        for u in self.units:
            if u.start == section.start and u.end == section.end:
                return u
        for u in self.units:
            if u.start <= section.start < u.end or u.start < section.end <= u.end:
                return u
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            'is_aggregated': self.is_aggregated,
            'confidence': round(self.confidence, 4),
            'is_headline_index': self.is_headline_index,
            'unit_count': len(self.units),
            'mean_independence': round(self.mean_independence, 4),
            'inter_unit_distance': round(self.inter_unit_distance, 4),
            'signals': {k: round(v, 4) for k, v in self.signals.items()},
            'units': [u.to_dict() for u in self.units[:24]],
        }

@dataclass
class AggregationContext:
    profile: AggregationProfile | None = None
    accumulator: CorpusStatsAccumulator | None = None
    template_miner: TemplateMiner | None = None

    def miner(self) -> TemplateMiner:
        if self.template_miner is None:
            self.template_miner = TemplateMiner(self.accumulator)
        return self.template_miner

    def local_position(self, section: RecoveredSection) -> float:
        if self.profile is None or not self.profile.is_aggregated:
            return section.position_ratio
        unit = self.profile.unit_for(section)
        if unit is not None:
            return unit.local_position
        return section.position_ratio

    def is_aggregated(self) -> bool:
        return bool(self.profile and self.profile.is_aggregated)

_CTX = AggregationContext()

def set_aggregation_context(ctx: AggregationContext | None) -> AggregationContext | None:
    global _CTX
    prev = _CTX
    _CTX = ctx or AggregationContext()
    return prev

def get_aggregation_context() -> AggregationContext:
    return _CTX

def _separator_density(text: str) -> float:
    if not text:
        return 0.0
    tokens = max(len(text.split()), 1)
    sep = sum(text.count(c) for c in _SEP_CHARS)
    return min(1.0, sep / tokens)

def _embed_dist(a: str, b: str) -> float:
    va = _EMBED._one(a)
    vb = _EMBED._one(b)
    na = float((va * va).sum()) ** 0.5
    nb = float((vb * vb).sum()) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    cos = float((va @ vb) / (na * nb))
    return max(0.0, 1.0 - cos)

def _paragraph_spans(text: str, *, min_chars: int = 40) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for block in text.split('\n\n'):
        stripped = block.strip()
        if not stripped:
            cursor += len(block) + 2
            continue
        start = text.find(stripped, cursor)
        if start < 0:
            start = cursor
        end = start + len(stripped)
        if len(stripped) >= min_chars:
            spans.append((stripped, start, end))
        cursor = end
    if not spans and text.strip():
        s = text.strip()
        spans.append((s, 0, len(s)))
    return spans

@dataclass
class _SectionFeatures:
    ev: Any
    structural: Any
    layout: Any
    profile: Any
    raw: Any


def _section_features(text: str) -> _SectionFeatures:
    from indw.extract.core.context import get_document_context
    from indw.extract.core.profile import ke_record, ke_timed
    from indw.clean.document.value import resolve_analysis_bundle

    with ke_timed('agg_section_features', payload_bytes=len(text.encode('utf-8', 'surrogatepass'))):
        dctx = get_document_context()
        bundle = resolve_analysis_bundle(text)
        ev = bundle.evidence(text)
        if dctx is not None:
            structural = dctx.structure_analysis(text, lambda: analyze_structure(text))
            layout = dctx.layout(text, lambda: compute_layout(text))
            profile = dctx.structure_profile(
                text, lambda: compute_structure_profile(text, evidence=ev),
            )
        else:
            structural = analyze_structure(text)
            layout = compute_layout(text)
            profile = compute_structure_profile(text, evidence=ev)
        raw = shared_feature_extractor().extract(text)
    ke_record('agg_evidence', dedupe_key=(len(text), text[:64]))
    return _SectionFeatures(ev=ev, structural=structural, layout=layout, profile=profile, raw=raw)


def _independence_from_features(feat: _SectionFeatures) -> float:
    baseline = AdaptiveBaselineEstimator()
    noise = baseline.baseline(list(feat.ev.negative.values()) or [0.0])
    return baseline.baseline([
        feat.ev.utility,
        feat.structural.sentence_completeness_mean,
        feat.structural.information_density,
        feat.profile.explanation_ratio,
        1.0 - noise * 0.35,
    ])


def _independence_score(text: str) -> float:
    if not text.strip():
        return 0.0
    return _independence_from_features(_section_features(text))


def _headline_from_features(feat: _SectionFeatures) -> float:
    baseline = AdaptiveBaselineEstimator()
    if feat.raw.word_count >= 12 and feat.structural.sentence_completeness_mean >= 0.65:
        return baseline.baseline([0.06, feat.layout.line_count / 10.0])
    short = max(0.0, 1.0 - feat.raw.word_count / 9.0)
    incomplete = 1.0 - feat.structural.sentence_completeness_mean
    per_line = feat.raw.word_count / max(feat.layout.line_count, 1)
    listing = 0.35 if feat.layout.line_count >= 2 and per_line < 7 else 0.0
    return baseline.baseline([short, incomplete, listing, max(0.0, 0.18 - feat.ev.utility)])


def _headline_score(text: str) -> float:
    return _headline_from_features(_section_features(text))


def _wrapper_from_features(
    text: str,
    feat: _SectionFeatures,
    *,
    local_position: float,
    ctx: AggregationContext,
) -> float:
    tmpl = ctx.miner().analyze(text)
    sep = _separator_density(text)
    baseline = AdaptiveBaselineEstimator()
    admin = feat.ev.negative.get('administrative', 0.0)
    trans = feat.ev.negative.get('transactional', 0.0)
    edge = 0.0
    if local_position < 0.12:
        edge = 1.0 - local_position / 0.12
    if local_position > 0.88:
        edge = max(edge, (local_position - 0.88) / 0.12)
    return baseline.baseline([
        admin * 0.35,
        trans * 0.25,
        sep * 0.30,
        tmpl.template_density * 0.25,
        edge * 0.20,
        (1.0 - feat.ev.utility) * 0.15 if feat.layout.line_count <= 3 else 0.0,
    ])


def _wrapper_score(text: str, *, local_position: float, ctx: AggregationContext) -> float:
    feat = _section_features(text)
    return _wrapper_from_features(text, feat, local_position=local_position, ctx=ctx)


def _assign_unit_role_from_features(
    text: str,
    feat: _SectionFeatures,
    *,
    independence: float,
    headline: float,
    wrapper: float,
    profile: AggregationProfile,
) -> str:
    if profile.is_headline_index or (headline > 0.55 and independence < 0.52):
        return 'headline'
    if independence >= 0.55 and feat.structural.sentence_completeness_mean >= 0.55 and feat.ev.utility >= 0.08:
        return 'article'
    if wrapper > 0.42 and independence < 0.40:
        return 'wrapper'
    if feat.layout.line_count <= 2 and _separator_density(text) > 0.14 and feat.ev.utility < 0.13:
        return 'subscription'
    if independence >= 0.28 and feat.ev.utility >= 0.08:
        return 'article'
    if wrapper > 0.35 and independence < 0.45:
        return 'wrapper'
    return 'unknown'


def _inter_unit_distance(units: list[AggregationUnit]) -> float:
    if len(units) < 2:
        return 0.0
    dists: list[float] = []
    for i in range(len(units) - 1):
        dists.append(_embed_dist(units[i].text, units[i + 1].text))
    return sum(dists) / len(dists)

def _aggregation_signals(
    text: str,
    sections: list[RecoveredSection],
    units: list[AggregationUnit],
    *,
    inter_dist: float | None = None,
    section_layouts: list[Any] | None = None,
) -> dict[str, float]:
    if len(sections) < 2:
        return {'combined': 0.0}

    baseline = AdaptiveBaselineEstimator()
    lens = [len(s.text) for s in sections]
    mean_len = sum(lens) / len(lens)
    cv = (sum((l - mean_len) ** 2 for l in lens) / len(lens)) ** 0.5 / max(mean_len, 1)

    para_count = len(_paragraph_spans(text, min_chars=30))
    if inter_dist is None:
        inter_dist = _inter_unit_distance(units)
    unit_sim = 1.0 - inter_dist
    mean_indep = sum(u.independence for u in units) / max(len(units), 1)
    mean_headline = sum(u.headline_score for u in units) / max(len(units), 1)

    single_para = 0.0
    if section_layouts is not None and len(section_layouts) == len(sections):
        for lay in section_layouts:
            if lay.line_count <= 3:
                single_para += 1.0
    else:
        for s in sections:
            lay = compute_layout(s.text)
            if lay.line_count <= 3:
                single_para += 1.0
    single_para /= len(sections)

    blank_sep = min(1.0, para_count / max(len(sections), 1))
    count_sig = min(1.0, len(sections) / 6.0)
    length_sig = 1.0 - min(1.0, abs(mean_len - 110) / 180.0)
    diversity = baseline.baseline([inter_dist, 1.0 - unit_sim])

    combined = baseline.baseline([
        count_sig * 0.22,
        single_para * 0.18,
        blank_sep * 0.12,
        length_sig * 0.10,
        diversity * 0.18,
        mean_indep * 0.12,
        (1.0 - cv) * 0.08,
    ])
    headline_hits = sum(1 for u in units if u.headline_score > 0.55)
    headline_frac = headline_hits / len(units)
    if headline_frac > 0.65:
        combined *= 0.55

    return {
        'combined': combined,
        'count': count_sig,
        'single_para': single_para,
        'blank_sep': blank_sep,
        'length': length_sig,
        'diversity': diversity,
        'mean_independence': mean_indep,
        'mean_headline': mean_headline,
        'headline_frac': headline_frac,
        'cv': cv,
    }

def build_aggregation_units(
    text: str,
    sections: list[RecoveredSection],
    *,
    ctx: AggregationContext | None = None,
) -> tuple[list[AggregationUnit], list[Any]]:
    ctx = ctx or get_aggregation_context()
    if not sections:
        return [], []

    n = len(sections)
    units: list[AggregationUnit] = []
    section_layouts: list[Any] = []
    section_feats: list[_SectionFeatures] = []
    for i, sec in enumerate(sections):
        local_pos = i / max(n - 1, 1) if n > 1 else 0.5
        feat = _section_features(sec.text)
        section_feats.append(feat)
        section_layouts.append(feat.layout)
        indep = _independence_from_features(feat)
        headline = _headline_from_features(feat)
        wrapper = _wrapper_from_features(sec.text, feat, local_position=local_pos, ctx=ctx)
        units.append(AggregationUnit(
            text=sec.text,
            start=sec.start,
            end=sec.end,
            local_index=i,
            local_position=local_pos,
            independence=indep,
            headline_score=headline,
            wrapper_score=wrapper,
        ))

    profile_stub = AggregationProfile(units=units)
    for u, feat in zip(units, section_feats):
        u.role = _assign_unit_role_from_features(
            u.text,
            feat,
            independence=u.independence,
            headline=u.headline_score,
            wrapper=u.wrapper_score,
            profile=profile_stub,
        )
    return units, section_layouts

def analyze_aggregation(
    text: str,
    sections: list[RecoveredSection],
    *,
    ctx: AggregationContext | None = None,
    topic_split: bool = False,
) -> AggregationProfile:
    ctx = ctx or get_aggregation_context()
    if not text.strip() or len(sections) < 2:
        return AggregationProfile()

    units, section_layouts = build_aggregation_units(text, sections, ctx=ctx)
    inter_dist = _inter_unit_distance(units)
    signals = _aggregation_signals(
        text, sections, units, inter_dist=inter_dist, section_layouts=section_layouts,
    )
    baseline = AdaptiveBaselineEstimator()
    thr = baseline.baseline([0.32, signals['combined'] * 0.88, 0.26])
    mean_len = sum(len(u.text) for u in units) / max(len(units), 1)
    is_agg = (
        signals['combined'] >= thr and len(sections) >= 3
    ) or (
        topic_split
        and len(sections) >= 2
        and mean_len < 78
        and inter_dist >= 0.28
        and signals['mean_independence'] >= 0.48
    )

    headline_hits = sum(1 for u in units if u.headline_score > 0.55)
    is_headline_index = (
        is_agg
        and headline_hits / len(units) > 0.65
        and mean_len < 72
    )

    if is_headline_index:
        for u in units:
            if u.headline_score > 0.55 and u.independence < 0.52:
                u.role = 'headline'
            elif u.role == 'headline':
                u.role = 'unknown'

    mean_indep = sum(u.independence for u in units) / len(units)
    return AggregationProfile(
        is_aggregated=is_agg,
        confidence=signals['combined'],
        is_headline_index=is_headline_index,
        units=units,
        mean_independence=mean_indep,
        inter_unit_distance=inter_dist,
        topic_split=topic_split,
        signals=signals,
    )

@dataclass
class TopicSpan:
    text: str
    start: int
    end: int
    confidence: float = 0.0
    coherence: float = 0.0

    def topic_signature(self) -> tuple[float, ...]:
        v = _EMBED._one(self.text[:512])
        n = float((v * v).sum()) ** 0.5
        if n <= 0:
            return ()
        return tuple(round(float(x / n), 4) for x in v[:12])

def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    if not text.strip():
        return []
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    buf: list[str] = []
    buf_start = 0

    def flush(end: int) -> None:
        nonlocal buf, buf_start, cursor
        if not buf:
            return
        chunk = ' '.join(buf).strip()
        if chunk:
            start = text.find(chunk, buf_start)
            if start < 0:
                start = buf_start
            spans.append((chunk, start, start + len(chunk)))
        buf = []
        buf_start = end
        cursor = end

    tokens = text.replace('?', '?\x00').replace('!', '!\x00').replace('.', '.\x00').split('\x00')
    pos = 0
    for piece in tokens:
        piece = piece.strip()
        if not piece:
            pos += 1
            continue
        idx = text.find(piece, pos)
        if idx < 0:
            idx = pos
        if ':' in piece and piece.index(':') < 80:
            colon = piece.index(':')
            if colon + 1 < len(piece) and piece[colon + 1].isdigit():
                colon = -1
            elif colon > 0 and piece[colon - 1].isdigit():
                colon = -1
            if colon >= 0:
                head = piece[:colon + 1].strip()
            tail = piece[colon + 1:].strip() if colon >= 0 else ''
            head_words = head.rstrip(':').split() if colon >= 0 else []
            if (
                colon >= 0
                and head
                and tail
                and len(tail) > 25
                and len(head_words) <= 12
                and len(head_words) >= 2
                and head_words[0][:1].isupper()
                and _separator_density(head) < 0.10
                and '|' not in head
                and '>' not in head
            ):
                if buf:
                    flush(idx)
                spans.append((head, idx, idx + len(head)))
                rest_start = idx + colon + 1
                while rest_start < len(text) and text[rest_start].isspace():
                    rest_start += 1
                tail_end = idx + len(piece)
                spans.append((tail, rest_start, tail_end))
                pos = tail_end
                buf_start = tail_end
                continue
        if not buf:
            buf_start = idx
        buf.append(piece)
        pos = idx + len(piece)
        if piece[-1] in '.?!':
            if piece[-1] == '.':
                dot = text.rfind('.', 0, pos)
                if dot >= 0 and not period_ends_sentence(text, dot):
                    continue
            flush(pos)
    if buf:
        flush(len(text))
    if not spans:
        s = text.strip()
        return [(s, 0, len(s))]
    return spans

def _entity_shift(left: str, right: str) -> float:
    lw = {w.lower() for w in left.split() if len(w) > 3 and w[0].isupper()}
    rw = {w.lower() for w in right.split() if len(w) > 3 and w[0].isupper()}
    if not lw and not rw:
        return 0.0
    union = lw | rw
    if not union:
        return 0.0
    overlap = len(lw & rw) / len(union)
    return 1.0 - overlap

def _topic_transition(left: str, right: str) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    baseline = AdaptiveBaselineEstimator()
    embed = _embed_dist(left, right)
    ent_l = shannon_entropy(left) / 8.0
    ent_r = shannon_entropy(right) / 8.0
    ev_l = resolve_semantic_evidence(left)
    ev_r = resolve_semantic_evidence(right)
    utility = abs(ev_l.utility - ev_r.utility)
    entity = _entity_shift(left, right)
    caps_l = sum(1 for w in left.split()[:4] if w[:1].isupper()) / max(min(4, len(left.split())), 1)
    caps_r = sum(1 for w in right.split()[:4] if w[:1].isupper()) / max(min(4, len(right.split())), 1)
    title_shift = max(0.0, caps_r - caps_l)
    return baseline.baseline([embed, abs(ent_l - ent_r), utility, entity, title_shift * 0.35])

def segment_topics(text: str, *, min_unit_chars: int = 40) -> list[TopicSpan]:
    if not text or not text.strip():
        return []
    spans = _sentence_spans(text)
    if len(spans) <= 1:
        s = text.strip()
        return [TopicSpan(s, 0, len(s), 0.5, 1.0)]

    baseline = AdaptiveBaselineEstimator()
    cuts: list[int] = [0]
    trans_scores: list[float] = []
    for i in range(1, len(spans)):
        left = spans[i - 1][0]
        right = spans[i][0]
        trans_scores.append(_topic_transition(left, right))
    thr = baseline.baseline([
        baseline.spread(trans_scores) if trans_scores else 0.0,
        baseline.baseline(trans_scores) * 1.1 if trans_scores else 0.22,
        0.24,
    ])
    acc = spans[0][0]
    for i in range(1, len(spans)):
        if trans_scores[i - 1] >= thr and len(acc) >= min_unit_chars:
            cuts.append(i)
            acc = spans[i][0]
        else:
            acc = f'{acc} {spans[i][0]}'

    if len(cuts) == 1 and len(spans) >= 2:
        ranked = sorted(enumerate(trans_scores), key=lambda x: -x[1])
        for idx, score in ranked[: max(1, len(spans) // 3)]:
            if score >= thr * 0.88 and idx + 1 not in cuts:
                cuts.append(idx + 1)
        cuts = sorted(set(cuts))

    groups: list[list[tuple[str, int, int]]] = []
    prev = 0
    for c in cuts[1:] if len(cuts) > 1 else []:
        groups.append(spans[prev:c])
        prev = c
    groups.append(spans[prev:])

    out: list[TopicSpan] = []
    for grp in groups:
        if not grp:
            continue
        chunk = ' '.join(s[0] for s in grp).strip()
        if len(chunk) < min_unit_chars and out:
            prev_span = out[-1]
            merged = f'{prev_span.text} {chunk}'.strip()
            from indw.extract.sections.semantic import analyze_completion_cached
            if analyze_completion_cached(merged).incomplete_probability < 0.42:
                out[-1] = TopicSpan(
                    merged,
                    prev_span.start,
                    grp[-1][2],
                    prev_span.confidence,
                    prev_span.coherence,
                )
            continue
        start, end = grp[0][1], grp[-1][2]
        if len(grp) > 1:
            internal = [
                _topic_transition(grp[j][0], grp[j + 1][0])
                for j in range(len(grp) - 1)
            ]
            coherence = 1.0 - baseline.baseline(internal)
            conf = baseline.baseline(trans_scores) if trans_scores else 0.5
        else:
            coherence = 1.0
            conf = 0.72
        out.append(TopicSpan(chunk, start, end, conf, coherence))
    return out or [TopicSpan(text.strip(), 0, len(text.strip()), 0.5, 1.0)]

def _expand_section_topics(
    sec: RecoveredSection,
    doc_text: str,
    *,
    min_section_chars: int,
) -> list[RecoveredSection]:
    local = sec.text
    spans = _sentence_spans(local)
    if len(spans) < 2:
        return [sec]
    trans = [_topic_transition(spans[i][0], spans[i + 1][0]) for i in range(len(spans) - 1)]
    baseline = AdaptiveBaselineEstimator()
    max_trans = max(trans) if trans else 0.0
    if max_trans < baseline.baseline([0.42, baseline.baseline(trans) * 1.08 if trans else 0.4]):
        return [sec]
    topics = segment_topics(local, min_unit_chars=min_section_chars)
    if len(topics) <= 1:
        return [sec]
    if _inter_topic_distance(topics) < 0.20:
        return [sec]
    total = max(len(doc_text), 1)
    return [
        RecoveredSection(
            text=t.text,
            start=sec.start + t.start,
            end=sec.start + t.end,
            position_ratio=(sec.start + t.start + sec.start + t.end) / (2 * total),
            structural_role='body',
            layout_kind='paragraph',
        )
        for t in topics
    ]

def expand_topic_sections(
    text: str,
    sections: list[RecoveredSection],
    *,
    min_section_chars: int = 40,
) -> list[RecoveredSection]:
    if not sections:
        return []
    out: list[RecoveredSection] = []
    for sec in sections:
        out.extend(_expand_section_topics(sec, text, min_section_chars=min_section_chars))
    return out if len(out) > len(sections) else sections

def _inter_topic_distance(topics: list[TopicSpan]) -> float:
    if len(topics) < 2:
        return 0.0
    dists = [_embed_dist(topics[i].text, topics[i + 1].text) for i in range(len(topics) - 1)]
    return sum(dists) / len(dists)

def refine_aggregation_sections(
    text: str,
    sections: list[RecoveredSection],
    *,
    min_section_chars: int = 60,
) -> list[RecoveredSection]:
    if len(sections) >= 3:
        return sections

    spans = _paragraph_spans(text, min_chars=min_section_chars)
    if len(spans) <= len(sections):
        return sections

    total = max(len(text), 1)
    out: list[RecoveredSection] = []
    for chunk, start, end in spans:
        out.append(RecoveredSection(
            text=chunk,
            start=start,
            end=end,
            position_ratio=(start + end) / (2 * total),
            structural_role='body',
            layout_kind='paragraph',
        ))
    return out

def effective_position(
    section: RecoveredSection,
    *,
    agg_ctx: AggregationContext | None = None,
) -> float:
    agg_ctx = agg_ctx or get_aggregation_context()
    return agg_ctx.local_position(section)

def trim_structural_tail(text: str) -> str:
    if not text or not text.strip():
        return text
    from indw.extract.sections.semantic import analyze_completion_cached

    parts: list[str] = []
    for chunk in text.replace('!', '!\n').replace('?', '?\n').replace('.', '.\n').splitlines():
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    if len(parts) >= 2:
        tail = parts[-1]
        body = ' '.join(parts[:-1]).strip()
        tail_feat = _section_features(tail)
        body_feat = _section_features(body)
        if body and _headline_from_features(tail_feat) > 0.52 and _independence_from_features(tail_feat) < _independence_from_features(body_feat) * 0.75:
            if analyze_completion_cached(body).incomplete_probability <= analyze_completion_cached(tail).incomplete_probability + 0.08:
                return body

    words = text.split()
    if len(words) < 8:
        return text
    run = 0
    for w in reversed(words):
        core = w.rstrip('.,;:!?')
        if not core:
            break
        if (len(core) > 1 and core[0].isupper() and not core.isupper()) or core.isupper():
            run += 1
        else:
            break
    if run < 2 or run > 6:
        return text
    tail = ' '.join(words[-run:])
    body = ' '.join(words[:-run]).strip().rstrip('.,;:!?')
    if not body or len(body.split()) < 6:
        return text
    if _headline_score(tail) > 0.42:
        ev_tail = resolve_semantic_evidence(tail)
        ev_body = resolve_semantic_evidence(body)
        if ev_tail.utility <= ev_body.utility * 1.05:
            if analyze_completion_cached(body).incomplete_probability <= analyze_completion_cached(text).incomplete_probability:
                return body
    return text
