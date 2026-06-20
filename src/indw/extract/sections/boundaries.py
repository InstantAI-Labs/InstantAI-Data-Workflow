from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from indw.dedup.embed.providers import HashEmbeddingProvider
from indw.clean.artifact.decompose import LayoutVector, compute_layout
from indw.extract.nav.context import (
    nav_transition_score,
    score_navigation_role,
    structural_listing_score,
)
from indw.extract.roles.forum import conversation_role_boundary
from indw.extract.roles.publication import (
    pipe_split_offsets,
    publication_role_boundary,
)
from indw.extract.roles.education import educational_role_boundary
from indw.filter.score.signals import shannon_entropy
from indw.clean.artifact.evidence_engine import compute_semantic_evidence
from indw.clean.artifact.evidence_features import DocumentFeatureExtractor
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator

_EMBED = HashEmbeddingProvider(dimension=96)
_EXTRACTOR = DocumentFeatureExtractor()
_SEP_CHARS = '|>»·•→←/\\'
_LAYOUT_SPLITS = frozenset('×✕✖')

@dataclass
class TextUnit:
    text: str
    start: int
    end: int
    line_count: int = 1

@dataclass
class BoundaryCandidate:
    offset: int
    score: float
    role_boundary: bool = False
    signals: dict[str, float] = field(default_factory=dict)

def _separator_density(text: str) -> float:
    if not text:
        return 0.0
    tokens = max(len(text.split()), 1)
    sep = sum(text.count(c) for c in _SEP_CHARS)
    return min(1.0, sep / tokens)

def _lines_with_offsets(text: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    pos = 0
    for line in text.splitlines():
        stripped = line.strip()
        start = pos + (line.index(stripped) if stripped and stripped in line else 0)
        end = start + len(stripped) if stripped else pos + len(line)
        if stripped:
            out.append((stripped, start, end))
        pos += len(line) + 1
    if not out and text.strip():
        s = text.strip()
        out.append((s, 0, len(s)))
    return out

def _layout_split_offsets(text: str) -> list[int]:
    cuts: list[int] = []
    for i, ch in enumerate(text):
        if ch in _LAYOUT_SPLITS:
            cuts.append(i)
            cuts.append(i + 1)
    return sorted(set(cuts))

def _paren_prose_offsets(text: str) -> list[int]:
    cuts: list[int] = []
    for i in range(2, len(text) - 4):
        if text[i - 1] != ')' or text[i] != ' ':
            continue
        j = i + 1
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or not text[j].isupper():
            continue
        word_end = j
        while word_end < len(text) and text[word_end].isalpha():
            word_end += 1
        if word_end - j < 3:
            continue
        left = text[:i].strip()
        right = text[j:].strip()
        if len(left) < 12 or len(right) < 30:
            continue
        left_nav = score_navigation_role(left, position_ratio=0.05).nav_mass()
        right_art = score_navigation_role(right, position_ratio=0.35).article
        util_l = compute_semantic_evidence(left).utility
        util_r = compute_semantic_evidence(right).utility
        if left_nav > 0.2 or util_l + 0.12 < util_r or right_art > 0.28:
            cuts.append(i)
    return cuts

def _is_time_colon(text: str, colon_idx: int) -> bool:
    tail_start = colon_idx + 1
    while tail_start < len(text) and text[tail_start].isspace():
        tail_start += 1
    if tail_start < len(text) and text[tail_start].isdigit():
        return True
    j = colon_idx - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    return j >= 0 and text[j].isdigit()

def _colon_offsets(text: str) -> list[int]:
    cuts: list[int] = []
    for i, ch in enumerate(text):
        if ch != ':' or i >= 120:
            continue
        if _is_time_colon(text, i):
            continue
        head = text[:i + 1].strip()
        tail_start = i + 1
        while tail_start < len(text) and text[tail_start].isspace():
            tail_start += 1
        tail = text[tail_start:].strip()
        if not head or not tail or len(tail) < 25:
            continue
        head_words = head.rstrip(':').split()
        if len(head_words) < 2 or len(head_words) > 14:
            continue
        if not head_words[0][:1].isupper():
            continue
        if _separator_density(head) >= 0.12:
            continue
        if '|' in head or '>' in head:
            continue
        left_nav = score_navigation_role(head, position_ratio=0.06).nav_mass()
        left_list = structural_listing_score(head, position_ratio=0.06)
        util_l = compute_semantic_evidence(head).utility
        util_r = compute_semantic_evidence(tail[:400]).utility
        if text[:i].count(':') >= 1:
            if len(head_words) <= 6 and left_list < 0.35 and util_r - util_l < 0.14:
                continue
        if left_nav > 0.22 or left_list > 0.38 or (util_l + 0.15 < util_r and util_r > 0.3):
            cuts.append(tail_start)
    return cuts

_ABBREV_PERIOD = re.compile(
    r'\b(?:'
    r'Mr|Mrs|Ms|Miss|Dr|Prof|Sr|Jr|St|Mt|Gen|Col|Capt|Lt|Sgt|Rep|Sen|Gov|Pres|Rev|Hon'
    r'|vs|etc|e\.g|i\.e|a\.k\.a|Ph\.D|M\.D|B\.A|M\.A|U\.S|U\.K|No'
    r')\.$',
    re.I,
)

def period_ends_sentence(text: str, period_idx: int) -> bool:
    if period_idx < 0 or period_idx >= len(text) or text[period_idx] != '.':
        return False
    prefix = text[:period_idx + 1]
    if _ABBREV_PERIOD.search(prefix):
        return False
    words = prefix.rstrip().split()
    if words:
        last = words[-1]
        if re.match(r'^[A-Z]\.$', last):
            return False
        if re.match(r'^(?:[A-Z]\.){2,}$', last):
            return False
    nxt = period_idx + 1
    while nxt < len(text) and text[nxt].isspace():
        nxt += 1
    if nxt >= len(text):
        return True
    return text[nxt].isupper()

def _sentence_offsets(text: str) -> list[int]:
    cuts: list[int] = []
    for i, ch in enumerate(text):
        if ch == '.':
            if not period_ends_sentence(text, i):
                continue
        elif ch not in '?!':
            continue
        nxt = i + 1
        while nxt < len(text) and text[nxt].isspace():
            nxt += 1
        if nxt >= len(text) or not text[nxt].isupper():
            continue
        cuts.append(nxt)
    return cuts

def _decompose_inline_spans(text: str, *, base_offset: int = 0) -> list[TextUnit]:
    blob = text.strip()
    if not blob:
        return []

    cut_points = sorted(set(
        _layout_split_offsets(blob)
        + _colon_offsets(blob)
        + _paren_prose_offsets(blob)
        + _sentence_offsets(blob)
        + pipe_split_offsets(blob)
    ))
    cut_points = [c for c in cut_points if 0 < c < len(blob)]

    if not cut_points:
        return [TextUnit(blob, base_offset, base_offset + len(blob), 1)]

    bounds = [0, *cut_points, len(blob)]
    units: list[TextUnit] = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        chunk = blob[start:end].strip()
        if not chunk or chunk in _LAYOUT_SPLITS:
            continue
        abs_start = base_offset + start
        units.append(TextUnit(chunk, abs_start, abs_start + len(chunk), 1))
    return units if units else [TextUnit(blob, base_offset, base_offset + len(blob), 1)]

def _decompose_spans(text: str) -> list[TextUnit]:
    if not text or not text.strip():
        return []

    if '\n\n' in text:
        units: list[TextUnit] = []
        cursor = 0
        for para in text.split('\n\n'):
            stripped = para.strip()
            if not stripped:
                continue
            idx = text.find(stripped, cursor)
            if idx < 0:
                idx = cursor
            if '\n' not in stripped and len(stripped) < 480:
                units.extend(_decompose_inline_spans(stripped, base_offset=idx))
            else:
                units.append(TextUnit(stripped, idx, idx + len(stripped), stripped.count('\n') + 1))
            cursor = idx + len(stripped)
        return units if units else [TextUnit(text.strip(), 0, len(text.strip()), 1)]

    lines = _lines_with_offsets(text)
    if len(lines) > 1:
        units: list[TextUnit] = []
        for line_text, line_start, _line_end in lines:
            units.extend(_decompose_inline_spans(line_text, base_offset=line_start))
        return units if units else [TextUnit(text.strip(), 0, len(text.strip()), 1)]

    blob = text.strip()
    return _decompose_inline_spans(blob, base_offset=text.find(blob))

def _feature_vector(raw) -> list[float]:
    return [
        raw.nav_line_ratio,
        raw.uniform_line_ratio,
        raw.table_line_ratio,
        raw.structured_line_ratio,
        raw.contact_token_ratio,
        raw.first_person_ratio,
        raw.avg_line_len / 200.0,
        raw.line_len_cv,
        raw.qa_line_hits / max(raw.line_count, 1),
        raw.anchor_density,
        raw.numeric_token_ratio,
        raw.exclaim_line_ratio,
    ]

def _vec_dist(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)) / n)

def _layout_dist(a: LayoutVector, b: LayoutVector) -> float:
    return _vec_dist(list(a.to_tuple()), list(b.to_tuple()))

def _embed_dist(a: str, b: str) -> float:
    va = _EMBED._one(a)
    vb = _EMBED._one(b)
    na = float((va * va).sum()) ** 0.5
    nb = float((vb * vb).sum()) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    cos = float((va @ vb) / (na * nb))
    return max(0.0, 1.0 - cos)

def _role_boundary_score(
    left_text: str,
    right_text: str,
    *,
    left_pos: float,
    right_pos: float,
) -> float:
    if not left_text.strip() or not right_text.strip():
        return 0.0
    left_nav = score_navigation_role(left_text, position_ratio=left_pos)
    right_nav = score_navigation_role(right_text, position_ratio=right_pos)
    left_list = structural_listing_score(left_text, position_ratio=left_pos)
    right_list = structural_listing_score(right_text, position_ratio=right_pos)
    ev_l = compute_semantic_evidence(left_text)
    ev_r = compute_semantic_evidence(right_text)
    util_shift = max(0.0, ev_r.utility - ev_l.utility)
    nav_cut = nav_transition_score(left_text, right_text, left_pos=left_pos, right_pos=right_pos)
    wrapper_l = max(left_nav.nav_mass(), left_list, left_nav.footer, left_nav.menu)
    article_r = max(right_nav.article, ev_r.utility * 0.85)
    role = 0.0
    if wrapper_l > 0.28 and article_r > 0.25:
        role = min(1.0, wrapper_l * 0.45 + article_r * 0.35 + util_shift * 0.35 + nav_cut * 0.25)
    elif left_nav.nav_mass() > 0.22 and right_nav.article > 0.28:
        role = min(1.0, (right_nav.article - left_nav.nav_mass()) * 0.7 + util_shift * 0.4 + 0.15)
    elif left_list > 0.35 and right_list < 0.25 and util_shift > 0.12:
        role = min(1.0, left_list * 0.5 + util_shift * 0.45 + 0.1)
    elif util_shift > 0.22 and right_nav.article > 0.22:
        role = min(1.0, util_shift * 0.65 + nav_cut * 0.35)
    conv = conversation_role_boundary(left_text, right_text, left_pos=left_pos, right_pos=right_pos)
    if conv > role:
        role = conv
    pub = publication_role_boundary(left_text, right_text, left_pos=left_pos, right_pos=right_pos)
    if pub > role:
        role = pub
    edu = educational_role_boundary(left_text, right_text, left_pos=left_pos, right_pos=right_pos)
    if edu > role:
        role = edu
    return max(role, nav_cut)

def _window_signal(units: list[TextUnit], i: int, width: int = 2) -> dict[str, float]:
    left = max(0, i - width)
    right = min(len(units), i + width)
    chunk = '\n'.join(u.text for u in units[left:i])
    nxt = '\n'.join(u.text for u in units[i:right])
    if not chunk.strip() or not nxt.strip():
        return {'combined': 0.0, 'role_boundary': 0.0}

    raw_a = _EXTRACTOR.extract(chunk)
    raw_b = _EXTRACTOR.extract(nxt)
    lay_a = compute_layout(chunk)
    lay_b = compute_layout(nxt)
    ev_a = compute_semantic_evidence(chunk)
    ev_b = compute_semantic_evidence(nxt)
    ent_a = shannon_entropy(chunk) / 8.0
    ent_b = shannon_entropy(nxt) / 8.0
    baseline = AdaptiveBaselineEstimator()
    feat = _vec_dist(_feature_vector(raw_a), _feature_vector(raw_b))
    layout = _layout_dist(lay_a, lay_b)
    embed = _embed_dist(chunk, nxt)
    entropy = abs(ent_a - ent_b)
    utility = abs(ev_a.utility - ev_b.utility)
    neg_a = baseline.baseline(list(ev_a.negative.values()) or [0.0])
    neg_b = baseline.baseline(list(ev_b.negative.values()) or [0.0])
    noise_shift = abs(neg_a - neg_b)
    left_pos = units[left].start / max(units[-1].end, 1) if units else 0.0
    right_pos = units[i].start / max(units[-1].end, 1) if units else 0.0
    role = _role_boundary_score(chunk, nxt, left_pos=left_pos, right_pos=right_pos)
    nav_cut = nav_transition_score(chunk, nxt, left_pos=left_pos, right_pos=right_pos)
    if nxt.strip() and nxt.strip()[0] in _LAYOUT_SPLITS:
        role = max(role, 0.55)
    if chunk.rstrip() and chunk.rstrip()[-1] in _LAYOUT_SPLITS:
        role = max(role, 0.55)
    combined = baseline.baseline([feat, layout, embed, entropy, utility, noise_shift, nav_cut, role])
    return {
        'combined': combined,
        'feat': feat,
        'layout': layout,
        'embed': embed,
        'entropy': entropy,
        'utility': utility,
        'noise_shift': noise_shift,
        'nav_transition': nav_cut,
        'role_boundary': role,
    }

def detect_boundaries(text: str, *, min_section_chars: int = 60) -> list[int]:
    if not text or not text.strip():
        return []
    units = _decompose_spans(text)
    if len(units) <= 1:
        return []

    scores: list[BoundaryCandidate] = []
    for i in range(1, len(units)):
        sig = _window_signal(units, i)
        role = sig.get('role_boundary', 0.0)
        scores.append(BoundaryCandidate(
            offset=units[i].start,
            score=sig['combined'],
            role_boundary=role >= 0.42,
            signals=sig,
        ))

    if not scores:
        return []

    baseline = AdaptiveBaselineEstimator()
    vals = [c.score for c in scores]
    role_vals = [c.signals.get('role_boundary', 0.0) for c in scores]
    thr = baseline.baseline([baseline.spread(vals), baseline.baseline(vals) * 1.12, 0.18])
    role_thr = baseline.baseline([baseline.spread(role_vals), 0.38, 0.42])

    cuts = [0]
    acc = units[0].text
    acc_start = units[0].start
    for i, unit in enumerate(units[1:], start=1):
        cand = scores[i - 1]
        acc_len = len(acc)
        role_hit = cand.signals.get('role_boundary', 0.0) >= role_thr
        para_break = unit.start - (units[i - 1].end) > 1
        if para_break and acc_len >= max(20, min_section_chars // 4):
            cuts.append(unit.start)
            acc = unit.text
            acc_start = unit.start
        elif role_hit and acc_len >= max(20, min_section_chars // 3):
            cuts.append(cand.offset)
            acc = unit.text
            acc_start = unit.start
        elif cand.score >= thr and acc_len >= min_section_chars:
            cuts.append(cand.offset)
            acc = unit.text
            acc_start = unit.start
        else:
            acc = f'{acc}\n{unit.text}' if acc else unit.text

    if len(cuts) <= 1 and len(units) >= 2:
        ranked = sorted(scores, key=lambda c: (-c.signals.get('role_boundary', 0.0), -c.score))
        for cand in ranked[: max(2, len(units) // 3)]:
            role = cand.signals.get('role_boundary', 0.0)
            if role >= role_thr * 0.9 and cand.offset not in cuts:
                cuts.append(cand.offset)
            elif cand.score >= thr * 0.82 and cand.offset not in cuts:
                cuts.append(cand.offset)
        cuts = sorted(set(cuts))

    return cuts[1:] if len(cuts) > 1 else []

def force_role_splits(text: str, *, min_section_chars: int = 60) -> list[int]:
    if not text or not text.strip():
        return []
    units = _decompose_spans(text)
    if len(units) <= 1:
        return []
    baseline = AdaptiveBaselineEstimator()
    role_vals: list[float] = []
    scores: list[BoundaryCandidate] = []
    for i in range(1, len(units)):
        sig = _window_signal(units, i)
        role = sig.get('role_boundary', 0.0)
        role_vals.append(role)
        scores.append(BoundaryCandidate(offset=units[i].start, score=sig['combined'], signals=sig))
    if not scores:
        return []
    role_thr = baseline.baseline([baseline.spread(role_vals), 0.36, 0.38])
    cuts: list[int] = []
    for i, cand in enumerate(scores, start=1):
        role = cand.signals.get('role_boundary', 0.0)
        if role >= role_thr:
            cuts.append(units[i].start)
    return sorted(set(cuts))

def decompose_span_sections(
    text: str,
    *,
    min_section_chars: int = 60,
) -> list[tuple[str, int, int]]:
    forced = force_role_splits(text, min_section_chars=min_section_chars)
    if not forced:
        return []
    strength = boundary_cut_strength(text, forced)
    return split_at_boundaries(
        text,
        forced,
        min_section_chars=max(20, min_section_chars // 3),
        cut_strength=strength,
    )

def boundary_cut_strength(text: str, cuts: list[int]) -> dict[int, float]:
    if not cuts:
        return {}
    units = _decompose_spans(text)
    if len(units) <= 1:
        return {cuts[0]: 1.0} if cuts else {}
    out: dict[int, float] = {}
    unit_starts = {u.start for u in units}
    for cut in cuts:
        idx = next((i for i, u in enumerate(units) if u.start == cut), None)
        if idx is None:
            for i, u in enumerate(units):
                if abs(u.start - cut) <= 2:
                    idx = i
                    break
        if idx is None or idx <= 0:
            out[cut] = 0.5
            continue
        sig = _window_signal(units, idx)
        out[cut] = max(sig.get('role_boundary', 0.0), sig.get('combined', 0.0))
    return out

def split_at_boundaries(
    text: str,
    cuts: list[int],
    *,
    min_section_chars: int = 60,
    cut_strength: dict[int, float] | None = None,
) -> list[tuple[str, int, int]]:
    if not text.strip():
        return []
    if not cuts:
        s = text.strip()
        return [(s, 0, len(s))]

    strength = cut_strength or boundary_cut_strength(text, cuts)
    bounds = [0, *sorted(c for c in cuts if 0 < c < len(text)), len(text)]
    out: list[tuple[str, int, int]] = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        chunk = text[start:end].strip()
        if not chunk:
            continue
        while chunk and chunk[-1] in _LAYOUT_SPLITS:
            chunk = chunk[:-1].rstrip()
        protected = strength.get(start, 0.0) >= 0.38
        if out and len(chunk) < min_section_chars and not protected:
            prev_text, prev_start, _ = out[-1]
            merged = f'{prev_text}\n\n{chunk}'
            from indw.extract.sections.semantic import (
                analyze_completion,
                last_complete_boundary,
            )
            comp = analyze_completion(merged)
            if comp.incomplete_probability >= 0.42:
                boundary = last_complete_boundary(merged, min_chars=min_section_chars)
                if boundary > 0:
                    merged = merged[:boundary].strip()
            out[-1] = (merged, prev_start, end)
            continue
        out.append((chunk, start, end))
    if not out:
        s = text.strip()
        return [(s, 0, len(s))]
    return out
