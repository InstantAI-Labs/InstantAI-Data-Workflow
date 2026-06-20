from __future__ import annotations

import math
import zlib
from collections import Counter
from dataclasses import dataclass, field

from indw.clean.artifact.decompose import LayoutVector, compute_layout
from indw.clean.semantic.boilerplate import analyze_statistical_boilerplate
from indw.clean.semantic.structure import infer_section_role
from indw.filter.score.signals import shannon_entropy
from indw.clean.artifact.evidence_features import shared_feature_extractor


@dataclass
class StructuralProfile:
    layout: LayoutVector = field(default_factory=LayoutVector)
    char_entropy: float = 0.0
    word_entropy: float = 0.0
    unique_token_ratio: float = 0.0
    compression_ratio: float = 1.0
    repeated_trigram_ratio: float = 0.0
    repeated_line_ratio: float = 0.0
    title_density: float = 0.0
    metadata_density: float = 0.0
    navigation_density: float = 0.0
    table_density: float = 0.0
    boilerplate_density: float = 0.0
    template_density: float = 0.0
    content_density: float = 0.0
    link_density: float = 0.0
    header_footer_repetition: float = 0.0
    paragraph_quality_mean: float = 0.0
    sentence_completeness_mean: float = 0.0
    information_density: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items() if isinstance(v, float)}


def _ngram_repeat_ratio(tokens: list[str], n: int) -> float:
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return min(1.0, repeated / len(grams))


def _sentence_completeness(line: str) -> float:
    from indw.extract.sections.boundaries import period_ends_sentence
    from indw.filter.refine.truncation import _DANGLING_END

    s = line.strip()
    if not s:
        return 0.0
    if s[-1] in '.!?':
        if s[-1] == '.':
            dot = s.rfind('.')
            if dot >= 0 and not period_ends_sentence(s, dot):
                return 0.25
        words = s.split()
        if words:
            last = words[-1].strip('\'",;:)').lower()
            if last in _DANGLING_END or (len(last) == 1 and last.isalpha()):
                return 0.22
        return 0.88
    words = s.split()
    if len(words) >= 6:
        return 0.65
    if len(words) <= 2:
        return 0.2
    return 0.45


def analyze_structure(text: str) -> StructuralProfile:
    if not text or not text.strip():
        return StructuralProfile()
    from indw.clean.artifact.evidence_cache import get_structure_cache, structure_cache_key
    key = structure_cache_key(text)
    if key is not None:
        cache = get_structure_cache()
        hit = cache.get(key)
        if hit is not None:
            return hit
        result = _analyze_structure_impl(text)
        cache.put(key, result)
        return result
    return _analyze_structure_impl(text)


def _analyze_structure_impl(text: str) -> StructuralProfile:
    layout = compute_layout(text)
    raw = shared_feature_extractor().extract(text)
    words = raw.words
    lines = [ln.strip() for ln in raw.lines if ln.strip()]
    prof = StructuralProfile(layout=layout)

    prof.char_entropy = shannon_entropy(text) / 8.0
    if words:
        prof.word_entropy = shannon_entropy(' '.join(words)) / 8.0
        prof.unique_token_ratio = len(set(w.lower() for w in words)) / len(words)
    payload = text.encode('utf-8', errors='ignore')
    if payload:
        prof.compression_ratio = len(zlib.compress(payload, 9)) / max(len(payload), 1)
    prof.repeated_trigram_ratio = _ngram_repeat_ratio([w.lower() for w in words], 3)
    if lines:
        line_keys = [ln.lower() for ln in lines]
        prof.repeated_line_ratio = 1.0 - len(set(line_keys)) / len(line_keys)

    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    role_counts: Counter[str] = Counter()
    completeness: list[float] = []
    for i, para in enumerate(paras):
        pos = i / max(len(paras) - 1, 1)
        role = infer_section_role(
            para,
            layout=compute_layout(para),
            position_ratio=pos,
            structural_kind='paragraph',
        )
        role_counts[role] += 1
        completeness.extend(_sentence_completeness(ln) for ln in para.splitlines() if ln.strip())

    n = max(len(paras), 1)
    prof.title_density = role_counts.get('title', 0) / n
    prof.metadata_density = (role_counts.get('metadata', 0) + role_counts.get('author_info', 0)) / n
    prof.navigation_density = role_counts.get('navigation', 0) / n
    prof.table_density = role_counts.get('table', 0) / n
    prof.boilerplate_density = analyze_statistical_boilerplate(text).boilerplate_score
    prof.template_density = max(prof.repeated_line_ratio, prof.repeated_trigram_ratio)
    prof.content_density = (role_counts.get('body', 0) + role_counts.get('introduction', 0)) / n
    prof.link_density = min(1.0, raw.url_char_ratio + raw.anchor_density)
    if len(lines) >= 4:
        head = lines[:2]
        tail = lines[-2:]
        prof.header_footer_repetition = sum(
            1 for ln in head + tail if len(ln.split()) <= 6 and raw.nav_line_ratio > 0.05
        ) / 4.0
    if completeness:
        prof.sentence_completeness_mean = sum(completeness) / len(completeness)
    prof.paragraph_quality_mean = prof.content_density * prof.sentence_completeness_mean
    prof.information_density = (
        prof.unique_token_ratio * 0.35
        + prof.paragraph_quality_mean * 0.35
        + (1.0 - prof.template_density) * 0.30
    )
    return prof
