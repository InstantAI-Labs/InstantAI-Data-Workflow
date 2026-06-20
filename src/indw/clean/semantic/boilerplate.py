from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from indw.clean.artifact.evidence import DocumentFeatureExtractor, RawDocumentFeatures

_WS = re.compile(r'\s+')
_WORD = re.compile(r'\b\w+\b', re.UNICODE)
_URL = re.compile(r'https?://|www\.', re.I)
_PHONE = re.compile(r'\+?\d[\d\s().-]{7,}\d')

@dataclass
class StatisticalBoilerplateSignals:
    token_entropy: float = 0.0
    phrase_repetition: float = 0.0
    line_uniformity: float = 0.0
    url_density: float = 0.0
    phone_density: float = 0.0
    link_density: float = 0.0
    metadata_density: float = 0.0
    template_similarity: float = 0.0
    boilerplate_score: float = 0.0

def _entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    n = len(tokens)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h

def _phrase_repeat(lines: list[str]) -> float:
    if len(lines) < 2:
        return 0.0
    norm = [_WS.sub(' ', ln.strip().lower()) for ln in lines if ln.strip()]
    if len(norm) < 2:
        return 0.0
    counts = Counter(norm)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return dup / max(len(norm), 1)

def _line_uniformity(lines: list[str]) -> float:
    lens = [len(ln) for ln in lines if ln.strip()]
    if len(lens) < 2:
        return 0.0
    mean = sum(lens) / len(lens)
    if mean <= 1e-6:
        return 0.0
    var = sum((x - mean) ** 2 for x in lens) / len(lens)
    cv = math.sqrt(var) / mean
    return max(0.0, min(1.0, 1.0 - cv))

def analyze_statistical_boilerplate(text: str, raw: RawDocumentFeatures | None = None) -> StatisticalBoilerplateSignals:
    if not text or not text.strip():
        return StatisticalBoilerplateSignals()
    raw = raw or DocumentFeatureExtractor().extract(text)
    lines = text.splitlines()
    tokens = _WORD.findall(text.lower())
    words = max(len(tokens), 1)
    chars = max(len(text), 1)

    ent = _entropy(tokens)
    ent_norm = min(1.0, ent / 6.5) if tokens else 0.0
    low_entropy = 1.0 - ent_norm

    phrase_rep = _phrase_repeat(lines)
    uniform = _line_uniformity(lines)
    url_d = len(_URL.findall(text)) / words
    phone_d = len(_PHONE.findall(text)) / words
    link_d = raw.anchor_density
    meta_d = raw.structured_line_ratio * 0.5 + raw.uniform_line_ratio * 0.5
    template = max(phrase_rep, uniform, raw.uniform_line_ratio)

    parts = [
        low_entropy * 0.18,
        phrase_rep * 0.22,
        uniform * 0.12,
        min(1.0, url_d * 8.0) * 0.14,
        min(1.0, phone_d * 12.0) * 0.10,
        min(1.0, link_d * 4.0) * 0.10,
        meta_d * 0.08,
        template * 0.06,
    ]
    score = max(0.0, min(1.0, sum(parts)))

    return StatisticalBoilerplateSignals(
        token_entropy=round(ent_norm, 4),
        phrase_repetition=round(phrase_rep, 4),
        line_uniformity=round(uniform, 4),
        url_density=round(url_d, 4),
        phone_density=round(phone_d, 4),
        link_density=round(link_d, 4),
        metadata_density=round(meta_d, 4),
        template_similarity=round(template, 4),
        boilerplate_score=round(score, 4),
    )
