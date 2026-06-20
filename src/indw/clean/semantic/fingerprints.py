from __future__ import annotations

import math
from dataclasses import dataclass

from indw.clean.artifact.evidence import RawDocumentFeatures
from indw.clean.artifact.evidence_features import shared_feature_extractor

_ARTIFACT_AXES = (
    'url_char_ratio',
    'nav_line_ratio',
    'uniform_line_ratio',
    'contact_token_ratio',
    'schedule_token_ratio',
    'numeric_token_ratio',
    'uppercase_token_ratio',
    'line_len_cv',
    'exclaim_line_ratio',
    'first_person_ratio',
    'table_line_ratio',
    'fence_char_ratio',
    'structured_line_ratio',
    'anchor_density',
)

@dataclass(frozen=True)
class ArtifactArchetype:
    name: str
    vector: tuple[float, ...]

def _vec(raw: RawDocumentFeatures) -> tuple[float, ...]:
    return tuple(float(getattr(raw, k, 0.0)) for k in _ARTIFACT_AXES)

def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))

_ARCHETYPES: list[ArtifactArchetype] = [
    ArtifactArchetype('navigation', (0.35, 0.55, 0.40, 0.02, 0.01, 0.08, 0.15, 0.25, 0.02, 0.01, 0.0, 0.0, 0.15, 0.02)),
    ArtifactArchetype('footer', (0.12, 0.18, 0.35, 0.08, 0.05, 0.06, 0.10, 0.15, 0.01, 0.02, 0.0, 0.0, 0.05, 0.03)),
    ArtifactArchetype('contact', (0.08, 0.05, 0.20, 0.45, 0.25, 0.20, 0.05, 0.30, 0.02, 0.01, 0.0, 0.0, 0.02, 0.01)),
    ArtifactArchetype('license', (0.15, 0.02, 0.55, 0.01, 0.01, 0.03, 0.08, 0.08, 0.0, 0.0, 0.0, 0.05, 0.35, 0.04)),
    ArtifactArchetype('seo', (0.22, 0.12, 0.30, 0.02, 0.02, 0.05, 0.25, 0.35, 0.15, 0.08, 0.0, 0.0, 0.10, 0.02)),
    ArtifactArchetype('cookie_banner', (0.28, 0.22, 0.25, 0.03, 0.01, 0.04, 0.12, 0.20, 0.05, 0.02, 0.0, 0.0, 0.08, 0.01)),
]

def _calibrate_scores(raw: RawDocumentFeatures, scores: dict[str, float]) -> dict[str, float]:
    out = dict(scores)
    prose = raw.word_count >= 35 and raw.fence_char_ratio < 0.05
    if prose and raw.uniform_line_ratio < 0.55:
        for k in ('license', 'footer', 'seo', 'cookie_banner'):
            out[k] = round(out.get(k, 0.0) * 0.45, 4)
    if raw.contact_token_ratio > 0.08:
        out['contact'] = round(min(1.0, out.get('contact', 0.0) + raw.contact_token_ratio * 1.2), 4)
    if raw.schedule_token_ratio > 0.06:
        out['contact'] = round(min(1.0, out.get('contact', 0.0) + raw.schedule_token_ratio), 4)
    if raw.nav_line_ratio > 0.25:
        out['navigation'] = round(min(1.0, out.get('navigation', 0.0) + raw.nav_line_ratio * 0.8), 4)
    if raw.anchor_density > 0.08:
        out['navigation'] = round(min(1.0, out.get('navigation', 0.0) + raw.anchor_density * 0.5), 4)
    if raw.structured_line_ratio > 0.35 and raw.uniform_line_ratio > 0.45:
        out['license'] = round(min(1.0, out.get('license', 0.0) + 0.15), 4)
    return out

class SemanticFingerprintMatcher:
    def __init__(self) -> None:
        self._extractor = shared_feature_extractor()

    def match(self, text: str, *, raw: RawDocumentFeatures | None = None) -> dict[str, float]:
        if not text or len(text.strip()) < 20:
            return {}
        if raw is None:
            raw = self._extractor.extract(text)
        v = _vec(raw)
        scores: dict[str, float] = {}
        for arch in _ARCHETYPES:
            scores[arch.name] = round(_cosine(v, arch.vector), 4)
        return _calibrate_scores(raw, scores)

    def dominant_artifact(self, text: str, *, min_score: float = 0.55) -> tuple[str, float]:
        scores = self.match(text)
        if not scores:
            return '', 0.0
        val = scores[name]
        return (name, val) if val >= min_score else ('', val)
