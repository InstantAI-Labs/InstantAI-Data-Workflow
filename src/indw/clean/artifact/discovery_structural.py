from __future__ import annotations

import math

from indw.clean.artifact.decompose import LayoutVector

def _layout_distance(a: LayoutVector, b: LayoutVector) -> float:
    ta = a.to_tuple()
    tb = b.to_tuple()
    if len(ta) != len(tb):
        return 1.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(ta, tb))) / max(len(ta), 1)

def structural_signature(layout: LayoutVector) -> str:
    return layout.bucket()

def html_remnant_score(layout: LayoutVector, text: str) -> float:
    if '<' not in text and '&' not in text:
        return 0.0
    angle = text.count('<') + text.count('>')
    entity = text.count('&')
    tagish = min(1.0, (angle + entity) / max(len(text), 1) * 12)
    return min(1.0, tagish * 0.6 + layout.punct_ratio * 0.4)

class StructuralLearner:
    def cohesion(self, layout: LayoutVector, cluster_centroid: LayoutVector | None) -> float:
        if cluster_centroid is None:
            return 0.5
        dist = _layout_distance(layout, cluster_centroid)
        return max(0.0, min(1.0, 1.0 - dist / 2.0))

    def score(
        self,
        layout: LayoutVector,
        doc_frequency: int,
        docs_seen: int,
        *,
        cluster_centroid: LayoutVector | None = None,
        text: str = '',
    ) -> float:
        cohesion = self.cohesion(layout, cluster_centroid)
        freq = doc_frequency / max(docs_seen, 1)
        nav_hint = min(1.0, layout.url_ratio * 8 + layout.punct_ratio * 2)
        uniform_hint = 1.0 if layout.line_count >= 2 and layout.avg_len < 60 else 0.0
        sep_hint = min(1.0, layout.punct_ratio * 4) if layout.alpha_ratio < 0.3 else 0.0
        html_hint = html_remnant_score(layout, text)
        template = max(nav_hint, uniform_hint, sep_hint, html_hint) * 0.4
        return min(1.0, cohesion * 0.35 + freq * 0.35 + template * 0.30)
