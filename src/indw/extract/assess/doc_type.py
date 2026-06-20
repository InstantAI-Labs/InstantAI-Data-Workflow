from __future__ import annotations

from dataclasses import dataclass, field

from indw.clean.artifact.evidence_engine import compute_semantic_evidence
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator

@dataclass
class DiscoveredClass:
    label: str
    weight: float
    source: str = 'latent'

@dataclass
class DocumentClassProfile:
    primary: str = 'unknown'
    confidence: float = 0.0
    classes: list[DiscoveredClass] = field(default_factory=list)
    mixed: bool = False

    def to_dict(self) -> dict:
        return {
            'primary': self.primary,
            'confidence': round(self.confidence, 4),
            'mixed': self.mixed,
            'classes': [{'label': c.label, 'weight': round(c.weight, 4), 'source': c.source} for c in self.classes[:8]],
        }

def classify_document(text: str) -> DocumentClassProfile:
    if not text or not text.strip():
        return DocumentClassProfile()

    evidence = compute_semantic_evidence(text)
    baseline = AdaptiveBaselineEstimator()
    classes: list[DiscoveredClass] = []

    for axis, val in sorted(evidence.positive.items(), key=lambda x: -x[1]):
        if val > baseline.baseline([val, 0.08]):
            classes.append(DiscoveredClass(label=axis, weight=val, source='positive'))
    for axis, val in sorted(evidence.negative.items(), key=lambda x: -x[1]):
        if val > baseline.baseline([val, 0.12]):
            classes.append(DiscoveredClass(label=f'noise_{axis}', weight=val, source='negative'))

    rep = evidence.representation
    if rep is not None:
        for axis in ('factual', 'referential', 'reasoning', 'narrative', 'procedural'):
            val = getattr(rep, axis, 0.0)
            if val > baseline.baseline([val, 0.1]):
                classes.append(DiscoveredClass(label=f'rep_{axis}', weight=val, source='representation'))

    classes.sort(key=lambda c: -c.weight)
    if not classes:
        return DocumentClassProfile(primary='unknown', confidence=0.0)

    primary = classes[0].label
    conf = classes[0].weight
    spread = baseline.spread([c.weight for c in classes[:4]])
    mixed = spread > baseline.baseline([spread, 0.15]) and len(classes) > 2
    return DocumentClassProfile(
        primary=primary,
        confidence=conf,
        classes=classes[:12],
        mixed=mixed,
    )
