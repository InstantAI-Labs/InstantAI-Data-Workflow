from __future__ import annotations

from dataclasses import dataclass

from indw.clean.semantic.classifier import ChunkClassification

@dataclass
class SectionSignals:
    knowledge_value: float = 0.0
    educational_value: float = 0.0
    technical_value: float = 0.0
    reference_value: float = 0.0
    promotional_likelihood: float = 0.0
    navigation_likelihood: float = 0.0
    metadata_likelihood: float = 0.0
    boilerplate_likelihood: float = 0.0
    noise_level: float = 0.0
    evergreen_value: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}

def compute_section_signals(
    classification: ChunkClassification,
    *,
    section_role: str = 'body',
) -> SectionSignals:
    probs = classification.probabilities
    evidence = classification.evidence
    boiler = classification.boilerplate
    fp = classification.fingerprint

    edu = probs.get('educational', 0.0)
    tech = probs.get('technical', 0.0) + probs.get('code', 0.0)
    ref = probs.get('reference', 0.0) + probs.get('documentation', 0.0)
    promo = probs.get('advertisement', 0.0)
    nav = probs.get('navigation', 0.0) + fp.get('navigation', 0.0) * 0.5
    meta = probs.get('metadata', 0.0) + probs.get('seo', 0.0)
    boiler_s = (boiler.boilerplate_score if boiler else 0.0) + probs.get('boilerplate', 0.0)

    knowledge = classification.utility
    rep = evidence.representation if evidence else None
    quality = evidence.quality if evidence else None
    if rep:
        knowledge = max(knowledge, rep.factual * 0.35 + rep.reasoning * 0.25 + rep.structural * 0.15)
    if quality:
        knowledge = max(knowledge, quality.information_density_per_token * 0.4 + quality.coherence * 0.2)

    noise = 0.0
    if evidence and evidence.negative:
        noise = max(evidence.negative.values())
    noise = max(noise, boiler_s * 0.6, promo * 0.4, nav * 0.35)

    evergreen = 0.0
    if rep:
        evergreen = rep.factual * 0.4 + rep.structural * 0.3 + (1.0 - rep.temporal) * 0.3
    if section_role in ('body', 'code', 'table', 'introduction', 'reference'):
        evergreen = max(evergreen, edu + tech * 0.5)

    if section_role == 'navigation':
        nav = min(1.0, nav + 0.25)
    elif section_role == 'footer':
        nav = min(1.0, nav + 0.2)
        boiler_s = min(1.0, boiler_s + 0.15)
    elif section_role == 'contact':
        meta = min(1.0, meta + 0.2)
        boiler_s = min(1.0, boiler_s + 0.1)
    elif section_role == 'legal':
        boiler_s = min(1.0, boiler_s + 0.3)
    elif section_role in ('metadata', 'author_info'):
        meta = min(1.0, meta + 0.2)
    elif section_role == 'related_content':
        nav = min(1.0, nav + 0.2)

    return SectionSignals(
        knowledge_value=knowledge,
        educational_value=edu,
        technical_value=tech,
        reference_value=ref,
        promotional_likelihood=promo,
        navigation_likelihood=nav,
        metadata_likelihood=meta,
        boilerplate_likelihood=boiler_s,
        noise_level=noise,
        evergreen_value=evergreen,
    )
