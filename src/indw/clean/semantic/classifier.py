from __future__ import annotations

import math
from dataclasses import dataclass, field

from indw.clean.artifact.safeguards import is_protected_unit
from indw.clean.semantic.boilerplate import StatisticalBoilerplateSignals, analyze_statistical_boilerplate
from indw.clean.semantic.fingerprints import SemanticFingerprintMatcher
from indw.clean.artifact.evidence_features import shared_feature_extractor
from indw.clean.artifact.evidence import (
    LatentSemanticRepresentation,
    RawDocumentFeatures,
    SemanticEvidenceBundle,
    compute_semantic_evidence,
)

LABELS = (
    'educational',
    'technical',
    'code',
    'documentation',
    'conversation',
    'reference',
    'news',
    'advertisement',
    'navigation',
    'footer',
    'contact',
    'legal',
    'license',
    'seo',
    'metadata',
    'boilerplate',
)

@dataclass
class ChunkClassification:
    probabilities: dict[str, float] = field(default_factory=dict)
    dominant: str = ''
    confidence: float = 0.0
    utility: float = 0.0
    preserve: bool = True
    action: str = 'keep'
    discard_reason: str = ''
    fingerprint: dict[str, float] = field(default_factory=dict)
    boilerplate: StatisticalBoilerplateSignals | None = None
    evidence: SemanticEvidenceBundle | None = None

    def to_dict(self) -> dict:
        return {
            'probabilities': {k: round(v, 4) for k, v in self.probabilities.items()},
            'dominant': self.dominant,
            'confidence': round(self.confidence, 4),
            'utility': round(self.utility, 4),
            'preserve': self.preserve,
            'action': self.action,
            'discard_reason': self.discard_reason,
            'fingerprint': self.fingerprint,
            'boilerplate_score': self.boilerplate.boilerplate_score if self.boilerplate else 0.0,
        }

def _softmax_dict(raw: dict[str, float]) -> dict[str, float]:
    vals = list(raw.values())
    if not vals:
        return {}
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) or 1.0
    keys = list(raw.keys())
    return {k: exps[i] / s for i, k in enumerate(keys)}

def _map_rep_to_labels(
    rep: LatentSemanticRepresentation,
    quality: DynamicQualityScores,
    raw: RawDocumentFeatures,
    boiler: StatisticalBoilerplateSignals,
    fingerprint: dict[str, float],
) -> dict[str, float]:
    raw_scores = {
        'educational': rep.educational * 0.7 + quality.educational * 0.3,
        'technical': rep.technical * 0.6 + quality.technical * 0.4,
        'code': quality.code + raw.fence_char_ratio * 0.4,
        'documentation': quality.reference * 0.5 + rep.structural * 0.3 + raw.structured_line_ratio * 0.2,
        'conversation': rep.narrative * 0.5 + rep.opinion * 0.5,
        'reference': quality.reference * 0.6 + rep.factual * 0.4,
        'news': rep.temporal * 0.5 + rep.narrative * 0.3,
        'advertisement': rep.promotional * 0.7 + rep.transactional * 0.3,
        'navigation': rep.navigational * 0.6 + fingerprint.get('navigation', 0.0) * 0.4,
        'footer': fingerprint.get('footer', 0.0) * 0.7 + rep.navigational * 0.3,
        'contact': fingerprint.get('contact', 0.0) * 0.6 + raw.contact_token_ratio * 0.4,
        'legal': fingerprint.get('license', 0.0) * 0.4 + rep.structural * 0.2,
        'license': fingerprint.get('license', 0.0) * 0.8 + boiler.metadata_density * 0.2,
        'seo': fingerprint.get('seo', 0.0) * 0.7 + boiler.template_similarity * 0.3,
        'metadata': boiler.metadata_density * 0.5 + rep.corruption * 0.3,
        'boilerplate': boiler.boilerplate_score * 0.7 + rep.redundancy * 0.3,
    }
    return _softmax_dict(raw_scores)

class SemanticChunkClassifier:
    def __init__(self) -> None:
        self._fingerprints = SemanticFingerprintMatcher()
        self._features = shared_feature_extractor()

    def classify(
        self,
        text: str,
        *,
        position_ratio: float = 0.5,
        in_fence: bool = False,
        enabled: bool = True,
    ) -> ChunkClassification:
        if not text or not text.strip():
            return ChunkClassification(action='remove', preserve=False, confidence=1.0)

        evidence = compute_semantic_evidence(text, enabled=enabled)
        rep = evidence.representation
        quality = evidence.quality
        raw_feats = self._features.extract(text)

        boiler = analyze_statistical_boilerplate(text, raw_feats)
        fp = self._fingerprints.match(text)

        probs = _map_rep_to_labels(rep, quality, raw_feats, boiler, fp)
        dominant = max(probs, key=probs.get) if probs else ''
        conf = probs.get(dominant, 0.0) if dominant else 0.0

        neg_labels = {'navigation', 'footer', 'contact', 'license', 'seo', 'metadata', 'boilerplate', 'advertisement'}
        neg_mass = sum(probs.get(l, 0.0) for l in neg_labels)
        pos_labels = {'educational', 'technical', 'code', 'documentation', 'reference'}
        pos_mass = sum(probs.get(l, 0.0) for l in pos_labels)

        action = 'keep'
        preserve = evidence.preserve
        if in_fence or is_protected_unit(text, kind='code' if in_fence else '', in_fence=in_fence):
            preserve = True
            action = 'keep'
        elif evidence.preserve:
            action = 'keep'
            preserve = True
        elif neg_mass > pos_mass + 0.12 and evidence.utility < 0.35:
            action = 'remove'
            preserve = False
        elif neg_mass > pos_mass and evidence.utility < 0.48:
            action = 'downweight'
            preserve = True

        artifact_fp = max(
            fp.get('navigation', 0.0),
            fp.get('footer', 0.0),
            fp.get('contact', 0.0),
            fp.get('license', 0.0),
            fp.get('seo', 0.0),
            fp.get('cookie_banner', 0.0),
        )
        if not evidence.preserve and artifact_fp >= 0.72:
            if position_ratio > 0.88 and fp.get('footer', 0.0) > 0.6:
                action = 'remove'
                preserve = False
            if position_ratio < 0.08 and fp.get('navigation', 0.0) > 0.65 and pos_mass < 0.25:
                action = 'remove'
                preserve = False

        return ChunkClassification(
            probabilities=probs,
            dominant=dominant,
            confidence=conf,
            utility=evidence.utility,
            preserve=preserve,
            action=action,
            discard_reason=evidence.discard_reason or evidence.discard_code,
            fingerprint=fp,
            boilerplate=boiler,
            evidence=evidence,
        )
