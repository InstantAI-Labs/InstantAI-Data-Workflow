from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from indw.clean.artifact.safeguards import is_protected_unit
from indw.clean.semantic.classifier import ChunkClassification
from indw.clean.semantic.config import SemanticCleaningConfig
from indw.clean.semantic.scoring import SectionSignals, compute_section_signals
from indw.clean.semantic.thresholds import CorpusThresholdCalibrator

RoutingAction = Literal['KEEP', 'KEEP_AFTER_CLEANING', 'DOWNWEIGHT', 'REMOVE']

ARTIFACT_ROLES = frozenset({
    'navigation', 'footer', 'contact', 'legal', 'metadata', 'author_info',
    'related_content', 'promotional',
})
KNOWLEDGE_ROLES = frozenset({
    'title', 'introduction', 'body', 'code', 'table', 'examples', 'references',
})

@dataclass
class RoutingDecision:
    action: RoutingAction = 'KEEP'
    section_role: str = 'body'
    signals: SectionSignals = field(default_factory=SectionSignals)
    confidence: float = 0.0
    reason: str = ''
    classification: ChunkClassification | None = None

    def to_dict(self) -> dict:
        return {
            'action': self.action,
            'section_role': self.section_role,
            'confidence': round(self.confidence, 4),
            'reason': self.reason,
            'signals': self.signals.to_dict(),
            'dominant_label': self.classification.dominant if self.classification else '',
        }

class SectionRouter:
    def __init__(self, config: SemanticCleaningConfig, calibrator: CorpusThresholdCalibrator):
        self.config = config
        self._calibrator = calibrator

    def route(
        self,
        text: str,
        classification: ChunkClassification,
        *,
        section_role: str = 'body',
        position_ratio: float = 0.5,
        in_fence: bool = False,
    ) -> RoutingDecision:
        cfg = self.config
        signals = compute_section_signals(classification, section_role=section_role)

        if cfg.calibrate_thresholds and classification.evidence:
            neg = max(classification.evidence.negative.values()) if classification.evidence.negative else 0.0
            self._calibrator.observe(classification.utility, neg)

        remove_thr = self._calibrator.remove_threshold()
        down_thr = self._calibrator.downweight_threshold()
        noise_ceil = self._calibrator.noise_ceiling()

        pos_mass = (
            signals.educational_value + signals.technical_value
            + signals.reference_value + signals.knowledge_value * 0.5
        )
        neg_mass = (
            signals.promotional_likelihood + signals.navigation_likelihood
            + signals.metadata_likelihood + signals.boilerplate_likelihood + signals.noise_level * 0.5
        )

        action: RoutingAction = 'KEEP'
        reason = 'default_keep'

        if classification.evidence and classification.evidence.preserve:
            if signals.noise_level > 0.42:
                action, reason = 'KEEP_AFTER_CLEANING', 'evidence_preserve_trim'
            else:
                action, reason = 'KEEP', 'evidence_preserve'
        elif in_fence or (cfg.preserve_code and section_role in ('code', 'table')):
            action, reason = 'KEEP', 'protected_structure'
        elif is_protected_unit(text, kind=section_role, in_fence=in_fence):
            action, reason = 'KEEP', 'protected_content'
        elif section_role == 'legal' and signals.boilerplate_likelihood > 0.45:
            action, reason = 'REMOVE', 'legal_boilerplate'
        elif section_role in ('navigation', 'footer', 'contact', 'promotional') and neg_mass > pos_mass + 0.08:
            if signals.knowledge_value < remove_thr or classification.utility < remove_thr:
                action, reason = 'REMOVE', f'artifact_{section_role}'
            else:
                action, reason = 'KEEP_AFTER_CLEANING', f'artifact_{section_role}_mixed'
        elif section_role in ('metadata', 'author_info', 'related_content', 'promotional'):
            if signals.knowledge_value < remove_thr and signals.evergreen_value < down_thr:
                action, reason = 'REMOVE', f'low_value_{section_role}'
            elif signals.knowledge_value >= down_thr:
                action, reason = 'KEEP_AFTER_CLEANING', f'trim_{section_role}'
            else:
                action, reason = 'DOWNWEIGHT', f'marginal_{section_role}'
        elif signals.educational_value >= cfg.preserve_educational_floor and pos_mass > neg_mass:
            if signals.noise_level > noise_ceil * 0.7 and signals.promotional_likelihood > 0.25:
                action, reason = 'KEEP_AFTER_CLEANING', 'educational_with_noise'
            else:
                action, reason = 'KEEP', 'educational_preserve'
        elif signals.knowledge_value < remove_thr and signals.noise_level >= noise_ceil * 0.85:
            action, reason = 'REMOVE', 'low_utility_high_noise'
        elif signals.knowledge_value < down_thr or neg_mass > pos_mass + 0.1:
            action, reason = 'DOWNWEIGHT', 'marginal_utility'
        elif signals.noise_level > 0.38 or signals.promotional_likelihood > 0.35:
            action, reason = 'KEEP_AFTER_CLEANING', 'light_trim'

        fp = classification.fingerprint
        fp_name = max(fp, key=fp.get) if fp else ''
        fp_score = fp.get(fp_name, 0.0) if fp_name else 0.0
        if action == 'REMOVE' and pos_mass > neg_mass + 0.15 and signals.educational_value > 0.2:
            action, reason = 'KEEP_AFTER_CLEANING', 'educational_override'

        knowledge_roles = KNOWLEDGE_ROLES
        if (
            action in ('REMOVE', 'DOWNWEIGHT')
            and fp_score >= cfg.fingerprint_similarity_remove
            and fp_name in {'navigation', 'footer', 'contact', 'license', 'cookie_banner', 'seo'}
            and signals.educational_value < cfg.preserve_educational_floor
            and pos_mass < neg_mass
        ):
            if section_role in knowledge_roles and (
                signals.knowledge_value >= 0.20
                or signals.educational_value + signals.technical_value >= 0.14
            ):
                action, reason = 'KEEP_AFTER_CLEANING', f'knowledge_{fp_name}_trim'
            elif fp_name in {'navigation', 'footer', 'contact'}:
                action, reason = 'REMOVE', f'fingerprint_{fp_name}'

        conf = classification.confidence
        if classification.evidence:
            conf = max(conf, classification.evidence.confidence)

        return RoutingDecision(
            action=action,
            section_role=section_role,
            signals=signals,
            confidence=conf,
            reason=reason,
            classification=classification,
        )
