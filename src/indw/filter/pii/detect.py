from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from indw.filter.pii.config import PiiPolicyConfig
from indw.filter.pii.context import PiiContextAnalyzer, PiiContextResult
from indw.filter.pii.entities import EntityExtractor, EntityExtractionResult
from indw.filter.pii.redaction import redact_text
from indw.filter.pii.risk import PiiRiskResult, score_risk
from indw.filter.pii.secrets import SecretAnalyzer, SecretAnalysisResult

@dataclass
class PiiAssessment:
    entities: EntityExtractionResult
    secrets: SecretAnalysisResult
    context: PiiContextResult
    risk: PiiRiskResult
    redacted_text: Optional[str] = None

    @property
    def pii_score(self) -> float:
        return self.risk.pii_score

    @property
    def pii_reason(self) -> Optional[str]:
        return self.risk.reason

    def to_dict(self) -> dict[str, Any]:
        return {
            'entities': self.entities.to_dict(),
            'secrets': self.secrets.to_dict(),
            'context': self.context.to_dict(),
            'pii_score': round(self.risk.pii_score, 4),
            'band': self.risk.band,
            'reason': self.risk.reason,
            'redacted': self.redacted_text is not None,
        }

class PiiDetector:
    def __init__(self, policy: Optional[PiiPolicyConfig] = None):
        self.policy = policy or PiiPolicyConfig.resolve()
        self._entities = EntityExtractor(self.policy.ner)
        self._secrets = SecretAnalyzer(self.policy.secrets)
        self._context = PiiContextAnalyzer(self.policy.context)

    def assess(self, text: str) -> PiiAssessment:
        if not self.policy.enabled or not text:
            empty_ent = EntityExtractionResult()
            empty_sec = SecretAnalysisResult()
            empty_ctx = PiiContextResult()
            risk = score_risk(
                entities=empty_ent,
                secrets=empty_sec,
                context=empty_ctx,
                policy=self.policy,
            )
            return PiiAssessment(empty_ent, empty_sec, empty_ctx, risk)
        entities = self._entities.extract(text)
        secrets = self._secrets.analyze(text)
        nearby = text[:400] + ' ' + text[max(0, len(text) // 2) : len(text) // 2 + 200]
        context = self._context.analyze(text, nearby=nearby)
        risk = score_risk(
            entities=entities,
            secrets=secrets,
            context=context,
            policy=self.policy,
        )
        redacted: Optional[str] = None
        if self.policy.redaction_enabled and (risk.should_redact or risk.should_reject):
            redacted = redact_text(
                text,
                entities=entities.entities,
                secrets=secrets.spans,
            )
        return PiiAssessment(entities, secrets, context, risk, redacted)
