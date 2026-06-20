from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from indw.filter.pii.config import PiiPolicyConfig
from indw.filter.pii.context import PiiContextResult
from indw.filter.pii.entities import EntityExtractionResult
from indw.filter.pii.secrets import SecretAnalysisResult

_REASON_BY_CONTEXT = {
    'credential_leak': 'credential_leak',
    'customer_data': 'customer_data',
    'production_secret': 'production_secret',
}

@dataclass
class PiiRiskResult:
    pii_score: float = 0.0
    entity_score: float = 0.0
    secret_score: float = 0.0
    context_score: float = 0.0
    band: str = 'accept'
    reason: Optional[str] = None
    should_reject: bool = False
    should_hard_reject: bool = False
    should_redact: bool = False

    def to_public_dict(self, *, entity_count: int, secret_count: int) -> dict:
        return {
            'pii_score': round(self.pii_score, 4),
            'entities': entity_count,
            'secrets': secret_count,
            'reason': self.reason,
        }

def score_risk(
    *,
    entities: EntityExtractionResult,
    secrets: SecretAnalysisResult,
    context: PiiContextResult,
    policy: PiiPolicyConfig,
) -> PiiRiskResult:
    weights = policy.scoring_weights
    w_ent = float(weights.get('entities', 0.40))
    w_sec = float(weights.get('secrets', 0.38))
    w_ctx = float(weights.get('context', 0.22))
    entity_score = entities.entity_risk()
    secret_score = secrets.secret_probability
    ctx_boost = 1.0 if context.context in ('credential_leak', 'customer_data', 'production_secret') else 0.0
    raw = (
        w_ent * entity_score
        + w_sec * secret_score
        + w_ctx * ctx_boost * context.confidence
    )
    raw = min(1.0, raw * context.risk_multiplier)
    peak = secret_score if w_ent <= 0.0 else max(entity_score, secret_score)
    if context.context in ('example', 'documentation', 'educational', 'configuration'):
        if peak < 0.45:
            peak = min(peak, 0.22 * (1.0 - context.confidence * 0.5))
            raw = min(raw, 0.18)
        else:
            damp = max(0.35, 0.55 - context.confidence * 0.25)
            peak *= damp
            raw *= damp
    pii_score = min(1.0, max(raw, peak * 0.88))
    th = policy.thresholds
    band = th.band(pii_score)
    if policy.max_pii_score is not None and pii_score > policy.max_pii_score:
        band = 'reject' if pii_score < th.hard_reject else 'hard_reject'
    should_hard = band == 'hard_reject'
    should_reject = band in ('reject', 'hard_reject')
    should_redact = band == 'redact' and not should_reject
    reason = None
    if should_reject or should_redact:
        reason = _REASON_BY_CONTEXT.get(context.context)
        if not reason and secret_score >= entity_score:
            reason = 'secret_exposure'
        elif not reason:
            reason = 'pii_entities'
    return PiiRiskResult(
        pii_score=pii_score,
        entity_score=entity_score,
        secret_score=secret_score,
        context_score=context.confidence,
        band=band,
        reason=reason,
        should_reject=should_reject,
        should_hard_reject=should_hard,
        should_redact=should_redact,
    )
