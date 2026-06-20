from __future__ import annotations

from typing import Any


def reject_stage(reason: str) -> str:
    if reason.startswith('validation_'):
        return 'validation'
    if reason in ('exact_dup', 'near_dup_fuzzy', 'near_dup_semantic', 'near_dup_embed'):
        return 'dedup'
    if reason in (
        'domain_cap', 'language_cap', 'curriculum_balance', 'low_value', 'toxicity',
        'invalid_code', 'secret_detected', 'rejected',
    ):
        return 'quality_scoring'
    if reason in ('too_short', 'document_too_large', 'document_budget_exceeded'):
        return 'document_gate'
    if 'language' in reason:
        return 'language_detection'
    if reason.startswith('empty') or reason in (
        'no_knowledge_units', 'no_knowledge', 'no_knowledge_after_filter', 'cleaning',
    ):
        return 'cleaning'
    if 'novelty' in reason or 'coherence' in reason:
        return 'semantic_cleaning'
    return 'quality_scoring'


def record_merge_reject(
    reject_log: Any,
    *,
    reason: str,
    source: str,
    text: str = '',
    doc: Any = None,
    quality_score: float = 0.0,
    domain: str = '',
    language: str = '',
) -> None:
    if reject_log is None:
        return
    score = quality_score
    dom = domain
    lang = language
    if doc is not None:
        score = float(getattr(doc, 'score', score) or score)
        dom = str(getattr(doc, 'domain', dom) or dom)
        lang = str(getattr(doc, 'language', lang) or lang)
    reject_log.record(
        reason=reason,
        stage=reject_stage(reason),
        source=source,
        domain=dom,
        language=lang,
        quality_score=score,
        chars=len(text),
        preview=text[:240],
    )
