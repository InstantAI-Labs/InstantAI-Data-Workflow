from __future__ import annotations

from typing import Any

from indw.config.defaults import (
    MAX_CHARS_GATE,
    MAX_MERGE_CHUNK_SIZE,
    MIN_CHARS_GATE,
    MIN_MERGE_CHUNK_SIZE,
)
class ConfigValidationError(ValueError):
    pass


class ConfigResolutionError(RuntimeError):
    pass


def _require_range(name: str, value: float, *, lo: float, hi: float) -> None:
    if not lo <= value <= hi:
        raise ConfigValidationError(f'{name} must be in [{lo}, {hi}], got {value}')


def validate_quality_config(cfg) -> None:
    th = cfg.thresholds
    if th.min_chars < 1:
        raise ConfigValidationError(f'thresholds.min_chars must be >= 1, got {th.min_chars}')
    if th.max_chars < th.min_chars:
        raise ConfigValidationError(
            f'thresholds.max_chars ({th.max_chars}) must be >= min_chars ({th.min_chars})'
        )
    if th.max_chars > MAX_CHARS_GATE * 2:
        raise ConfigValidationError(f'thresholds.max_chars unreasonably large: {th.max_chars}')

    clean = cfg.cleaning
    if clean.min_chars_after_clean < MIN_CHARS_GATE:
        raise ConfigValidationError(
            f'cleaning.min_chars_after_clean must be >= {MIN_CHARS_GATE}, got {clean.min_chars_after_clean}'
        )
    if clean.hard_max_chars < clean.min_chars_after_clean:
        raise ConfigValidationError(
            'cleaning.hard_max_chars must be >= min_chars_after_clean'
        )

    _require_range('dedup.fuzzy_threshold', cfg.dedup.fuzzy_threshold, lo=0.0, hi=1.0)
    _require_range('dedup.semantic_jaccard_threshold', cfg.dedup.semantic_jaccard_threshold, lo=0.0, hi=1.0)
    _require_range('balance.soft_cap_overflow', cfg.balance.soft_cap_overflow, lo=0.0, hi=1.0)
    _require_range('balance.quality_cap_bypass_score', cfg.balance.quality_cap_bypass_score, lo=0.0, hi=1.0)
    _require_range(
        'balance.quality_high_value_domain_bypass_score',
        cfg.balance.quality_high_value_domain_bypass_score,
        lo=0.0,
        hi=1.0,
    )

    if cfg.dedup.fuzzy and cfg.dedup.fuzzy_threshold < 0.5:
        raise ConfigValidationError('dedup.fuzzy_threshold too low for stable near-dup detection')

    if th.max_toxicity is not None:
        _require_range('thresholds.max_toxicity', th.max_toxicity, lo=0.0, hi=1.0)
    if th.max_pii_score is not None:
        _require_range('thresholds.max_pii_score', th.max_pii_score, lo=0.0, hi=1.0)


def validate_merge_runtime(*, workers: int, chunk_size: int) -> None:
    if workers < 1:
        raise ConfigValidationError(f'merge workers must be >= 1, got {workers}')
    if not MIN_MERGE_CHUNK_SIZE <= chunk_size <= MAX_MERGE_CHUNK_SIZE:
        raise ConfigValidationError(
            f'merge chunk_size must be in [{MIN_MERGE_CHUNK_SIZE}, {MAX_MERGE_CHUNK_SIZE}], got {chunk_size}'
        )


def validate_pipeline_policy(policy) -> None:
    from indw.filter.spec.pipeline import DecisionHeuristicsPolicy, StructuralRepairThresholds

    th: StructuralRepairThresholds = policy.structural_thresholds
    if not 0.0 <= th.trunc_repair_probability <= 1.0:
        raise ConfigValidationError('structural_repair.trunc_repair_probability out of range')
    if th.trunc_remove_probability < th.trunc_repair_probability:
        raise ConfigValidationError('trunc_remove_probability must be >= trunc_repair_probability')

    d: DecisionHeuristicsPolicy = policy.decision
    if d.prose_ratio_for_code <= 0 or d.prose_ratio_for_code > 1:
        raise ConfigValidationError('decision.prose_ratio_for_code out of range')
    if d.secret_pii_score < 0 or d.secret_pii_score > 1:
        raise ConfigValidationError('decision.secret_pii_score out of range')
    if d.confidence_min >= d.confidence_max:
        raise ConfigValidationError('decision.confidence_min must be < confidence_max')


def validate_legacy_aliases(raw: dict[str, Any]) -> list[str]:
    legacy: list[str] = []
    if 'deduplication' in raw:
        legacy.append('deduplication')
    if 'corpus_cleaning' in raw:
        legacy.append('corpus_cleaning')
    th = raw.get('thresholds') or {}
    if 'min_quality_score' in th and 'min_score' not in th:
        legacy.append('thresholds.min_quality_score')
    if raw.get('tokenizer_validation') and not raw.get('track_token_efficiency'):
        legacy.append('tokenizer_validation')
    return legacy
