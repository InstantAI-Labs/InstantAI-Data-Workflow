from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from indw.filter.spec.quality import QualityThresholds
    from indw.filter.gate.quality import QualityGate

logger = logging.getLogger(__name__)

_SCORE_BIN_EDGES = tuple(round(i * 0.1, 1) for i in range(11))

def score_histogram(scores: list[float], *, edges: tuple[float, ...] = _SCORE_BIN_EDGES) -> dict[str, int]:
    if not scores:
        return {}
    bins: dict[str, int] = {}
    for score in scores:
        idx = min(int(score * 10), 9)
        lo = edges[idx]
        hi = edges[idx + 1]
        key = f'{lo:.1f}-{hi:.1f}'
        bins[key] = bins.get(key, 0) + 1
    return dict(sorted(bins.items()))

def threshold_snapshot(th: 'QualityThresholds') -> dict[str, Any]:
    return {
        'mode': 'signal_based_adaptive',
        'high_quality_only': th.high_quality_only,
        'max_boilerplate_score': th.max_boilerplate_score,
        'max_commercial_score': th.max_commercial_score,
        'max_seo_spam_score': th.max_seo_spam_score,
        'max_low_information_score': th.max_low_information_score,
        'min_alpha_ratio': th.min_alpha_ratio,
    }

def log_gate_diagnostics(
    gate: 'QualityGate',
    *,
    total_scanned: int,
    merge_kept: int,
    merge_rejected: int,
    thresholds: 'QualityThresholds | None' = None,
) -> None:
    stats = gate.stats
    qs = stats.to_dict()
    th = thresholds or gate.config.thresholds
    pre_hist = score_histogram(list(stats.evaluated_score_samples))
    kept_hist = score_histogram(list(stats.score_samples))
    top_rejects = sorted(
        dict(qs.get('reject_reasons') or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    keep_rate = merge_kept / max(total_scanned, 1)
    logger.info(
        '[quality] scanned=%d kept=%d rejected=%d keep_rate=%.2f%% '
        'evaluated=%d pre_score_mean=%.3f kept_score_mean=%.3f',
        total_scanned,
        merge_kept,
        merge_rejected,
        keep_rate * 100.0,
        stats.evaluated,
        stats.pre_filter_score_mean,
        float(qs.get('score_mean', 0.0)),
    )
    calibration = gate.calibrator.distribution_stats() if hasattr(gate, 'calibrator') else {}
    qs = stats.to_dict()
    logger.info('[quality] thresholds=%s calibration=%s', threshold_snapshot(th), calibration)
    if top_rejects:
        logger.info('[quality] reject_reasons=%s', top_rejects)
    ev_discard = sorted(
        dict(qs.get('evidence_discard_reasons') or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    if ev_discard:
        logger.info(
            '[quality] evidence utility_mean=%.3f preserve_rate=%.2f%% discard=%s',
            float(qs.get('utility_mean', 0.0)),
            float(qs.get('preserve_rate', 0.0)) * 100.0,
            ev_discard,
        )
    if pre_hist:
        logger.info('[quality] pre_filter_score_histogram=%s', pre_hist)
    if kept_hist:
        logger.info('[quality] kept_score_histogram=%s', kept_hist)
    util_hist = qs.get('utility_histogram') or {}
    if util_hist:
        logger.info('[quality] utility_histogram=%s', util_hist)
