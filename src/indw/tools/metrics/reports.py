from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from indw.tools.metrics.alerts import Alert, alerts_from_regression
from indw.tools.metrics.config import ObservabilityPolicyConfig
from indw.tools.metrics.regression import RegressionResult, compare_corpora
from indw.tools.metrics.snapshot import CorpusSnapshot
from indw.tools.metrics.storage import append_snapshot, load_snapshots, previous_snapshot
from indw.tools.metrics.trends import build_trends, write_trend_histories
from indw.filter.gate.quality import QualityGate
from indw.filter.gate.reports import CorpusQualityReport

def record_corpus_observability(
    gate: QualityGate,
    report: CorpusQualityReport,
    *,
    policy: Optional[ObservabilityPolicyConfig] = None,
    dedup_stats: Optional[dict[str, Any]] = None,
    merge_stats: Optional[dict[str, Any]] = None,
    corpus_manifest_version: Optional[int] = None,
    compare_version: Optional[str] = None,
) -> dict[str, Any]:
    from indw.tools.metrics.regression import analyze_regression
    from indw.tools.metrics.snapshot import build_snapshot
    from indw.tools.metrics.storage import latest_snapshot, next_version

    pol = policy or ObservabilityPolicyConfig.resolve()
    if not pol.enabled:
        return {'enabled': False}

    out = Path(pol.output_dir)
    version = next_version(out)
    snapshot = build_snapshot(
        gate,
        report,
        version=version,
        dedup_stats=dedup_stats,
        merge_stats=merge_stats,
        corpus_manifest_version=corpus_manifest_version,
    )
    prev = previous_snapshot(out)
    regression = analyze_regression(snapshot, prev, policy=pol)
    alerts = alerts_from_regression(regression)
    append_snapshot(out, snapshot)
    trends = build_trends(out)
    write_trend_histories(out, load_snapshots(out))

    comparison: dict[str, Any] = {}
    if compare_version:
        snaps = {s.version: s for s in load_snapshots(out)}
        other = snaps.get(compare_version)
        if other:
            comparison = compare_corpora(snapshot, other)

    now = datetime.now(timezone.utc).isoformat()
    metrics = {
        'total_documents': snapshot.total_documents,
        'accepted_documents': snapshot.accepted_documents,
        'rejected_documents': snapshot.rejected_documents,
        'duplicate_rate': snapshot.duplicate_rate,
        'quality_score_distribution': snapshot.quality_score_distribution,
        'quality_score_mean': snapshot.quality_score_mean,
        'toxicity_rate': snapshot.toxicity_rate,
        'pii_rate': snapshot.pii_rate,
        'language_distribution': snapshot.language_distribution,
        'average_document_length': snapshot.average_document_length,
        'source_distribution': snapshot.source_distribution,
    }
    payload = {
        'created_at': now,
        'snapshot': snapshot.to_dict(),
        'metrics': metrics,
        'regression': regression.to_dict(),
        'alerts': [a.to_dict() for a in alerts],
        'trends': trends,
        'comparison': comparison,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / 'observability_report.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (out / 'regression_report.json').write_text(json.dumps(regression.to_dict(), indent=2), encoding='utf-8')
    (out / 'alerts.json').write_text(json.dumps({'created_at': now, 'alerts': [a.to_dict() for a in alerts]}, indent=2), encoding='utf-8')
    return payload

def write_observability_artifacts(
    gate: QualityGate,
    report: CorpusQualityReport,
    *,
    work_dir: Optional[Path] = None,
    policy: Optional[ObservabilityPolicyConfig] = None,
    dedup_stats: Optional[dict[str, Any]] = None,
    merge_stats: Optional[dict[str, Any]] = None,
    corpus_manifest_version: Optional[int] = None,
) -> dict[str, Any]:
    pol = policy or ObservabilityPolicyConfig.resolve()
    payload = record_corpus_observability(
        gate,
        report,
        policy=pol,
        dedup_stats=dedup_stats,
        merge_stats=merge_stats,
        corpus_manifest_version=corpus_manifest_version,
    )
    if work_dir:
        run_dir = Path(work_dir) / 'observability'
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'observability_report.json').write_text(
            json.dumps(payload, indent=2),
            encoding='utf-8',
        )
    return payload
