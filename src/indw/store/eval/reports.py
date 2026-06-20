from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from indw.store.eval.config import CorpusEvaluationConfig
from indw.store.eval.evaluator import CorpusEvaluationResult, CorpusEvaluator
from indw.store.eval.metrics import collect_corpus_metrics
from indw.tools.metrics.snapshot import CorpusSnapshot
from indw.tools.metrics.storage import latest_snapshot, load_snapshots, previous_snapshot
from indw.filter.gate.quality import QualityGate
from indw.filter.gate.reports import CorpusQualityReport

def write_corpus_evaluation(
    gate: QualityGate,
    snapshot: CorpusSnapshot,
    *,
    work_dir: Optional[Path] = None,
    config: Optional[CorpusEvaluationConfig] = None,
) -> dict[str, Any]:
    pol = config or CorpusEvaluationConfig.resolve()
    obs_dir = Path(pol.output_dir).parent / 'observability'
    prev_snap = previous_snapshot(obs_dir) if obs_dir.exists() else None

    prev_metrics = None
    if prev_snap:
        from indw.store.eval.evaluator import _metrics_from_snapshot

        prev_metrics = _metrics_from_snapshot(prev_snap)

    evaluator = CorpusEvaluator(pol)
    result = evaluator.evaluate(gate, snapshot, previous_metrics=prev_metrics)

    now = datetime.now(timezone.utc).isoformat()
    payload = {'created_at': now, 'version': snapshot.version, **result.to_dict()}
    out = Path(pol.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'corpus_evaluation_report.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (out / 'corpus_evaluation_decision.json').write_text(
        json.dumps({'created_at': now, **result.decision.to_dict()}, indent=2),
        encoding='utf-8',
    )

    hist_path = out / 'corpus_evaluation_history.json'
    history: list[dict[str, Any]] = []
    if hist_path.exists():
        try:
            loaded = json.loads(hist_path.read_text(encoding='utf-8'))
            if isinstance(loaded, list):
                history = loaded
        except json.JSONDecodeError:
            history = []
    history.append(
        {
            'created_at': now,
            'version': snapshot.version,
            'decision': result.decision.decision,
            'corpus_score': result.score.corpus_score,
            'metrics': result.metrics.to_dict(),
        }
    )
    if len(history) > 200:
        history = history[-200:]
    hist_path.write_text(json.dumps(history, indent=2), encoding='utf-8')

    if work_dir:
        run_dir = Path(work_dir) / 'corpus_evaluation'
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'corpus_evaluation_report.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    return payload

def evaluate_after_merge(
    gate: QualityGate,
    report: CorpusQualityReport,
    *,
    work_dir: Optional[Path] = None,
    dedup_stats: Optional[dict[str, Any]] = None,
    merge_stats: Optional[dict[str, Any]] = None,
    corpus_manifest_version: Optional[int] = None,
    config: Optional[CorpusEvaluationConfig] = None,
) -> dict[str, Any]:
    pol = config or CorpusEvaluationConfig.resolve()
    if not pol.enabled:
        return {'enabled': False}
    obs_dir = Path(pol.output_dir).parent / 'observability'
    snap = latest_snapshot(obs_dir)
    if snap is None:
        from indw.tools.metrics.snapshot import build_snapshot
        from indw.tools.metrics.storage import next_version

        snap = build_snapshot(
            gate,
            report,
            version=next_version(obs_dir),
            dedup_stats=dedup_stats,
            merge_stats=merge_stats,
            corpus_manifest_version=corpus_manifest_version,
        )
    return write_corpus_evaluation(gate, snap, work_dir=work_dir, config=pol)
