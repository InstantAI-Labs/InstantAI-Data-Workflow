from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from indw.store.eval.compare import CorpusComparison, compare_versions
from indw.store.eval.config import CorpusEvaluationConfig
from indw.store.eval.decision import AcceptanceDecision, decide_acceptance
from indw.store.eval.diversity import DiversityResult, _entropy, compute_diversity
from indw.store.eval.metrics import CorpusMetrics, collect_corpus_metrics
from indw.store.eval.scoring import CorpusScoreResult, compute_corpus_score
from indw.tools.metrics.snapshot import CorpusSnapshot
from indw.tools.metrics.storage import load_snapshots, previous_snapshot
from indw.filter.gate.quality import QualityGate
from indw.filter.gate.reports import CorpusQualityReport


@dataclass
class CorpusEvaluationResult:
    metrics: CorpusMetrics
    diversity: DiversityResult
    score: CorpusScoreResult
    comparison: CorpusComparison
    decision: AcceptanceDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            'metrics': self.metrics.to_dict(),
            'diversity': self.diversity.to_dict(),
            'knowledge_density': {'knowledge_density': round(self.metrics.knowledge_density, 4)},
            'corpus_score': self.score.to_dict(),
            'comparison': self.comparison.to_dict(),
            'decision': self.decision.to_dict(),
        }


class CorpusEvaluator:
    def __init__(self, config: Optional[CorpusEvaluationConfig] = None):
        self.config = config or CorpusEvaluationConfig.resolve()

    def evaluate(
        self,
        gate: QualityGate,
        snapshot: CorpusSnapshot,
        *,
        previous_metrics: Optional[CorpusMetrics] = None,
        observability_dir: Optional[str] = None,
    ) -> CorpusEvaluationResult:
        metrics = collect_corpus_metrics(gate, snapshot)
        if self.config.lightweight:
            metrics.knowledge_density = metrics.quality_score * 0.9
            lang_div = _entropy(metrics.language_distribution)
            diversity = DiversityResult(
                diversity_score=lang_div,
                language_diversity=lang_div,
                source_diversity=_entropy(metrics.source_distribution),
                domain_diversity=_entropy(metrics.domain_distribution),
                topic_diversity=lang_div,
            )
            score = compute_corpus_score(metrics, diversity, config=self.config)
            comparison = CorpusComparison(skipped=True, mode='lightweight')
            decision = decide_acceptance(
                score,
                comparison,
                config=self.config,
                has_previous=False,
            )
            return CorpusEvaluationResult(
                metrics=metrics,
                diversity=diversity,
                score=score,
                comparison=comparison,
                decision=decision,
            )
        diversity = compute_diversity(
            language_distribution=metrics.language_distribution,
            source_distribution=metrics.source_distribution,
            domain_distribution=metrics.domain_distribution,
        )
        score = compute_corpus_score(metrics, diversity, config=self.config)

        prev = previous_metrics
        if prev is None and observability_dir:
            from pathlib import Path

            prev_snap = previous_snapshot(Path(observability_dir))
            if prev_snap:
                prev = _metrics_from_snapshot(prev_snap)

        comparison = compare_versions(metrics, prev, config=self.config)
        decision = decide_acceptance(
            score,
            comparison,
            config=self.config,
            has_previous=prev is not None,
        )
        return CorpusEvaluationResult(
            metrics=metrics,
            diversity=diversity,
            score=score,
            comparison=comparison,
            decision=decision,
        )

    def evaluate_from_observability(
        self,
        gate: QualityGate,
        report: CorpusQualityReport,
        *,
        version: str,
        dedup_stats: Optional[dict] = None,
        merge_stats: Optional[dict] = None,
        corpus_manifest_version: Optional[int] = None,
    ) -> CorpusEvaluationResult:
        from indw.tools.metrics.snapshot import build_snapshot

        snap = build_snapshot(
            gate,
            report,
            version=version,
            dedup_stats=dedup_stats,
            merge_stats=merge_stats,
            corpus_manifest_version=corpus_manifest_version,
        )
        obs_dir = self.config.output_dir.replace('corpus_evaluation', 'observability')
        return self.evaluate(gate, snap, observability_dir=obs_dir)


def _metrics_from_snapshot(snap: CorpusSnapshot) -> CorpusMetrics:
    meta = snap.metadata or {}
    qs = meta.get('quality_signals') or {}
    return CorpusMetrics(
        quality_score=snap.quality_score_mean,
        duplicate_rate=snap.duplicate_rate,
        toxicity_rate=snap.toxicity_rate,
        pii_rate=snap.pii_rate,
        language_distribution=dict(snap.language_distribution),
        source_distribution=dict(snap.source_distribution),
        domain_distribution=dict(qs.get('domain_distribution') or {}),
        document_length_distribution=dict(qs.get('document_length_distribution') or {}),
        knowledge_density=float(qs.get('knowledge_density', snap.quality_score_mean * 0.9)),
        version=snap.version,
        accepted_documents=snap.accepted_documents,
    )


def load_previous_metrics(observability_dir: str) -> Optional[CorpusMetrics]:
    from pathlib import Path

    prev = previous_snapshot(Path(observability_dir))
    if not prev:
        return None
    return _metrics_from_snapshot(prev)
