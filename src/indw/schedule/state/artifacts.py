from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from indw.store.eval.reports import evaluate_after_merge
from indw.filter.language.reports import write_language_reports
from indw.filter.language.script_policy import MultilingualPolicyConfig
from indw.filter.language.telemetry import sample_tokenizer_telemetry_by_bucket
from indw.tools.metrics.pipeline_health import (
    checkpoint_stats_from_path,
    gate_stats_from_gate,
    record_pipeline_health,
)
from indw.tools.metrics.reports import write_observability_artifacts
from indw.schedule.mix.config import MixtureOrchestrationConfig
from indw.schedule.mix.mixture_planner import build_corpus_mixture_plan
from indw.filter.spec.quality import QualityPipelineConfig
from indw.filter.gate.quality import QualityGate
from indw.filter.gate.reports import append_quality_history, build_quality_report
from indw.filter.language.stats import sample_tokenizer_efficiency
from indw.filter.license.manifest import write_dataset_manifest
from indw.filter.license.reports import write_license_reports
from indw.filter.pii.reports import write_pii_reports
from indw.filter.toxicity.reports import write_toxicity_reports

if TYPE_CHECKING:
    from indw.store.corpus.registry import CorpusRegistry
    from indw.dedup.fuzzy import StreamingFuzzyDedup
    from indw.dedup.exact import PersistentHashIndex
    from indw.dedup.semantic import StreamingSemanticDedup
    from indw.clean.corpus import CorpusCleaningPipeline
    from indw.ingest.hash import ExactHashDedup
    from indw.schedule.state.checkpoint import MergeCheckpoint

logger = logging.getLogger(__name__)


def merge_quality_report_path(merge_work: Path) -> Path:
    return Path(merge_work) / 'quality' / 'corpus_quality_report.json'


def _seed_gate_stats_from_progress(gate: QualityGate, *, merge_work: Path, checkpoint: 'MergeCheckpoint') -> None:
    from indw.schedule.state.checkpoint import load_run_progress, restore_balancers_from_checkpoint

    restore_balancers_from_checkpoint(gate, checkpoint)
    totals = checkpoint.totals()
    gate.stats.kept = int(totals.get('kept', 0))
    gate.stats.rejected = int(totals.get('rejected', 0))
    progress = load_run_progress(merge_work)
    if not progress:
        return
    rejects = progress.get('reject_reasons')
    if isinstance(rejects, dict):
        gate.stats.reject_reasons = defaultdict(int)
        for key, value in rejects.items():
            try:
                gate.stats.reject_reasons[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    try:
        score_mean = float(progress.get('score_mean', 0.0))
    except (TypeError, ValueError):
        score_mean = 0.0
    if score_mean > 0:
        gate.stats.score_samples = [score_mean]


def run_merge_finalize(
    *,
    gate: QualityGate,
    cfg: QualityPipelineConfig,
    out_path: Path,
    work_dir: Optional[Path],
    merge_work: Path,
    checkpoint: 'MergeCheckpoint',
    exact: 'ExactHashDedup',
    cleaning_pipeline: 'CorpusCleaningPipeline',
    skipped_parse: int,
    elapsed: float,
    pipeline_metrics: Any = None,
    corpus_registry: Optional['CorpusRegistry'] = None,
    index: Optional['PersistentHashIndex'] = None,
    fuzzy: Optional['StreamingFuzzyDedup'] = None,
    semantic: Optional['StreamingSemanticDedup'] = None,
    embed_semantic: Any = None,
    parallel_workers: Optional[int] = None,
    worker_failures: int = 0,
    pci_stats: Optional[dict[str, Any]] = None,
    doc_monitor_stats: Optional[dict[str, Any]] = None,
    seed_from_progress: bool = False,
    log_message: Optional[str] = None,
) -> dict[str, Any]:
    from indw.ingest.log import human_bytes
    from indw.store.io.jsonl import checkpoint_kept_lines, count_jsonl_lines
    from indw.schedule.state.checkpoint import load_run_progress

    if seed_from_progress:
        _seed_gate_stats_from_progress(gate, merge_work=merge_work, checkpoint=checkpoint)
    if index is not None:
        index.flush()
    progress = load_run_progress(merge_work)
    exact_duplicates = exact.duplicates
    if seed_from_progress:
        try:
            exact_duplicates = int(progress.get('exact_duplicates', exact_duplicates))
        except (TypeError, ValueError):
            pass
    totals = checkpoint.totals()
    kept_lines = checkpoint_kept_lines(checkpoint) or count_jsonl_lines(out_path)
    total_bytes = out_path.stat().st_size if out_path.exists() else 0
    dedup_stats = {
        'exact_duplicates': exact_duplicates,
        'unique_hashes': len(exact),
        'skipped_trained': exact.skipped_trained,
    }
    if fuzzy is not None:
        dedup_stats.update(fuzzy.summary())
    if semantic is not None:
        dedup_stats.update(semantic.summary())
    if embed_semantic is not None:
        dedup_stats.update(embed_semantic.summary())
    cleaning_stats = cleaning_pipeline.snapshot()
    if work_dir:
        cleaning_report = Path(work_dir) / 'quality' / 'cleaning_report.json'
        cleaning_report.parent.mkdir(parents=True, exist_ok=True)
        cleaning_report.write_text(json.dumps(cleaning_stats, indent=2), encoding='utf-8')
    from indw.schedule.monitor.obs import stage_metrics_on_finalize
    if stage_metrics_on_finalize():
        from indw.tools.metrics.stage_profile import MergeStageProfile, write_stage_metrics
        write_stage_metrics(
            merge_work,
            MergeStageProfile(),
            cleaning_stats=cleaning_stats,
            merge_wall_sec=elapsed,
            docs_scanned=int(totals.get('scanned', 0)),
        )
    merge_stats: dict[str, Any] = {
        'docs': kept_lines,
        'skipped_parse': skipped_parse,
        'bytes': total_bytes,
        'elapsed_sec': round(elapsed, 2),
        'scanned': totals['scanned'],
        'kept': totals['kept'],
        'rejected': totals['rejected'],
        'session_kept': gate.stats.kept,
        'session_rejected': gate.stats.rejected,
        'cleaning': cleaning_stats,
    }
    if parallel_workers is not None:
        merge_stats['parallel_workers'] = parallel_workers
    if worker_failures:
        merge_stats['worker_failures'] = worker_failures
    if pci_stats is not None:
        merge_stats['pci'] = pci_stats
    if doc_monitor_stats is not None:
        merge_stats['doc_monitor'] = doc_monitor_stats
    if cleaning_pipeline.discovery_engine is not None:
        cleaning_pipeline.discovery_engine.close()
    from indw.clean.artifact.discovery_engine import reset_discovery_engines
    reset_discovery_engines()
    try:
        from indw.store.io.columnar import write_mixture_index_parquet
        write_mixture_index_parquet(Path(merge_work))
    except Exception:
        pass
    result = finalize_merge(
        gate=gate,
        cfg=cfg,
        out_path=out_path,
        work_dir=work_dir,
        dedup_stats=dedup_stats,
        merge_stats=merge_stats,
        elapsed=elapsed,
        pipeline_metrics=pipeline_metrics,
    )
    if corpus_registry is not None:
        corpus_registry.close()
    if log_message:
        logger.info(
            '%s: kept=%d rejected=%d exact_dup=%d %s (%.1fs)',
            log_message,
            totals['kept'],
            totals['rejected'],
            dedup_stats['exact_duplicates'],
            human_bytes(total_bytes),
            elapsed,
        )
    return result


def finalize_merge(
    *,
    gate: QualityGate,
    cfg: QualityPipelineConfig,
    out_path: Path,
    work_dir: Optional[Path],
    dedup_stats: dict[str, Any],
    merge_stats: dict[str, Any],
    elapsed: float,
    pipeline_metrics: Any = None,
) -> dict[str, Any]:
    tok_stats: dict[str, Any] = {}
    gate_tok = gate.stats.tokenizer_telemetry.to_dict()
    if gate_tok.get('global', {}).get('samples', 0) > 0:
        tok_stats = gate_tok
    elif cfg.track_token_efficiency and cfg.tokenizer_path and out_path.exists():
        mpol = MultilingualPolicyConfig.from_dict(cfg.multilingual)
        tok_stats = sample_tokenizer_telemetry_by_bucket(
            out_path,
            Path(cfg.tokenizer_path),
            max_samples=cfg.sample_scores,
            target_chars_per_token=mpol.target_chars_per_token,
        )
        if not tok_stats.get('per_bucket'):
            tok_stats = sample_tokenizer_efficiency(out_path, Path(cfg.tokenizer_path), max_samples=cfg.sample_scores)
    report = build_quality_report(
        gate,
        dedup_stats=dedup_stats,
        merge_stats=merge_stats,
        tokenizer_stats=tok_stats,
        elapsed_sec=elapsed,
    )
    if pipeline_metrics is not None:
        pipeline_metrics.publish_final(
            gate,
            report,
            dedup_stats=dedup_stats,
            merge_stats=merge_stats,
        )
    if work_dir:
        _write_workdir_artifacts(
            gate=gate,
            cfg=cfg,
            work_dir=Path(work_dir),
            report=report,
            dedup_stats=dedup_stats,
            merge_stats=merge_stats,
            tok_stats=tok_stats,
        )
    from indw.schedule.state.survivor import release_survivor_mmap
    if work_dir:
        release_survivor_mmap(work_dir)
    return {
        **merge_stats,
        **dedup_stats,
        'quality': gate.stats.to_dict(),
        'recommendations': report.recommendations,
    }


def _write_workdir_artifacts(
    *,
    gate: QualityGate,
    cfg: QualityPipelineConfig,
    work_dir: Path,
    report: Any,
    dedup_stats: dict[str, Any],
    merge_stats: dict[str, Any],
    tok_stats: dict[str, Any],
) -> None:
    report_dir = work_dir / 'quality'
    report.save(report_dir / 'corpus_quality_report.json')
    append_quality_history(work_dir, report)
    if gate._toxicity_detector is not None and gate._toxicity_policy.enabled:
        write_toxicity_reports(
            gate.toxicity_stats,
            output_dir=work_dir / 'toxicity',
            report_detail={'merge_stats': merge_stats, 'work_dir': str(work_dir)},
        )
    if gate._pii_detector is not None and gate._pii_policy.enabled:
        write_pii_reports(
            gate.pii_stats,
            output_dir=work_dir / 'pii',
            report_detail={'merge_stats': merge_stats, 'work_dir': str(work_dir)},
        )
    if gate._license_detector is not None and gate._license_policy.enabled:
        lic_dir = work_dir / gate._license_policy.output_dir
        write_license_reports(
            gate.license_stats,
            output_dir=lic_dir,
            report_detail={'merge_stats': merge_stats, 'work_dir': str(work_dir)},
        )
        write_dataset_manifest(
            gate.license_stats,
            work_dir / 'dataset_manifest.json',
            source_distribution=gate.stats.source_distribution(),
            extra={'quality_report': str(report_dir / 'corpus_quality_report.json')},
        )
    if gate._language_identifier is not None and gate._language_policy.enabled:
        lang_dir = work_dir / 'language'
        write_language_reports(
            gate.language_stats,
            output_dir=lang_dir,
            report_detail={'merge_stats': merge_stats, 'work_dir': str(work_dir)},
        )
        pol = gate._language_policy
        write_language_reports(
            gate.language_stats,
            output_dir=pol.reporting_output_dir,
            report_detail={'merge_stats': merge_stats, 'work_dir': str(work_dir)},
        )
    orch_cfg = MixtureOrchestrationConfig.from_dict(cfg.orchestration)
    if orch_cfg.enabled:
        plan = build_corpus_mixture_plan(gate, cfg=orch_cfg, tokenizer_stats=tok_stats)
        plan_path = work_dir / 'quality' / 'corpus_mixture_plan.json'
        plan.save(plan_path)
        logger.info('Corpus mixture plan → %s (digest=%s)', plan_path, plan.plan_digest)
    obs_pol = cfg.observability_policy()
    manifest_ver = None
    latest_manifest = work_dir / 'latest.json'
    if latest_manifest.exists():
        try:
            meta = json.loads(latest_manifest.read_text(encoding='utf-8'))
            manifest_ver = int(meta.get('version', 0)) or None
        except (json.JSONDecodeError, TypeError, ValueError):
            manifest_ver = None
    if obs_pol.enabled:
        def _obs_async() -> None:
            write_observability_artifacts(
                gate,
                report,
                work_dir=work_dir,
                policy=obs_pol,
                dedup_stats=dedup_stats,
                merge_stats=merge_stats,
                corpus_manifest_version=manifest_ver,
            )
        threading.Thread(target=_obs_async, daemon=True, name='merge-obs-finalize').start()
    eval_pol = cfg.corpus_evaluation_policy()
    if eval_pol.enabled:
        eval_payload = evaluate_after_merge(
            gate,
            report,
            work_dir=work_dir,
            dedup_stats=dedup_stats,
            merge_stats=merge_stats,
            corpus_manifest_version=manifest_ver,
            config=eval_pol,
        )
        merge_stats['corpus_evaluation'] = eval_payload.get('decision', {})
        decision_body = eval_payload.get('decision', {})
        if isinstance(decision_body, dict) and decision_body.get('decision') == 'REJECT':
            merge_stats['corpus_evaluation_blocked'] = True
    merge_work = work_dir / 'merge' if (work_dir / 'merge').exists() else work_dir
    obs_out = Path(obs_pol.output_dir) if obs_pol.enabled else work_dir / 'observability'
    if not obs_out.is_absolute():
        obs_out = work_dir / obs_out
    record_pipeline_health(
        obs_out,
        gate_stats=gate_stats_from_gate(gate),
        merge_stats=merge_stats,
        checkpoint_stats=checkpoint_stats_from_path(merge_work),
    )
