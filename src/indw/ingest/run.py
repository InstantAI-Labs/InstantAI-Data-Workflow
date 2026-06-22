from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Optional
from indw.config.defaults import DEFAULT_QUALITY_SPEC, DEFAULT_WRITE_BUFFER_BYTES
from indw.store.corpus.registry import CorpusRegistry
from indw.ingest.download import DatasetDownloader
from indw.ingest.hf_env import configure_hf_fast
from indw.ingest.resume import run_incremental_stage
from indw.config.resolve import PipelineConfigContext
from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule import merge_with_quality
logger = logging.getLogger(__name__)

class FastDatasetPipeline:

    def __init__(self, work_dir: str | Path, config: Optional[dict[str, Any]]=None, *, corpus_id: str='default', write_buffer_mb: int | None=None, write_buffer_bytes: int | None=None, quality_config_path: Optional[str | Path]=None, repo_root: Optional[Path]=None):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        if write_buffer_bytes is not None:
            self.write_buffer_bytes = write_buffer_bytes
        elif write_buffer_mb is not None:
            self.write_buffer_bytes = write_buffer_mb * 1024 * 1024
        else:
            self.write_buffer_bytes = DEFAULT_WRITE_BUFFER_BYTES
        self.raw_dir = self.work_dir / 'raw'
        self.corpus = CorpusRegistry(self.work_dir, corpus_id=corpus_id)
        self.downloader = DatasetDownloader(self.raw_dir, write_buffer_bytes=self.write_buffer_bytes)
        self.quality_config = self._load_quality_config(quality_config_path)

    def _load_quality_config(self, path: Optional[str | Path]) -> QualityPipelineConfig:
        spec = str(path) if path else DEFAULT_QUALITY_SPEC
        self.ctx = PipelineConfigContext.resolve(quality_spec=spec)
        qc = self.ctx.quality
        if 'min_chars' in self.config:
            qc.thresholds.min_chars = int(self.config['min_chars'])
        if 'max_chars' in self.config:
            qc.thresholds.max_chars = int(self.config['max_chars'])
        if self.config.get('fuzzy_dedup'):
            qc.dedup.fuzzy = True
        return qc

    def run(self, sources: dict[str, Any], *, sources_ref_path: Path | None=None, skip_download: bool=False, incremental_sources: Optional[list[str]]=None, append_filtered: bool=False, resume_merge: bool=True, fresh_merge: bool=False, merge_workers: Optional[int]=None, merge_chunk_size: Optional[int]=None) -> Path:
        configure_hf_fast()
        pipeline_t0 = time.perf_counter()
        logger.info('=' * 72)
        logger.info(
            'QUALITY DATASET PIPELINE | work_dir=%s | corpus=%s | quality=%s | fresh=%s',
            self.work_dir,
            self.corpus.corpus_id,
            self.quality_config.enabled,
            fresh_merge,
        )
        logger.info('Sources: %s', (sources.get('meta') or {}).get('id', 'inline'))
        th = self.quality_config.thresholds
        cal = self.quality_config.adaptive_calibration
        logger.info(
            'Quality profile: adaptive=%s high_quality_only=%s exact_dedup=%s semantic_selection=%s',
            cal.enabled,
            th.high_quality_only,
            self.quality_config.dedup.exact,
            self.quality_config.semantic_selection.enabled,
        )
        logger.info('=' * 72)
        if not skip_download:
            dl_t0 = time.perf_counter()
            logger.info('Phase 1/3: Streaming download ...')
            if incremental_sources:
                self._fetch_incremental_sources(sources, incremental_sources)
            else:
                self.downloader.fetch_all(sources)
            logger.info('Phase 1/3 done in %.1fs', time.perf_counter() - dl_t0)
        else:
            logger.info('Phase 1/3: Skipping download (using %s)', self.raw_dir)
        filtered = self.work_dir / 'filtered.jsonl'
        merge_t0 = time.perf_counter()
        logger.info('Phase 2/3: Quality merge + dedup + balance -> %s', filtered)
        if incremental_sources:
            stats = run_incremental_stage(
                self.corpus,
                sources,
                new_source_names=incremental_sources,
                filtered_path=filtered,
                append_filtered=append_filtered,
                quality_config=self.quality_config,
                write_buffer_bytes=self.write_buffer_bytes,
                work_dir=self.work_dir,
                workers=merge_workers,
                chunk_size=merge_chunk_size,
            )
        else:
            stats = merge_with_quality(
                self.raw_dir,
                filtered,
                quality_config=self.quality_config,
                corpus_registry=self.corpus,
                write_buffer_bytes=self.write_buffer_bytes,
                work_dir=self.work_dir,
                resume=resume_merge,
                fresh=fresh_merge,
                workers=merge_workers,
                chunk_size=merge_chunk_size,
            )
        logger.info(
            'Phase 2/3 done in %.1fs | kept=%s rejected=%s exact_dup=%s',
            time.perf_counter() - merge_t0,
            stats.get('kept', stats.get('docs', '?')),
            stats.get('rejected', '?'),
            stats.get('exact_duplicates', 0),
        )
        manifest_t0 = time.perf_counter()
        logger.info('Phase 3/3: Corpus manifest + lineage')
        dedup_summary = self.corpus.open_index().summary()
        dedup_summary['merge_stats'] = stats
        from indw.schedule.config.pin import resolve_config_fingerprints
        import json as _json

        quality_fp, _sources_fp = resolve_config_fingerprints(self.work_dir)
        plan_digest = ''
        plan_path = self.work_dir / 'quality' / 'corpus_mixture_plan.json'
        if plan_path.is_file():
            try:
                plan_digest = str(_json.loads(plan_path.read_text(encoding='utf-8')).get('plan_digest', ''))
            except (OSError, _json.JSONDecodeError, TypeError, ValueError):
                plan_digest = ''
        dataset_manifest = str(self.work_dir / 'dataset_manifest.json')
        if not Path(dataset_manifest).is_file():
            dataset_manifest = ''
        self.corpus.save_manifest(
            sources_yaml=sources_ref_path or Path((sources.get('meta') or {}).get('id', 'inline')),
            stats=stats,
            dedup_summary=dedup_summary,
            update_replay_pool_from=filtered,
            quality_config_fingerprint=quality_fp,
            filtered_path=filtered,
            mixture_plan_digest=plan_digest,
            dataset_manifest_path=dataset_manifest,
            incremental_sources=incremental_sources,
        )
        self.corpus.close()
        logger.info('Phase 3/3 done in %.1fs', time.perf_counter() - manifest_t0)
        logger.info('=' * 72)
        logger.info(
            'PIPELINE COMPLETE in %.1fs: %s',
            time.perf_counter() - pipeline_t0,
            filtered,
        )
        logger.info('Quality report: %s', self.work_dir / 'quality' / 'corpus_quality_report.json')
        logger.info('Live progress: %s', self.work_dir / 'pipeline_progress.json')
        logger.info(
            'Metrics: Grafana http://127.0.0.1:3001 | raw http://127.0.0.1:9093/metrics'
        )
        logger.info('=' * 72)
        return filtered

    def _fetch_incremental_sources(self, cfg: dict[str, Any], names: list[str]) -> None:
        all_sources = {s['name']: s for s in cfg.get('sources', []) if isinstance(s, dict) and s.get('name')}
        subset = [all_sources[n] for n in names if n in all_sources]
        if len(subset) != len(names):
            missing = set(names) - set(all_sources)
            raise ValueError(f'Unknown incremental sources: {missing}')
        partial = {**cfg, 'sources': subset}
        self.downloader.fetch_all(partial)
