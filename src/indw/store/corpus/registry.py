from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional
from indw.store.corpus.manifest import CorpusManifest, file_sha256, next_version, corpus_build_id
from indw.dedup.exact import PersistentHashIndex
logger = logging.getLogger(__name__)

class CorpusRegistry:

    def __init__(self, work_dir: str | Path, *, corpus_id: str='default'):
        self.work_dir = Path(work_dir)
        self.corpus_id = corpus_id
        self.corpus_dir = self.work_dir / 'corpus'
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir = self.corpus_dir / 'manifests'
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.replay_pool_path = self.corpus_dir / 'replay_pool.jsonl'
        self._index: Optional[PersistentHashIndex] = None

    @property
    def index_path(self) -> Path:
        return PersistentHashIndex.default_path(self.work_dir)

    def open_index(self) -> PersistentHashIndex:
        if self._index is None:
            self._index = PersistentHashIndex(self.index_path)
        return self._index

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None

    def latest_manifest(self) -> Optional[CorpusManifest]:
        files = sorted(self.manifest_dir.glob('manifest_v*.json'))
        if not files:
            return None
        return CorpusManifest.load(files[-1])

    def save_manifest(
        self,
        *,
        sources_yaml: Path,
        stats: dict[str, Any],
        dedup_summary: dict[str, Any],
        tokenizer_name: str = '',
        tokenizer_version: str = '',
        tokenizer_path: str = '',
        shard_glob: str = '',
        update_replay_pool_from: Optional[Path] = None,
        quality_config_fingerprint: str = '',
        filtered_path: Optional[Path] = None,
        mixture_plan_digest: str = '',
        dataset_manifest_path: str = '',
        incremental_sources: Optional[list[str]] = None,
    ) -> CorpusManifest:
        sources_yaml = Path(sources_yaml)
        prev = self.latest_manifest()
        version = next_version(self.manifest_dir)
        if update_replay_pool_from and update_replay_pool_from.exists():
            shutil.copy2(update_replay_pool_from, self.replay_pool_path)
            logger.info('Updated replay pool → %s', self.replay_pool_path)
        filtered_sha = ''
        filtered_lines = 0
        if filtered_path and filtered_path.exists():
            filtered_sha = file_sha256(filtered_path)
            filtered_lines = int(stats.get('kept', 0) or 0)
            if filtered_lines <= 0:
                with filtered_path.open('rb') as fin:
                    filtered_lines = sum(1 for ln in fin if ln.strip())
        build_id = corpus_build_id(
            sources_yaml_sha256=file_sha256(sources_yaml) if sources_yaml.exists() else '',
            quality_config_fingerprint=quality_config_fingerprint,
            filtered_sha256=filtered_sha,
        )
        manifest = CorpusManifest(
            corpus_id=self.corpus_id,
            version=version,
            sources_yaml=str(sources_yaml),
            sources_yaml_sha256=file_sha256(sources_yaml) if sources_yaml.exists() else '',
            quality_config_fingerprint=quality_config_fingerprint,
            filtered_sha256=filtered_sha,
            filtered_line_count=filtered_lines,
            mixture_plan_digest=mixture_plan_digest,
            corpus_build_id=build_id,
            dataset_manifest_path=dataset_manifest_path,
            tokenizer_name=tokenizer_name,
            tokenizer_version=tokenizer_version,
            tokenizer_path=tokenizer_path,
            stats={
                **stats,
                **({'incremental_sources': list(incremental_sources)} if incremental_sources else {}),
            },
            dedup=dedup_summary,
            shard_glob=shard_glob,
            replay_pool=str(self.replay_pool_path) if self.replay_pool_path.exists() else '',
            parent_version=prev.version if prev else None,
        )
        out = self.manifest_dir / f'manifest_v{version}.json'
        manifest.save(out)
        (self.corpus_dir / 'latest.json').write_text(json.dumps({'manifest': str(out), 'version': version}, indent=2), encoding='utf-8')
        logger.info('Corpus manifest v%s → %s (build_id=%s)', version, out, build_id)
        return manifest

    def rollback_to(self, version: int) -> CorpusManifest:
        path = self.manifest_dir / f'manifest_v{version}.json'
        if not path.is_file():
            raise FileNotFoundError(f'No manifest_v{version}.json under {self.manifest_dir}')
        manifest = CorpusManifest.load(path)
        (self.corpus_dir / 'latest.json').write_text(
            json.dumps({'manifest': str(path), 'version': version}, indent=2),
            encoding='utf-8',
        )
        logger.info('Rolled back corpus pointer to manifest v%s', version)
        return manifest

    def mark_corpus_trained(self, jsonl_path: Path, *, source: str='trained') -> int:
        idx = self.open_index()
        n = idx.ingest_jsonl_hashes(jsonl_path, source=source)
        logger.info('Marked %d documents as trained in dedup index', n)
        return n

    def status(self) -> dict[str, Any]:
        idx = self.open_index()
        return         {
            'corpus_id': self.corpus_id,
            'work_dir': str(self.work_dir),
            'dedup': idx.summary(),
            'latest_manifest': self.latest_manifest().to_dict() if self.latest_manifest() else None,
            'replay_pool_bytes': self.replay_pool_path.stat().st_size if self.replay_pool_path.exists() else 0
        }
