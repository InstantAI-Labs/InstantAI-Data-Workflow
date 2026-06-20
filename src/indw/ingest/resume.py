from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional
from indw.config.defaults import DEFAULT_WRITE_BUFFER_BYTES
from indw.store.corpus.registry import CorpusRegistry
from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule import merge_with_quality
logger = logging.getLogger(__name__)

def merge_incremental(registry: CorpusRegistry, raw_dir: Path, out_path: Path, *, source_names: Optional[list[str]]=None, quality_config: Optional[QualityPipelineConfig]=None, write_buffer_bytes: int=DEFAULT_WRITE_BUFFER_BYTES, append: bool=False, work_dir: Optional[Path]=None, workers: Optional[int]=None, chunk_size: Optional[int]=None, **_: Any) -> dict[str, Any]:
    return merge_with_quality(
        raw_dir,
        out_path,
        quality_config=quality_config,
        corpus_registry=registry,
        write_buffer_bytes=write_buffer_bytes,
        source_filter=source_names,
        append=append,
        work_dir=work_dir or registry.work_dir,
        workers=workers,
        chunk_size=chunk_size,
    )

def run_incremental_stage(registry: CorpusRegistry, sources_yaml: Path, *, new_source_names: list[str], filtered_path: Optional[Path]=None, append_filtered: bool=True, quality_config: Optional[QualityPipelineConfig]=None, write_buffer_bytes: int=DEFAULT_WRITE_BUFFER_BYTES, work_dir: Optional[Path]=None, workers: Optional[int]=None, chunk_size: Optional[int]=None, **_: Any) -> dict[str, Any]:
    filtered = filtered_path or registry.work_dir / 'filtered.jsonl'
    return merge_with_quality(
        registry.work_dir / 'raw',
        filtered,
        quality_config=quality_config,
        corpus_registry=registry,
        write_buffer_bytes=write_buffer_bytes,
        source_filter=new_source_names,
        append=append_filtered,
        work_dir=work_dir or registry.work_dir,
        resume=True,
        workers=workers,
        chunk_size=chunk_size,
    )
