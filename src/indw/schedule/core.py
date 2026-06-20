from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, TextIO

from indw.filter.spec.quality import QualityPipelineConfig
from indw.ingest.sink import DEFAULT_WRITE_BUFFER
from indw.schedule.row.resolve import resolve_merge_chunk_size, resolve_merge_workers
from indw.schedule.state.checkpoint import MergeCheckpoint
from indw.store.corpus.registry import CorpusRegistry

logger = logging.getLogger(__name__)


def _weighted_schedule(weights: dict[str, int]) -> list[str]:
    schedule: list[str] = []
    for name in sorted(weights):
        schedule.extend([name] * weights[name])
    return schedule


@dataclass
class _SourceHandle:
    path: Path
    name: str
    stream: TextIO
    line_no: int = -1
    exhausted: bool = False


@dataclass
class _InterleavedSources:
    handles: dict[str, _SourceHandle] = field(default_factory=dict)
    schedule: list[str] = field(default_factory=list)
    tick: int = 0

    @classmethod
    def open(cls, sources: list[Path], checkpoint: MergeCheckpoint, weights: dict[str, int]) -> '_InterleavedSources':
        handles: dict[str, _SourceHandle] = {}
        for src in sources:
            name = src.parent.name
            stream = src.open(encoding='utf-8')
            skip = checkpoint.line_offset(name)
            for _ in range(skip):
                stream.readline()
            handles[name] = _SourceHandle(path=src, name=name, stream=stream, line_no=skip - 1)
        schedule = _weighted_schedule({n: weights.get(n, 1) for n in handles})
        return cls(handles=handles, schedule=schedule)

    def close(self) -> None:
        for handle in self.handles.values():
            if not handle.exhausted:
                handle.stream.close()
                handle.exhausted = True

    def __iter__(self) -> Iterator[tuple[str, Path, int, str]]:
        if not self.schedule:
            return
        active = sum(1 for h in self.handles.values() if not h.exhausted)
        while active > 0:
            name = self.schedule[self.tick % len(self.schedule)]
            self.tick += 1
            handle = self.handles.get(name)
            if handle is None or handle.exhausted:
                continue
            line = handle.stream.readline()
            if not line:
                handle.stream.close()
                handle.exhausted = True
                active -= 1
                continue
            handle.line_no += 1
            yield name, handle.path, handle.line_no, line


def merge_with_quality(raw_dir: Path, out_path: Path, *, quality_config: Optional[QualityPipelineConfig]=None, corpus_registry: Optional[CorpusRegistry]=None, write_buffer_bytes: int=DEFAULT_WRITE_BUFFER, source_filter: Optional[list[str]]=None, append: bool=False, work_dir: Optional[Path]=None, resume: bool=True, fresh: bool=False, checkpoint_interval: int | None=None, workers: Optional[int]=None, chunk_size: Optional[int]=None, time_limit_sec: Optional[float]=None, validation_collector: Optional[Any]=None, merge_lock_owner: Optional[str]=None, merge_lock_force: bool=False) -> dict[str, Any]:
    merge_work = Path(work_dir) if work_dir else Path(out_path).parent
    lock_owner = merge_lock_owner or f'merge:{Path(out_path).name}'
    from indw.schedule.state.lock import merge_run_lock
    with merge_run_lock(merge_work, owner=lock_owner, force=merge_lock_force):
        return _merge_with_quality_locked(
            raw_dir,
            out_path,
            quality_config=quality_config,
            corpus_registry=corpus_registry,
            write_buffer_bytes=write_buffer_bytes,
            source_filter=source_filter,
            append=append,
            work_dir=work_dir,
            resume=resume,
            fresh=fresh,
            checkpoint_interval=checkpoint_interval,
            workers=workers,
            chunk_size=chunk_size,
            time_limit_sec=time_limit_sec,
            validation_collector=validation_collector,
        )


def _merge_with_quality_locked(raw_dir: Path, out_path: Path, *, quality_config: Optional[QualityPipelineConfig]=None, corpus_registry: Optional[CorpusRegistry]=None, write_buffer_bytes: int=DEFAULT_WRITE_BUFFER, source_filter: Optional[list[str]]=None, append: bool=False, work_dir: Optional[Path]=None, resume: bool=True, fresh: bool=False, checkpoint_interval: int | None=None, workers: Optional[int]=None, chunk_size: Optional[int]=None, time_limit_sec: Optional[float]=None, validation_collector: Optional[Any]=None) -> dict[str, Any]:
    from indw.schedule.dispatch.parallel import merge_with_quality_parallel

    resolved_workers = resolve_merge_workers(workers)
    resolved_chunk = resolve_merge_chunk_size(chunk_size)
    logger.info(
        'Using canonical merge graph (%d workers, chunk=%d)',
        resolved_workers,
        resolved_chunk,
    )
    return merge_with_quality_parallel(
        raw_dir,
        out_path,
        quality_config=quality_config,
        corpus_registry=corpus_registry,
        write_buffer_bytes=write_buffer_bytes,
        source_filter=source_filter,
        append=append,
        work_dir=work_dir,
        resume=resume,
        fresh=fresh,
        workers=resolved_workers,
        chunk_size=resolved_chunk,
        checkpoint_interval=checkpoint_interval,
        time_limit_sec=time_limit_sec,
        validation_collector=validation_collector,
    )
