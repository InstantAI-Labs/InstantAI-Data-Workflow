from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_PREFETCH_FACTOR = 4
DEFAULT_READ_CHUNK_TOKENS = 262144
DEFAULT_HOST_PREFETCH_QUEUE = 2
DEFAULT_TRAIN_MAX_WORKERS = 8
DEFAULT_TRAIN_MIN_WORKERS = 2
DEFAULT_DATALOADER_NUM_WORKERS = 4
DEFAULT_DATALOADER_NUM_WORKERS_LIGHT = 2

@dataclass
class DataPipelineConfig:
    num_workers: Optional[int] = None
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR
    persistent_workers: bool = True
    pin_memory: bool = True
    pin_memory_device: Optional[str] = None
    non_blocking_h2d: bool = True
    mmap_cache_shards: bool = True
    read_chunk_tokens: int = DEFAULT_READ_CHUNK_TOKENS
    background_preprocess: bool = True
    host_prefetch: bool = True
    host_prefetch_queue: int = DEFAULT_HOST_PREFETCH_QUEUE
    cuda_prefetch: bool = True
    auto_tune_workers: bool = True
    max_workers: int = DEFAULT_TRAIN_MAX_WORKERS
    min_workers: int = DEFAULT_TRAIN_MIN_WORKERS
    multiprocessing_context: Optional[str] = None
    allow_legacy_uint16_shards: bool = False
    expected_tokenizer_fingerprint: Optional[str] = None
    verify_shard_checksum: bool = True

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> DataPipelineConfig:
        if not raw:
            return cls()
        return cls(
            num_workers=raw.get('num_workers'),
            prefetch_factor=int(raw.get('prefetch_factor', DEFAULT_PREFETCH_FACTOR)),
            persistent_workers=bool(raw.get('persistent_workers', True)),
            pin_memory=bool(raw.get('pin_memory', True)),
            pin_memory_device=raw.get('pin_memory_device'),
            non_blocking_h2d=bool(raw.get('non_blocking_h2d', True)),
            mmap_cache_shards=bool(raw.get('mmap_cache_shards', True)),
            read_chunk_tokens=int(raw.get('read_chunk_tokens', DEFAULT_READ_CHUNK_TOKENS)),
            background_preprocess=bool(raw.get('background_preprocess', True)),
            host_prefetch=bool(raw.get('host_prefetch', True)),
            host_prefetch_queue=int(raw.get('host_prefetch_queue', DEFAULT_HOST_PREFETCH_QUEUE)),
            cuda_prefetch=bool(raw.get('cuda_prefetch', True)),
            auto_tune_workers=bool(raw.get('auto_tune_workers', True)),
            max_workers=int(raw.get('max_workers', DEFAULT_TRAIN_MAX_WORKERS)),
            min_workers=int(raw.get('min_workers', DEFAULT_TRAIN_MIN_WORKERS)),
            multiprocessing_context=raw.get('multiprocessing_context'),
            allow_legacy_uint16_shards=bool(raw.get('allow_legacy_uint16_shards', False)),
            expected_tokenizer_fingerprint=raw.get('expected_tokenizer_fingerprint'),
            verify_shard_checksum=bool(raw.get('verify_shard_checksum', True)),
        )

    def resolve_num_workers(self, explicit: Optional[int]=None) -> int:
        if explicit is not None and explicit > 0:
            return explicit
        if self.num_workers is not None and self.num_workers > 0:
            return self.num_workers
        if self.auto_tune_workers:
            return suggest_num_workers(max_workers=self.max_workers, min_workers=self.min_workers)
        return 0

def suggest_num_workers(*, cpu_count: Optional[int]=None, max_workers: int=DEFAULT_TRAIN_MAX_WORKERS, min_workers: int=DEFAULT_TRAIN_MIN_WORKERS) -> int:
    cpus = cpu_count or os.cpu_count() or 4
    return max(min_workers, min(max_workers, cpus // 2))
