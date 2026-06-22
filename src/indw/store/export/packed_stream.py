from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from indw.store.export.config import (
    DEFAULT_DATALOADER_NUM_WORKERS,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_READ_CHUNK_TOKENS,
)
from indw.store.export.memmap_stream import resolve_shards
from indw.store.export.shard_io import MemmapShardCache, fast_chunk_tensors, tokens_to_tensors
from indw.store.export.packing.binpack import DynamicSequencePacker, PackingStats, pack_density
from indw.store.export.packing import PackingConfig

class MultiSamplePackedDataset(IterableDataset):

    def __init__(self, data_paths: list[str | Path], seq_len: int, pack_cfg: PackingConfig, rank: int=0, world_size: int=1, seed: int=42, infinite: bool=True, shard_cache: Optional[MemmapShardCache]=None, read_chunk_tokens: int=DEFAULT_READ_CHUNK_TOKENS):
        self.data_paths = [Path(p) for p in data_paths]
        self.seq_len = seq_len
        self.pack_cfg = pack_cfg
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.infinite = infinite
        self._epoch = 0
        self._shard_cache = shard_cache or MemmapShardCache(enabled=True)
        self._read_chunk = read_chunk_tokens
        self._packer = DynamicSequencePacker(seq_len, pack_cfg)
        self._stage_policy: dict[str, Any] = {}

    @property
    def packing_stats(self) -> PackingStats:
        return self._packer.stats

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict[str, Any]:
        return {'epoch': int(self._epoch), 'seed': int(self.seed), 'stage_policy': dict(self._stage_policy)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        d = dict(state or {})
        self._epoch = int(d.get('epoch', self._epoch))
        self.seed = int(d.get('seed', self.seed))
        self._stage_policy = dict(d.get('stage_policy') or {})

    def apply_stage_policy(self, *, stage_ref: str, dataset_overrides: dict[str, Any], curriculum_policy: dict[str, Any]) -> None:
        self._stage_policy = {'stage_ref': str(stage_ref), 'dataset_overrides': dict(dataset_overrides), 'curriculum_policy': dict(curriculum_policy)}
        if dataset_overrides.get('seed') is not None:
            self.seed = int(dataset_overrides['seed'])
        if dataset_overrides.get('read_chunk_tokens') is not None:
            self._read_chunk = int(dataset_overrides['read_chunk_tokens'])

    def _shard_paths(self) -> list[Path]:
        return self.data_paths[self.rank::self.world_size]

    def _feed_path(self, path: Path) -> None:
        length = self._shard_cache.shard_length(path)
        offset = 0
        while offset < length:
            n = min(self._read_chunk, length - offset)
            arr = self._shard_cache.read_range(path, offset, n)
            if len(arr) == 0:
                break
            self._packer.feed_tokens(arr)
            offset += n

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        paths = self._shard_paths()
        if worker is not None:
            paths = paths[worker.id::worker.num_workers]
        rng = np.random.default_rng(self.seed + self.rank + self._epoch * 9973)
        while True:
            order = paths.copy()
            rng.shuffle(order)
            for path in order:
                if not path.exists():
                    continue
                if self._shard_cache.shard_length(path) < self.pack_cfg.min_document_tokens:
                    continue
                self._feed_path(path)
                while True:
                    packed = self._packer.pack_one()
                    if packed is None:
                        break
                    x_np = np.asarray(packed.input_ids, dtype=np.int64)
                    y_np = np.asarray(packed.labels, dtype=np.int64)
                    tensors = {'input_ids': torch.from_numpy(x_np), 'labels': torch.from_numpy(y_np)}
                    tensors['segment_lengths'] = packed.segment_lengths
                    tensors['packing_efficiency'] = pack_density(packed)
                    tensors['documents_packed'] = packed.documents_packed
                    yield tensors
            if not self.infinite:
                break

class ContiguousPackedDataset(IterableDataset):

    def __init__(self, data_paths: list[str | Path], seq_len: int, rank: int=0, world_size: int=1, seed: int=42, infinite: bool=True, shard_cache: Optional[MemmapShardCache]=None, read_chunk_tokens: int=DEFAULT_READ_CHUNK_TOKENS):
        self.data_paths = [Path(p) for p in data_paths]
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.infinite = infinite
        self._epoch = 0
        self._stats = PackingStats()
        self._shard_cache = shard_cache or MemmapShardCache(enabled=True)
        self._read_chunk = read_chunk_tokens
        self._stage_policy: dict[str, Any] = {}

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict[str, Any]:
        return {'epoch': int(self._epoch), 'seed': int(self.seed), 'stage_policy': dict(self._stage_policy)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        d = dict(state or {})
        self._epoch = int(d.get('epoch', self._epoch))
        self.seed = int(d.get('seed', self.seed))
        self._stage_policy = dict(d.get('stage_policy') or {})

    def apply_stage_policy(self, *, stage_ref: str, dataset_overrides: dict[str, Any], curriculum_policy: dict[str, Any]) -> None:
        self._stage_policy = {'stage_ref': str(stage_ref), 'dataset_overrides': dict(dataset_overrides), 'curriculum_policy': dict(curriculum_policy)}
        if dataset_overrides.get('seed') is not None:
            self.seed = int(dataset_overrides['seed'])
        if dataset_overrides.get('read_chunk_tokens') is not None:
            self._read_chunk = int(dataset_overrides['read_chunk_tokens'])

    @property
    def packing_stats(self) -> PackingStats:
        return self._stats

    def _shard_paths(self) -> list[Path]:
        return self.data_paths[self.rank::self.world_size]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        paths = self._shard_paths()
        if worker is not None:
            paths = paths[worker.id::worker.num_workers]
        rng = np.random.default_rng(self.seed + self.rank + self._epoch * 9973)
        need = self.seq_len + 1
        carry = np.empty(need * 2, dtype=np.int64)
        carry_len = 0
        while True:
            order = paths.copy()
            rng.shuffle(order)
            for path in order:
                if not path.exists():
                    continue
                length = self._shard_cache.shard_length(path)
                offset = 0
                while offset < length:
                    n = min(self._read_chunk, length - offset)
                    block = self._shard_cache.read_range(path, offset, n)
                    offset += n
                    blen = len(block)
                    if blen == 0:
                        continue
                    if carry_len + blen > len(carry):
                        new_carry = np.empty(max(len(carry) * 2, carry_len + blen + need), dtype=np.int64)
                        new_carry[:carry_len] = carry[:carry_len]
                        carry = new_carry
                    carry[carry_len:carry_len + blen] = block
                    carry_len += blen
                    while carry_len >= need:
                        chunk = carry[:need]
                        carry[:carry_len - need + 1] = carry[need - 1:carry_len]
                        carry_len -= need - 1
                        out = tokens_to_tensors(chunk)
                        self._stats.record_stride_window(self.seq_len)
                        out['segment_lengths'] = [self.seq_len]
                        out['packing_efficiency'] = 1.0
                        out['documents_packed'] = 1
                        yield out
            if not self.infinite:
                break

class TokenShardDatasetStride(IterableDataset):

    def __init__(self, data_paths: list[str | Path], seq_len: int, rank: int=0, world_size: int=1, seed: int=42, infinite: bool=True, shard_cache: Optional[MemmapShardCache]=None):
        self.data_paths = [Path(p) for p in data_paths]
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.infinite = infinite
        self._epoch = 0
        self._stats = PackingStats()
        self._shard_cache = shard_cache or MemmapShardCache(enabled=True)
        self._stage_policy: dict[str, Any] = {}

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict[str, Any]:
        return {'epoch': int(self._epoch), 'seed': int(self.seed), 'stage_policy': dict(self._stage_policy)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        d = dict(state or {})
        self._epoch = int(d.get('epoch', self._epoch))
        self.seed = int(d.get('seed', self.seed))
        self._stage_policy = dict(d.get('stage_policy') or {})

    def apply_stage_policy(self, *, stage_ref: str, dataset_overrides: dict[str, Any], curriculum_policy: dict[str, Any]) -> None:
        self._stage_policy = {'stage_ref': str(stage_ref), 'dataset_overrides': dict(dataset_overrides), 'curriculum_policy': dict(curriculum_policy)}
        if dataset_overrides.get('seed') is not None:
            self.seed = int(dataset_overrides['seed'])

    @property
    def packing_stats(self) -> PackingStats:
        return self._stats

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        paths = self.data_paths[self.rank::self.world_size]
        if worker is not None:
            paths = paths[worker.id::worker.num_workers]
        rng = np.random.default_rng(self.seed + self.rank + self._epoch * 9973)
        while True:
            order = paths.copy()
            rng.shuffle(order)
            for path in order:
                if not path.exists():
                    continue
                data = self._shard_cache.open(path)
                n = len(data) - self.seq_len
                if n <= 0:
                    continue
                for start in np.arange(0, n, self.seq_len):
                    out = fast_chunk_tensors(data, int(start), self.seq_len)
                    self._stats.record_stride_window(self.seq_len)
                    out['segment_lengths'] = [self.seq_len]
                    out['packing_efficiency'] = self._stats.eff()
                    out['documents_packed'] = 1
                    yield out
            if not self.infinite:
                break

def _build_dataset(
    paths: list[Path],
    seq_len: int,
    pack_cfg: PackingConfig,
    rank: int,
    world_size: int,
    seed: int,
    pipeline_cfg: Optional[Any] = None,
    *,
    infinite: bool = True,
) -> IterableDataset:
    cache = None
    read_chunk = DEFAULT_READ_CHUNK_TOKENS
    if pipeline_cfg is not None:
        from indw.store.export.config import DataPipelineConfig
        if isinstance(pipeline_cfg, DataPipelineConfig):
            from indw.store.export.shard_io import MemmapShardCache
            cache = MemmapShardCache(
                enabled=pipeline_cfg.mmap_cache_shards,
                allow_legacy_uint16=pipeline_cfg.allow_legacy_uint16_shards,
                verify_checksum=pipeline_cfg.verify_shard_checksum,
            )
            read_chunk = pipeline_cfg.read_chunk_tokens
    if not pack_cfg.enabled or pack_cfg.mode == 'stride':
        return TokenShardDatasetStride(
            paths, seq_len, rank, world_size, seed, infinite=infinite, shard_cache=cache
        )
    if pack_cfg.mode == 'contiguous':
        return ContiguousPackedDataset(
            paths,
            seq_len,
            rank,
            world_size,
            seed,
            infinite=infinite,
            shard_cache=cache,
            read_chunk_tokens=read_chunk,
        )
    return MultiSamplePackedDataset(
        paths,
        seq_len,
        pack_cfg,
        rank,
        world_size,
        seed,
        infinite=infinite,
        shard_cache=cache,
        read_chunk_tokens=read_chunk,
    )

def build_packed_dataloader(data_glob: str, seq_len: int, batch_size: int, *, rank: int=0, world_size: int=1, num_workers: int=DEFAULT_DATALOADER_NUM_WORKERS, seed: int=42, sequence_packing: bool=True, packing: Optional[PackingConfig]=None, delimiter_token: Optional[int]=None, prefetch_factor: int=DEFAULT_PREFETCH_FACTOR, persistent_workers: bool=True, pin_memory: bool=True, pipeline: Optional[Any]=None, device: Optional[torch.device]=None) -> DataLoader:
    from indw.store.export.config import DataPipelineConfig
    from indw.store.export.pipeline import build_optimized_train_dataloader
    pack_cfg = packing or PackingConfig(enabled=sequence_packing, mode='multi_sample' if sequence_packing else 'stride', delimiter_token=delimiter_token)
    if delimiter_token is not None and pack_cfg.delimiter_token is None:
        from dataclasses import replace
        pack_cfg = replace(pack_cfg, delimiter_token=delimiter_token)
    pipe = pipeline
    if pipe is None:
        pipe = DataPipelineConfig(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )
    elif isinstance(pipe, DataPipelineConfig):
        if num_workers > 0 and pipe.num_workers is None:
            from dataclasses import replace
            pipe = replace(pipe, num_workers=num_workers)
    return build_optimized_train_dataloader(
        data_glob,
        seq_len,
        batch_size,
        rank=rank,
        world_size=world_size,
        seed=seed,
        pipeline=pipe,
        packing=pack_cfg,
        device=device
    )
