from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Optional
import torch
from torch.utils.data import DataLoader
from indw.store.export.config import DataPipelineConfig
from indw.store.export.memmap_stream import resolve_shards
from indw.store.export.packed_stream import _build_dataset
from indw.store.export.prefetch import wrap_prefetch
from training.packing.collate import collate_packed_batch, collate_varlen_batch
from training.packing import PackingConfig
from indw.store.export.shard_meta import read_shard_meta

def build_optimized_train_dataloader(
    data_glob: str,
    seq_len: int,
    batch_size: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    pipeline: Optional[DataPipelineConfig] = None,
    packing: Optional[PackingConfig] = None,
    device: Optional[torch.device] = None,
    infinite: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    cfg = pipeline or DataPipelineConfig()
    pack_cfg = packing or PackingConfig(enabled=True, mode='contiguous')
    paths = resolve_shards(data_glob)
    if paths:
        meta0 = read_shard_meta(paths[0])
        if cfg.expected_tokenizer_fingerprint:
            if meta0.tokenizer_fingerprint != cfg.expected_tokenizer_fingerprint:
                raise ValueError(
                    f'shard/tokenizer mismatch: expected={cfg.expected_tokenizer_fingerprint} shard={meta0.tokenizer_fingerprint} path={paths[0]}'
                )
    ds = _build_dataset(
        paths, seq_len, pack_cfg, rank, world_size, seed, pipeline_cfg=cfg, infinite=infinite
    )
    if hasattr(ds, '_packer'):
        ds._packer._reader = getattr(ds, '_reader', None)
    _attach_io_cache(ds, cfg)
    num_workers = cfg.resolve_num_workers()
    if sys.platform == 'win32' and num_workers > 0:
        num_workers = 0
    use_pin = cfg.pin_memory and (num_workers > 0 or cfg.host_prefetch)
    collate_fn = collate_varlen_batch if pack_cfg.mode == 'varlen_batch' else collate_packed_batch
    loader_kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = max(2, cfg.prefetch_factor)
        loader_kwargs['persistent_workers'] = cfg.persistent_workers
        if cfg.multiprocessing_context:
            import torch.multiprocessing as mp
            loader_kwargs['multiprocessing_context'] = mp.get_context(cfg.multiprocessing_context)
    if cfg.pin_memory and cfg.pin_memory_device and (num_workers > 0):
        loader_kwargs['pin_memory_device'] = cfg.pin_memory_device
    loader = DataLoader(ds, **loader_kwargs)
    use_host_prefetch = cfg.host_prefetch
    use_cuda_prefetch = cfg.cuda_prefetch and device is not None and getattr(device, 'type', '') == 'cuda'
    wrapped = wrap_prefetch(
        loader,
        device=device,
        host_prefetch=use_host_prefetch,
        host_queue=max(4, cfg.host_prefetch_queue),
        cuda_prefetch=use_cuda_prefetch,
        non_blocking=cfg.non_blocking_h2d
    )
    loader = wrapped
    loader.pipeline_config = cfg
    loader.pack_dataset = ds
    loader.packing_stats = lambda: getattr(ds, 'packing_stats', None)
    return loader

def _attach_io_cache(dataset: Any, cfg: DataPipelineConfig) -> None:
    from indw.store.export.shard_io import MemmapShardCache
    cache = MemmapShardCache(
        enabled=cfg.mmap_cache_shards,
        allow_legacy_uint16=cfg.allow_legacy_uint16_shards,
        verify_checksum=cfg.verify_shard_checksum,
    )
    if hasattr(dataset, '_shard_cache'):
        dataset._shard_cache = cache
    if hasattr(dataset, '_read_chunk'):
        dataset._read_chunk = cfg.read_chunk_tokens

def build_optimized_val_dataloader(
    data_glob: str,
    seq_len: int,
    batch_size: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    pipeline: Optional[DataPipelineConfig] = None,
    packing: Optional[PackingConfig] = None,
    device: Optional[torch.device] = None,
) -> DataLoader:
    return build_optimized_train_dataloader(
        data_glob,
        seq_len,
        batch_size,
        rank=rank,
        world_size=world_size,
        seed=0,
        pipeline=pipeline,
        packing=packing,
        device=device,
        infinite=False,
        drop_last=False,
    )
