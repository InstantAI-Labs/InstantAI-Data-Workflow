from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from indw.store.export.config import DEFAULT_DATALOADER_NUM_WORKERS, DEFAULT_DATALOADER_NUM_WORKERS_LIGHT
from indw.store.export.shard_io import tokens_to_tensors
from indw.store.export.shard_meta import read_shard_meta, sha256_file, shard_dtype_from_meta

class TokenShardDataset(IterableDataset):

    def __init__(
        self,
        data_paths: list[str | Path],
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        infinite: bool = True,
        *,
        allow_legacy_uint16: bool = False,
    ):
        self.data_paths = [Path(p) for p in data_paths]
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.infinite = infinite
        self.allow_legacy_uint16 = allow_legacy_uint16
        self._epoch = 0
        self._validated: set[str] = set()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _shard_paths(self) -> list[Path]:
        return self.data_paths[self.rank::self.world_size]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        paths = self._shard_paths()
        if worker is not None:
            paths = paths[worker.id::worker.num_workers]
        rng = np.random.default_rng(self.seed + self.rank + self._epoch * 9973)
        while True:
            perm = rng.permutation(len(paths))
            for path_idx in perm:
                path = paths[int(path_idx)]
                if not path.exists():
                    continue
                dt = None
                try:
                    meta = read_shard_meta(path)
                    if meta.vocab_size <= 0:
                        raise ValueError(f'invalid shard metadata vocab_size for {path}: {meta.vocab_size}')
                    if meta.eos_id < 0 or meta.eos_id >= meta.vocab_size:
                        raise ValueError(
                            f'invalid shard metadata eos_id for {path}: eos={meta.eos_id} vocab={meta.vocab_size}'
                        )
                    k = str(path.resolve())
                    if k not in self._validated:
                        expected = str(meta.checksum_sha256 or '')
                        if not expected:
                            raise ValueError(f'missing shard checksum in metadata: {path}')
                        got = sha256_file(path)
                        if got != expected:
                            raise ValueError(
                                f'shard checksum mismatch for {path}: expected={expected} got={got}'
                            )
                        self._validated.add(k)
                    dt = shard_dtype_from_meta(meta)
                except FileNotFoundError:
                    if not self.allow_legacy_uint16:
                        raise
                    dt = np.uint16
                data = np.memmap(path, dtype=dt, mode='r')
                n = len(data) - self.seq_len
                if n <= 0:
                    continue
                indices = np.arange(0, n, self.seq_len)
                rng.shuffle(indices)
                for start in indices:
                    chunk = data[start:start + self.seq_len + 1]
                    yield tokens_to_tensors(np.asarray(chunk, dtype=np.int64))
            if not self.infinite:
                break

def resolve_shards(data_glob: str) -> list[Path]:
    if ',' in data_glob:
        paths: list[Path] = []
        for part in data_glob.split(','):
            part = part.strip()
            if part:
                paths.extend(resolve_shards(part))
        return sorted(set(paths), key=lambda p: p.name)
    path = Path(data_glob)
    if '*' in data_glob:
        root = path.parent
        pattern = path.name
        paths = sorted(root.glob(pattern))
    else:
        paths = sorted(path.glob('*.bin')) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(
            f'No shards match {data_glob}\nRun indw merge to prepare corpus output first.'
        )
    return paths

def build_pretrain_dataloader(
    data_glob: str,
    seq_len: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = DEFAULT_DATALOADER_NUM_WORKERS,
    seed: int = 42,
) -> DataLoader:
    paths = resolve_shards(data_glob)
    ds = TokenShardDataset(
        paths,
        seq_len,
        rank=rank,
        world_size=world_size,
        seed=seed,
        infinite=True,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

def build_val_dataloader(
    data_glob: str,
    seq_len: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = DEFAULT_DATALOADER_NUM_WORKERS_LIGHT,
    *,
    pipeline: Optional[Any] = None,
    packing: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> DataLoader:
    from indw.store.export.pipeline import build_optimized_val_dataloader

    return build_optimized_val_dataloader(
        data_glob,
        seq_len,
        batch_size,
        rank=rank,
        world_size=world_size,
        pipeline=pipeline,
        packing=packing,
        device=device,
    )
