from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from indw.store.export.shard_meta import read_shard_meta, sha256_file, shard_dtype_from_meta

def _slice_as_int64(data: np.ndarray, start: int, end: int) -> np.ndarray:
    sl = data[start:end]
    if sl.size == 0:
        return np.empty(0, dtype=np.int64)
    if sl.dtype == np.int64:
        return sl
    return sl.astype(np.int64, copy=False)

class MemmapShardCache:

    def __init__(
        self,
        *,
        enabled: bool = True,
        dtype: type = np.uint32,
        allow_legacy_uint16: bool = False,
        verify_checksum: bool = True,
    ):
        self.enabled = enabled
        self.dtype = dtype
        self.allow_legacy_uint16 = allow_legacy_uint16
        self.verify_checksum = verify_checksum
        self._handles: dict[tuple[str, str], np.memmap] = {}
        self._verified: set[str] = set()

    def _validate_meta(self, path: str | Path) -> np.dtype:
        meta = read_shard_meta(path)
        if meta.vocab_size <= 0:
            raise ValueError(f'invalid shard metadata vocab_size for {path}: {meta.vocab_size}')
        if meta.eos_id < 0 or meta.eos_id >= meta.vocab_size:
            raise ValueError(
                f'invalid shard metadata eos_id for {path}: eos={meta.eos_id} vocab={meta.vocab_size}'
            )
        dt = shard_dtype_from_meta(meta)
        p = str(Path(path).resolve())
        if self.verify_checksum and p not in self._verified:
            got = sha256_file(path)
            exp = str(meta.checksum_sha256 or '')
            if not exp:
                raise ValueError(f'missing shard checksum in metadata: {path}')
            if got != exp:
                raise ValueError(f'shard checksum mismatch for {path}: expected={exp} got={got}')
            self._verified.add(p)
        return dt

    def _dtype_for_path(self, path: str | Path) -> np.dtype:
        try:
            return self._validate_meta(path)
        except FileNotFoundError:
            if self.allow_legacy_uint16:
                return np.dtype(np.uint16)
            raise

    def open(self, path: str | Path) -> np.memmap:
        p = Path(path).resolve()
        dt = self._dtype_for_path(p)
        key = (str(p), str(dt.name))
        if not self.enabled:
            return np.memmap(str(p), dtype=dt, mode='r')
        if key not in self._handles:
            self._handles[key] = np.memmap(str(p), dtype=dt, mode='r')
        return self._handles[key]

    def clear(self) -> None:
        for mm in list(self._handles.values()):
            try:
                mm._mmap.close()
            except AttributeError:
                pass
        self._handles.clear()

    def read_range(self, path: str | Path, start: int, length: int) -> np.ndarray:
        data = self.open(path)
        end = min(start + length, len(data))
        if start >= end:
            return np.empty(0, dtype=np.int64)
        return _slice_as_int64(data, start, end)

    def read_into_buffer(self, path: str | Path, offset: int, out: np.ndarray) -> int:
        data = self.open(path)
        n = min(len(out), len(data) - offset)
        if n <= 0:
            return 0
        sl = _slice_as_int64(data, offset, offset + n)
        out[:n] = sl
        return n

    def shard_length(self, path: str | Path) -> int:
        return len(self.open(path))

def tokens_to_tensors(token_ids: np.ndarray, labels_offset: int = 1) -> dict[str, torch.Tensor]:
    if len(token_ids) < 2:
        raise ValueError('need at least 2 tokens')
    if token_ids.dtype != np.int64:
        token_ids = token_ids.astype(np.int64, copy=False)
    x_np = token_ids[:-labels_offset]
    y_np = token_ids[labels_offset:]
    return {
        'input_ids': torch.from_numpy(x_np),
        'labels': torch.from_numpy(y_np),
    }

def fast_chunk_tensors(mmap_arr: np.memmap, start: int, seq_len: int) -> dict[str, torch.Tensor]:
    end = start + seq_len + 1
    chunk = _slice_as_int64(mmap_arr, start, end)
    return tokens_to_tensors(chunk)
