from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from indw.config.defaults import DEFAULT_WRITE_BUFFER_BYTES, TOKEN_SHARD_VERSION

def tokenizer_fingerprint(tokenizer_path: str | Path) -> str:
    p = Path(tokenizer_path)
    blob = p.read_bytes()
    return hashlib.sha256(blob).hexdigest()[:16]

def _infer_tokenizer_id_from_path(tokenizer_path: Path) -> tuple[str, str] | None:
    parts = tokenizer_path.as_posix().split('/')
    try:
        i = parts.index('vocabularies')
    except ValueError:
        return None
    if i + 3 >= len(parts):
        return None
    name = parts[i + 1]
    version = parts[i + 2]
    if parts[i + 3] != 'tokenizer.json':
        return None
    return name, version

def shard_meta_path(bin_path: str | Path) -> Path:
    p = Path(bin_path)
    return p.with_suffix(p.suffix + '.meta.json')

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(DEFAULT_WRITE_BUFFER_BYTES), b''):
            h.update(chunk)
    return h.hexdigest()

@dataclass(frozen=True)
class TokenShardMeta:
    version: str
    dtype: str
    vocab_size: int
    eos_id: int
    tokenizer_path: str
    tokenizer_fingerprint: str
    tokenizer_name: str = ''
    tokenizer_version: str = ''
    checksum_sha256: str = ''
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'version': self.version,
            'dtype': self.dtype,
            'vocab_size': int(self.vocab_size),
            'eos_id': int(self.eos_id),
            'tokens': int(self.tokens),
            'tokenizer_path': self.tokenizer_path,
            'tokenizer_fingerprint': self.tokenizer_fingerprint,
            'tokenizer_name': self.tokenizer_name,
            'tokenizer_version': self.tokenizer_version,
            'checksum_sha256': self.checksum_sha256,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TokenShardMeta:
        return cls(
            version=str(raw.get('version', '')),
            dtype=str(raw.get('dtype', '')),
            vocab_size=int(raw.get('vocab_size', 0)),
            eos_id=int(raw.get('eos_id', -1)),
            tokens=int(raw.get('tokens', 0)),
            tokenizer_path=str(raw.get('tokenizer_path', '')),
            tokenizer_fingerprint=str(raw.get('tokenizer_fingerprint', '')),
            tokenizer_name=str(raw.get('tokenizer_name', '')),
            tokenizer_version=str(raw.get('tokenizer_version', '')),
            checksum_sha256=str(raw.get('checksum_sha256', '')),
        )

def write_shard_meta(
    *,
    bin_path: str | Path,
    tokenizer_path: str | Path,
    dtype: np.dtype,
    vocab_size: int,
    eos_id: int,
    tokens: int,
    checksum_sha256: str | None = None,
) -> Path:
    bp = Path(bin_path)
    tp = Path(tokenizer_path)
    fp = tokenizer_fingerprint(tp)
    name_ver = _infer_tokenizer_id_from_path(tp)
    name, ver = (name_ver if name_ver is not None else ('', ''))
    meta = TokenShardMeta(
        version=TOKEN_SHARD_VERSION,
        dtype=str(np.dtype(dtype).name),
        vocab_size=int(vocab_size),
        eos_id=int(eos_id),
        tokens=int(tokens),
        tokenizer_path=str(tp),
        tokenizer_fingerprint=str(fp),
        tokenizer_name=name,
        tokenizer_version=ver,
        checksum_sha256=checksum_sha256 or sha256_file(bp),
    )
    from indw.store.io.atomic import atomic_write_text

    out = shard_meta_path(bp)
    atomic_write_text(out, json.dumps(meta.to_dict(), indent=2))
    return out

def read_shard_meta(bin_path: str | Path) -> TokenShardMeta:
    mp = shard_meta_path(bin_path)
    if not mp.exists():
        raise FileNotFoundError(f'missing shard metadata: {mp} for {bin_path}')
    raw = json.loads(mp.read_text(encoding='utf-8'))
    return TokenShardMeta.from_dict(raw)

def shard_dtype_from_meta(meta: TokenShardMeta) -> np.dtype:
    dt = np.dtype(meta.dtype)
    if dt not in (np.uint16, np.uint32, np.int32, np.int64):
        raise ValueError(f'unsupported shard dtype: {meta.dtype}')
    return dt
