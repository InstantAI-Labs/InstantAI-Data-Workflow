from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any


def _canonicalize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (tuple, list)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, dict):
        items = [[str(k), _canonicalize(v)] for k, v in obj.items()]
        items.sort(key=lambda x: x[0])
        return items
    return str(obj)


def stable_digest_hex(obj: Any, *, algo: str = 'sha256') -> str:
    canon = _canonicalize(obj)
    blob = json.dumps(canon, separators=(',', ':'), sort_keys=False, ensure_ascii=True).encode('utf-8')
    h = hashlib.new(algo)
    h.update(blob)
    return h.hexdigest()


def stable_digest_int(obj: Any, *, algo: str = 'sha256', bits: int = 128) -> int:
    if bits <= 0:
        raise ValueError('bits must be > 0')
    hex_digest = stable_digest_hex(obj, algo=algo)
    needed_hex = (bits + 3) // 4
    part = hex_digest[:needed_hex]
    return int(part, 16)
