from __future__ import annotations

import hashlib
from typing import Any


def ingest_line_meta(
    *,
    line: str,
    src_name: str,
    line_no: int,
    seq: int,
) -> dict[str, Any]:
    raw = line.encode('utf-8', 'surrogatepass')
    return {
        'seq': seq,
        'source_id': src_name,
        'line_no': line_no,
        'document_id': f'{src_name}:{line_no}',
        'raw_bytes': len(raw),
        'line_checksum': hashlib.blake2b(raw, digest_size=16).hexdigest(),
    }


def enrich_ingest_meta(
    meta: dict[str, Any],
    *,
    row: dict[str, Any] | None,
    text: str,
) -> dict[str, Any]:
    out = dict(meta)
    if row:
        for key in ('title', 'url', 'mime', 'lang', 'source', 'id'):
            val = row.get(key)
            if val is not None and key not in out:
                out[key] = val
    out['meaningful_chars'] = len(text.strip())
    return out
