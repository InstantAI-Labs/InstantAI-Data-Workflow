from __future__ import annotations

import hashlib
from pathlib import Path


def artifact_digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def pin_artifact(merge_work: Path | str, name: str, data: bytes) -> str:
    root = Path(merge_work) / 'artifacts'
    root.mkdir(parents=True, exist_ok=True)
    digest = artifact_digest(data)
    path = root / digest
    if not path.is_file():
        path.write_bytes(data)
    manifest = root / 'manifest.json'
    if manifest.is_file():
        import json
        entries = json.loads(manifest.read_text(encoding='utf-8'))
    else:
        entries = {}
    entries[name] = digest
    manifest.write_text(json.dumps(entries, indent=2), encoding='utf-8')
    return digest


def resolve_artifact(merge_work: Path | str, digest: str) -> bytes:
    path = Path(merge_work) / 'artifacts' / digest
    if not path.is_file():
        raise FileNotFoundError(f'artifact missing: {digest}')
    return path.read_bytes()
