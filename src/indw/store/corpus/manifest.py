from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from indw.config.defaults import PREPROCESSING_VERSION

def corpus_build_id(
    *,
    sources_yaml_sha256: str,
    quality_config_fingerprint: str,
    filtered_sha256: str = '',
    preprocessing_version: str = PREPROCESSING_VERSION,
) -> str:
    payload = '|'.join([
        preprocessing_version,
        sources_yaml_sha256 or '',
        quality_config_fingerprint or '',
        filtered_sha256 or '',
    ])
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


from indw.store.export.shard_meta import sha256_file as file_sha256


@dataclass
class CorpusManifest:
    corpus_id: str
    version: int
    sources_yaml: str
    sources_yaml_sha256: str
    quality_config_fingerprint: str = ''
    filtered_sha256: str = ''
    filtered_line_count: int = 0
    mixture_plan_digest: str = ''
    corpus_build_id: str = ''
    dataset_manifest_path: str = ''
    preprocessing_version: str = PREPROCESSING_VERSION
    tokenizer_name: str = ''
    tokenizer_version: str = ''
    tokenizer_path: str = ''
    stats: dict[str, Any] = field(default_factory=dict)
    dedup: dict[str, Any] = field(default_factory=dict)
    shard_glob: str = ''
    replay_pool: str = ''
    parent_version: Optional[int] = None
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusManifest:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')
        return path

    @classmethod
    def load(cls, path: Path) -> CorpusManifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

def next_version(manifest_dir: Path) -> int:
    existing = sorted(manifest_dir.glob('manifest_v*.json'))
    if not existing:
        return 1
    last = existing[-1].stem
    try:
        return int(last.split('_v')[-1]) + 1
    except ValueError:
        return len(existing) + 1
