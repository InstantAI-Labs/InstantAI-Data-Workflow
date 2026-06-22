from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import yaml

JsonMap = dict[str, Any]
Validator = Callable[[JsonMap], None]

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = _PACKAGE_ROOT / 'configs'


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def freeze(x: Any) -> Any:
    if isinstance(x, Mapping):
        return MappingProxyType({str(k): freeze(v) for k, v in x.items()})
    if isinstance(x, (list, tuple)):
        return tuple(freeze(v) for v in x)
    return x


def thaw(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): thaw(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return [thaw(v) for v in x]
    return x


@dataclass(frozen=True)
class ConfigRef:
    kind: str
    id: str
    version: str | None = None
    overlays: tuple['ConfigRef', ...] = ()

    def to_dict(self) -> JsonMap:
        return {
            'kind': self.kind,
            'id': self.id,
            'version': self.version,
            'overlays': [o.to_dict() for o in self.overlays],
        }


@dataclass(frozen=True)
class ResolvedConfig:
    kind: str
    raw: Mapping[str, Any]
    fingerprint: str
    trace: Mapping[str, Any]


def _normalize_spec_id(config_id: str) -> str:
    s = str(config_id or '').strip()
    if not s:
        raise ValueError('Empty config id')
    if s.startswith('cfg://'):
        s = s[len('cfg://'):].strip('/')
    elif s.startswith('cfg:'):
        s = s[len('cfg:'):].strip('/')
    if s.startswith('data/'):
        s = s[len('data/'):]
    return s


def _resolve_config_path(config_id: str) -> Path:
    s = str(config_id or '').strip()
    if not s:
        raise ValueError('Empty config id')
    if s.startswith('path:'):
        p = Path(s[len('path:'):].strip())
        return p if p.is_absolute() else (CONFIG_ROOT / p)
    if s.startswith('cfg://'):
        s = s[len('cfg://'):].strip('/')
    elif s.startswith('cfg:'):
        s = s[len('cfg:'):].strip('/')
    if s.startswith('data/'):
        s = s[len('data/'):]
    p = Path(s)
    if p.is_absolute() or (len(s) > 2 and s[1] == ':' and s[2] in '/\\'):
        return p
    if p.suffix.lower() in {'.yaml', '.yml', '.json'}:
        return CONFIG_ROOT / p
    for suffix in ('.yaml', '.yml', '.json'):
        cand = CONFIG_ROOT / f'{s}{suffix}'
        if cand.exists():
            return cand
    raise FileNotFoundError(f'Unknown config id: {config_id}')


def _json_canonical(raw: JsonMap) -> str:
    return json.dumps(raw, sort_keys=True, separators=(',', ':'))


def _config_fingerprint(raw: JsonMap) -> str:
    return sha256(_json_canonical(raw).encode('utf-8')).hexdigest()


def _compose_path(path: Path, *, seen: set[str] | None = None) -> JsonMap:
    seen = seen or set()
    raw = load_yaml(path) if path.suffix.lower() != '.json' else json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raw = {}
    cfg_name = str(path)
    if cfg_name in seen:
        raise ValueError(f'Config inheritance cycle detected at {cfg_name}')
    seen.add(cfg_name)
    parents = raw.get('extends')
    if not parents:
        return dict(raw)
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise TypeError('extends must be str or list[str]')
    merged: JsonMap = {}
    for parent in parents:
        pp = _resolve_config_path(str(parent))
        pdata = _compose_path(pp, seen=seen)
        merged = merge_dicts(merged, pdata)
    child = dict(raw)
    child.pop('extends', None)
    return merge_dicts(merged, child)


@dataclass(frozen=True)
class SchemaRegistry:
    _validators: dict[str, Validator] = field(default_factory=dict)

    def register(self, kind: str, fn: Validator) -> SchemaRegistry:
        kind_s = str(kind)
        if not kind_s:
            raise ValueError('schema kind required')
        if kind_s in self._validators:
            raise KeyError(f'schema validator already registered: {kind_s}')
        return SchemaRegistry(_validators={**self._validators, kind_s: fn})

    def validate(self, kind: str, raw: JsonMap) -> None:
        kind_s = str(kind)
        if kind_s not in self._validators:
            raise KeyError(f'No schema validator for kind={kind_s}')
        self._validators[kind_s](raw)


def _validate_dict(raw: JsonMap) -> None:
    if not isinstance(raw, dict):
        raise TypeError('config must be dict')


@dataclass(frozen=True)
class Resolver:
    schemas: SchemaRegistry

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> Resolver:
        kinds = (
            'quality', 'pipeline', 'refining', 'safety', 'language',
            'observability', 'corpus', 'language_manifest', 'curriculum_manifest',
            'dataset_sources',
        )
        reg = SchemaRegistry()
        for kind in kinds:
            reg = reg.register(kind, _validate_dict)
        return cls(schemas=reg)

    def resolve(self, ref: ConfigRef) -> ResolvedConfig:
        base_raw = self._load_ref(ref)
        merged = dict(base_raw)
        overlays: list[ConfigRef] = []
        composed_ids: list[str] = [ref.id]
        for overlay in ref.overlays:
            merged = merge_dicts(merged, self._load_ref(overlay))
            overlays.append(overlay)
            composed_ids.append(overlay.id)
        self.schemas.validate(ref.kind, merged)
        return ResolvedConfig(
            kind=ref.kind,
            raw=freeze(merged),
            fingerprint=_config_fingerprint(merged),
            trace={
                'base': ref.to_dict(),
                'overlays': [o.to_dict() for o in overlays],
                'composed_ids': composed_ids,
            },
        )

    def _load_ref(self, ref: ConfigRef) -> JsonMap:
        ref_id = str(ref.id)
        if ref_id.startswith(('http://', 'https://', 's3://', 'gs://', 'az://', 'artifact://')):
            raise NotImplementedError(f'remote config refs are not supported: {ref_id}')
        return _compose_path(_resolve_config_path(ref_id))
