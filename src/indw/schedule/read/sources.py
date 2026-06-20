from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import yaml


def iter_resolved_sources(raw_dir: Path) -> Iterator[dict[str, Any]]:
    for path in (raw_dir.parent / '_resolved_sources.yaml', raw_dir / '_resolved_sources.yaml'):
        if not path.exists():
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(raw, dict):
            yield raw


def load_mix_weights(raw_dir: Path, source_names: list[str]) -> dict[str, int]:
    for raw in iter_resolved_sources(raw_dir):
        weights: dict[str, int] = {}
        for entry in raw.get('sources') or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get('name') or '').strip()
            if name in source_names:
                weights[name] = max(1, int(entry.get('mix_pct') or 1))
        if weights:
            return weights
    return {name: 1 for name in source_names}


def corpus_has_multilingual_sources(source_names: list[str]) -> bool:
    for name in source_names:
        low = name.lower()
        for tag in ('-hi', '-ar', '-de', '-es', '-fr', '-zh', '-ja', '-ko', '-pt', '-ru', '-tr', '-vi', '-id'):
            if tag in low:
                return True
    return False


def load_source_registry(raw_dir: Path) -> dict[str, dict[str, Any]]:
    from indw.filter.license.source_policy import source_meta_from_yaml_entry

    registry: dict[str, dict[str, Any]] = {}
    for raw in iter_resolved_sources(raw_dir):
        for entry in raw.get('sources') or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get('name') or '').strip()
            if name:
                registry[name] = source_meta_from_yaml_entry(entry)
        if registry:
            return registry
    return registry
