from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from indw.tools.metrics.snapshot import CorpusSnapshot
from indw.tools.metrics.storage import load_snapshots

def _window(snaps: list[CorpusSnapshot], n: int) -> list[CorpusSnapshot]:
    return snaps[-n:] if n > 0 else snaps

def _series(snaps: list[CorpusSnapshot], field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in snaps:
        val = getattr(s, field, None)
        if val is None and field in s.to_dict():
            val = s.to_dict()[field]
        rows.append({'version': s.version, 'timestamp': s.timestamp, 'value': val})
    return rows

def build_trends(output_dir: Path) -> dict[str, Any]:
    snaps = load_snapshots(output_dir)
    return {
        'runs': len(snaps),
        '7_run': {
            'quality_score_mean': _series(_window(snaps, 7), 'quality_score_mean'),
            'duplicate_rate': _series(_window(snaps, 7), 'duplicate_rate'),
            'toxicity_rate': _series(_window(snaps, 7), 'toxicity_rate'),
            'pii_rate': _series(_window(snaps, 7), 'pii_rate'),
        },
        '30_run': {
            'quality_score_mean': _series(_window(snaps, 30), 'quality_score_mean'),
            'duplicate_rate': _series(_window(snaps, 30), 'duplicate_rate'),
            'toxicity_rate': _series(_window(snaps, 30), 'toxicity_rate'),
            'pii_rate': _series(_window(snaps, 30), 'pii_rate'),
        },
        'all_time': {
            'quality_score_mean': _series(snaps, 'quality_score_mean'),
            'duplicate_rate': _series(snaps, 'duplicate_rate'),
            'toxicity_rate': _series(snaps, 'toxicity_rate'),
            'pii_rate': _series(snaps, 'pii_rate'),
            'accepted_documents': _series(snaps, 'accepted_documents'),
        },
    }

def write_trend_histories(output_dir: Path, snapshots: list[CorpusSnapshot]) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _write(name: str, rows: list[dict[str, Any]]) -> None:
        path = output_dir / name
        existing: list[dict[str, Any]] = []
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(loaded, list):
                    existing = loaded
            except json.JSONDecodeError:
                existing = []
        merged = existing + rows
        if len(merged) > 200:
            merged = merged[-200:]
        path.write_text(json.dumps(merged, indent=2), encoding='utf-8')
        paths[name] = path

    for snap in snapshots[-1:]:
        base = {'created_at': snap.timestamp, 'version': snap.version}
        _write(
            'quality_history.json',
            [{**base, 'quality_score_mean': snap.quality_score_mean, 'accepted': snap.accepted_documents}],
        )
        _write(
            'language_history.json',
            [{**base, 'language_distribution': snap.language_distribution}],
        )
        _write(
            'toxicity_history.json',
            [{**base, 'toxicity_rate': snap.toxicity_rate}],
        )
        _write(
            'pii_history.json',
            [{**base, 'pii_rate': snap.pii_rate}],
        )
    return paths
