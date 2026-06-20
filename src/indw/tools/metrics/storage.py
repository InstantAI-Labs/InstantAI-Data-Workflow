from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from indw.tools.metrics.snapshot import CorpusSnapshot

SNAPSHOTS_FILE = 'corpus_snapshots.json'

def _snapshots_path(output_dir: Path) -> Path:
    return output_dir / SNAPSHOTS_FILE

def load_snapshots(output_dir: Path) -> list[CorpusSnapshot]:
    path = _snapshots_path(output_dir)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    out: list[CorpusSnapshot] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        out.append(
            CorpusSnapshot(
                version=str(row.get('version', 'v0')),
                timestamp=str(row.get('timestamp', '')),
                total_documents=int(row.get('total_documents', row.get('documents', 0))),
                accepted_documents=int(row.get('accepted_documents', row.get('documents', 0))),
                rejected_documents=int(row.get('rejected_documents', 0)),
                duplicate_rate=float(row.get('duplicate_rate', 0.0)),
                quality_score_mean=float(row.get('quality_score_mean', 0.0)),
                quality_score_p10=float(row.get('quality_score_p10', 0.0)),
                quality_score_distribution=dict(row.get('quality_score_distribution') or {}),
                toxicity_rate=float(row.get('toxicity_rate', 0.0)),
                pii_rate=float(row.get('pii_rate', 0.0)),
                language_distribution=dict(row.get('language_distribution') or {}),
                average_document_length=float(row.get('average_document_length', 0.0)),
                source_distribution=dict(row.get('source_distribution') or {}),
                reject_reasons=dict(row.get('reject_reasons') or {}),
                metadata=dict(row.get('metadata') or {}),
            )
        )
    return out

def append_snapshot(output_dir: Path, snapshot: CorpusSnapshot) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = load_snapshots(output_dir)
    history.append(snapshot)
    path = _snapshots_path(output_dir)
    rows = [s.to_dict() for s in history]
    path.write_text(json.dumps(rows, separators=(',', ':')), encoding='utf-8')
    return path

def previous_snapshot(output_dir: Path) -> Optional[CorpusSnapshot]:
    snaps = load_snapshots(output_dir)
    if len(snaps) < 2:
        return snaps[0] if len(snaps) == 1 else None
    return snaps[-2]

def latest_snapshot(output_dir: Path) -> Optional[CorpusSnapshot]:
    snaps = load_snapshots(output_dir)
    return snaps[-1] if snaps else None

def next_version(output_dir: Path) -> str:
    return f'v{len(load_snapshots(output_dir)) + 1}'
