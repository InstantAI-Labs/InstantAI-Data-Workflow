from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

RECOVERY_LOG = 'recovery_events.jsonl'

def recovery_log_path(output_dir: Path) -> Path:
    return Path(output_dir) / RECOVERY_LOG

def record_recovery_event(
    output_dir: Path,
    event: str,
    **fields: Any,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'event': str(event),
        **fields,
    }
    path = recovery_log_path(output_dir)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, default=str) + '\n')
    return path

def load_jsonl_tail(path: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]

def load_recovery_events(output_dir: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    return load_jsonl_tail(recovery_log_path(output_dir), limit=limit)

def recovery_stats_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(e.get('event', '')) for e in events if e.get('event'))
    return {
        'total_events': len(events),
        'worker_crashes': int(counts.get('worker_failure_recovered', 0)),
        'sqlite_retries': int(counts.get('sqlite_retry', 0)),
        'disk_failures': int(counts.get('disk_full', 0)),
        'checkpoint_corrupt': int(counts.get('checkpoint_corrupt', 0)),
        'checkpoint_recoveries': int(counts.get('checkpoint_recovered', 0)),
        'export_failures': int(counts.get('export_partial_aborted', 0)),
        'stream_interruptions': int(counts.get('merge_interrupted', 0)),
        'by_type': dict(counts),
        'last_event': events[-1] if events else None,
    }

def load_recovery_stats(output_dir: Path, *, limit: int = 200) -> dict[str, Any]:
    return recovery_stats_from_events(load_recovery_events(output_dir, limit=limit))
