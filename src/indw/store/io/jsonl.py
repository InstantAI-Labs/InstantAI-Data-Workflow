from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from indw.store.io.json_codec import dumps, dumps_line, loads

SOURCE_META_NAME = 'data.meta.json'
_READ_CHUNK = 1024 * 1024


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open('rb') as fin:
        for line in fin:
            if line.strip():
                count += 1
    return count


def iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = loads(line)
            if isinstance(row, dict):
                yield row


def parse_jsonl_line(line: str) -> tuple[str, dict[str, Any] | None]:
    if not line.strip():
        return 'blank', None
    try:
        row = loads(line)
    except (ValueError, TypeError):
        return 'parse_error', None
    if not isinstance(row, dict):
        return 'parse_error', None
    return 'ok', row


def parse_jsonl_batch(lines: list[str]) -> list[tuple[str, dict[str, Any] | None]]:
    return [parse_jsonl_line(line) for line in lines]


def source_meta_path(jsonl_path: Path) -> Path:
    return jsonl_path.parent / SOURCE_META_NAME


def write_source_line_meta(jsonl_path: Path, *, line_count: int, bytes_written: int = 0) -> Path:
    meta_path = source_meta_path(jsonl_path)
    payload = {
        'path': jsonl_path.name,
        'line_count': int(line_count),
        'bytes': int(bytes_written or (jsonl_path.stat().st_size if jsonl_path.exists() else 0)),
    }
    meta_path.write_text(dumps(payload), encoding='utf-8')
    return meta_path


def resolve_jsonl_line_count(path: Path, *, recompute: bool = False) -> int:
    if not path.exists():
        return 0
    meta_path = source_meta_path(path)
    if not recompute and meta_path.exists():
        try:
            meta = loads(meta_path.read_bytes())
            if (
                isinstance(meta, dict)
                and str(meta.get('path', '')) == path.name
                and meta.get('line_count') is not None
            ):
                stored_bytes = int(meta.get('bytes', 0) or 0)
                actual_bytes = path.stat().st_size
                if stored_bytes <= 0 or abs(stored_bytes - actual_bytes) < max(4096, actual_bytes // 200):
                    return int(meta['line_count'])
        except (OSError, TypeError, ValueError):
            pass
    count = count_jsonl_lines(path)
    write_source_line_meta(path, line_count=count)
    return count


def resolve_raw_line_total(sources: list[Path]) -> int:
    return sum(resolve_jsonl_line_count(src) for src in sources)


def checkpoint_kept_lines(checkpoint: Any) -> int:
    return int(checkpoint.totals().get('kept', 0))
