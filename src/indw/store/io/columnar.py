from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def jsonl_to_parquet(
    jsonl_path: Path,
    parquet_path: Path,
    *,
    batch_size: int = 8192,
) -> bool:
    if not jsonl_path.exists():
        return False
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    batch: list[dict[str, Any]] = []
    writer: Any = None
    try:
        with jsonl_path.open(encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                batch.append(row)
                if len(batch) < batch_size:
                    continue
                table = pa.Table.from_pylist(batch)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table.schema, compression='zstd')
                writer.write_table(table)
                batch = []
        if batch:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                pq.write_table(table, parquet_path, compression='zstd')
            else:
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return parquet_path.exists()


def write_mixture_index_parquet(merge_work: Path) -> Path | None:
    index_path = merge_work / 'filtered.mixture_index.jsonl'
    if not index_path.exists():
        return None
    out = index_path.with_suffix('.parquet')
    if jsonl_to_parquet(index_path, out):
        logger.info('Wrote mixture analytics parquet → %s', out)
        return out
    return None
