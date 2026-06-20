from __future__ import annotations
import logging
import random
from pathlib import Path
from typing import Iterator, Optional
from indw.store.export.export_items import ExportRecord
logger = logging.getLogger(__name__)

def _cycle_jsonl(path: Path) -> Iterator[str]:
    from indw.dedup.replay import iter_jsonl_text

    while True:
        empty = True
        for text in iter_jsonl_text(path):
            empty = False
            yield text
        if empty:
            return

def mixed_text_iterator(
    primary_jsonl: Path,
    replay_jsonl: Optional[Path],
    *,
    replay_ratio: float = 0.0,
    seed: int = 42,
) -> Iterator[ExportRecord]:
    from indw.dedup.normalize import content_hash
    from indw.dedup.replay import iter_jsonl_text

    primary_jsonl = Path(primary_jsonl)
    rng = random.Random(seed)
    replay_iter: Optional[Iterator[str]] = None
    if replay_jsonl and Path(replay_jsonl).exists() and (replay_ratio > 0):
        replay_iter = _cycle_jsonl(Path(replay_jsonl))
    for idx, text in enumerate(iter_jsonl_text(primary_jsonl)):
        if replay_iter is not None and rng.random() < replay_ratio:
            try:
                replay_text = next(replay_iter)
                if replay_text:
                    yield ExportRecord(text=replay_text, split_key='__replay__', is_replay=True)
                continue
            except StopIteration:
                replay_iter = None
        if text:
            yield ExportRecord(
                text=text,
                split_key=content_hash(text),
                is_replay=False,
            )
