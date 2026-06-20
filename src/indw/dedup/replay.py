from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from indw.dedup.fuzzy import StreamingFuzzyDedup
    from indw.dedup.semantic import StreamingSemanticDedup

logger = logging.getLogger(__name__)


def iter_jsonl_text(path: Path) -> Iterator[str]:
    with path.open(encoding='utf-8') as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get('text', '')
            if isinstance(text, str) and text.strip():
                yield text


def restore_dedup_from_jsonl(
    path: Path,
    *,
    exact: Any | None = None,
    fuzzy: StreamingFuzzyDedup | None = None,
    semantic: StreamingSemanticDedup | None = None,
    embed_semantic: Any | None = None,
    skip_lines: int = 0,
    near_layers_full_replay: bool = False,
) -> int:
    if not path.exists():
        return 0
    count = 0
    skipped = 0
    for text in iter_jsonl_text(path):
        if not near_layers_full_replay and skipped < skip_lines:
            skipped += 1
            continue
        if exact is not None:
            if near_layers_full_replay and skip_lines > 0:
                exact.seed_text_if_missing(text, source='resume')
            else:
                exact.seed_text(text, source='resume')
        if fuzzy is not None:
            fuzzy.register(text)
        if semantic is not None:
            semantic.register(text)
        if embed_semantic is not None and embed_semantic.enabled:
            embed_semantic.register(text)
        count += 1
    if count:
        logger.info(
            'Restored dedup state from %d documents in %s (skipped %d already indexed)',
            count,
            path.name,
            skip_lines if not near_layers_full_replay else 0,
        )
    elif skip_lines and not near_layers_full_replay:
        logger.info('Dedup index already covers %s (%d lines)', path.name, skip_lines)
    return count
