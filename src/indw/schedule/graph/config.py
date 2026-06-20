from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def pipeline_graph_enabled() -> bool:
    raw = os.environ.get('INSTANT_PIPELINE_GRAPH', 'v2').strip().lower()
    if raw in ('v1', 'legacy', '0', 'false', 'no', 'off'):
        logger.warning(
            'INSTANT_PIPELINE_GRAPH=%s ignored; canonical graph scheduler is always active',
            raw,
        )
    return True


def pipeline_queue_backend() -> str:
    return os.environ.get('INSTANT_PIPELINE_QUEUE', 'local').strip().lower()


def pipeline_dedup_shards() -> int:
    raw = os.environ.get('INSTANT_DEDUP_SHARDS', '').strip()
    if not raw:
        return 0
    try:
        return max(1, int(raw))
    except ValueError:
        return 0
