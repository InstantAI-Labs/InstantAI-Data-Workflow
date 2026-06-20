from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_merge_dedup_stack(cfg: Any, index: Any) -> tuple[Any, Any, Any, Any]:
    from indw.dedup.embed.config import EmbeddingDedupConfig
    from indw.dedup.embed.pipeline import create_embedding_dedup
    from indw.dedup.fuzzy import StreamingFuzzyDedup
    from indw.dedup.semantic import create_semantic_dedup
    from indw.ingest.hash import ExactHashDedup

    exact = ExactHashDedup(index)
    fuzzy = (
        StreamingFuzzyDedup(
            threshold=cfg.dedup.fuzzy_threshold,
            num_perm=cfg.dedup.fuzzy_num_perm,
            quality_margin=cfg.dedup.fuzzy_quality_margin,
        )
        if cfg.dedup.fuzzy
        else None
    )
    semantic = (
        create_semantic_dedup(
            hamming_threshold=cfg.dedup.semantic_hamming_threshold,
            jaccard_threshold=cfg.dedup.semantic_jaccard_threshold,
            recent_jaccard_threshold=cfg.dedup.semantic_recent_jaccard_threshold,
        )
        if cfg.dedup.semantic
        else None
    )
    embed_cfg = cfg.dedup.embedding
    if not isinstance(embed_cfg, EmbeddingDedupConfig):
        embed_cfg = EmbeddingDedupConfig.from_dict(embed_cfg if isinstance(embed_cfg, dict) else None)
    embed_semantic = create_embedding_dedup(embed_cfg) if embed_cfg.enabled else None
    return exact, fuzzy, semantic, embed_semantic


def restore_merge_gate_balancers(
    *,
    gate: Any,
    checkpoint: Any,
    index_path: Path,
    resuming: bool,
    append: bool,
) -> None:
    from indw.schedule.state.checkpoint import (
        restore_balancers_from_checkpoint,
        restore_gate_balancers,
    )

    should_restore = (resuming or append) and index_path.exists()
    if should_restore:
        restored = restore_gate_balancers(gate, index_path)
        if restored:
            logger.info('Restored balancer state from %d kept documents', restored)
    elif resuming:
        if restore_balancers_from_checkpoint(gate, checkpoint):
            logger.info('Restored balancer state from merge checkpoint')


def restore_merge_dedup_from_output(
    *,
    out_path: Path,
    checkpoint: Any,
    cfg: Any,
    exact: Any,
    fuzzy: Any,
    semantic: Any,
    index: Any,
    resuming: bool,
    append: bool,
    embed_semantic: Any = None,
) -> None:
    from indw.dedup.replay import restore_dedup_from_jsonl
    from indw.store.io.jsonl import checkpoint_kept_lines, count_jsonl_lines

    dedup_layers = cfg.dedup.exact or cfg.dedup.fuzzy or cfg.dedup.semantic
    if getattr(cfg.dedup.embedding, 'enabled', False):
        dedup_layers = True
    if not ((resuming or append) and out_path.exists() and dedup_layers):
        return

    file_lines = checkpoint.filtered_line_count or checkpoint_kept_lines(checkpoint)
    if file_lines <= 0:
        file_lines = count_jsonl_lines(out_path)
    if file_lines <= 0:
        return

    index_count = len(exact) if cfg.dedup.exact else 0
    needs_near = (
        (cfg.dedup.fuzzy and fuzzy is not None)
        or (cfg.dedup.semantic and semantic is not None)
        or (getattr(cfg.dedup.embedding, 'enabled', False) and embed_semantic is not None)
    )

    if cfg.dedup.exact and not needs_near and index_count >= file_lines:
        logger.info(
            'Dedup index already seeded (%d hashes, %d file lines); skipping restore',
            index_count,
            file_lines,
        )
        return

    near_full_replay = needs_near and cfg.dedup.exact and index_count > 0
    skip_lines = index_count if cfg.dedup.exact and not near_full_replay else 0
    seeded = restore_dedup_from_jsonl(
        out_path,
        exact=exact if cfg.dedup.exact else None,
        fuzzy=fuzzy if cfg.dedup.fuzzy else None,
        semantic=semantic if cfg.dedup.semantic else None,
        embed_semantic=embed_semantic if getattr(cfg.dedup.embedding, 'enabled', False) else None,
        skip_lines=skip_lines,
        near_layers_full_replay=near_full_replay,
    )
    if seeded and index is not None:
        index.flush()
