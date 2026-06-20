from __future__ import annotations

from typing import Any, Optional

from indw.config.defaults import MAX_MERGE_CHUNK_SIZE, MIN_MERGE_CHUNK_SIZE


def resolve_merge_workers(workers: Optional[int] = None) -> int:
    if workers is not None:
        w = int(workers)
        if w <= 0:
            from indw.schedule.config.policy import active_or_built_policy
            return active_or_built_policy().workers
        return w
    from indw.schedule.config.policy import active_or_built_policy
    return active_or_built_policy().workers


def resolve_merge_chunk_size(chunk_size: Optional[int] = None) -> int:
    if chunk_size is not None:
        size = int(chunk_size)
        return max(MIN_MERGE_CHUNK_SIZE, min(MAX_MERGE_CHUNK_SIZE, size))
    from indw.config.defaults import DEFAULT_MERGE_CHUNK_SIZE
    from indw.schedule.config.resolve import env_optional_int
    from indw.schedule.config.policy import active_policy

    pol = active_policy()
    if pol is not None:
        return pol.chunk_size
    env_c = env_optional_int('INSTANT_MERGE_CHUNK_SIZE')
    if env_c is not None:
        return max(MIN_MERGE_CHUNK_SIZE, min(MAX_MERGE_CHUNK_SIZE, env_c))
    return DEFAULT_MERGE_CHUNK_SIZE


def resolve_merge_batch_timeout_sec(chunk_size: Optional[int] = None) -> float:
    from indw.schedule.config.policy import active_or_built_policy, build_runtime_policy
    if chunk_size is None:
        return active_or_built_policy().batch_timeout_sec
    return build_runtime_policy(chunk_size=chunk_size).batch_timeout_sec
