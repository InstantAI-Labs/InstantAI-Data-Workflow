from __future__ import annotations

import threading
import time

_deadline = threading.local()


def resolve_doc_wall_budget_sec() -> float:
    from indw.schedule.config.policy import active_or_built_policy
    return active_or_built_policy().doc_wall_budget_sec


def resolve_doc_max_chars() -> int:
    from indw.schedule.config.policy import active_or_built_policy
    return active_or_built_policy().doc_max_chars


def set_doc_deadline(deadline: float | None) -> None:
    _deadline.deadline = deadline


def clear_doc_deadline() -> None:
    _deadline.deadline = None


def doc_deadline() -> float | None:
    return getattr(_deadline, 'deadline', None)


def doc_budget_exceeded() -> bool:
    d = doc_deadline()
    return d is not None and time.perf_counter() > d
