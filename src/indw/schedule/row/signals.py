from __future__ import annotations

import logging
import signal
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MergeStopState:
    requested: bool = False
    signals: int = 0


def merge_stop_handler(
    state: MergeStopState,
    on_pause: Callable[[], None],
    *,
    stop_event: Any = None,
) -> Callable[..., None]:
    def handler(*_) -> None:
        state.signals += 1
        state.requested = True
        if stop_event is not None:
            stop_event.set()
        if state.signals >= 2:
            logger.warning('Force stop — saving checkpoint now …')
            try:
                on_pause()
            finally:
                raise SystemExit(130)
        logger.warning(
            'Merge stop requested — saving after current step (Ctrl+C again to force quit) …'
        )

    return handler


def install_merge_signal_handlers(handler: Callable[..., None]) -> tuple[Any, Any | None]:
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, 'SIGTERM'):
        previous_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, handler)
    else:
        previous_term = None
    return previous_handler, previous_term


def restore_merge_signal_handlers(previous_handler: Any, previous_term: Any | None) -> None:
    signal.signal(signal.SIGINT, previous_handler)
    if previous_term is not None:
        signal.signal(signal.SIGTERM, previous_term)
