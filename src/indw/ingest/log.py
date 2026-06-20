from __future__ import annotations

import logging
import sys

_CONSOLE_SAFE_REPLACEMENTS: tuple[tuple[str, str], ...] = ()


class _ConsoleSafeFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if sys.platform == 'win32':
            for old, new in _CONSOLE_SAFE_REPLACEMENTS:
                msg = msg.replace(old, new)
        return msg


def _dataset_log_formatter() -> _ConsoleSafeFormatter:
    return _ConsoleSafeFormatter(
        fmt='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )


def _has_console_safe_handler(root: logging.Logger) -> bool:
    return any(
        isinstance(handler, logging.StreamHandler)
        and handler.stream is sys.stdout
        and isinstance(handler.formatter, _ConsoleSafeFormatter)
        for handler in root.handlers
    )


def setup_dataset_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if _has_console_safe_handler(root):
        root.setLevel(level)
    else:
        root.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_dataset_log_formatter())
        root.addHandler(handler)
        root.setLevel(level)
    logging.getLogger('datasets').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.INFO)


def human_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f'{n / 1_000_000_000:.2f} GB'
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f} MB'
    if n >= 1_000:
        return f'{n / 1_000:.1f} KB'
    return f'{n} B'
