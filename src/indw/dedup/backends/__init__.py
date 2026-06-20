from __future__ import annotations

import os

_BACKEND = os.environ.get('INSTANT_FUZZY_BACKEND', 'custom').strip().lower()


def fuzzy_backend_available() -> bool:
    if _BACKEND != 'datasketch':
        return False
    try:
        import datasketch  # noqa: F401
        return True
    except ImportError:
        return False


def backend_name() -> str:
    if fuzzy_backend_available():
        return 'datasketch'
    return 'custom'
