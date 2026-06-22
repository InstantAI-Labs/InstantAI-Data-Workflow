import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from indw._compat import install_compat

install_compat()

import indw as _indw

__all__ = list(_indw.__all__)


def __getattr__(name: str):
    return getattr(_indw, name)
