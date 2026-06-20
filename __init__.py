import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from indw._compat import install_compat

install_compat()

import indw

for _name in indw.__all__:
    globals()[_name] = getattr(indw, _name)

__all__ = list(indw.__all__)
