from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

_installed = False


class _IndwCompatFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "data" and not fullname.startswith("data."):
            return None
        indw_name = "indw" if fullname == "data" else "indw" + fullname[4:]
        if fullname in sys.modules:
            return importlib.util.spec_from_loader(fullname, loader=None)
        try:
            mod = importlib.import_module(indw_name)
        except ModuleNotFoundError:
            return None
        sys.modules[fullname] = mod
        return importlib.util.spec_from_loader(fullname, loader=None)


def install_compat() -> None:
    global _installed
    if _installed:
        return
    for finder in sys.meta_path:
        if isinstance(finder, _IndwCompatFinder):
            _installed = True
            return
    sys.meta_path.insert(0, _IndwCompatFinder())
    _installed = True
