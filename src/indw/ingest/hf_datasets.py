from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[2]
_HF_MODULE: ModuleType | None = None


def _is_shadowed_datasets_module(mod: ModuleType) -> bool:
    f = getattr(mod, '__file__', None)
    if not f:
        return False
    p = Path(f).resolve()
    if 'site-packages' in p.parts:
        return False
    return p.name == '__init__.py' and p.parent.name == 'datasets'


def _clear_shadowed_modules() -> None:
    for name in list(sys.modules):
        if name == 'datasets' or name.startswith('datasets.'):
            mod = sys.modules.get(name)
            if mod is not None and _is_shadowed_datasets_module(mod):
                del sys.modules[name]


def _path_without_src_root() -> list[str]:
    root = _SRC_ROOT.resolve()
    cwd = Path.cwd().resolve()
    out: list[str] = []
    for entry in sys.path:
        if entry in ('', '.'):
            if cwd == root:
                continue
            out.append(entry)
            continue
        try:
            if Path(entry).resolve() == root:
                continue
        except OSError:
            pass
        out.append(entry)
    return out


def get_hf_datasets_module() -> ModuleType:
    global _HF_MODULE
    if _HF_MODULE is not None and hasattr(_HF_MODULE, 'load_dataset'):
        return _HF_MODULE
    _clear_shadowed_modules()
    old_path = sys.path
    try:
        sys.path = _path_without_src_root()
        hf = importlib.import_module('datasets')
    finally:
        sys.path = old_path
    if not hasattr(hf, 'load_dataset'):
        raise ImportError(
            "HuggingFace `datasets` is not installed. Run:\n"
            "  pip install 'datasets>=2.18.0' huggingface_hub"
        )
    _HF_MODULE = hf
    return hf


def load_dataset(*args: Any, **kwargs: Any):
    return get_hf_datasets_module().load_dataset(*args, **kwargs)


def hf_datasets_available() -> bool:
    try:
        get_hf_datasets_module()
        return True
    except ImportError:
        return False
