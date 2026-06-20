from __future__ import annotations

import argparse
import importlib.util
import platform


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("doctor", help="check install and backend availability")
    p.set_defaults(_handler=run)


def run(_args: argparse.Namespace) -> int:
    import indw
    from indw.schedule.backends.config import pipeline_execution_backend
    from indw.schedule.backends.factory import resolve_execution_backend

    print(f"indw={indw.__version__} python={platform.python_version()} platform={platform.platform()}")
    print(f"backend={pipeline_execution_backend()} resolved={resolve_execution_backend().name}")
    for pkg in ("orjson", "trafilatura", "dask"):
        print(f"{pkg}={'ok' if importlib.util.find_spec(pkg) else 'missing'}")
    return 0
