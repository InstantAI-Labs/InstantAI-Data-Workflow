from __future__ import annotations

import argparse


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("benchmark", help="production scale benchmark")
    p.add_argument("--workers", default="1 2 4", help="worker counts")
    p.set_defaults(_handler=run)


def run(args: argparse.Namespace) -> int:
    from app.workflows import run_benchmark

    return run_benchmark(workers=args.workers)
