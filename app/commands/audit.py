from __future__ import annotations

import argparse
from pathlib import Path


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("audit", help="run pipeline audit reports")
    p.add_argument(
        "--kind",
        choices=("pipeline", "dask", "production", "library", "stage0"),
        default="pipeline",
    )
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(_handler=run)


def run(args: argparse.Namespace) -> int:
    from app.workflows import run_audit

    return run_audit(kind=args.kind, work_dir=args.work_dir, workers=args.workers)
