from __future__ import annotations

import argparse
import os
from pathlib import Path


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("merge", help="run quality merge on raw corpus")
    p.add_argument("raw_dir", type=Path)
    p.add_argument("out_path", type=Path)
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--fresh", action="store_true")
    p.add_argument(
        "--backend",
        choices=("local", "thread", "multiprocess", "dask"),
        default=None,
        help="execution backend (INSTANT_PIPELINE_BACKEND)",
    )
    p.set_defaults(_handler=run)


def run(args: argparse.Namespace) -> int:
    from indw.filter.spec.quality import QualityPipelineConfig
    from indw.schedule.core import merge_with_quality

    if args.backend:
        os.environ["INSTANT_PIPELINE_BACKEND"] = args.backend
    os.environ.setdefault("INSTANT_MERGE_HW_PROBE", "0")
    cfg = QualityPipelineConfig()
    merge_with_quality(
        args.raw_dir,
        args.out_path,
        quality_config=cfg,
        work_dir=args.work_dir,
        fresh=args.fresh,
        resume=not args.fresh,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )
    return 0
