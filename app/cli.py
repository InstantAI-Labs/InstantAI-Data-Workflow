from __future__ import annotations

import argparse
import sys

from app.commands import (
    register_audit,
    register_benchmark,
    register_doctor,
    register_merge,
    register_test,
    register_validate,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="indw", description="INDW — Instant Data Workflow")
    sub = ap.add_subparsers(dest="command", required=True)
    register_merge(sub)
    register_test(sub)
    register_validate(sub)
    register_audit(sub)
    register_benchmark(sub)
    register_doctor(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args._handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
