from __future__ import annotations

import argparse


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("test", help="run framework test suite")
    p.add_argument(
        "--profile",
        choices=("unit", "critical", "parity", "integration", "smoke"),
        default="unit",
    )
    p.add_argument("pytest_args", nargs="*", help="extra pytest arguments")
    p.set_defaults(_handler=run)


def run(args: argparse.Namespace) -> int:
    from app.workflows import run_tests

    return run_tests(args.profile, extra_args=args.pytest_args or None)
