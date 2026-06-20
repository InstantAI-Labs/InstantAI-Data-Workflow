#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from indw.tools.reports.dask_integration import build_dask_integration_report


def main() -> int:
    ap = argparse.ArgumentParser(description='Dask execution backend integration report')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--work-dir', type=Path, default=None)
    args = ap.parse_args()
    report = build_dask_integration_report(workers=args.workers)
    text = json.dumps(report, indent=2)
    if args.write and args.work_dir:
        out = args.work_dir / 'dask_integration_report.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8')
        print(out)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
