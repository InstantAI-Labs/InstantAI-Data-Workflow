#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Library-first pipeline migration report')
    parser.add_argument('--work-dir', type=Path, default=None, help='Merge work dir for runtime metrics')
    parser.add_argument('--workers', type=int, default=8, help='Worker count for allocation spec')
    parser.add_argument('--out', type=Path, default=None, help='Output JSON path')
    args = parser.parse_args(argv)

    from indw.tools.reports.library_migration import build_library_migration_report

    report = build_library_migration_report(args.work_dir, workers=args.workers)
    out = args.out
    if out is None and args.work_dir is not None:
        out = args.work_dir / 'library_migration_report.json'
    elif out is None:
        out = Path(__file__).resolve().parents[1] / 'reports' / 'library_migration_report.json'

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'out': str(out),
        'commodity_stages': report.get('commodity_stage_count'),
        'intelligence_stages': report.get('intelligence_stage_count'),
        'json_backend': report.get('json_backend'),
        'opportunities': len(report.get('library_adoption_opportunities') or []),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
