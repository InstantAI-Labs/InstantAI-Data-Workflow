#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Admission tier cost audit for merge work dir')
    parser.add_argument('work_dir', type=Path, help='Merge work directory')
    parser.add_argument('--baseline', type=Path, default=None, help='Baseline work dir for delta')
    parser.add_argument('--out', type=Path, default=None, help='Write JSON report path')
    args = parser.parse_args(argv)

    from indw.tools.reports.admission_cost import build_admission_cost_report

    baseline = None
    if args.baseline is not None and args.baseline.is_dir():
        baseline = build_admission_cost_report(args.baseline)
    report = build_admission_cost_report(args.work_dir, baseline=baseline)
    out_path = args.out or (args.work_dir / 'admission_cost_report.json')
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'out': str(out_path),
        'scanned': report.get('scanned'),
        'accepted': report.get('accepted'),
        'reject_pct_by_tier': report.get('reject_pct_by_tier'),
        'throughput_improvement_pct_est': (report.get('savings_projection') or {}).get(
            'throughput_improvement_pct_est',
        ),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
