#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Full pipeline architecture and cost audit')
    parser.add_argument('--work-dir', type=Path, default=None, help='Merge work dir with runtime metrics')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)

    from indw.tools.reports.pipeline_audit import build_pipeline_audit_report

    report = build_pipeline_audit_report(args.work_dir, workers=args.workers)
    out = args.out
    if out is None and args.work_dir is not None:
        out = args.work_dir / 'pipeline_audit_report.json'
    elif out is None:
        out = Path('pipeline_audit_report.json')

    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'out': str(out),
        'commodity_stages': report.get('commodity_count'),
        'intelligence_stages': report.get('intelligence_count'),
        'throughput_docs_per_sec': (report.get('throughput_estimate') or {}).get('docs_per_sec'),
        'gate_recommendations': len(report.get('gate_recommendations') or []),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
