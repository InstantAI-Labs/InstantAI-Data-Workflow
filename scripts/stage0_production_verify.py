#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

from indw.ingest.log import setup_dataset_logging
from indw.store.io.json_codec import loads
from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule.monitor.audit import sorted_output_hash
from indw.schedule.core import merge_with_quality
from indw.filter.stage0.audit import bind_audit_dir, publish_report, build_report
from indw.tools.reports.stage0_cost import build_stage0_cost_report
from indw.filter.stage0.verify import (
    build_production_verification_report,
    human_verification_summary,
)
from tests.fixtures.pipeline_corpus import MERGE_PASSAGE_A, MERGE_PASSAGE_B, write_resolved_quality


def _verification_config() -> QualityPipelineConfig:
    cfg_path = ROOT / "configs" / "filtering" / "quality_fast_first.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return QualityPipelineConfig.from_dict(raw)


def _build_verification_corpus(raw_dir: Path) -> dict[str, int]:
    html_junk = (
        "<html><head><title>Index</title></head><body>"
        + "<nav>" + "".join(f'<a href="/p{i}">Link {i}</a>' for i in range(60)) + "</nav>"
        + "<script>function noop(){}</script>" * 30
        + "</body></html>"
    )
    metadata_only = "\n".join([
        "Copyright 2024 Example Corp. All rights reserved.",
        "Privacy Policy | Terms of Service | Cookie Settings",
        "Sign in to view this content. Register now.",
        "Contact: support@example.com | Follow us on social media",
    ] * 12)
    corruption = "\ufffd" * 40 + "asdfghjkl qwertyuiop " + "z" * 120
    error_page = "\n".join([
        "404 Not Found",
        "The page you requested could not be found.",
        "404 Not Found",
        "Go back to home",
    ] * 5)
    nav_page = "\n".join([
        "Home | About | Contact | Login | Register | Privacy | Terms | FAQ | Help | Support",
    ] * 40 + [
        "Section: Products",
        "Section: Services",
    ])
    too_short = "Hello world."
    duplicate = MERGE_PASSAGE_A

    docs: dict[str, list[tuple[str, str]]] = {
        "valid": [
            ("valid_a", MERGE_PASSAGE_A),
            ("valid_b", MERGE_PASSAGE_B),
        ],
        "junk": [
            ("html_dump", html_junk),
            ("metadata_only", metadata_only),
            ("corruption", corruption),
            ("error_page", error_page),
            ("nav_boilerplate", nav_page),
            ("too_short", too_short),
            ("duplicate_a", duplicate),
        ],
    }
    counts: dict[str, int] = {}
    for source, entries in docs.items():
        src_dir = raw_dir / source
        src_dir.mkdir(parents=True, exist_ok=True)
        out = src_dir / "data.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for doc_id, text in entries:
                fh.write(json.dumps({"id": doc_id, "text": text}) + "\n")
        counts[source] = len(entries)
    return counts


def _text_by_seq(raw_dir: Path, work_dir: Path) -> dict[int, str]:
    del work_dir
    texts: list[str] = []
    for src in sorted(raw_dir.glob("*/data.jsonl")):
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = loads(line)
                texts.append(str(row.get("text") or ""))
    return {i: t for i, t in enumerate(texts)}


def _run_merge(
    raw_dir: Path,
    work_dir: Path,
    cfg: QualityPipelineConfig,
    *,
    workers: int,
) -> dict[str, str]:
    os.environ["INSTANT_MERGE_STAGE0_AUDIT"] = "1"
    os.environ.setdefault("INSTANT_SKIP_METRICS_PROBE", "1")
    bind_audit_dir(work_dir)
    write_resolved_quality(work_dir, cfg)
    out = work_dir / "filtered.jsonl"
    merge_with_quality(
        raw_dir,
        out,
        quality_config=cfg,
        work_dir=work_dir,
        fresh=True,
        resume=False,
        workers=workers,
        chunk_size=2,
    )
    publish_report(work_dir, build_report(work_dir))
    return {"output_hash": sorted_output_hash(out)}


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 0 production verification audit")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--cost", action="store_true", help="Include Stage 0 cost audit breakdown")
    p.add_argument("--baseline-dir", type=Path, default=None, help="Baseline work_dir for before/after comparison")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    setup_dataset_logging(__import__("logging").DEBUG if args.verbose else __import__("logging").INFO)

    cfg = _verification_config()
    with tempfile.TemporaryDirectory(prefix="stage0_verify_") as td:
        root = Path(td)
        raw_dir = root / "raw"
        counts = _build_verification_corpus(raw_dir)
        seq_dir = root / "seq"
        par_dir = root / "par"

        seq_result = _run_merge(raw_dir, seq_dir, cfg, workers=1)
        par_result = {}
        if args.workers > 1:
            par_result = _run_merge(raw_dir, par_dir, cfg, workers=args.workers)

        parity = {
            "hash_match": not par_result or seq_result["output_hash"] == par_result["output_hash"],
            "sequential_hash": seq_result["output_hash"],
            "parallel_hash": par_result.get("output_hash", ""),
        }
        text_map = _text_by_seq(raw_dir, par_dir if par_result else seq_dir)
        report = build_production_verification_report(
            par_dir if par_result else seq_dir,
            parity=parity,
            text_by_seq=text_map,
        )
        report["corpus"] = counts
        report["corpus_total"] = sum(counts.values())

        if args.workers > 1 and par_result:
            seq_report = build_production_verification_report(
                seq_dir, parity=parity, text_by_seq=text_map,
            )
            report["sequential_waterfall"] = seq_report.get("waterfall")
            report["parallel_waterfall"] = report.get("waterfall")

        if args.cost or args.baseline_dir:
            baseline = None
            if args.baseline_dir and args.baseline_dir.is_dir():
                baseline = build_production_verification_report(args.baseline_dir)
            report["stage0_cost"] = build_stage0_cost_report(
                par_dir if par_result else seq_dir,
                baseline=baseline,
            )

        summary = human_verification_summary(report)
        payload = json.dumps(report, indent=2)
        if args.out:
            args.out.write_text(payload, encoding="utf-8")
            args.out.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
        if args.json:
            print(payload)
        else:
            print(summary)
            print("\nreject_buckets:", json.dumps(report.get("reject_buckets"), indent=2))
            bottlenecks = report.get("bottleneck_ranking_by_wall") or []
            if bottlenecks:
                print("\ntop_wall_stages:")
                for row in bottlenecks[:6]:
                    print(f"  {row['stage']:30s} wall={row['wall_sec']}s dps={row['docs_per_sec']}")
            cost = report.get("stage0_cost") or {}
            if cost:
                cold = cost.get("cold_start") or {}
                print("\nstage0_cost:")
                print(f"  cold_ms={cold.get('cold_wall_ms', 0)} steady_avg_ms={cold.get('steady_avg_ms', 0)}")
                ba = cost.get("before_after") or {}
                if ba:
                    print(
                        f"  before_after_avg_ms: {ba.get('baseline_stage0_avg_ms')} -> "
                        f"{ba.get('optimized_stage0_avg_ms')} ({ba.get('delta_pct')}%)"
                    )
        return 0 if parity["hash_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
