from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


STAGE_METRICS_FILE = 'pipeline_stage_metrics.json'


@dataclass
class StageTiming:
    wall_sec: float = 0.0
    max_wall_sec: float = 0.0
    calls: int = 0
    in_docs: int = 0
    out_docs: int = 0
    dropped: int = 0

    def record(self, *, wall_sec: float, in_docs: int = 0, out_docs: int = 0, dropped: int = 0) -> None:
        self.wall_sec += max(0.0, wall_sec)
        self.max_wall_sec = max(self.max_wall_sec, wall_sec)
        self.calls += 1
        self.in_docs += in_docs
        self.out_docs += out_docs
        self.dropped += dropped

    def to_dict(self) -> dict[str, Any]:
        avg = self.wall_sec / max(self.calls, 1)
        reject = self.in_docs - self.out_docs if self.in_docs else 0
        return {
            'wall_sec': round(self.wall_sec, 4),
            'max_wall_sec': round(self.max_wall_sec, 4),
            'avg_wall_sec': round(avg, 6),
            'calls': self.calls,
            'in_docs': self.in_docs,
            'out_docs': self.out_docs,
            'dropped': self.dropped,
            'reject_rate': round(reject / max(self.in_docs, 1), 4),
        }


@dataclass
class MergeStageProfile:
    stages: dict[str, StageTiming] = field(default_factory=dict)
    merge_wall_sec: float = 0.0
    docs_scanned: int = 0

    def stage(self, name: str) -> StageTiming:
        if name not in self.stages:
            self.stages[name] = StageTiming()
        return self.stages[name]

    @contextmanager
    def timed(
        self,
        name: str,
        *,
        in_docs: int = 1,
        out_docs: int | None = None,
        dropped: int = 0,
    ) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            timing = self.stage(name)
            timing.record(
                wall_sec=elapsed,
                in_docs=in_docs,
                out_docs=out_docs if out_docs is not None else in_docs,
                dropped=dropped,
            )

    def absorb_cleaning_stats(self, cleaning_stats: dict[str, Any], *, total_wall_sec: float = 0.0) -> None:
        stage_rows = cleaning_stats.get('stages') or {}
        for name, row in stage_rows.items():
            if not isinstance(row, dict):
                continue
            timing = self.stage(name)
            timing.in_docs += int(row.get('in_docs', 0))
            timing.out_docs += int(row.get('out_docs', 0))
            timing.dropped += int(row.get('dropped', 0))
            timing.wall_sec += float(row.get('wall_sec', 0.0))
            timing.calls += int(row.get('calls', 0))
        ke = cleaning_stats.get('knowledge_extraction')
        if isinstance(ke, dict):
            for name, row in ke.items():
                if not isinstance(row, dict):
                    continue
                timing = self.stage(f'ke_{name}')
                timing.wall_sec += float(row.get('wall_sec', 0.0))
                timing.calls += int(row.get('calls', 0))
        if total_wall_sec > 0:
            self.merge_wall_sec = total_wall_sec

    def to_dict(self) -> dict[str, Any]:
        total_wall = sum(s.wall_sec for s in self.stages.values()) or self.merge_wall_sec or 1e-9
        stage_rows: dict[str, Any] = {}
        for name, timing in sorted(self.stages.items()):
            row = timing.to_dict()
            row['wall_pct'] = round(100.0 * timing.wall_sec / total_wall, 2)
            stage_rows[name] = row
        return {
            'merge_wall_sec': round(self.merge_wall_sec, 2),
            'docs_scanned': self.docs_scanned,
            'stages': stage_rows,
        }


def stage_metrics_path(work_dir: Path) -> Path:
    return Path(work_dir) / STAGE_METRICS_FILE


def write_stage_metrics(
    work_dir: Path,
    profile: MergeStageProfile,
    *,
    cleaning_stats: dict[str, Any] | None = None,
    merge_wall_sec: float = 0.0,
    docs_scanned: int = 0,
) -> Path:
    if cleaning_stats:
        profile.absorb_cleaning_stats(cleaning_stats, total_wall_sec=merge_wall_sec)
    profile.docs_scanned = docs_scanned
    profile.merge_wall_sec = merge_wall_sec
    path = stage_metrics_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding='utf-8')
    return path
