from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indw.schedule.admission.tiers import TIER_COST, stage_tier
from indw.store.io.json_codec import dumps_pretty


@dataclass
class StageCostRow:
    stage: str
    tier: int = 0
    entered: int = 0
    rejected: int = 0
    survived: int = 0
    wall_sec: float = 0.0
    latency_ms_sum: float = 0.0
    calls: int = 0
    queue_depth_peak: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cpu_sec: float = 0.0
    memory_mb_peak: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_sum / max(self.calls, 1)

    @property
    def reject_rate(self) -> float:
        return self.rejected / max(self.entered, 1)

    @property
    def cost_per_rejection(self) -> float:
        weight = TIER_COST.get(self.tier, 1.0)
        return round(weight * self.wall_sec / max(self.rejected, 1), 4)

    @property
    def cost_per_survivor(self) -> float:
        weight = TIER_COST.get(self.tier, 1.0)
        return round(weight * self.wall_sec / max(self.survived, 1), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            'stage': self.stage,
            'tier': self.tier,
            'entered': self.entered,
            'rejected': self.rejected,
            'survived': self.survived,
            'reject_rate': round(self.reject_rate, 4),
            'wall_sec': round(self.wall_sec, 4),
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'calls': self.calls,
            'queue_depth_peak': self.queue_depth_peak,
            'cache_hit_rate': round(
                self.cache_hits / max(self.cache_hits + self.cache_misses, 1), 4,
            ),
            'cost_per_rejection': self.cost_per_rejection,
            'cost_per_survivor': self.cost_per_survivor,
        }


class StageCostLedger:
    def __init__(self) -> None:
        self._rows: dict[str, StageCostRow] = {}
        self._lock = threading.RLock()
        self._started = time.perf_counter()

    def row(self, stage: str) -> StageCostRow:
        with self._lock:
            if stage not in self._rows:
                self._rows[stage] = StageCostRow(stage=stage, tier=stage_tier(stage))
            return self._rows[stage]

    def record_batch(
        self,
        stage: str,
        *,
        entered: int,
        rejected: int = 0,
        survived: int = 0,
        wall_sec: float,
        queue_depth: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0,
    ) -> None:
        with self._lock:
            r = self.row(stage)
            r.entered += entered
            r.rejected += rejected
            r.survived += survived
            r.wall_sec += max(0.0, wall_sec)
            r.latency_ms_sum += max(0.0, wall_sec) * 1000.0
            r.calls += 1
            r.queue_depth_peak = max(r.queue_depth_peak, queue_depth)
            r.cache_hits += cache_hits
            r.cache_misses += cache_misses

    def absorb_payload(self, payload: dict[str, Any]) -> None:
        for row in payload.get('_cost_rows') or []:
            if not isinstance(row, dict):
                continue
            self.record_batch(
                str(row.get('stage') or ''),
                entered=int(row.get('entered') or 0),
                rejected=int(row.get('rejected') or 0),
                survived=int(row.get('survived') or 0),
                wall_sec=float(row.get('wall_sec') or 0.0),
                queue_depth=int(row.get('queue_depth') or 0),
                cache_hits=int(row.get('cache_hits') or 0),
                cache_misses=int(row.get('cache_misses') or 0),
            )

    def gate_recommendations(self) -> list[dict[str, str]]:
        recs: list[dict[str, str]] = []
        with self._lock:
            for name, r in self._rows.items():
                if r.entered < 10:
                    continue
                if r.reject_rate < 0.02 and r.tier >= 3:
                    recs.append({
                        'stage': name,
                        'issue': 'low_reject_high_cost',
                        'action': 'verify earlier tier gate coverage; parity-locked if dedup-order dependent',
                    })
                if r.reject_rate > 0.4 and r.tier >= 3:
                    recs.append({
                        'stage': name,
                        'issue': 'high_reject_late_stage',
                        'action': f'candidate to strengthen tier {max(r.tier - 1, 0)} gate if subset of rejects',
                    })
        recs.append({
            'stage': 'language_gate',
            'issue': 'parity_lock',
            'action': 'must precede exact dedup (check-and-insert ordering)',
        })
        return recs

    def bottleneck_tree(self) -> list[dict[str, Any]]:
        with self._lock:
            ranked = sorted(self._rows.values(), key=lambda r: -r.wall_sec)
            total = sum(r.wall_sec for r in ranked) or 1e-9
            out: list[dict[str, Any]] = []
            for i, r in enumerate(ranked[:16], start=1):
                out.append({
                    'rank': i,
                    'stage': r.stage,
                    'tier': r.tier,
                    'wall_sec': round(r.wall_sec, 4),
                    'wall_pct': round(100.0 * r.wall_sec / total, 2),
                    'entered': r.entered,
                    'reject_rate': round(r.reject_rate, 4),
                    'cost_per_survivor': r.cost_per_survivor,
                })
            return out

    def summary(self) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self._started, 1e-9)
        with self._lock:
            stages = {name: r.to_dict() for name, r in sorted(self._rows.items())}
            total_entered = sum(r.entered for r in self._rows.values())
            total_rejected = sum(r.rejected for r in self._rows.values())
            total_wall = sum(r.wall_sec for r in self._rows.values())
        return {
            'elapsed_sec': round(elapsed, 2),
            'total_entered': total_entered,
            'total_rejected': total_rejected,
            'total_wall_sec': round(total_wall, 4),
            'throughput_docs_per_sec': round(total_entered / elapsed, 3),
            'stages': stages,
            'bottleneck_tree': self.bottleneck_tree(),
            'gate_recommendations': self.gate_recommendations(),
        }

    def publish(self, merge_work: Path | str) -> Path:
        path = Path(merge_work) / 'pipeline_cost_accounting.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_pretty(self.summary()), encoding='utf-8')
        return path
