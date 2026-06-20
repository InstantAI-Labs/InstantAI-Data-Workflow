from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indw.schedule.admission.tiers import TIER_COST
from indw.store.io.json_codec import dumps_pretty


@dataclass
class TierTracker:
    rejects: dict[int, int] = field(default_factory=lambda: {t: 0 for t in TIER_COST})
    survivors: dict[int, int] = field(default_factory=lambda: {t: 0 for t in TIER_COST})
    accepted: int = 0
    scanned: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_reject(self, tier: int, *, reason: str = '') -> None:
        with self._lock:
            self.rejects[tier] = self.rejects.get(tier, 0) + 1
            self.scanned += 1

    def record_survivor(self, tier: int) -> None:
        with self._lock:
            self.survivors[tier] = self.survivors.get(tier, 0) + 1

    def record_accept(self) -> None:
        with self._lock:
            self.accepted += 1

    def record_scan(self) -> None:
        with self._lock:
            self.scanned += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            total_rejects = sum(self.rejects.values())
            tier_reject_pct = {
                str(t): round(100.0 * c / max(self.scanned, 1), 2)
                for t, c in self.rejects.items() if c
            }
            reject_cost = sum(self.rejects[t] * TIER_COST[t] for t in self.rejects)
            accept_cost_est = self.accepted * (
                TIER_COST[0] + TIER_COST[1] + TIER_COST[2] + TIER_COST[3] + TIER_COST[4] * 0.15
            )
            return {
                'scanned': self.scanned,
                'accepted': self.accepted,
                'rejects_by_tier': {str(t): c for t, c in self.rejects.items() if c},
                'reject_pct_by_tier': tier_reject_pct,
                'avg_cost_rejected': round(reject_cost / max(total_rejects, 1), 2),
                'avg_cost_accepted_est': round(accept_cost_est / max(self.accepted, 1), 2),
                'survivors_by_tier': {str(t): c for t, c in self.survivors.items() if c},
            }

    def publish(self, merge_work: Path | str) -> None:
        path = Path(merge_work) / 'admission_tier_report.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_pretty(self.summary()), encoding='utf-8')
