from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indw.filter.stage0.admission import (
    DOC_TIER_HUGE,
    DOC_TIER_LARGE,
)

LANE_NORMAL = 'normal'
LANE_LARGE = 'large'
LANE_HUGE = 'huge'

ALL_LANES = (LANE_NORMAL, LANE_LARGE, LANE_HUGE)


def _survivor_seq(payload: dict[str, Any]) -> int:
    return int(payload['seq'])


def _insert_sorted(buf: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    seq = _survivor_seq(payload)
    lo, hi = 0, len(buf)
    while lo < hi:
        mid = (lo + hi) // 2
        if _survivor_seq(buf[mid]) <= seq:
            lo = mid + 1
        else:
            hi = mid
    buf.insert(lo, payload)


def _merge_sorted(buf: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> None:
    if not incoming:
        return
    incoming.sort(key=_survivor_seq)
    if not buf:
        buf.extend(incoming)
        return
    merged: list[dict[str, Any]] = []
    i = j = 0
    while i < len(buf) and j < len(incoming):
        if _survivor_seq(buf[i]) <= _survivor_seq(incoming[j]):
            merged.append(buf[i])
            i += 1
        else:
            merged.append(incoming[j])
            j += 1
    merged.extend(buf[i:])
    merged.extend(incoming[j:])
    buf[:] = merged


def survivor_lane(payload: dict[str, Any]) -> str:
    tier = str(payload.get('doc_tier') or '')
    if not tier:
        adm = payload.get('admission')
        if isinstance(adm, dict):
            tier = str(adm.get('tier') or '')
    if tier == DOC_TIER_HUGE:
        return LANE_HUGE
    if tier == DOC_TIER_LARGE:
        return LANE_LARGE
    return LANE_NORMAL


@dataclass
class LaneBuffers:
    normal: list[dict[str, Any]] = field(default_factory=list)
    large: list[dict[str, Any]] = field(default_factory=list)
    huge: list[dict[str, Any]] = field(default_factory=list)

    def route_many(self, survivors: list[dict[str, Any]]) -> None:
        if not survivors:
            return
        if len(survivors) == 1:
            self.route_one(survivors[0])
            return
        by_lane: dict[str, list[dict[str, Any]]] = {
            LANE_NORMAL: [],
            LANE_LARGE: [],
            LANE_HUGE: [],
        }
        for s in survivors:
            by_lane[survivor_lane(s)].append(s)
        for lane, batch in by_lane.items():
            if batch:
                _merge_sorted(self.lane_buffer(lane), batch)

    def route_one(self, payload: dict[str, Any]) -> None:
        _insert_sorted(self.lane_buffer(survivor_lane(payload)), payload)

    def lane_buffer(self, lane: str) -> list[dict[str, Any]]:
        if lane == LANE_HUGE:
            return self.huge
        if lane == LANE_LARGE:
            return self.large
        return self.normal

    def total(self) -> int:
        return len(self.normal) + len(self.large) + len(self.huge)

    def depths(self) -> dict[str, int]:
        return {
            'lane_normal': len(self.normal),
            'lane_large': len(self.large),
            'lane_huge': len(self.huge),
        }

    def empty(self) -> bool:
        return self.total() == 0


@dataclass(frozen=True)
class LaneWorkerSlots:
    normal: int
    large: int
    huge: int

    @classmethod
    def from_heavy_workers(cls, n: int) -> LaneWorkerSlots:
        from indw.schedule.config.tune import lane_worker_slots
        normal, large, huge = lane_worker_slots(n)
        return cls(normal=normal, large=large, huge=huge)

    def slot_limit(self, lane: str) -> int:
        if lane == LANE_HUGE:
            return self.huge
        if lane == LANE_LARGE:
            return self.large
        return self.normal

    def to_dict(self) -> dict[str, int]:
        return {'normal': self.normal, 'large': self.large, 'huge': self.huge}


def lane_min_seq(buffers: LaneBuffers, lane: str) -> int | None:
    buf = buffers.lane_buffer(lane)
    if not buf:
        return None
    return _survivor_seq(buf[0])


def has_buffered_seq(buffers: LaneBuffers, target_seq: int) -> bool:
    for lane in ALL_LANES:
        for payload in buffers.lane_buffer(lane):
            if int(payload['seq']) == target_seq:
                return True
    return False


def pop_target_seq(buffers: LaneBuffers, target_seq: int) -> tuple[dict[str, Any], str] | None:
    for lane in ALL_LANES:
        buf = buffers.lane_buffer(lane)
        for i, payload in enumerate(buf):
            if int(payload['seq']) == target_seq:
                return buf.pop(i), lane
    return None


def pick_lane_batch(
    buffer: list[dict[str, Any]],
    lane: str,
    threshold: int,
    *,
    ordering_gap: int = 0,
    batch_cap: int | None = None,
) -> list[dict[str, Any]]:
    if not buffer:
        return []
    cap = batch_cap if batch_cap is not None else threshold
    if lane != LANE_NORMAL or threshold <= 1 or ordering_gap > 0:
        return [buffer.pop(0)]
    count = min(threshold, cap, len(buffer))
    batch = buffer[:count]
    del buffer[:count]
    return batch
