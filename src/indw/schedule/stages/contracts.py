from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

StageVerdict = Literal['pass', 'reject', 'terminal']


@dataclass(frozen=True)
class StageEnvelope:
    seq: int
    src_name: str
    line_no: int
    stage: str
    verdict: StageVerdict
    reject_reason: str | None = None
    text_ref: str | None = None
    artifact_refs: tuple[str, ...] = ()
    stage_trace: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            'seq': self.seq,
            'src_name': self.src_name,
            'line_no': self.line_no,
            'stage': self.stage,
            'verdict': self.verdict,
            'reject_reason': self.reject_reason,
            'text_ref': self.text_ref,
            'artifact_refs': list(self.artifact_refs),
            'stage_trace': list(self.stage_trace),
            'metrics': dict(self.metrics),
        }
        if self.payload is not None:
            out['payload'] = self.payload
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StageEnvelope:
        refs = raw.get('artifact_refs') or ()
        trace = raw.get('stage_trace') or ()
        metrics = raw.get('metrics') or {}
        payload = raw.get('payload')
        return cls(
            seq=int(raw['seq']),
            src_name=str(raw['src_name']),
            line_no=int(raw['line_no']),
            stage=str(raw['stage']),
            verdict=raw['verdict'],
            reject_reason=raw.get('reject_reason'),
            text_ref=raw.get('text_ref'),
            artifact_refs=tuple(str(r) for r in refs),
            stage_trace=tuple(str(t) for t in trace),
            metrics={str(k): float(v) for k, v in metrics.items()},
            payload=payload if isinstance(payload, dict) else None,
        )


@runtime_checkable
class StageQueue(Protocol):
    def put(self, item: Any, *, block: bool = True, timeout: float | None = None) -> None: ...
    def get(self, *, block: bool = True, timeout: float | None = None) -> Any: ...
    def qsize(self) -> int: ...
    def full(self) -> bool: ...


@runtime_checkable
class StageWorker(Protocol):
    def process_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]: ...
