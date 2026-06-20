from __future__ import annotations
import queue
import threading
from typing import Any, Iterator, Optional
import torch

PREFETCH_ATTR = '_instant_prefetch'

def is_cuda_prefetch_wrapped(loader: Any) -> bool:
    if isinstance(loader, CudaPrefetcher):
        return True
    if getattr(loader, PREFETCH_ATTR, None) == 'cuda':
        return True
    inner = getattr(loader, 'loader', None)
    if inner is not None and inner is not loader:
        return is_cuda_prefetch_wrapped(inner)
    return False

class HostPrefetcher:

    def __init__(self, loader, *, queue_size: int = 2):
        self.loader = loader
        self.queue_size = max(1, queue_size)
        setattr(self, PREFETCH_ATTR, 'host')

    def __iter__(self) -> Iterator[Any]:
        q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        stop = object()

        def producer() -> None:
            try:
                for batch in self.loader:
                    q.put(batch)
            finally:
                q.put(stop)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is stop:
                break
            yield item
        thread.join(timeout=1.0)

class CudaPrefetcher:

    def __init__(self, loader, device: torch.device, *, non_blocking: bool = True):
        self.loader = loader
        self.device = device
        self.non_blocking = non_blocking
        self.stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        setattr(self, PREFETCH_ATTR, 'cuda')

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                out[key] = val.to(self.device, non_blocking=self.non_blocking)
            else:
                out[key] = val
        return out

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.stream is None:
            for batch in self.loader:
                yield self._to_device(batch)
            return
        loader_iter = iter(self.loader)
        next_batch: Optional[dict[str, Any]] = None

        def preload() -> Optional[dict[str, Any]]:
            try:
                batch = next(loader_iter)
            except StopIteration:
                return None
            with torch.cuda.stream(self.stream):
                return self._to_device(batch)

        next_batch = preload()
        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            batch = next_batch
            next_batch = preload()
            yield batch

def wrap_prefetch(
    loader,
    *,
    device: Optional[torch.device] = None,
    host_prefetch: bool = False,
    host_queue: int = 2,
    cuda_prefetch: bool = False,
    non_blocking: bool = True,
) -> Any:
    out = loader
    if host_prefetch:
        out = HostPrefetcher(out, queue_size=host_queue)
    if cuda_prefetch and device is not None and getattr(device, 'type', '') == 'cuda':
        out = CudaPrefetcher(out, device, non_blocking=non_blocking)
    return out
