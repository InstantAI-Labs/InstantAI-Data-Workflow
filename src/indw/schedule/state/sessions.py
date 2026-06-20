from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indw.schedule.intel.session import bind_acim_session
from indw.schedule.monitor.doc import DocMonitorSession, bind_doc_monitor
from indw.schedule.intel.merge_session import open_merge_intelligence_session
from indw.schedule.config.policy import MergeRuntime, bind_merge_runtime


@dataclass
class MergeCoordinator:
    merge_work: Path
    runtime: MergeRuntime
    intel: Any
    acim: Any
    doc_monitor: DocMonitorSession

    def refresh_signals(self, **kwargs: Any) -> None:
        from indw.schedule.config.policy import PipelineSignals
        if self.acim is not None:
            self.acim.update_hardware(
                queue_depth=int(kwargs.get('queue_depth', 0)),
                docs_per_sec=float(kwargs.get('docs_per_sec', 0.0)),
                cpu_pct=kwargs.get('cpu_pct'),
                rss_mb=kwargs.get('rss_mb'),
                cache_hit_rate=kwargs.get('cache_hit_rate'),
            )
        sig = PipelineSignals(
            cpu_pct=float(kwargs.get('cpu_pct', 0.0)),
            rss_mb=float(kwargs.get('rss_mb', 0.0)),
            cache_hit_rate=float(kwargs.get('cache_hit_rate', 0.0)),
            queue_depth=int(kwargs.get('queue_depth', 0)),
            docs_per_sec=float(kwargs.get('docs_per_sec', 0.0)),
            active_workers=int(kwargs.get('active_workers', 0)),
        )
        self.runtime.refresh(sig)

    def close(self) -> tuple[dict[str, Any], dict[str, Any]]:
        intel = self.intel
        if intel is None:
            intel_stats: dict[str, Any] = {}
        elif hasattr(intel, 'pci_compat_stats'):
            intel_stats = intel.pci_compat_stats()
        else:
            intel_stats = intel.stats()
        doc_stats = self.doc_monitor.stats()
        if self.intel is not None:
            self.intel.close()
        bind_acim_session(None)
        self.doc_monitor.close()
        bind_merge_runtime(None)
        return intel_stats, doc_stats


def open_merge_coordinator(
    merge_work: Path,
    *,
    workers: int | None = None,
    chunk_size: int | None = None,
    checkpoint_interval: int | None = None,
) -> MergeCoordinator:
    runtime = MergeRuntime.bootstrap(
        workers=workers,
        chunk_size=chunk_size,
        checkpoint_interval=checkpoint_interval,
        work_dir=merge_work,
    )
    bind_merge_runtime(runtime)
    intel, acim = open_merge_intelligence_session(merge_work)
    doc_monitor = DocMonitorSession(merge_work)
    bind_doc_monitor(doc_monitor)
    return MergeCoordinator(
        merge_work=Path(merge_work),
        runtime=runtime,
        intel=intel,
        acim=acim,
        doc_monitor=doc_monitor,
    )
