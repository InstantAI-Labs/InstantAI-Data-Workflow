from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from indw.config.defaults import ACIM_ENABLED, PCI_ENABLED
from indw.schedule.config.resolve import env_flag


@runtime_checkable
class MergeIntelligenceSession(Protocol):
    def observe_preprocessed(self, line: dict[str, Any]) -> Any: ...
    def close(self) -> None: ...
    def stats(self) -> dict[str, Any]: ...


def acim_enabled() -> bool:
    return env_flag('INSTANT_ACIM_ENABLED', ACIM_ENABLED)


def pci_enabled() -> bool:
    if acim_enabled():
        return False
    return env_flag('INSTANT_PCI_ENABLED', PCI_ENABLED)


def open_merge_intelligence_session(
    merge_work: Path,
) -> tuple[MergeIntelligenceSession | None, Any]:
    from indw.schedule.intel.session import ACIMSession, bind_acim_session
    from indw.schedule.intel.pci import PCISession

    acim: ACIMSession | None = None
    intel: MergeIntelligenceSession | None = None
    if acim_enabled():
        acim = ACIMSession(merge_work)
        if acim.enabled:
            bind_acim_session(acim)
            intel = acim
        else:
            acim.close()
            acim = None
    if intel is None and pci_enabled():
        pci = PCISession(merge_work)
        if pci.enabled:
            intel = pci
        else:
            pci.close()
    return intel, acim


def open_worker_intelligence_session(merge_work: Path) -> Any:
    from indw.schedule.intel.session import ACIMSession, bind_acim_session

    if not acim_enabled():
        return None
    sess = ACIMSession.for_worker(merge_work)
    if sess is None:
        return None
    bind_acim_session(sess)
    return sess


def close_worker_intelligence_session() -> None:
    from indw.schedule.intel.session import bind_acim_session, get_acim_session

    sess = get_acim_session()
    if sess is not None and getattr(sess, 'readonly', False):
        sess.close()
    bind_acim_session(None)


