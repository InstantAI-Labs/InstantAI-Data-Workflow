from __future__ import annotations

import os
from typing import Any

from indw.schedule.config.resolve import env_explicit_off, env_explicit_on, env_optional_int, env_yes

MODE_PRODUCTION = 'production'
MODE_VALIDATION = 'validation'
MODE_DEBUG = 'debug'

_VALID = frozenset({MODE_PRODUCTION, MODE_VALIDATION, MODE_DEBUG})


def resolve_obs_mode() -> str:
    raw = os.environ.get('INSTANT_OBSERVABILITY_MODE', '').strip().lower()
    if raw in ('prod', 'production'):
        return MODE_PRODUCTION
    if raw in ('validate', 'validation'):
        return MODE_VALIDATION
    if raw in ('debug', 'dev'):
        return MODE_DEBUG
    if raw in _VALID:
        return raw
    if env_yes('INSTANT_PIPELINE_VALIDATION'):
        return MODE_VALIDATION
    return MODE_PRODUCTION


def obs_mode() -> str:
    return resolve_obs_mode()


def is_production() -> bool:
    return obs_mode() == MODE_PRODUCTION


def is_validation() -> bool:
    return obs_mode() == MODE_VALIDATION


def is_debug() -> bool:
    return obs_mode() == MODE_DEBUG


def heartbeat_stdout_enabled() -> bool:
    if env_explicit_on('INSTANT_MERGE_HEARTBEAT_LOG'):
        return True
    if env_explicit_off('INSTANT_MERGE_HEARTBEAT_LOG'):
        return False
    return not is_production()


def gate_diagnostics_enabled() -> bool:
    if env_explicit_on('INSTANT_MERGE_GATE_DIAGNOSTICS'):
        return True
    if env_explicit_off('INSTANT_MERGE_GATE_DIAGNOSTICS'):
        return False
    return is_debug()


def pci_events_enabled() -> bool:
    if env_explicit_on('INSTANT_PCI_EVENTS'):
        return True
    if env_explicit_off('INSTANT_PCI_EVENTS'):
        return False
    return is_debug()


def doc_stalls_enabled() -> bool:
    if env_explicit_off('INSTANT_DOC_STALLS'):
        return False
    return True


def cache_stats_enabled() -> bool:
    if env_explicit_on('INSTANT_CACHE_STATS'):
        return True
    if env_explicit_off('INSTANT_CACHE_STATS'):
        return False
    return is_debug()


def reject_log_enabled(cfg_observability: dict[str, Any] | None = None) -> bool:
    if cfg_observability is not None and not cfg_observability.get('enabled', True):
        return False
    if env_explicit_off('INSTANT_REJECT_LOG'):
        return False
    return True


def reject_log_flush_every() -> int:
    raw = env_optional_int('INSTANT_REJECT_LOG_FLUSH_EVERY')
    if raw is not None:
        return max(1, raw)
    if is_debug():
        return 25
    if is_validation():
        return 50
    return 100


def periodic_stdout_progress_enabled() -> bool:
    return is_validation() or is_debug()


def stage_metrics_on_finalize() -> bool:
    if env_explicit_on('INSTANT_STAGE_METRICS'):
        return True
    if env_explicit_off('INSTANT_STAGE_METRICS'):
        return False
    return is_validation() or is_debug()


def progress_active_doc_detail() -> bool:
    return is_validation() or is_debug()


def progress_reject_reasons(work_dir: Any, gate: Any) -> dict[str, int]:
    if is_production():
        return {str(k): int(v) for k, v in dict(gate.stats.reject_reasons).items()}
    from indw.schedule.monitor.invariants import merge_progress_reject_reasons
    return merge_progress_reject_reasons(work_dir, gate)
