from __future__ import annotations

import os


def env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, '1' if default else '0')).strip().lower()
    return raw not in ('0', 'false', 'no', 'off', '')


def env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = env_optional_int(name)
    if raw is None:
        return default
    return max(minimum, raw)


def env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = env_optional_float(name)
    if raw is None:
        return default
    return max(minimum, raw)


def env_bool(name: str, default: bool) -> bool:
    return env_flag(name, default)


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()


def env_yes(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes')


def env_explicit_on(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


def env_explicit_off(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in ('0', 'false', 'no', 'off')


def resolve_explicit_or_env(
    explicit: int | None,
    env_name: str,
    default: int,
    *,
    minimum: int = 1,
) -> int:
    if explicit is not None:
        return max(minimum, int(explicit))
    env_val = env_optional_int(env_name)
    if env_val is not None:
        return max(minimum, env_val)
    return default
