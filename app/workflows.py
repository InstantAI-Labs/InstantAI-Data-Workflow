from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]


def test_profiles() -> dict[str, dict[str, Any]]:
    return {
        "unit": {"markers": "not integration and not slow", "parallel": True, "paths": ["tests/"]},
        "critical": {"markers": "critical and not integration", "parallel": False, "paths": ["tests/subsystems/"]},
        "parity": {
            "markers": "integration",
            "parallel": False,
            "paths": [
                "tests/subsystems/test_stage_pool_parity.py",
                "tests/subsystems/test_parallel_merge_parity.py",
                "tests/subsystems/test_tier_admission_parity.py",
                "tests/subsystems/test_execution_backend.py",
            ],
        },
        "integration": {"markers": "integration or slow", "parallel": False, "paths": ["tests/"]},
        "smoke": {"markers": "smoke", "parallel": False, "paths": ["tests/"]},
    }


def run_tests(profile: str = "unit", *, extra_args: list[str] | None = None) -> int:
    profiles = test_profiles()
    if profile not in profiles:
        raise ValueError(f"unknown profile {profile}; choose from {sorted(profiles)}")
    spec = profiles[profile]
    cmd = [sys.executable, "-m", "pytest", *spec["paths"], "-m", spec["markers"], "--tb=short", "--strict-markers"]
    if spec.get("parallel"):
        cmd.extend(["-n", "auto", "--dist", "loadfile", "-q"])
    else:
        cmd.append("-v")
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=_ROOT).returncode


def run_benchmark(*, workers: str = "1 2 4") -> int:
    script = _ROOT / "scripts" / "production_scale_audit.py"
    if not script.is_file():
        print("benchmark script missing", file=sys.stderr)
        return 1
    cmd = [sys.executable, str(script), "--workers", *workers.split()]
    return subprocess.run(cmd, cwd=_ROOT).returncode


def run_audit(*, kind: str = "pipeline", work_dir: Path | None = None, workers: int = 4) -> int:
    scripts = {
        "pipeline": ("scripts/pipeline_audit.py", []),
        "dask": ("scripts/dask_integration_report.py", []),
        "production": ("scripts/production_scale_audit.py", ["--workers", "1", "2"]),
        "library": ("scripts/library_migration_report.py", []),
        "stage0": ("scripts/stage0_production_verify.py", ["--workers", str(workers)]),
    }
    rel, extra = scripts.get(kind, scripts["pipeline"])
    script = _ROOT / rel
    if not script.is_file():
        print(f"audit script not found: {kind}", file=sys.stderr)
        return 1
    cmd = [sys.executable, str(script), *extra]
    if work_dir is not None and kind == "pipeline":
        cmd.extend(["--work-dir", str(work_dir)])
    return subprocess.run(cmd, cwd=_ROOT).returncode
