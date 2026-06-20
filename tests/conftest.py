"""Framework test configuration."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _framework_test_env() -> None:
    os.environ.setdefault("INSTANT_MERGE_HW_PROBE", "0")
    os.environ.setdefault("INSTANT_PIPELINE_PUSHGATEWAY", "off")
