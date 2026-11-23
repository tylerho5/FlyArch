"""Run-ID helpers: the sweep key and the test-run suffix; outside ``train.py``'s ``_CODE_FILES``."""

from __future__ import annotations

import re

_SEED_SEGMENT = re.compile(r"-s\d{3,}-")

TEST_RUN_SUFFIX = "-test"


def sweep_key(run_id: str) -> str:
    """The run ID without its seed segment, shared by every seed of one configuration."""
    return _SEED_SEGMENT.sub("-", run_id)


def is_test_run(run_id: str) -> bool:
    return run_id.endswith(TEST_RUN_SUFFIX)
