"""The registered suite digests and Phase 2 verdicts, as committed in ``results/registered.json``."""

from __future__ import annotations

import json
from pathlib import Path

REGISTERED = Path("results/registered.json")


def read_registered(repo_root: Path) -> dict:
    """``{"suites": {region: {split: digest}}, "boundary_suites": {name: {"suite", "selector"}},
    "phase2_verdicts": {region: "PASS" | "FAIL"}}``."""
    return json.loads((Path(repo_root) / REGISTERED).read_text(encoding="utf-8"))
