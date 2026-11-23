"""The shared Phase 2 training cohort identity."""

from __future__ import annotations

import hashlib
import json
from typing import Any

_COHORT_FIELDS = (
    "config", "code_digest", "graph_digest", "feature_digest", "schema_digest",
    "suite_digest",
)


def cohort_digest(manifest: dict[str, Any]) -> str:
    """Hash the scientific run inputs used by the registry, independent of seed."""
    missing = [name for name in _COHORT_FIELDS if name not in manifest]
    if missing:
        raise ValueError(f"run manifest is missing cohort field(s): {missing}")
    payload = {name: manifest[name] for name in _COHORT_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
