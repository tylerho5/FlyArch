"""File hashing shared by the scoring and report scripts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()
