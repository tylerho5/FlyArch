"""Same CLI as scripts/wiring_null.py (run / analyse, same arguments and outputs), but S2 is
computed in row blocks (src/wiring_null_chunked.py)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import wiring_null as frozen  # noqa: E402
from src import wiring_null as wn  # noqa: E402
from src.wiring_null_chunked import stats_chunked  # noqa: E402


def main(argv=None) -> int:
    orig = wn.stats
    wn.stats = stats_chunked  # run_draw / real_row resolve `stats` in this module at call time
    try:
        return frozen.main(argv)
    finally:
        wn.stats = orig


if __name__ == "__main__":
    raise SystemExit(main())
