"""Export completed M1 run directories into the tracked milestone CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train.milestones import export_experts, export_milestones  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        metavar="RUN_DIR",
        help="completed run directories under the runs root",
    )
    parser.add_argument("--out", required=True, type=Path, help="CSV path to replace")
    parser.add_argument(
        "--registry", action="store_true",
        help="validate five-seed regional cohorts and export results/experts.csv format",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        (export_experts if args.registry else export_milestones)(args.run_dirs, args.out)
    except (OSError, ValueError) as exc:
        print(f"milestone export failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
