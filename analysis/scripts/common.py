"""Shared command-line arguments for the analysis scripts.

    --repo-root     checkout holding src/, scripts/, data/suites/, results/experts.csv, results/phase3/ (default: this repo)
    --data-root     connectome data: raw/, interim/, processed/, runs/ (default: <repo-root>/data)
    --results-root  results/ tree holding <region>/scored/*.parquet (default: <repo-root>/results)
    --out           directory the scripts write their CSVs and intermediates to (default: analysis/out)
"""
import argparse
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
REPO = ANALYSIS.parent


def parse_args(doc=None, extra=None):
    ap = argparse.ArgumentParser(description=(doc or "").strip().split("\n")[0] or None)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--results-root", type=Path)
    ap.add_argument("--out", type=Path, default=ANALYSIS / "out")
    if extra:
        extra(ap)
    a = ap.parse_args()
    a.repo_root = a.repo_root.resolve()
    a.data_root = (a.data_root or a.repo_root / "data").resolve()
    a.results_root = (a.results_root or a.repo_root / "results").resolve()
    a.suites = a.repo_root / "data" / "suites"
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.repo_root))
    return a
