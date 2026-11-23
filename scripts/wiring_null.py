"""CLI for the wiring-only rewiring null; logic in src/wiring_null.py.

  run      one region/split, draws [start, end): writes real row (if start == 0), N0 and N1 draws
  analyse  F1 and F2 CSVs -> results table (z, p, saturation, Holm, verdict, replication)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.scoring_common import resolve_data_root  # noqa: E402
from src import wiring_null as wn  # noqa: E402


def cmd_run(a) -> int:
    root = resolve_data_root(a.data_root)
    src, dst, w, n, n_loops = wn.load_graph(root, a.split, a.region)
    seed = wn.derive_seed(a.seed, a.region, a.split)  # one seed per (region, dataset)
    meta = {"region": a.region, "split": a.split, "master_seed": a.seed, "seed": seed,
            "n_loops_dropped": n_loops, "n_nodes": n, "n_edges": len(src)}
    seed_sets = wn.make_seed_sets(n, np.random.SeedSequence(seed, spawn_key=(0,)))
    out = Path(a.out)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    have = {(r["kind"], r["perm"]) for r in rows}
    if a.start == 0 and ("real", -1) not in have:
        rows.append({**meta, **wn.real_row(src, dst, w, n, seed_sets)})
    for idx in range(a.start, min(a.end, a.n_perm)):
        if ("N0", idx) in have and ("N1", idx) in have:
            continue
        try:
            n0, n1 = wn.run_draw(src, dst, w, n, seed_sets, seed, idx)
        except RuntimeError as e:  # null unreachable: recorded, not relaxed
            rows.append({**meta, "kind": "N0_unreachable", "perm": idx})
            print(e, file=sys.stderr)
            break
        rows += [{**meta, **n0}, {**meta, **n1}]
        if idx % 10 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    return 0


def cmd_analyse(a) -> int:
    def load(paths):
        return pd.concat([pd.read_csv(p) for p in paths]).drop_duplicates(["region", "kind", "perm"])

    wn.decide(load(a.f1), load(a.f2), expected_b=a.expected_b).to_csv(a.out, index=False)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--region", required=True)
    r.add_argument("--split", required=True, choices=["f1", "f2"])
    r.add_argument("--n-perm", type=int, default=10_000)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--end", type=int, default=10_000)
    r.add_argument("--seed", type=int, default=20261121)
    r.add_argument("--data-root", type=Path)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("analyse")
    s.add_argument("--f1", required=True, nargs="+")
    s.add_argument("--f2", required=True, nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--expected-b", type=int, default=10_000,
                   help="draws required per null; a region with fewer is flagged 'incomplete'")
    s.set_defaults(fn=cmd_analyse)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
