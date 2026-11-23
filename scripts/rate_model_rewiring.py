"""CLI for the rate-model rewiring test; logic in src/rate_model.py.

  run      one region/split, draws [start, end): one CSV row per draw (resumable, shard-independent)
  analyse  all shards -> per (region, dataset) results with the full decision rule and labels
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.scoring_common import resolve_data_root  # noqa: E402
from src import rate_model as rm  # noqa: E402


def _commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def check_frozen(sha):
    """Refuse to run unless HEAD is exactly the registered commit and no tracked file is modified."""
    head = _commit()
    dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if head != sha or dirty:
        raise SystemExit(f"refusing to run: HEAD {head} != --frozen-sha {sha}" if head != sha
                         else "refusing to run: tracked files modified:\n" + dirty)


def _plan(region, split):
    """Registered (B, B_sens) per cell; the extension cells are fixed here so a CLI slip cannot change them."""
    if region in rm.EXT_REGIONS:
        return rm.B_EXT, rm.B_SENS_EXT
    return rm.B_MAIN, rm.B_SENS


def cmd_run(a) -> int:
    check_frozen(a.frozen_sha)
    b, bs = _plan(a.region, a.split)
    if a.split == "f3" and a.region != "lx":
        raise SystemExit("f3 is registered only for lx (extension Part A)")
    if a.n_perm is None:
        a.n_perm = b
    if a.b_sens is None:
        a.b_sens = bs
    is_ext = a.region in rm.EXT_REGIONS or a.split == "f3"   # original cells keep the original CLI behaviour
    if is_ext and (a.n_perm, a.b_sens) != (b, bs):
        raise SystemExit(f"{a.region}/{a.split} is registered with B={b}, B_sens={bs}; got {a.n_perm}, {a.b_sens}")
    root = resolve_data_root(a.data_root)
    g = rm.load_graph(root, a.split, a.region)
    if not rm.eligible(g):
        print(f"{a.region}/{a.split} excluded (A={g.meta['n_A']}, E={g.meta['n_E']}); nothing to run")
        return 0
    flips = rm.flip_sets(g, a.seed, a.region, a.split)
    out = Path(a.out)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    have = {int(r["draw"]) for r in rows}
    commit = _commit()
    last_save = len(rows)
    for i in range(a.start, min(a.end, a.n_perm)):
        if i in have:
            continue
        row = rm.run_draw(g, a.region, a.split, a.seed, i, flips, commit, b_sens=a.b_sens)
        rows.append(row)
        if row["cap_hit"]:
            print(f"null unreachable: draw {i} hit the attempt cap ({row['accepted']} accepted)", file=sys.stderr)
            break
        if len(rows) - last_save >= 10:
            pd.DataFrame(rows, columns=rm.DRAW_COLUMNS).to_csv(out, index=False)
            last_save = len(rows)
    pd.DataFrame(rows, columns=rm.DRAW_COLUMNS).to_csv(out, index=False)
    return 0


def cmd_analyse(a) -> int:
    check_frozen(a.frozen_sha)
    root = resolve_data_root(a.data_root)
    df = pd.concat([pd.read_csv(p) for p in a.shards], ignore_index=True)
    cells, meta, problems = {}, {}, []
    for region in rm.REGIONS:
        for split in rm.SPLITS:
            g = rm.load_graph(root, split, region)
            real = rm.real_result(g, region, split, a.seed)
            d = df[(df.region == region) & (df.split == split)]
            c = rm.cell_from_draws(d, real, rm.eligible(g), b=a.n_perm)
            c.update({k: v for k, v in g.meta.items()}, real_dense_fallback=real["dense"], master_seed=a.seed,
                     analysis_commit=_commit())
            cells[(region, split)] = c
    res = rm.decide(cells)
    res.to_csv(a.out, index=False)
    for _, r in res.iterrows():
        if r.status != "defined":
            problems.append(f"{r.region}/{r.split}: {r.status} ({r.reason})")
        elif not r.sens_complete:
            problems.append(f"{r.region}/{r.split}: sensitivity incomplete (draws 0..{rm.B_SENS - 1})")
    print(res[["region", "split", "status", "z", "p", "d", "p_holm", "verdict", "sign_label", "amp_label"]]
          .to_string(index=False))
    print("PROBLEMS (undefined / incomplete / unreachable / excluded):")
    print("\n".join(problems) or "none")
    return 0


def cmd_analyse_ext(a) -> int:
    check_frozen(a.frozen_sha)
    root = resolve_data_root(a.data_root)
    df = pd.concat([pd.read_csv(p) for p in a.shards], ignore_index=True)
    plan = [("lx", "f3", rm.B_MAIN, rm.B_SENS)] + [(r, sp, rm.B_EXT, rm.B_SENS_EXT) for r in rm.EXT_REGIONS for sp in rm.SPLITS]
    cells, problems = {}, []
    for region, split, b, bs in plan:
        g = rm.load_graph(root, split, region)
        real = rm.real_result(g, region, split, a.seed)
        d = df[(df.region == region) & (df.split == split)]
        c = rm.cell_from_draws(d, real, rm.eligible(g), b=b, bs=bs)
        c.update({k: v for k, v in g.meta.items()}, real_dense_fallback=real["dense"], master_seed=a.seed,
                 analysis_commit=_commit(), B=b, B_sens=bs)
        c["p_flip"], c["p_amp"] = rm.sens_p_ext(d, real, bs)
        cells[(region, split)] = c
    res = rm.decide_ext(cells)
    res.to_csv(a.out, index=False)
    for _, r in res.iterrows():
        if r.status != "defined":
            problems.append(f"{r.region}/{r.split}: {r.status} ({r.reason})")
        elif not r.sens_complete:
            problems.append(f"{r.region}/{r.split}: sensitivity incomplete")
    print(res[["family", "region", "split", "status", "z", "p", "d", "p_holm", "verdict", "sign_label", "amp_label"]]
          .to_string(index=False))
    print("PROBLEMS:")
    print("\n".join(problems) or "none")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--region", required=True, choices=rm.ALL_REGIONS)
    r.add_argument("--split", required=True, choices=rm.ALL_SPLITS)
    r.add_argument("--b-sens", type=int, default=None,
                   help="draws 0..b_sens-1 also run the sensitivity simulations (extension LH/MB/PENP: 500)")
    r.add_argument("--n-perm", type=int, default=None, help="defaults to the registered B for the cell")
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--end", type=int, default=rm.B_MAIN)
    r.add_argument("--seed", type=int, default=rm.DEFAULT_SEED)
    r.add_argument("--data-root", type=Path)
    r.add_argument("--out", required=True)
    r.add_argument("--frozen-sha", required=True, help="the registered merge commit; must equal HEAD")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("analyse")
    s.add_argument("--shards", required=True, nargs="+")
    s.add_argument("--n-perm", type=int, default=rm.B_MAIN)
    s.add_argument("--seed", type=int, default=rm.DEFAULT_SEED)
    s.add_argument("--data-root", type=Path)
    s.add_argument("--out", required=True)
    s.add_argument("--frozen-sha", required=True, help="the registered merge commit; must equal HEAD")
    s.set_defaults(fn=cmd_analyse)
    x = sub.add_parser("analyse-ext", help="extension: male CNS LX (B=5000) and LH/MB/PENP (B=2000)")
    x.add_argument("--shards", required=True, nargs="+")
    x.add_argument("--seed", type=int, default=rm.DEFAULT_SEED)
    x.add_argument("--data-root", type=Path)
    x.add_argument("--out", required=True)
    x.add_argument("--frozen-sha", required=True, help="the registered merge commit; must equal HEAD")
    x.set_defaults(fn=cmd_analyse_ext)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
