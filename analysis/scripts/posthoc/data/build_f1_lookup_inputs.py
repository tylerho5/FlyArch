"""Build the FlyWire inputs of the matched-neuron lookup (match_transfer.py) from the public v783 release.

Outputs, in --out (default analysis/out):
  f1_ids_sorted.npy      sorted int64 root ids of the 139,255 v783 proofread neurons
                         (proofread_root_ids_783.npy, the release's own list)
  f1_pairs_<region>.npz  keys   = i * 2**18 + j (int64, sorted), i/j = positions in f1_ids_sorted of pre/post
                         totals = synapses from pre to post summed over the region's neuropils (src/regions.py),
                                  no threshold (match_transfer.py thresholds at 5 itself for its binary variant)
Self-check (always run): for each region, the pairs with total >= the region's FlyWire threshold must be exactly the
edges, with the same weights, of the processed FlyWire region graph built by the pipeline (processed/f1/<r>/edges.parquet).
SHA-256 of every output is printed and written to f1_lookup_inputs.sha256.

Usage: python build_f1_lookup_inputs.py --raw <dir with proofread_connections_783.feather and
       proofread_root_ids_783.npy> --processed <data/processed> [--out <dir>]   (--raw/--processed default to the repo's data/)
"""
import argparse, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import REPO  # noqa: E402
sys.path.insert(0, str(REPO))
from src.regions import REGIONS, RegionGraph, region_labels  # noqa: E402


SHIFT = np.int64(1 << 18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--processed", type=Path, default=REPO / "data" / "processed")
    ap.add_argument("--out", type=Path, default=REPO / "analysis" / "out")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    ids = np.sort(np.load(a.raw / "proofread_root_ids_783.npy").astype(np.int64))
    assert len(ids) < SHIFT and (np.diff(ids) > 0).all()
    np.save(a.out / "f1_ids_sorted.npy", ids)

    con = pd.read_feather(a.raw / "proofread_connections_783.feather",
                          columns=["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"])
    written = ["f1_ids_sorted.npy"]
    for code in REGIONS:
        c = code.lower()
        sub = con[con["neuropil"].isin(region_labels(code))]
        tot = sub.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False)["syn_count"].sum()
        pre = tot.index.get_level_values(0).to_numpy(np.int64)
        post = tot.index.get_level_values(1).to_numpy(np.int64)
        pi, qi = np.searchsorted(ids, pre), np.searchsorted(ids, post)
        ok = (pi < len(ids)) & (qi < len(ids))
        ok[ok] &= (ids[pi[ok]] == pre[ok]) & (ids[qi[ok]] == post[ok])
        keys = pi[ok] * SHIFT + qi[ok]
        vals = tot.to_numpy()[ok]
        order = np.argsort(keys, kind="stable")
        keys, vals = keys[order], vals[order].astype(np.int64)
        np.savez(a.out / f"f1_pairs_{c}.npz", keys=keys, totals=vals)
        written.append(f"f1_pairs_{c}.npz")

        # self-check against the processed FlyWire region graph
        e = pd.read_parquet(a.processed / "f1" / c / "edges.parquet")
        node_ids = RegionGraph.read(a.processed / "f1" / c).node_ids.astype(np.int64)
        src, dst, w = (e[k].to_numpy() for k in _edge_cols(e))
        src, dst = node_ids[src], node_ids[dst]   # edges.parquet stores node indices
        thr = int(w.min())
        gk = np.searchsorted(ids, src.astype(np.int64)) * SHIFT + np.searchsorted(ids, dst.astype(np.int64))
        sel = vals >= thr
        same_set = np.array_equal(np.sort(gk), keys[sel])
        same_w = same_set and np.array_equal(w[np.argsort(gk)].astype(np.int64), vals[sel])
        print(f"{c}: {len(keys):,} pairs; >= {thr}: {sel.sum():,} vs graph {len(gk):,}; "
              f"edge set {'OK' if same_set else 'MISMATCH'}, weights {'OK' if same_w else 'MISMATCH'}")
        if not same_w:
            sys.exit(f"self-check failed for {c}")

    with open(a.out / "f1_lookup_inputs.sha256", "w", encoding="utf-8", newline="\n") as fh:
        for f in written:
            h = hashlib.sha256((a.out / f).read_bytes()).hexdigest()
            fh.write(f"{h}  {f}\n")
            print(h, f)


def _edge_cols(e):
    for s, d, w in (("pre", "post", "weight"), ("src", "dst", "weight"), ("pre_root_id", "post_root_id", "syn_count"),
                    ("source", "target", "weight")):
        if {s, d, w} <= set(e.columns):
            return s, d, w
    raise SystemExit(f"unknown edge columns: {list(e.columns)}")


if __name__ == "__main__":
    main()
