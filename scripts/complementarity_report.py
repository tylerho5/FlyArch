"""Seed versus visitor within one region (Phase 3 plan 05, question 1).

    python scripts/complementarity_report.py --region VMNP --suite f1_val \
        --connections data/raw/proofread_connections_783.feather --out phase3-complementarity-vmnp.md

From ``scripts/score_union.py`` caches on the region's own suite: ``home`` mode for its
five seeds, ``regional`` mode for every visiting region's three gate seeds. On the
suite's single-masked rows it compares, size-matched:

- home 3 seeds + one visitor against home 4 seeds, for every visitor;
- home 3 seeds + the two best visitors against home 5 seeds.

A visitor is one model standing in for one extra home seed: each comparison is run once
per visitor gate seed (0, 1, 2 in turn) and the three results are averaged, so every
fusion is model-for-model the size of its home ensemble.

A visitor's weight is fit leave-one-instance-out (``train.union_scoring.loo_b0``), and
"best" is ranked by those held-out scores, which is still a choice made after seeing
every visitor, so the best-visitor rows are optimistic. Each row is also split by
whether its masked neuron is in a chosen visitor's graph (shared, an audit stratum that
is never a model input) or only in the home graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from scripts.union_census import read_connections  # noqa: E402
from src.masks import apply_mask  # noqa: E402
from src.regions import RegionGraph, build_region_graph  # noqa: E402
from src.suites import Suite  # noqa: E402
from scripts.b0_report import check_provenance  # noqa: E402
from train.union_scoring import fuse, instance_mrr, load_cache, loo_b0  # noqa: E402

GATE = (0, 1, 2)
LABELS = ("all", "shared", "home only")


def masked_roots(graph: RegionGraph, suite: Suite) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per instance: single-masked row ids, and each row's masked root id."""
    out = {}
    for inst in suite.resolve(graph):
        view = apply_mask(graph, inst)
        single = np.flatnonzero(~view.both_masked())
        edges = view.hidden_edges[single]
        node = np.where(view.is_masked[edges[:, 0]], edges[:, 0], edges[:, 1])
        out[inst.instance_id] = (single, graph.node_ids[node])
    return out


def strata(rows: dict, members: set) -> dict[str, dict]:
    """Row ids per stratum: all single-masked rows, then split by visitor membership."""
    members = np.asarray(sorted(members), dtype=np.int64)
    inside = {iid: np.isin(roots, members) for iid, (_, roots) in rows.items()}
    return {"all": {iid: ids for iid, (ids, _) in rows.items()},
            "shared": {iid: ids[inside[iid]] for iid, (ids, _) in rows.items()},
            "home only": {iid: ids[~inside[iid]] for iid, (ids, _) in rows.items()}}


def single_rows(cache: dict, rows: dict) -> dict:
    """``cache`` cut to each instance's single-masked rows, the rows the report measures."""
    out = {}
    for iid, (ids, scores) in cache.items():
        keep = np.isin(ids, rows[iid][0])
        out[iid] = (ids[keep], scores[keep])
    return out


#: What every home and visiting cache of one region's suite must share.
SHARED = ("suite_digest", "negatives_file", "region_graph", "region_features")


def score(cache: dict, split: dict) -> dict[str, float]:
    out = {}
    for label, rows in split.items():
        values = list(instance_mrr(cache, rows).values())
        out[label] = float(np.mean(values)) if values else float("nan")
    return out


def held_out_fusion(groups: dict[str, list[dict]]) -> tuple[dict, float]:
    """Each instance fused with the weights fit on the other instances, and the mean held-out MRR."""
    fold, weights = loo_b0(groups)
    fused = {iid: fuse({k: [{iid: c[iid]} for c in g] for k, g in groups.items()}, weights[iid])[iid]
             for iid in fold}
    return fused, float(np.mean(list(fold.values())))


def size_matched(home3: list[dict], visitors: dict[str, list[dict]], chosen) -> list[tuple[dict, float]]:
    """Home's three seeds plus one seed of each chosen visitor, once per gate seed: (fused, held-out MRR) each."""
    return [held_out_fusion({"home": home3, **{v: [visitors[v][k]] for v in chosen}}) for k in range(len(GATE))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", required=True)
    parser.add_argument("--suite", default="f1_val")
    parser.add_argument("--connections", required=True, type=Path, help="the dataset's raw connections, for membership")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "results/phase3/manifest.json")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--suites-root", type=Path, default=REPO_ROOT / "data/suites")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    dataset = "f1" if args.suite == "f1_val" else "f2"
    graph = RegionGraph.read(args.data_root / "processed" / dataset / args.region.lower())
    suite = Suite.read(args.suites_root / dataset / args.region.lower() / f"{args.suite}.json")

    home: dict[int, dict] = {}
    visiting: dict[str, dict[int, dict]] = defaultdict(dict)
    rows = masked_roots(graph, suite)
    entries = {}
    for path, entry in sorted(json.loads(args.manifest.read_text(encoding="utf-8")).items()):
        if entry["suite"] != args.suite or entry.get("suite_region") != args.region or entry.get("system"):
            continue  # A (plan 06) is a system, never a visiting expert
        if entry["mode"] == "home" and entry["region"] == args.region:
            entries[path] = entry
            home[entry["seed"]] = single_rows(load_cache(REPO_ROOT / path), rows)
        elif entry["mode"] == "regional" and entry["region"] != args.region and entry["seed"] in GATE:
            entries[path] = entry
            visiting[entry["region"]][entry["seed"]] = single_rows(load_cache(REPO_ROOT / path), rows)
    # Same code, suite, negatives and region inputs for every cache, and none changed since written;
    # weights are fit and visitors ranked on single-masked rows only, the rows the report measures.
    check_provenance(entries, None, REPO_ROOT, shared=SHARED)
    if {e.get("suite_digest") for e in entries.values()} != {suite.digest()}:
        raise SystemExit(f"{args.region}: the caches were scored on another {args.suite}")
    if sorted(home) != [0, 1, 2, 3, 4]:
        raise SystemExit(f"{args.region}: home caches for seeds 0-4 are required, found {sorted(home)}")
    visitors = {v: [seeds[s] for s in GATE] for v, seeds in visiting.items() if all(s in seeds for s in GATE)}
    if not visitors:
        raise SystemExit(f"{args.region}: no visitor has regional caches for all three gate seeds")

    connections = read_connections(args.connections)
    nodes = {v: set(map(int, build_region_graph(connections, region=v, dataset=dataset).node_ids)) for v in visitors}
    seeds = [home[s] for s in range(5)]

    lines = [f"# Seed versus visitor in {args.region} ({args.suite})", "",
             "Per-instance MRR on single-masked rows, averaged over instances. Each visitor is one seed, run once "
             "per gate seed and averaged, so every fusion matches its home ensemble's size. Visitor weights are fit "
             "leave-one-instance-out; best-visitor rows are optimistic (chosen after seeing every visitor).", "",
             "| Comparison | Rows | Fused | Size-matched home | Difference |", "|---|---|---|---|---|"]

    def add(name, fused_by_seed, matched_seeds, members):
        split = strata(rows, members)
        runs = [score(fused, split) for fused in fused_by_seed]
        mine = {label: float(np.mean([run[label] for run in runs])) for label in LABELS}
        matched = score(fuse({"home": matched_seeds}), split)
        for label in LABELS:
            lines.append(f"| {name} | {label} | {mine[label]:.4f} | {matched[label]:.4f} | "
                         f"{mine[label] - matched[label]:+.4f} |")

    held = {}
    for v in sorted(visitors):  # one visitor seed per fusion, each gate seed in turn
        runs = size_matched(seeds[:3], visitors, [v])
        held[v] = float(np.mean([m for _, m in runs]))
        add(f"home 3 + {v} vs home 4", [f for f, _ in runs], seeds[:4], nodes[v])
    if len(held) >= 2:
        best = sorted(held, key=held.get, reverse=True)[:2]
        runs = size_matched(seeds[:3], visitors, best)
        add(f"home 3 + {best[0]} + {best[1]} vs home 5", [f for f, _ in runs], seeds, nodes[best[0]] | nodes[best[1]])
    text = "\n".join(lines + [""])
    print(text) if args.out is None else args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
