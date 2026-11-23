"""Build a Phase 3 union graph's F1 data directory (Phase 3 plan 01).

    python scripts/build_union.py --union UNION12
    python scripts/build_union.py --union INP+PENP+VLNP+VMNP --steps features extras

Writes ``data/processed/f1/<union>/``: ``graph.json``, ``nodes.parquet`` and
``edges.parquet`` (extract), ``features.parquet``, ``schema.json`` and ``coverage.csv``
(features, through ``scripts/build_features.py``), and ``extras.npz`` (extras, through
``scripts/build_node_extras.py``). Any union other than a pinned one is named by its sorted
codes joined with ``+``. Regional directories are never touched.

``--dataset f2 --calibrate`` (plan 01 task 5) prints the F2 union threshold sweep and writes
nothing: ``src.harmonize.calibrate_threshold`` with candidates 1-15 and matched type-pair density,
each candidate graph built by summing the members' F2 counts. OL's lamina types are left out of
the densities when OL is a member. A person commits the threshold in ``F2_THRESHOLDS``, as
Phase 2 does. ``--dataset f2`` then builds ``data/processed/f2/<union>/`` at it (no extras:
those align F1 root IDs). Its boundary edges are union edges absent from every member's F2
graph at that member's own frozen ``f2_threshold``; every member's F2 graph rebuilt from the
same counts must carry the frozen digest its committed F2-confirm suite records. Its ``regional``
step writes ``<union>/regional/``: the union plus every member F2 graph's nodes it lacks, each
with no edge, and their features. A scores the regional F2 suites there,
since a member at a lower threshold than the union keeps neurons the union drops.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import build_features, build_node_extras  # noqa: E402
from scripts.union_census import read_connections  # noqa: E402
from src.regions import RegionGraph  # noqa: E402
from src.union import PairCounts, boundary_flags, pair_counts, union_graph, union_members, with_isolated  # noqa: E402

STEPS = ("extract", "features", "regional", "extras")
#: Committed F2 union thresholds (plan 01 task 5). UNION12: the 2026-11-07 sweep's choice,
#: median log2 density ratio -0.217, Spearman 0.838 over 34,579 type pairs.
F2_THRESHOLDS = {"UNION12": 3}
RAW = {
    "connections": "proofread_connections_783.feather",
    "annotations": "Supplemental_file1_neuron_annotations.tsv",
    "synapses": "flywire_synapses_783.feather",
    "innervation": ("per_neuron_neuropil_count_pre_783.feather", "per_neuron_neuropil_count_post_783.feather"),
}


def f2_pair_counts(raw: Path, members) -> PairCounts:
    """The members' F2 (BANC) pair counts, read as each region's build reads them."""
    import pandas as pd

    from scripts.build_region import RAW_FILES, read_f2_connections

    meta = build_features.read_table(raw / RAW_FILES["f2"]["annotations"])
    return pair_counts(pd.concat([read_f2_connections(raw, code, meta) for code in members], ignore_index=True))


def f2_regional_thresholds(members) -> dict[str, int]:
    """Every member's frozen ``f2_threshold``, from its committed config."""
    from train.region_config import load_region_config

    thresholds = {code: load_region_config(REPO_ROOT / f"train/configs/{code.lower()}.toml").f2_threshold
                  for code in members}
    missing = sorted(code for code, t in thresholds.items() if t is None)
    if missing:
        raise SystemExit(f"F2 boundary edges need every member's f2_threshold; missing {missing}")
    return thresholds


def committed_f2_graphs(members, suites_root: Path) -> dict[str, str]:
    """Each member's frozen F2 graph digest, from its F2-confirm suite checked against its committed digest."""
    from src.boundary_suites import regional_confirm_digests
    from src.registered import read_registered
    from src.suites import Suite

    digests = regional_confirm_digests(read_registered(REPO_ROOT))
    graphs = {}
    for code in members:
        suite = Suite.read(Path(suites_root) / "f2" / code.lower() / "f2_confirm.json")
        if suite.digest() != digests[code]:
            raise SystemExit(f"{code} f2_confirm digest {suite.digest()} is not the committed {digests[code]}")
        graphs[code] = suite.graph_digest
    return graphs


def verify_f2_members(pc: PairCounts, members, thresholds: dict[str, int],
                      committed: dict[str, str]) -> dict[str, RegionGraph]:
    """Every member's F2 graph, rebuilt from the union's counts at its own threshold, must carry its
    frozen digest: otherwise the boundary flags would not describe the frozen regional graphs."""
    from src.union import CODES, _graph

    graphs = {}
    for code in members:
        if code not in committed:
            raise SystemExit(f"{code} has no frozen F2 graph digest; F2 boundary edges are undefined without it")
        column = pc.counts[:, CODES.index(code)]
        rebuilt = _graph(pc, column >= thresholds[code], column, code, "f2", thresholds[code])
        if rebuilt.digest() != committed[code]:
            raise SystemExit(f"the union's F2 counts rebuild {code} as {rebuilt.digest()[:16]}..., "
                             f"not its frozen graph {committed[code][:16]}...")
        graphs[code] = rebuilt
    return graphs


def check_f2_union(graph: RegionGraph, name: str) -> None:
    """A loaded F2 union graph must be ``name`` at its committed threshold."""
    expected = ("f2", name, F2_THRESHOLDS.get(name))
    if (graph.dataset, graph.region, graph.threshold) != expected:
        raise SystemExit(f"{graph.dataset}/{graph.region} at threshold {graph.threshold} is not "
                         f"f2/{name} at its committed threshold {expected[2]}; rerun --steps extract")


def f2_members(raw: Path, name: str, suites_root: Path):
    """The members' F2 counts, their frozen thresholds, and each member's verified F2 graph."""
    members = union_members(name)
    pc = f2_pair_counts(raw, members)
    thresholds = f2_regional_thresholds(members)
    return pc, thresholds, verify_f2_members(pc, members, thresholds, committed_f2_graphs(members, suites_root))


def f2_boundary(raw: Path, name: str, graph: RegionGraph | None, suites_root: Path):
    """The F2 counts and the union's boundary flags, after every check above.

    ``graph`` None builds the union from the counts; otherwise the loaded graph is checked.
    """
    pc, thresholds, _ = f2_members(raw, name, suites_root)
    if graph is None:
        graph = union_graph(pc, name, dataset="f2", threshold=F2_THRESHOLDS[name])
    check_f2_union(graph, name)
    return pc, graph, boundary_flags(graph, pc, thresholds)


def f2_features(graph_dir: Path, raw: Path, schema: Path) -> None:
    """The F2 union's features in the F1 union's schema, through ``scripts/build_features.py``."""
    from scripts.build_region import RAW_FILES

    files = RAW_FILES["f2"]
    build_features.main([
        "--graph", str(graph_dir), "--annotations", str(raw / files["annotations"]), "--dataset", "f2",
        "--out", str(graph_dir), "--schema", str(schema),
        "--f1-annotations", str(raw / RAW["annotations"]), "--synapses", str(raw / files["synapses"]),
        "--innervation", *(str(raw / name) for name in files["innervation"]),
    ])


def calibrate(args, members) -> int:
    """Print the F2 union threshold sweep against the built F1 union; write nothing."""
    from scripts.build_region import crosswalked_annotations, lamina_exclusion
    from scripts.build_suites import load_column
    from src.harmonize import calibrate_threshold
    from train.region_config import load_region_config

    raw = args.raw_dir
    f1_graph = RegionGraph.read(args.out_root / args.union.lower())
    f1_types = load_column(args.out_root.parent.parent / "interim" / "f1" / args.union.lower() / "f1_types.parquet",
                           "cell_type")
    f1_meta = build_features.read_table(raw / RAW["annotations"])
    pc = f2_pair_counts(raw, members)
    f2_types = crosswalked_annotations(raw, "f2", f1_meta).set_index("root_id")["cell_type"]
    exclude = ()
    if "OL" in members:
        exclude, _ = lamina_exclusion(load_region_config(REPO_ROOT / "train/configs/ol.toml"), raw, f1_meta)
    result = calibrate_threshold(f1_graph, f1_types, None, f2_types, region=args.union, dataset="f2",
                                 exclude_types=exclude,
                                 graph_at=lambda t: union_graph(pc, args.union, dataset="f2", threshold=t))
    print(result.curve.to_string(index=False))
    print(f"[build_union] chosen f2 threshold for {args.union}: {result.chosen} (spearman {result.concordance:.3f} "
          f"over {result.n_pairs} type pairs). Nothing was written; commit the threshold by hand.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--union", required=True, help="UNION12, UNION11, or sorted codes joined with +")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "data/processed/f1")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--suites-root", type=Path, default=REPO_ROOT / "data/suites",
                        help="f2: the committed regional F2-confirm suites, for the members' frozen graph digests")
    parser.add_argument("--dataset", choices=("f1", "f2"), default="f1")
    parser.add_argument("--calibrate", action="store_true", help="f2 only: print the threshold sweep")
    args = parser.parse_args(argv)
    try:
        members = union_members(args.union)
    except ValueError as exc:
        parser.error(str(exc))
    if args.calibrate:
        if args.dataset != "f2":
            parser.error("--calibrate is an f2 step")
        return calibrate(args, members)
    if args.dataset == "f2" and args.union not in F2_THRESHOLDS:
        parser.error(f"{args.union} has no committed F2 threshold; run --calibrate and commit it")
    out_root = args.out_root.parent / args.dataset if args.dataset == "f2" else args.out_root
    out = out_root / args.union.lower()
    raw = args.raw_dir

    if "extract" in args.steps:
        if args.dataset == "f2":
            pc, graph, boundary = f2_boundary(raw, args.union, None, args.suites_root)
        else:
            pc = pair_counts(read_connections(raw / RAW["connections"]))
            graph = union_graph(pc, args.union)
            boundary = boundary_flags(graph, pc)
        graph.write(out)
        print(f"[build_union] {args.dataset}/{graph.region}: {graph.n_nodes} nodes, {graph.n_edges} edges "
              f"({int(boundary.sum())} boundary), digest {graph.digest()} -> {out}", flush=True)
    if "features" in args.steps:
        if args.dataset == "f2":
            check_f2_union(RegionGraph.read(out), args.union)
            f2_features(out, raw, args.out_root / args.union.lower() / "schema.json")
        else:
            build_features.main([
                "--graph", str(out), "--annotations", str(raw / RAW["annotations"]), "--dataset", "f1",
                "--out", str(out), "--synapses", str(raw / RAW["synapses"]),
                "--innervation", *(str(raw / name) for name in RAW["innervation"]),
            ])
        print(f"[build_union] features -> {out}", flush=True)
    if "regional" in args.steps and args.dataset == "f2":
        union = RegionGraph.read(out)
        check_f2_union(union, args.union)
        _, _, members = f2_members(raw, args.union, args.suites_root)
        regional = with_isolated(union, np.concatenate([g.node_ids for g in members.values()]))
        regional.write(out / "regional")
        print(f"[build_union] f2/{union.region} regional: {regional.n_nodes - union.n_nodes} member nodes the union "
              f"lacks added with no edge, digest {regional.digest()} -> {out / 'regional'}", flush=True)
        f2_features(out / "regional", raw, args.out_root / args.union.lower() / "schema.json")
    if "extras" in args.steps and args.dataset == "f1":  # extras.npz aligns F1 root IDs
        build_node_extras.main(["--annotations", str(raw / RAW["annotations"]), str(out)])
    RegionGraph.read(out)  # digest check on whatever is now on disk
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
