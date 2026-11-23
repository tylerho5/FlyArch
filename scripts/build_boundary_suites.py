"""Build one Phase 3 boundary suite on a union graph (Phase 3 plan 02).

    python scripts/prepare_suite_inputs.py --graph data/processed/f1/union12 \
        --annotations data/raw/Supplemental_file1_neuron_annotations.tsv \
        --synapses data/raw/flywire_synapses_783.feather --out data/interim/f1/union12
    python scripts/build_boundary_suites.py --union UNION12 --suite phase3_boundary_f1_val

Writes under ``data/suites/<dataset>/<union>/``: ``<suite>.json`` (the masks),
``<suite>.negatives.json`` (one negative digest per instance), ``<suite>.selector.json``
(the metric rows, with the block stratum) and ``excluded_types.json`` (OL's lamina
types, which no instance masks). Prints the digests.
An instance with no metric row aborts the build before anything is written.

F2 suites (``phase3_boundary_f2_dev``, then ``_confirm``) are built on the F2 union
(``build_union.py --dataset f2``). Their boundary edges are union edges absent from every
member's F2 graph at its own ``f2_threshold``. They draw no T instance and never mask a root,
or a neuron of a type, that a regional F2-confirm suite masks or draws (read by committed
digest), and F2-confirm also avoids every F2-dev root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_features import read_table  # noqa: E402
from scripts.build_region import lamina_exclusion  # noqa: E402
from scripts.build_suites import load_centroids, load_column  # noqa: E402
from scripts.union_census import read_connections  # noqa: E402
from scripts.build_union import f2_boundary  # noqa: E402
from src.registered import read_registered  # noqa: E402
from src.boundary_suites import (  # noqa: E402
    BLOCK, BoundarySuite, SUITES, boundary_selector, build_boundary_suite, regional_confirm_digests,
    regional_confirm_exclusions, selector_digest,
)
from src.regions import RegionGraph  # noqa: E402
from src.suites import suite_negatives  # noqa: E402
from src.union import boundary_flags, pair_counts, support_within, union_members  # noqa: E402
from train.region_config import load_region_config  # noqa: E402

RAW = {"connections": "proofread_connections_783.feather",
       "annotations": "Supplemental_file1_neuron_annotations.tsv"}


def log(message: str) -> None:
    print(f"[build_boundary_suites] {message}", flush=True)


def committed_phase3_digest(name: str, registered: dict) -> str:
    """A Phase 3 boundary suite's registered digest (``results/registered.json``); refuses a missing one."""
    digest = registered.get("boundary_suites", {}).get(name, {}).get("suite")
    if not digest:
        raise SystemExit(f"{name}'s digest is not registered in results/registered.json")
    return digest


def pinned_dev(path: Path, graph: RegionGraph, registered: dict) -> BoundarySuite:
    """F2-dev as committed: its digest must be the pinned one and it must be on this graph."""
    dev = BoundarySuite.read(path)
    pinned = committed_phase3_digest("phase3_boundary_f2_dev", registered)
    if dev.digest() != pinned or dev.graph_digest != graph.digest():
        raise SystemExit(f"{path} is not the committed phase3_boundary_f2_dev on {graph.region}; "
                         f"confirm's exclusions must come from the pinned dev suite")
    return dev


def f2_exclusions(args, code: str, graph: RegionGraph) -> dict:
    """``build_boundary_suite``'s F2 arguments: what the regional F2-confirm suites (and, for
    F2-confirm, the pinned F2-dev) mask or draw."""
    registered = read_registered(REPO_ROOT)
    digests = regional_confirm_digests(registered)
    roots, types = regional_confirm_exclusions(args.data_root / "suites", digests)
    if args.suite == "phase3_boundary_f2_confirm":
        dev = pinned_dev(args.data_root / "suites" / "f2" / code / "phase3_boundary_f2_dev.json", graph, registered)
        for inst in dev.instances:
            roots |= set(map(int, inst.masked_root_ids))
    return {"exclude_roots": roots, "exclude_types": types}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--union", required=True)
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data",
                        help="holds processed/ and interim/ (graph and suite inputs) and suites/ (output)")
    args = parser.parse_args(argv)
    union_members(args.union)
    dataset = SUITES[args.suite][0]
    code = args.union.lower()
    graph = RegionGraph.read(args.data_root / "processed" / dataset / code)
    interim = args.data_root / "interim" / dataset / code

    if dataset == "f2":  # checks the graph's threshold and every member's frozen F2 graph first
        pc, _, boundary = f2_boundary(args.raw_dir, args.union, graph, args.data_root / "suites")
    else:
        pc = pair_counts(read_connections(args.raw_dir / RAW["connections"]))
        boundary = boundary_flags(graph, pc)
    block = boundary & support_within(graph, pc, BLOCK)
    del pc
    log(f"{graph.region}: {int(boundary.sum())} boundary edges, {int(block.sum())} in the block")

    annotations = read_table(args.raw_dir / RAW["annotations"])
    lamina, exclusion = lamina_exclusion(load_region_config(REPO_ROOT / "train/configs/ol.toml"), args.raw_dir, annotations)
    types = load_column(interim / f"{dataset}_types.parquet", "cell_type")
    centroids = load_centroids(interim / f"{dataset}_centroids.parquet", graph)
    extra = {}
    if dataset == "f2":
        extra = f2_exclusions(args, code, graph)
        log(f"F2 exclusions: {len(extra['exclude_roots'])} roots, {len(extra['exclude_types'])} types")
    suite = build_boundary_suite(graph, boundary, name=args.suite, types=types, centroids=centroids,
                                 mask_exclude_types=lamina, **extra)
    selector = boundary_selector(graph, suite, boundary, block)  # raises before anything is written
    log("masks and selector built; sampling negatives")
    negatives = suite_negatives(graph, suite)

    out = args.data_root / "suites" / dataset / code
    out.mkdir(parents=True, exist_ok=True)
    suite.write(out / f"{args.suite}.json")
    spec = next(iter(negatives.values())).spec
    (out / f"{args.suite}.negatives.json").write_text(json.dumps({
        "name": suite.name, "spec": {"k": spec.k, "alpha": spec.alpha, "master_seed": spec.master_seed},
        "digests": {iid: sample.digest() for iid, sample in negatives.items()},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / f"{args.suite}.selector.json").write_text(json.dumps(selector, indent=2, sort_keys=True) + "\n",
                                                     encoding="utf-8")
    (out / "excluded_types.json").write_text(json.dumps(exclusion, indent=2) + "\n", encoding="utf-8")

    print(f"\n--- {args.suite} on {graph.region} ---")
    print(f"suite {suite.digest()}")
    print(f"selector {selector_digest(selector)}")
    for iid, sample in negatives.items():
        row = selector["instances"][iid]
        print(f"{iid}: negatives {sample.digest()}, {row['n_rows']} metric rows "
              f"({len(row['block_rows'])} block) of {row['n_positives']} hidden edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
