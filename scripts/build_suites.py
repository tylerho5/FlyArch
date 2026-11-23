"""Freeze one named mask suite for a region graph.

    python3.13 scripts/build_suites.py --graph data/processed/f1/<code> \
        --name f1_val --types data/interim/f1/<code>/f1_types.parquet \
        --centroids data/interim/f1/<code>/f1_centroids.parquet --out data/suites/f1/<code>

For CX, ``<code>`` is ``cx``; ``scripts/build_region.py`` runs this for every
suite of a region and dataset.

Writes ``<name>.json`` (the suite) and ``<name>.negatives.json`` (one digest
per instance) under ``--out``, then prints a paste-ready digest block
(or an exploratory block for ``f2_side``). That suite
requires ``--sides`` instead of ``--types`` and ``--centroids``. ``f2_dev`` and
``f2_confirm`` also require ``--f1-graph``, ``--f1-types`` and ``--f1-val``: their
T instances draw only the cell types the region's expert trains on.
Thin wrapper over ``src.suites`` -- inherits its
numpy/pandas/pyarrow deps only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.regions import RegionGraph  # noqa: E402
from src.suites import (  # noqa: E402
    F1_POOL_SUITES, SIDE_SUITES, SUITE_SIZES, Suite, build_suite, f1_train_types, suite_negatives,
)


def log(message: str) -> None:
    print(f"[build_suites] {message}", flush=True)


def load_column(path: Path, column: str) -> pd.Series:
    """A root_id/<column> parquet -> that column, indexed by root id."""
    return pd.read_parquet(path).set_index("root_id")[column]


def load_centroids(path: Path, graph: RegionGraph):
    """A root_id/x/y/z parquet -> (N, 3) float array in graph.node_ids order."""
    df = pd.read_parquet(path).set_index("root_id")[["x", "y", "z"]]
    if not df.index.is_unique:
        raise ValueError("centroids file has duplicate root ids")
    df = df.reindex(graph.node_ids)
    missing = df.index[df.isna().any(axis=1)]
    if len(missing):
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"centroids file is missing root ids: {list(missing[:5])}{suffix}")
    return df.to_numpy(dtype=float)


def draw_pool(graph_dir: Path, types_path: Path, val_path: Path) -> tuple:
    """The F1 cell types the region's expert trains on (``f1_train_types``)."""
    return f1_train_types(RegionGraph.read(graph_dir), load_column(types_path, "cell_type"), Suite.read(val_path))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph", required=True, type=Path, help="RegionGraph.read directory")
    parser.add_argument("--name", required=True, choices=sorted(SUITE_SIZES))
    parser.add_argument("--types", type=Path, help="root_id/cell_type parquet (required for R/T/S suites)")
    parser.add_argument(
        "--centroids", type=Path,
        help="root_id/x/y/z parquet, keyed by root_id",
    )
    parser.add_argument(
        "--exclude-types", type=Path,
        help="excluded_types.json: its types are never masked (needs --types)",
    )
    parser.add_argument("--f1-graph", type=Path, help="the region's F1 graph (f2_dev/f2_confirm T draw pool)")
    parser.add_argument("--f1-types", type=Path, help="root_id/cell_type parquet for --f1-graph")
    parser.add_argument("--f1-val", type=Path, help="the region's f1_val.json, whose masks F1-train drops")
    parser.add_argument("--sides", type=Path, help="root_id/side parquet, L/R/null (required for f2_side)")
    parser.add_argument(
        "--out", required=True, type=Path,
        help="output directory, e.g. data/suites/<dataset>/<region>/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    side_suite = args.name in SIDE_SUITES
    required = ("sides",) if side_suite else ("types", "centroids")
    beside = args.graph / "excluded_types.json"
    if args.exclude_types is None and beside.exists():
        parser.error(f"{beside} exists, so this region's suites must pass --exclude-types {beside}")
    if args.exclude_types is not None:
        required += ("types",)
    if args.name in F1_POOL_SUITES:
        required += ("f1_graph", "f1_types", "f1_val")
    for field in required:
        if getattr(args, field) is None:
            parser.error(f"--{field.replace('_', '-')} is required for {args.name}")

    graph = RegionGraph.read(args.graph)
    types = load_column(args.types, "cell_type") if args.types is not None else None
    centroids = load_centroids(args.centroids, graph) if args.centroids is not None else None
    sides = load_column(args.sides, "side") if args.sides is not None else None

    draw_types = draw_pool(args.f1_graph, args.f1_types, args.f1_val) if args.f1_graph is not None else None
    exclude = json.loads(args.exclude_types.read_text(encoding="utf-8"))["types"] if args.exclude_types else ()
    suite = build_suite(
        graph, name=args.name, types=types, centroids=centroids, soma_sides=sides,
        mask_exclude_types=exclude, draw_types=draw_types,
    )
    negatives = suite_negatives(graph, suite)

    args.out.mkdir(parents=True, exist_ok=True)
    suite_path = args.out / f"{args.name}.json"
    negatives_path = args.out / f"{args.name}.negatives.json"

    suite.write(suite_path)
    spec = next(iter(negatives.values())).spec
    negatives_doc = {
        "name": suite.name,
        "spec": {"k": spec.k, "alpha": spec.alpha, "master_seed": spec.master_seed},
        "digests": {iid: sample.digest() for iid, sample in negatives.items()},
    }
    negatives_path.write_text(
        json.dumps(negatives_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(f"wrote {suite_path}")
    log(f"wrote {negatives_path}")

    heading = "Exploratory side-suite digests" if side_suite else "Suite digests"
    print(f"\n--- {heading} ---")
    for iid, sample in negatives.items():
        print(f"{iid}: {sample.digest()}")
    print(f"suite ({suite.name}): {suite.digest()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
