"""Each region expert's geometry for a union's neurons (Phase 3 plan 03).

    python scripts/expert_geometry.py --union UNION12
    python scripts/expert_geometry.py --union UNION12 --dataset f2

A union table's geometry is computed over the union's labels and z-scored union-wide,
a frame no region expert trained on. For each member region this streams the union's
neurons' own synapse ends on that region's labels and z-scores them with the region
graph's node statistics (``train.union_scoring.expert_geometry``), so a region neuron
reads exactly the geometry in the region's ``features.parquet``.

Writes ``data/processed/<dataset>/<union>/expert_geometry/<region>.parquet``: ``root_id``
and the six geometry columns, one row per neuron with a synapse on the region's labels.
``--dataset`` picks whose raw tables, union and region graphs are read: F2's region graph
is rebuilt from the BANC edgelist at the region's frozen ``f2_threshold`` and must carry
its committed F2-dev graph digest. Where ``data/processed/<dataset>/<region>/`` exists, the
region's own rows are checked against its ``features.parquet`` before anything is written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.build_features import stream_synapse_geometry  # noqa: E402
from scripts.union_census import read_connections  # noqa: E402
from src.feature_io import read_feature_artifacts  # noqa: E402
from src.regions import RegionGraph, build_region_graph, region_labels  # noqa: E402
from src.union import union_members  # noqa: E402
from train.region_config import config_path, load_region_config  # noqa: E402
from train.union_scoring import GEOMETRY, expert_geometry  # noqa: E402


SUITE = "f2_dev"


def region_nodes(code: str, dataset: str, raw: Path, connections, data_root: Path) -> np.ndarray:
    """One region's graph nodes, built the way that dataset's regional build builds them.

    F2's frame is the frozen regional graph, so the rebuild must carry the region's
    committed F2-dev graph digest; a different frame would z-score the geometry
    over other neurons.
    """
    if dataset == "f1":
        return build_region_graph(connections, region=code, dataset="f1").node_ids
    from scripts.build_features import read_table
    from scripts.build_region import RAW_FILES, read_f2_connections
    from src.suites import Suite

    threshold = load_region_config(config_path(code)).f2_threshold
    if threshold is None:
        raise SystemExit(f"{code} has no committed f2_threshold")
    meta = read_table(raw / RAW_FILES["f2"]["annotations"])
    graph = build_region_graph(read_f2_connections(raw, code, meta), region=code, dataset="f2",
                               threshold=threshold)
    suite = Suite.read(data_root / "suites" / "f2" / code.lower() / f"{SUITE}.json")
    if graph.digest() != suite.graph_digest:
        raise SystemExit(f"{code}: the rebuilt F2 graph {graph.digest()[:16]}... is not its committed "
                         f"{suite.graph_digest[:16]}...; its geometry frame would differ")
    return graph.node_ids


def check_against_region(values: np.ndarray, present: np.ndarray, union: RegionGraph, folder: Path) -> float:
    """Largest difference between the union rows and the region's own table, on its nodes."""
    graph = RegionGraph.read(folder)
    table, schema = read_feature_artifacts(folder, graph)
    rows = pd.Index(union.node_ids).get_indexer(graph.node_ids)
    cols = [schema.column_names.index(c) for c in GEOMETRY]
    if not np.array_equal(present[rows], table.geometry_present):
        raise SystemExit(f"{folder}: geometry_present differs from the region's own table")
    return float(np.abs(values[rows] - table.values[:, cols]).max())


def load(union_dir: Path, code: str, node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One region's expert-frame geometry, aligned to the union's nodes (zeros where absent)."""
    path = union_dir / "expert_geometry" / f"{code.lower()}.parquet"
    if not path.is_file():
        raise SystemExit(f"{path} is missing: run scripts/expert_geometry.py --regions {code}")
    aligned = pd.read_parquet(path).set_index("root_id").reindex(node_ids)[list(GEOMETRY)]
    present = aligned.notna().all(axis=1).to_numpy()
    return np.nan_to_num(aligned.to_numpy(float)), present


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--union", default="UNION12")
    parser.add_argument("--dataset", choices=("f1", "f2"), default="f1")
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "data/raw",
                        help="the dataset's raw files; F2 reads the BANC tables from here")
    parser.add_argument("--synapses", type=Path, help="override the dataset's synapse table")
    parser.add_argument("--connections", type=Path, help="override F1's connections table")
    parser.add_argument("--regions", nargs="+", help="member regions (default all)")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args(argv)
    if args.dataset == "f2" and args.connections:
        parser.error("--connections is F1's; F2 reads each region's rows from --raw")
    from scripts.build_region import RAW_FILES

    union_dir = args.data_root / "processed" / args.dataset / args.union.lower()
    union = RegionGraph.read(union_dir)
    members = union_members(args.union)
    regions = args.regions or list(members)
    if set(regions) - set(members):
        raise SystemExit(f"{sorted(set(regions) - set(members))} are not members of {args.union}")
    synapses = args.synapses or args.raw / RAW_FILES[args.dataset]["synapses"]
    connections = read_connections(args.connections or args.raw / RAW_FILES["f1"]["connections"]) \
        if args.dataset == "f1" else None
    out = union_dir / "expert_geometry"
    out.mkdir(exist_ok=True)
    for code in regions:
        # ponytail: one pass over the synapse table per region; one pass aggregating per region if this is slow.
        region_ids = region_nodes(code, args.dataset, args.raw, connections, args.data_root)
        raw = stream_synapse_geometry(synapses, union, labels=region_labels(code))
        values, present = expert_geometry(raw, union.node_ids, region_ids)
        folder = args.data_root / "processed" / args.dataset / code.lower()
        if (folder / "features.parquet").is_file():
            diff = check_against_region(values, present, union, folder)
            if diff > 1e-6:
                raise SystemExit(f"{code}: union rows differ from the region's own geometry by {diff:.3g}")
            print(f"{code}: matches data/processed/{args.dataset}/{code.lower()} (max diff {diff:.1e})", flush=True)
        frame = pd.DataFrame(values[present], columns=list(GEOMETRY)).assign(root_id=union.node_ids[present])
        frame[["root_id", *GEOMETRY]].to_parquet(out / f"{code.lower()}.parquet", index=False)
        print(f"{code}: {int(present.sum()):,} of {union.n_nodes:,} union neurons have geometry", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
