"""Extract one region's graph from a connections table.

    py -3.13 scripts/extract_region.py --connections data/raw/proofread_connections_783.feather --region <code>
    py -3.13 scripts/extract_region.py --connections <file> --verify-labels
    py -3.13 scripts/extract_region.py --connections <file> --region CX --verify-counts 3219 70830

``<code>`` is a registry code such as ``CX``. ``scripts/build_region.py`` runs
the extraction for every region and both datasets from the region config.

The two verify modes carry Phase 1's done-when criteria for region extraction, and they are the only place the registry meets real data: the label
vocabulary is a hypothesis until ``--verify-labels`` runs green on the box that
holds the feather, and the CX size is a claim until
``--verify-counts`` agrees with it.

Unlike ``sync_*.py`` this is not stdlib-only -- it is a thin wrapper over
``src.regions`` and inherits pandas/pyarrow from it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402

from src.regions import (  # noqa: E402
    DEFAULT_THRESHOLD,
    REGIONS,
    ColumnMap,
    all_labels,
    build_region_graph,
)


def log(message: str) -> None:
    print(f"[extract] {message}", flush=True)


def _resolve_neuropil_column(connections_path: Path) -> str:
    """Resolve the neuropil column name from the feather file's schema.

    Feather V2 files are Arrow IPC "file format" streams: a footer at the end
    of the file records the schema (and record-batch offsets) without
    requiring any column data to be read. ``pyarrow.ipc.open_file(...).schema``
    reads only that footer, so this is genuinely cheap even for 852 MB files.
    (``pyarrow.feather.read_table()`` does NOT have a schema-only mode -- even
    with no ``columns=`` filter it materializes every column's data before
    returning, so calling it just to read ``.schema`` afterward pays the full
    read cost for nothing. Do not use it here.)
    """
    # Read only the Arrow IPC file footer -- no record batches are decoded.
    with ipc.open_file(str(connections_path)) as reader:
        available_cols = reader.schema.names

    # Resolve the neuropil column name using the same alias logic as ColumnMap
    # Candidates from src.regions._ALIASES["neuropil"], in preference order
    neuropil_candidates = ("neuropil", "neuropil_name", "roi", "compartment")
    lookup = {str(c).lower(): c for c in available_cols}
    for candidate in neuropil_candidates:
        if candidate in lookup:
            return lookup[candidate]

    # If we can't resolve it, fail with a clear error
    raise ValueError(
        f"could not resolve the 'neuropil' column: tried "
        f"{neuropil_candidates}, table has {sorted(map(str, available_cols))}"
    )


def verify_labels(connections: pd.DataFrame, cols: ColumnMap) -> int:
    """Compare the table's neuropil vocabulary against the registry's."""
    observed = {str(v) for v in connections[cols.neuropil].dropna().unique()}
    known = set(all_labels())
    unknown = sorted(observed - known)
    unseen = sorted(known - observed)

    log(f"labels in table: {len(observed)}; labels in registry: {len(known)}")
    if unknown:
        log(f"NOT IN REGISTRY ({len(unknown)}): {unknown}")
    if unseen:
        log(f"IN REGISTRY BUT ABSENT FROM TABLE ({len(unseen)}): {unseen}")
    if unknown or unseen:
        log(
            "FAIL -- fix src.regions.UNPAIRED_NEUROPILS / REGIONS together. "
            "Do not edit the test."
        )
        return 1
    log("OK -- every label is accounted for")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--connections", required=True, type=Path)
    parser.add_argument("--region", choices=sorted(REGIONS), help="required unless --verify-labels")
    parser.add_argument("--dataset", default="f1")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--verify-labels",
        action="store_true",
        help="audit the neuropil vocabulary and exit",
    )
    parser.add_argument(
        "--verify-counts",
        nargs=2,
        type=int,
        metavar=("N_NODES", "N_EDGES"),
        help="assert the extracted sizes, e.g. --verify-counts 3219 70830",
    )
    args = parser.parse_args(argv)

    # For --verify-labels, read only the neuropil column to avoid loading the
    # entire 852 MB file. Resolve the column name from the schema first.
    if args.verify_labels:
        log(f"reading schema from {args.connections}")
        neuropil_col = _resolve_neuropil_column(args.connections)
        log(f"neuropil column resolved: {neuropil_col}")
        log(f"reading {args.connections} (neuropil column only)")
        connections = pd.read_feather(args.connections, columns=[neuropil_col])
        # Create a minimal ColumnMap for the neuropil-only DataFrame
        cols = ColumnMap(pre="", post="", neuropil=neuropil_col, syn_count="")
        return verify_labels(connections, cols)

    if args.region is None:
        parser.error("--region is required to extract a graph")

    # For extraction modes, read the full file
    log(f"reading {args.connections}")
    connections = pd.read_feather(args.connections)
    cols = ColumnMap.resolve(connections.columns)
    log(f"columns resolved: {cols}")

    graph = build_region_graph(
        connections,
        region=args.region,
        dataset=args.dataset,
        threshold=args.threshold,
        columns=cols,
    )
    log(
        f"{args.dataset}/{args.region} at threshold {args.threshold}: "
        f"{graph.n_nodes} nodes, {graph.n_edges} edges, "
        f"{int(graph.weights.sum())} synapses"
    )

    out = args.out or REPO_ROOT / "data" / "processed" / args.dataset / args.region.lower()
    graph.write(out)
    log(f"wrote {out} (digest {graph.digest()[:16]}...)")

    if args.verify_counts:
        want_nodes, want_edges = args.verify_counts
        if (graph.n_nodes, graph.n_edges) != (want_nodes, want_edges):
            log(
                f"FAIL -- expected {want_nodes} nodes / {want_edges} edges, "
                f"got {graph.n_nodes} / {graph.n_edges}"
            )
            return 1
        log("OK -- counts match the expected sizes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
