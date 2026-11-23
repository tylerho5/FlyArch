"""Prepare types, soma sides, and centroids for scripts/build_suites.py.

    py -3.13 scripts/prepare_suite_inputs.py --graph data/processed/f1/<code> \
        --annotations data/raw/Supplemental_file1_neuron_annotations.tsv \
        --synapses data/raw/flywire_synapses_783.feather --out data/interim/f1/<code>

For CX, ``<code>`` is ``cx``. ``scripts/build_region.py`` runs this per region
and dataset; F2 and F3 add ``--f1-annotations``.

``build_suites.py`` takes three per-node inputs:

- ``<dataset>_types.parquet``: ``root_id``/``cell_type``, one row per graph
  node. A neuron with no annotation keeps a null type, which
  ``src.masks.typed_mask`` never draws and never masks, but which must still be
  present for the index to cover every node.
- ``<dataset>_sides.parquet``: ``root_id``/``side``, soma laterality normalized
  from ``left``/``right`` to ``L``/``R``. Other values and missing annotations
  are null; an absent ``side`` column yields all nulls. These are soma sides,
  not synapse or neuropil sides. H requires both L and R nodes.
- ``<dataset>_centroids.parquet``: ``root_id``/``x``/``y``/``z``, the mean of
  the neuron's own in-region synapse locations. That is the same quantity
  ``build_features.py`` streams for the geometry block, taken before the block
  z-scores it: the suite's spatial mask asks where the neuron is, and
  standardisation is a feature-scaling step rather than a property of the
  neuron. Which source is used changes the S05 mask and therefore the suite
  digest, so it is a pinned choice, not an implementation detail. F2
  and F3 centroids stay in their native axes: the S05 ball depends only on
  distances, which the axis maps (``src.harmonize.F2_AXIS_MAP``,
  ``F3_AXIS_MAP``) do not change.

Centroids require synapse positions, so a graph node with no in-region synapse
end cannot be placed and raises rather than silently writing a hole that
``build_suites.load_centroids`` would reject later.

F1 types come from F1's own annotation table. F2 suites take F1's names: BANC
meta goes through ``build_features.f2_annotations``, the same canonicalization,
value maps and ``src.harmonize.type_crosswalk`` its features use, so a T mask
hides one F1 type by its FAFB match, or by the neuron's own name when it has no
matched type, and a neuron with neither is never T-drawn. That needs ``--f1-annotations``. Scoring's
``type_matched``/``type_trained`` and ``siblings`` read types in the same F1
vocabulary. F3 is the same: ``build_features.f3_annotations`` crosswalks its
normalized table's ``cell_type`` into F1's names (decision 7).

Tables are read through ``build_features.read_table``, which keys BANC meta on
``banc_888_id`` rather than its later ``root_id``. The output filenames carry
only the dataset, and the inputs are region-specific, so one ``--out``
directory holds one region's inputs at a time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from scripts.build_features import f2_annotations, f3_annotations, read_table, stream_synapse_geometry  # noqa: E402
from src.regions import REGIONS, RegionGraph  # noqa: E402
from src.union import union_labels  # noqa: E402


def log(message: str) -> None:
    print(f"[prepare_suite_inputs] {message}", flush=True)


def _node_column(annotations: pd.DataFrame, graph: RegionGraph, column: str) -> pd.Series:
    """One annotation column reindexed onto the graph's nodes.

    A missing ``column`` yields all nulls; the caller decides whether that is
    allowed. Graph nodes absent from the table stay null and are counted in
    the log, so a partial id-space mismatch is visible rather than silent.
    """
    if "root_id" not in annotations.columns:
        raise ValueError("annotations table has no root_id column")
    # Same discipline as src.features: a repeated root id means the table is not
    # one row per neuron, and which row won would silently decide a mask.
    duplicated = annotations["root_id"][annotations["root_id"].duplicated()]
    if len(duplicated):
        raise ValueError(
            f"annotations contain duplicate root id(s): {list(duplicated[:5])}"
        )
    present = pd.Index(graph.node_ids).isin(annotations["root_id"])
    if len(annotations) and not present.any():
        raise ValueError(
            f"no annotation root id matches the {graph.region} graph; the tables "
            f"are in different id spaces, e.g. annotations carry "
            f"{annotations['root_id'].iloc[0]!r} and the graph carries "
            f"{graph.node_ids[0]!r}"
        )
    if not present.all():
        log(f"{int((~present).sum())} of {graph.n_nodes} graph nodes have no annotation row")
    if column not in annotations.columns:
        return pd.Series(index=pd.Index(graph.node_ids, name="root_id"), dtype=object, name=column)
    return annotations.set_index("root_id")[column].reindex(graph.node_ids).rename_axis("root_id")


def build_types(annotations: pd.DataFrame, graph: RegionGraph) -> pd.DataFrame:
    """``root_id``/``cell_type`` for every graph node, null where unannotated."""
    if "cell_type" not in annotations.columns:
        raise ValueError("annotations table has no cell_type column")
    return _node_column(annotations, graph, "cell_type").reset_index()


def build_sides(annotations: pd.DataFrame, graph: RegionGraph) -> pd.DataFrame:
    """Soma sides for every graph node, null where missing or indeterminate."""
    sides = _node_column(annotations, graph, "side").map({"left": "L", "right": "R"})
    return sides.rename("side").reset_index()


def build_centroids(graph: RegionGraph, synapses: str | Path) -> pd.DataFrame:
    """``root_id``/``x``/``y``/``z`` in node order, from in-region synapse ends."""
    # A union graph (src/union.py) places neurons by their synapses on its members' labels.
    labels = union_labels(graph.region) if graph.region not in REGIONS else None
    geometry = stream_synapse_geometry(synapses, graph, labels)
    centroids = (
        geometry.set_index("root_id")[["centroid_x", "centroid_y", "centroid_z"]]
        .reindex(graph.node_ids)
        .rename(columns={"centroid_x": "x", "centroid_y": "y", "centroid_z": "z"})
    )
    missing = centroids.index[centroids.isna().any(axis=1)]
    if len(missing):
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"{len(missing)} graph node(s) have no in-region synapse position, "
            f"so they cannot be placed for a spatial mask: "
            f"{list(missing[:5])}{suffix}"
        )
    return centroids.reset_index(names="root_id")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph", required=True, type=Path, help="RegionGraph.read directory")
    parser.add_argument("--annotations", required=True, type=Path, help="root_id/cell_type table")
    parser.add_argument("--synapses", required=True, type=Path, help="synapse positions")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--f1-annotations", type=Path,
        help="F1's root_id/cell_type table; required for f2, whose types are crosswalked into it",
    )
    args = parser.parse_args(argv)

    graph = RegionGraph.read(args.graph)
    if (graph.dataset in ("f2", "f3")) != (args.f1_annotations is not None):
        parser.error("--f1-annotations is required for f2/f3 and valid only for them")
    args.out.mkdir(parents=True, exist_ok=True)

    annotations = read_table(args.annotations)
    if graph.dataset == "f2":
        annotations, _ = f2_annotations(annotations, read_table(args.f1_annotations))
    elif graph.dataset == "f3":
        annotations = f3_annotations(annotations, read_table(args.f1_annotations))
    if graph.dataset in ("f1", "f2", "f3"):
        types = build_types(annotations, graph)
        types_path = args.out / f"{graph.dataset}_types.parquet"
        types.to_parquet(types_path, index=False)
        log(
            f"{graph.dataset} types: {int(types['cell_type'].notna().sum())} of "
            f"{graph.n_nodes} nodes typed -> {types_path}"
        )

    sides = build_sides(annotations, graph)
    sides_path = args.out / f"{graph.dataset}_sides.parquet"
    sides.to_parquet(sides_path, index=False)
    log(f"{graph.dataset} soma sides: {int(sides['side'].notna().sum())} of {graph.n_nodes} nodes -> {sides_path}")

    centroids = build_centroids(graph, args.synapses)
    centroids_path = args.out / f"{graph.dataset}_centroids.parquet"
    centroids.to_parquet(centroids_path, index=False)
    log(f"{graph.dataset} centroids: {len(centroids)} nodes -> {centroids_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
