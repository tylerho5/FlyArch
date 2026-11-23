"""Build the shared feature table from graph and annotation files.

The FlyWire synapse Feather is multi-gigabyte, so that input is reduced in
Arrow record batches. Only one accumulator per graph neuron stays in memory;
F2/F3 must load the schema produced from F1. A Phase 3 union graph
(``src/union.py``, F1 and F2) gets its members' labels for geometry and innervation.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import (
    FeatureSchema,
    NEUROPIL_ALIASES,
    _ANNOTATION_ALIASES,
    _nt_probability_columns,
    _per_node_statistics,
    _resolve_column,
    build_features,
)
from src.harmonize import (
    F2_AXIS_MAP,
    F2_COLUMNS,
    F3_AXIS_MAP,
    map_f2_values,
    map_neuropil_labels,
    normalize_f3_annotations,
    type_crosswalk,
)
from src.regions import REGIONS, RegionGraph, region_labels
from src.union import CODES, union_labels, union_members


def _canonicalize(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    if "nt_type" in fields and _nt_probability_columns(frame):
        fields = tuple(field for field in fields if field != "nt_type")
    return frame.rename(columns={_resolve_column(frame, f): f for f in fields})


def _banc_root_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Key BANC meta on ``banc_888_id``, the id its edges and synapses carry.

    Meta's own ``root_id`` is a later id that differs in 8.7% of rows,
    and every alias resolver takes ``root_id``
    first, so left in place it would mis-join those neurons with no error.
    """
    if "banc_888_id" not in frame.columns:
        return frame
    ids = frame["banc_888_id"]
    if not pd.api.types.is_integer_dtype(ids):
        ids = ids.astype("string").astype("Int64")
    return frame.drop(columns=["root_id"], errors="ignore").assign(banc_888_id=ids).rename(
        columns={"banc_888_id": "root_id"}
    )


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in (".feather", ".parquet"):
        frame = pd.read_feather(source) if suffix == ".feather" else pd.read_parquet(source)
        return _banc_root_ids(frame)
    separator = "\t" if suffix in (".tsv", ".tab") else ","
    names = pd.read_csv(source, sep=separator, nrows=0).columns
    id_aliases = (*_ANNOTATION_ALIASES["root_id"], *_ANNOTATION_ALIASES["fafb_match_id"], "banc_888_id")
    ids = [name for name in names if str(name).lower() in id_aliases]
    # Parse strings before nullable integers so a missing match never routes
    # 18-digit ids through a float64 intermediate.
    frame = pd.read_csv(source, sep=separator, dtype={name: "string" for name in ids})
    for name in ids:
        frame[name] = frame[name].str.replace(r"^([+-]?\d+)\.0+$", r"\1", regex=True).astype("Int64")
    return _banc_root_ids(frame)


def _bases_and_labels(region: str) -> tuple[list[str], tuple[str, ...]]:
    """A region's (or a union's members') base neuropil names and v783 labels."""
    codes = (region,) if region in REGIONS else union_members(region)
    return [b for code in codes for b in REGIONS[code].base_neuropils], tuple(
        label for code in codes for label in region_labels(code))


def read_banc_table(path: str | Path, region: str, ids: tuple[str, ...]) -> pd.DataFrame:
    """A BANC edgelist or per-neuron count table's rows on ``region``, in v783 terms.

    BANC files carry a base neuropil name plus a ``side`` column and string ids.
    Rows are filtered to the region's base names as they are read, the ``ids``
    columns become int64, and ``neuropil`` becomes the v783 label,
    dropping rows a paired base cannot place on a side.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    source = Path(path)
    dataset = ds.dataset(source, format="ipc" if source.suffix.lower() == ".feather" else "parquet")
    columns = {
        name: ds.field(name).cast(pa.int64()) if name in ids else ds.field(name)
        for name in dataset.schema.names
    }
    bases, labels = _bases_and_labels(region)
    table = dataset.to_table(columns=columns, filter=ds.field("neuropil").isin(bases))
    frame = table.to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get).dropna(subset=list(ids))
    frame["neuropil"] = map_neuropil_labels(frame["neuropil"], frame["side"])
    frame = frame[frame["neuropil"].isin(labels)].drop(columns="side")
    return frame.astype({name: "int64" for name in ids}).reset_index(drop=True)


def f2_annotations(
    annotations: pd.DataFrame, f1_annotations: pd.DataFrame, root_ids=None
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """BANC meta in F1's terms: canonical columns, F1 values, crosswalked F1 cell types.

    Features and suite inputs both read F2 through this, so a typed mask hides
    a type name the model knows from F1. ``root_ids`` restricts the rows, and
    so the reported values with no F1 counterpart, to one graph's nodes.
    """
    frame = _canonicalize(
        annotations.rename(columns=F2_COLUMNS),
        ("root_id", "fafb_match_id", "super_class", "class", "cell_type", "side", "flow", "nt_type"),
    )
    if root_ids is not None:
        frame = frame[frame["root_id"].isin(root_ids)]
    frame, unmapped = map_f2_values(frame)
    f1 = _canonicalize(f1_annotations, ("root_id", "cell_type"))
    mapped = type_crosswalk(frame, dataset="f2", f1_types=f1.set_index("root_id")["cell_type"])
    frame["cell_type"] = mapped.reindex(frame["root_id"]).to_numpy()
    return frame, unmapped


def f3_annotations(
    annotations: pd.DataFrame, f1_annotations: pd.DataFrame, root_ids=None
) -> pd.DataFrame:
    """The raw F3 table (``fetch_f3``) in F1's values, ``cell_type`` crosswalked to F1 names.

    ``normalize_f3_annotations`` maps the values here, as F2's are, so a change
    to the maps needs no refetch. The type name is kept where F1 has it, null
    elsewhere (``f3_type`` keeps the release's own). Features and suite inputs
    both read F3 through this.
    """
    frame = _canonicalize(
        normalize_f3_annotations(annotations)[0], ("root_id", "super_class", "class", "cell_type", "side", "flow", "nt_type")
    )
    if root_ids is not None:
        frame = frame[frame["root_id"].isin(root_ids)]
    f1 = _canonicalize(f1_annotations, ("root_id", "cell_type"))
    mapped = type_crosswalk(frame, dataset="f3", f1_types=f1.set_index("root_id")["cell_type"])
    frame = frame.copy()
    frame["cell_type"] = mapped.reindex(frame["root_id"]).to_numpy()
    return frame


def apply_axis_map(geometry: pd.DataFrame, axis_map=F2_AXIS_MAP) -> pd.DataFrame:
    """Synapse statistics re-expressed in F1's axes (``F2_AXIS_MAP`` or ``F3_AXIS_MAP``)."""
    out = geometry.copy()
    for axis, (source, sign) in zip("xyz", axis_map):
        out[f"centroid_{axis}"] = sign * geometry[f"centroid_{source}"]
        out[f"extent_{axis}"] = geometry[f"extent_{source}"]
    return out


def _resolve_innervation_root(frame: pd.DataFrame) -> str:
    lookup = {str(column).lower(): column for column in frame.columns}
    for candidate in ("root_id", "pt_root_id", "pre_pt_root_id", "post_pt_root_id"):
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(
        f"could not resolve a neuron root id column; available: {sorted(map(str, frame.columns))}"
    )


def _innervation_wide(frame: pd.DataFrame, labels: tuple[str, ...], root_ids=None) -> pd.DataFrame:
    """Normalize one innervation file to a root-id-indexed, per-neuropil count table.

    Restricted to the region's ``labels``: only those neuropils feed the
    innervation feature, and a long file spans every neuropil for millions of
    root ids, so a full-vocabulary pivot is neither needed nor affordable.
    ``root_ids`` also drops other neurons' rows before the pivot (a union's label set
    is too wide to pivot whole), keeping the columns the unrestricted pivot would have.
    """
    root = _resolve_innervation_root(frame)
    lookup = {str(column).lower(): column for column in frame.columns}
    if "neuropil" in lookup and "count" in lookup:
        in_region = frame[frame[lookup["neuropil"]].isin(labels)]
        if root_ids is not None:
            seen = sorted(in_region[lookup["neuropil"]].unique())
            in_region = in_region[in_region[root].isin(root_ids)]
        pivot = in_region.pivot_table(
            index=root, columns=lookup["neuropil"], values=lookup["count"],
            aggfunc="sum", fill_value=0,
        )
        if root_ids is not None:
            pivot = pivot.reindex(columns=seen, fill_value=0)
        pivot.index.name = "root_id"
        pivot.columns.name = None
        return pivot
    if frame[root].duplicated().any():
        duplicates = frame.loc[frame[root].duplicated(), root].head(5).tolist()
        raise ValueError(f"innervation table has duplicate root id(s): {duplicates}")
    wide = frame.rename(columns={root: "root_id"}).set_index("root_id")
    return wide[[column for column in wide.columns if column in labels]]


def read_innervation(
    paths: str | Path | list[str | Path], labels: tuple[str, ...], *, banc_region: str | None = None,
    root_ids=None,
) -> pd.DataFrame:
    """Read per-neuron innervation counts, summed across the given files.

    Each file is long form (a ``neuropil`` label column and a ``count`` column,
    as in the pinned ``per_neuron_neuropil_count_{pre,post}`` feathers) or wide
    (a root id plus one column per neuropil). Long files are pivoted to wide;
    all files are summed on ``(root_id, neuropil)`` so a neuron's pre and post
    counts combine into its total in-region innervation. ``banc_region`` reads
    BANC's per-neuron table, whose pre and post rows share one file, through
    :func:`read_banc_table`.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    read = read_table if banc_region is None else (
        lambda path: read_banc_table(path, banc_region, ("root_id",))
    )
    frames = [_innervation_wide(read(path), labels, root_ids) for path in paths]
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.add(frame, fill_value=0)
    return combined


def _resolve_synapse_endpoints(
    lookup: dict[str, str]
) -> list[tuple[str, tuple[str, str, str]]]:
    """The (root id, x/y/z) column groups carrying a neuron's own synapse ends.

    The raw FlyWire synapse table is one row per synapse, each with a
    presynaptic and a postsynaptic partner at its own position; a neuron's own
    locations are its pre ends (where it is presynaptic) unioned with its post
    ends (where it is postsynaptic). BANC names the same groups
    ``{pre,post}_root_id`` and ``{pre,post}_{x,y,z}``. A processed table with a
    single root id and x/y/z is the one-endpoint case. The synapse `id` column
    is never a neuron id, so it is not a candidate.
    """
    def coords(prefix: str) -> tuple[str, str, str] | None:
        picked = []
        for axis in ("x", "y", "z"):
            for candidate in (f"{prefix}{axis}", f"{prefix}{axis}_nm"):
                if candidate in lookup:
                    picked.append(lookup[candidate])
                    break
            else:
                return None
        return tuple(picked)

    endpoints = []
    for role in ("pre", "post"):
        for root, prefix in ((f"{role}_pt_root_id", f"{role}_pt_position_"), (f"{role}_root_id", f"{role}_")):
            group = coords(prefix) if root in lookup else None
            if group is not None:
                endpoints.append((lookup[root], group))
                break
    if endpoints:
        return endpoints
    root = next((lookup[c] for c in ("root_id", "pt_root_id") if c in lookup), None)
    group = coords("")
    if root is None or group is None:
        raise ValueError(
            "could not resolve neuron root id and x/y/z coordinate columns; "
            f"available columns: {sorted(lookup.values())}"
        )
    return [(root, group)]


def _merge_statistics(
    counts: np.ndarray, means: np.ndarray, m2s: np.ndarray,
    add_counts: np.ndarray, add_means: np.ndarray, add_m2: np.ndarray,
) -> None:
    """Fold one batch of per-node count/mean/M2 into the running totals (Chan et al.)."""
    active = add_counts > 0
    n_a, n_b = counts[active], add_counts[active]
    delta = add_means[active] - means[active]
    total = n_a + n_b
    means[active] = means[active] + delta * (n_b / total)[:, None]
    m2s[active] = m2s[active] + add_m2[active] + (delta ** 2) * (n_a * n_b / total)[:, None]
    counts[active] = total


def stream_synapse_geometry(path: str | Path, graph: RegionGraph, labels: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Reduce own in-region synapses using batched two-pass statistics.

    Reads the raw FlyWire synapse feather (per-synapse pre/post endpoints) or a
    processed single-endpoint table. Neuropil aliases match the direct builder;
    label-free input is accepted as already filtered to graph.region. For an
    F2 graph, BANC's base neuropil names and ``side`` column map to v783 labels,
    and string root ids are read as integers. ``labels`` overrides the region's own
    labels, for a union graph (``src/union.py``), whose name is not a region code.
    """
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - exercised on the data box
        raise RuntimeError("streaming synapses requires pyarrow") from exc

    source = Path(path)
    dataset = ds.dataset(source, format="ipc" if source.suffix.lower() == ".feather" else None)
    lookup = {str(n).lower(): n for n in dataset.schema.names}
    endpoints = _resolve_synapse_endpoints(lookup)
    neuropil = next((lookup[alias] for alias in NEUROPIL_ALIASES if alias in lookup), None)
    side = lookup.get("side") if graph.dataset == "f2" and neuropil is not None else None
    string_ids = {
        root for root, _ in endpoints
        if pa.types.is_string(dataset.schema.field(root).type)
        or pa.types.is_large_string(dataset.schema.field(root).type)
    }

    wanted = set(map(int, graph.node_ids))
    node_index = pd.Index(graph.node_ids)
    region = labels if labels is not None else region_labels(graph.region)
    # Per-node running count/mean/M2; batches and endpoints merge in via Chan et al.
    counts = np.zeros(graph.n_nodes)
    means = np.zeros((graph.n_nodes, 3))
    m2s = np.zeros((graph.n_nodes, 3))
    selected = sorted(
        {column for root, coords in endpoints for column in (root, *coords)}
        | ({neuropil} if neuropil is not None else set())
        | ({side} if side is not None else set())
    )
    columns = {
        name: ds.field(name).cast(pa.int64()) if name in string_ids else ds.field(name)
        for name in selected
    }
    in_bases = (
        ds.field(neuropil).isin(_bases_and_labels(graph.region)[0]) if side is not None else None
    )
    nullable = {pa.int64(): pd.Int64Dtype()}.get if string_ids else None
    for batch in dataset.to_batches(columns=columns, filter=in_bases, batch_size=1_000_000):
        frame = batch.to_pandas(types_mapper=nullable)
        if neuropil is not None:
            if side is not None:
                frame[neuropil] = map_neuropil_labels(frame[neuropil], frame[side])
            frame = frame[frame[neuropil].isin(region)]
        if frame.empty:
            continue
        for root, coordinates in endpoints:
            own = frame[frame[root].isin(wanted)]
            if own.empty:
                continue
            codes = node_index.get_indexer(own[root].to_numpy(dtype="int64"))
            coords = own[list(coordinates)].to_numpy(dtype=np.float64)
            _merge_statistics(counts, means, m2s, *_per_node_statistics(codes, coords, graph.n_nodes))

    rows = []
    for i, root_id in enumerate(graph.node_ids):
        count = counts[i]
        if not count:
            continue
        variance = np.maximum(m2s[i] / count, 0.0)
        rows.append(
            {
                "root_id": root_id,
                "centroid_x": means[i, 0], "centroid_y": means[i, 1], "centroid_z": means[i, 2],
                "extent_x": np.sqrt(variance[0]),
                "extent_y": np.sqrt(variance[1]),
                "extent_z": np.sqrt(variance[2]),
            }
        )
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--dataset", required=True, choices=("f1", "f2", "f3"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--schema")
    parser.add_argument("--synapses")
    parser.add_argument("--innervation", nargs="+")
    parser.add_argument("--f1-annotations")
    args = parser.parse_args(argv)

    if args.dataset == "f1" and args.schema:
        parser.error("--schema is not valid for f1")
    if args.dataset == "f1" and args.f1_annotations:
        parser.error("--f1-annotations is not valid for f1")
    if args.dataset in ("f2", "f3"):
        if not args.schema:
            parser.error("--schema is required for f2/f3")
        if not args.f1_annotations:
            parser.error("--f1-annotations is required for f2/f3")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    annotations = read_table(args.annotations)
    graph = RegionGraph.read(args.graph)
    if graph.dataset != args.dataset:
        parser.error(
            f"graph dataset {graph.dataset!r} does not match --dataset {args.dataset!r}"
        )
    union = graph.region not in REGIONS
    labels = union_labels(graph.region) if union else None
    if union and args.dataset == "f3":
        parser.error(f"{graph.region} is a union graph; only f1 and f2 union features are built")
    if args.dataset == "f1":
        annotations = _canonicalize(annotations, ("root_id", "super_class", "class", "cell_type", "side", "flow", "nt_type"))
        schema = FeatureSchema.from_f1(annotations, CODES[0] if union else graph.region)
        if union:
            # The vocabularies are F1-wide whatever the region; only the name and labels differ.
            schema = dataclasses.replace(schema, region=graph.region, labels=labels)
    elif args.dataset == "f2":
        annotations, unmapped = f2_annotations(annotations, read_table(args.f1_annotations), graph.node_ids)
        for field, counts in unmapped.items():
            print(
                f"[build_features] f2 {field}: {sum(counts.values())} graph node(s) carry a value "
                f"with no F1 counterpart: {counts}",
                flush=True,
            )
        schema = FeatureSchema.from_json(Path(args.schema).read_text(encoding="utf-8"))
    else:
        annotations = f3_annotations(annotations, read_table(args.f1_annotations), graph.node_ids)
        schema = FeatureSchema.from_json(Path(args.schema).read_text(encoding="utf-8"))

    synapses = stream_synapse_geometry(args.synapses, graph, labels) if args.synapses else None
    if synapses is not None and args.dataset in ("f2", "f3") and not synapses.empty:
        synapses = apply_axis_map(synapses, F2_AXIS_MAP if args.dataset == "f2" else F3_AXIS_MAP)
    banc_region = graph.region if args.dataset == "f2" else None
    table = build_features(
        annotations,
        graph,
        schema,
        dataset=args.dataset,
        synapses=synapses,
        innervation=(
            read_innervation(args.innervation, schema.labels, banc_region=banc_region,
                             root_ids=graph.node_ids if union else None)
            if args.innervation else None
        ),
    )
    features = pd.DataFrame(table.values, columns=table.columns)
    features.insert(0, "root_id", graph.node_ids)
    features.insert(1, "geometry_present", table.geometry_present)
    features.to_parquet(
        output / "features.parquet", index=False
    )
    (output / "schema.json").write_text(schema.to_json(), encoding="utf-8")
    table.coverage.to_csv(output / "coverage.csv", index=False)


if __name__ == "__main__":
    main()
