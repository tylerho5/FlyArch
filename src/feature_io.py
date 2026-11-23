"""Read processed feature artifacts aligned to an existing region graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FeatureSchema, FeatureTable
from src.regions import RegionGraph


_COVERAGE_COLUMNS = ("feature", "n_present", "n_total", "fraction_present")


def _read_extras(path: Path, n_nodes: int) -> dict[str, np.ndarray] | None:
    """``scripts/build_node_extras.py``'s per-node arrays, or None when the region has none."""
    if not path.is_file():
        return None
    with np.load(path) as archive:
        extras = {name: archive[name] for name in archive.files}
    wrong = [name for name, array in extras.items() if array.shape[0] != n_nodes]
    if wrong:
        raise ValueError(f"{path}: arrays {wrong} do not have one row per graph node ({n_nodes})")
    return extras


def read_feature_artifacts(
    directory: str | Path, graph: RegionGraph
) -> tuple[FeatureTable, FeatureSchema]:
    """Load builder-produced features, schema, and coverage without rebuilding."""
    directory = Path(directory)
    paths = {name: directory / name for name in ("features.parquet", "schema.json", "coverage.csv")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"required feature artifact is missing: {path}")

    schema = FeatureSchema.from_json(paths["schema.json"].read_text(encoding="utf-8"))
    if schema.region != graph.region:
        raise ValueError(
            f"schema region {schema.region!r} does not match graph region {graph.region!r}"
        )

    frame = pd.read_parquet(paths["features.parquet"])
    if "root_id" not in frame.columns:
        raise ValueError("features.parquet is missing root_id")
    ids = frame["root_id"]
    if ids.isna().any():
        raise ValueError("features.parquet contains missing root_id values")
    if not pd.api.types.is_integer_dtype(ids.dtype):
        raise ValueError("features.parquet root_id values must use an integer dtype")
    root_ids = ids.to_numpy(dtype=np.int64)
    if pd.Index(root_ids).has_duplicates:
        raise ValueError("features.parquet contains duplicate root_id values")
    if not np.array_equal(root_ids, graph.node_ids):
        raise ValueError("features.parquet root_id values do not match graph node order")

    geometry_present = None
    if "geometry_present" in frame.columns:
        flags = frame.pop("geometry_present")
        if not pd.api.types.is_bool_dtype(flags.dtype):
            raise ValueError("features.parquet geometry_present values must be boolean")
        geometry_present = flags.to_numpy(dtype=bool)
    feature_columns = tuple(column for column in frame.columns if column != "root_id")
    if feature_columns != schema.column_names:
        raise ValueError(
            "feature columns must match schema.column_names in name and order"
        )
    try:
        values = frame.loc[:, list(schema.column_names)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("feature values must be numeric and finite") from exc
    if not np.isfinite(values).all():
        raise ValueError("feature values must be finite")

    coverage = pd.read_csv(paths["coverage.csv"])
    if any(column not in coverage.columns for column in _COVERAGE_COLUMNS):
        raise ValueError(f"coverage.csv must contain columns {_COVERAGE_COLUMNS}")
    for column in ("n_present", "n_total", "fraction_present"):
        coverage[column] = pd.to_numeric(coverage[column], errors="coerce")
    present = coverage["n_present"].to_numpy(dtype=float)
    total = coverage["n_total"].to_numpy(dtype=float)
    fraction = coverage["fraction_present"].to_numpy(dtype=float)
    valid = (
        np.isfinite(present) & np.isfinite(total) & np.isfinite(fraction)
        & (present >= 0) & (total == graph.n_nodes) & (present <= total)
        & (present == np.floor(present)) & (total == np.floor(total))
        & (fraction >= 0) & (fraction <= 1)
        & np.isclose(fraction, present / total, rtol=0, atol=1e-12)
    ) if graph.n_nodes else (
        np.isfinite(present) & np.isfinite(total) & np.isfinite(fraction)
        & (present == 0) & (total == 0) & (fraction == 0)
    )
    if not valid.all():
        raise ValueError("coverage.csv contains invalid counts or fractions")

    geometry_width = 6 + len(schema.labels)
    nt_width = len(schema.nt_classes) + 1
    types_width = schema.n_features - geometry_width - nt_width
    if types_width < 0:
        raise ValueError("schema column layout has invalid feature block widths")
    geometry_end = geometry_width
    types_end = geometry_end + types_width
    nt_end = types_end + nt_width
    if nt_end != schema.n_features:
        raise ValueError("schema column layout has invalid feature block widths")
    table = FeatureTable(
        values=values,
        columns=schema.column_names,
        blocks={
            "geometry": slice(0, geometry_end),
            "types": slice(geometry_end, types_end),
            "nt": slice(types_end, nt_end),
            "context": slice(nt_end, nt_end),
        },
        coverage=coverage,
        geometry_present=geometry_present,
        extras=_read_extras(directory / "extras.npz", graph.n_nodes),
    )
    return table, schema
