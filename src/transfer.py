"""Re-encode and ablate feature tables for the cross-region transfer matrix.

Region j's table is rewritten column by column into region i's schema so an
expert trained on i can score j.  Everything is mapped by column name; nothing
is read from disk and inputs are never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import FeatureSchema, FeatureTable

_GEOMETRY = ("centroid_x", "centroid_y", "centroid_z", "extent_x", "extent_y", "extent_z")


def _check(table: FeatureTable, schema: FeatureSchema) -> None:
    if table.columns != schema.column_names:
        raise ValueError(f"table columns do not match the schema for region {schema.region!r}")


def _copy(values: np.ndarray, columns: tuple[str, ...], table: FeatureTable) -> FeatureTable:
    return FeatureTable(values, columns, dict(table.blocks), table.coverage.copy(), table.geometry_present)


def _zeroed(table: FeatureTable, schema: FeatureSchema, prefixes: tuple[str, ...]) -> FeatureTable:
    _check(table, schema)
    values = table.values.copy()
    values[:, [k for k, c in enumerate(schema.column_names) if c.startswith(prefixes)]] = 0.
    return _copy(values, table.columns, table)


def ablate_innervation(table: FeatureTable, schema: FeatureSchema) -> FeatureTable:
    """Set every innervation column to the no-input value, 1 / len(labels)."""
    _check(table, schema)
    values = table.values.copy()
    values[:, [k for k, c in enumerate(schema.column_names) if c.startswith("innervation_")]] = (
        1. / len(schema.labels))
    return _copy(values, table.columns, table)


def ablate_geometry(table: FeatureTable, schema: FeatureSchema) -> FeatureTable:
    """Zero the six centroid and extent columns."""
    return _zeroed(table, schema, _GEOMETRY)


def reencode_features(table_j: FeatureTable, schema_j: FeatureSchema, schema_i: FeatureSchema) -> FeatureTable:
    """Rewrite region j's table in region i's columns, mapping vocabularies by name.

    Geometry and neurotransmitter columns are copied; innervation is uniform.
    Super class and class values i lacks read as ``_OOV``; side and flow values
    i lacks read as all zeros; cell type ids go through the type name.
    """
    _check(table_j, schema_j)
    if schema_j.nt_classes != schema_i.nt_classes:
        raise ValueError("schemas must share nt_classes")
    source = {c: k for k, c in enumerate(schema_j.column_names)}
    out = np.zeros((len(table_j.values), schema_i.n_features))
    for k, column in enumerate(schema_i.column_names):
        if column.startswith("innervation_"):
            out[:, k] = 1. / len(schema_i.labels)
        elif column in source and column not in ("cell_type", "superclass_OOV", "class_OOV"):
            out[:, k] = table_j.values[:, source[column]]
    names = {v: n for n, v in schema_j.type_vocab.items()}
    ids = np.rint(table_j.values[:, source["cell_type"]]).astype(int)
    out[:, schema_i.column_names.index("cell_type")] = [
        schema_i.type_id(names[t]) if t in names else 0 for t in ids]
    for prefix in ("superclass", "class"):
        value_cols = [k for k, c in enumerate(schema_i.column_names)
                      if c.startswith(f"{prefix}_") and c != f"{prefix}_OOV"]
        out[:, schema_i.column_names.index(f"{prefix}_OOV")] = out[:, value_cols].sum(axis=1) == 0
    geometry = 6 + len(schema_i.labels)
    nt = len(schema_i.nt_classes) + 1
    end = schema_i.n_features
    blocks = {"geometry": slice(0, geometry), "types": slice(geometry, end - nt),
              "nt": slice(end - nt, end), "context": slice(end, end)}
    return FeatureTable(out, schema_i.column_names, blocks, table_j.coverage.copy(), table_j.geometry_present)


def masked_oov_fraction(model, table: FeatureTable, graph, suite) -> dict[str, float]:
    """Per mask instance, the share of masked nodes whose cell type the model reads as OOV."""
    oov = model.type_ids(table).numpy() == 0
    index = pd.Index(graph.node_ids)
    return {inst.instance_id: float(oov[index.get_indexer(inst.masked_root_ids)].mean())
            for inst in suite.instances}
