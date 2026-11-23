import json

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureSchema
from src.regions import RegionGraph
from src.feature_io import read_feature_artifacts


ROOT_IDS = np.array([123456789012345678, 223456789012345678], dtype=np.int64)


@pytest.fixture
def artifacts(tmp_path):
    graph = RegionGraph("f1", "CX", 5, ROOT_IDS.copy(), np.array([[0, 1]], dtype=np.int32), np.array([7]))
    graph.write(tmp_path)
    schema = FeatureSchema(
        region="CX", superclass_vocab=("central",), class_vocab=("interneuron",),
        side_vocab=("left",), flow_vocab=("intrinsic",),
        type_vocab={"cell-a": 1}, labels=("FB", "EB"),
    )
    values = np.arange(2 * schema.n_features, dtype=np.float64).reshape(2, schema.n_features)
    values[0, 0] = 0.25
    values[1, -1] = 1.0
    features = pd.DataFrame(values, columns=schema.column_names)
    features.insert(0, "root_id", ROOT_IDS)
    features.to_parquet(tmp_path / "features.parquet", index=False)
    (tmp_path / "schema.json").write_text(schema.to_json(), encoding="utf-8")
    pd.DataFrame([
        {"feature": "centroid_x", "n_present": 1, "n_total": 2, "fraction_present": 0.5},
        {"feature": "nt_type", "n_present": 2, "n_total": 2, "fraction_present": 1.0},
    ]).to_csv(tmp_path / "coverage.csv", index=False)
    return tmp_path, graph, schema, values


def test_reader_preserves_ids_values_schema_coverage_and_feature_slices(artifacts):
    directory, graph, schema, values = artifacts
    table, loaded_schema = read_feature_artifacts(directory, graph)

    np.testing.assert_array_equal(table.values, values)
    assert table.columns == schema.column_names
    assert loaded_schema.column_names == schema.column_names
    assert table.blocks == {
        "geometry": slice(0, 8),
        "types": slice(8, 8 + len(schema.superclass_vocab) + 1 + len(schema.class_vocab) + 1 + 1 + len(schema.side_vocab) + len(schema.flow_vocab)),
        "nt": slice(8 + len(schema.superclass_vocab) + 1 + len(schema.class_vocab) + 1 + 1 + len(schema.side_vocab) + len(schema.flow_vocab), schema.n_features),
        "context": slice(schema.n_features, schema.n_features),
    }
    assert table.coverage.to_dict("records") == [
        {"feature": "centroid_x", "n_present": 1, "n_total": 2, "fraction_present": 0.5},
        {"feature": "nt_type", "n_present": 2, "n_total": 2, "fraction_present": 1.0},
    ]
    assert table.values.dtype.kind == "f"


def test_reader_keeps_which_neurons_have_geometry(artifacts):
    directory, graph, _, _ = artifacts
    feature_path = directory / "features.parquet"
    frame = pd.read_parquet(feature_path)

    assert read_feature_artifacts(directory, graph)[0].geometry_present is None
    frame.insert(1, "geometry_present", [True, False])
    frame.to_parquet(feature_path, index=False)
    table, _ = read_feature_artifacts(directory, graph)
    np.testing.assert_array_equal(table.geometry_present, [True, False])
    frame["geometry_present"] = [1, 0]
    frame.to_parquet(feature_path, index=False)
    with pytest.raises(ValueError, match="geometry_present"):
        read_feature_artifacts(directory, graph)


@pytest.mark.parametrize("mutation, message", [
    ("missing", "features.parquet"),
    ("ids_missing", "root_id"),
    ("ids_duplicate", "duplicate"),
    ("ids_shuffled", "order"),
    ("wrong_region", "region"),
    ("columns_missing", "column"),
    ("columns_reordered", "column"),
    ("nonfinite", "finite"),
    ("coverage_counts", "count"),
    ("coverage_fraction", "fraction"),
    ("coverage_columns", "coverage"),
])
def test_reader_rejects_invalid_artifacts(artifacts, mutation, message):
    directory, graph, schema, _ = artifacts
    feature_path = directory / "features.parquet"
    frame = pd.read_parquet(feature_path)
    coverage_path = directory / "coverage.csv"
    coverage = pd.read_csv(coverage_path)
    if mutation == "missing":
        feature_path.unlink()
    elif mutation == "ids_missing":
        frame = frame.drop(columns="root_id")
    elif mutation == "ids_duplicate":
        frame.loc[1, "root_id"] = ROOT_IDS[0]
    elif mutation == "ids_shuffled":
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif mutation == "wrong_region":
        data = json.loads((directory / "schema.json").read_text())
        data["region"] = "OL"
        (directory / "schema.json").write_text(json.dumps(data))
    elif mutation == "columns_missing":
        frame = frame.drop(columns=[schema.column_names[0]])
    elif mutation == "columns_reordered":
        frame = frame[["root_id", *reversed(schema.column_names)]]
    elif mutation == "nonfinite":
        frame.loc[0, schema.column_names[0]] = np.inf
    elif mutation == "coverage_counts":
        coverage.loc[0, "n_present"] = 3
        coverage.to_csv(coverage_path, index=False)
    elif mutation == "coverage_fraction":
        coverage.loc[0, "fraction_present"] = 0.25
        coverage.to_csv(coverage_path, index=False)
    elif mutation == "coverage_columns":
        coverage.drop(columns="fraction_present").to_csv(coverage_path, index=False)
    if mutation not in {"missing", "wrong_region", "coverage_counts", "coverage_fraction", "coverage_columns"}:
        frame.to_parquet(feature_path, index=False)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        read_feature_artifacts(directory, graph)
