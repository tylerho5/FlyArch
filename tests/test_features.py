"""One schema, built from F1, applied unchanged to every dataset."""

import numpy as np
import pandas as pd
import pytest

from src.features import NT_CLASSES, FeatureSchema, build_features
from src.regions import build_region_graph


def _f1_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4],
            "super_class": ["central", "central", "central", "sensory"],
            "class": ["CX", "CX", "CX", "visual"],
            "cell_type": ["EPG", "EPG", "PEN_a", "R1"],
            "side": ["center", "center", "left", "right"],
            "flow": ["intrinsic", "intrinsic", "intrinsic", "afferent"],
            "nt_type": ["acetylcholine", "glutamate", "gaba", "histamine"],
        }
    )


def _graph_for(dataset: str):
    return build_region_graph(
        pd.DataFrame(
            {
                "pre_pt_root_id": [1, 2, 3],
                "post_pt_root_id": [2, 3, 1],
                "neuropil": ["FB", "EB", "FB"],
                "syn_count": [9, 9, 9],
            }
        ),
        region="CX",
        dataset=dataset,
        threshold=5,
    )


def _graph():
    return _graph_for("f1")


def test_schema_vocabularies_come_from_f1_and_reserve_zero_for_oov():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    assert schema.nt_classes == (
        "acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine"
    )
    assert NT_CLASSES == schema.nt_classes
    assert len(schema.nt_classes) == 6
    assert schema.type_id("EPG") > 0
    assert schema.type_id("a-type-that-does-not-exist") == 0


def test_schema_digest_and_json_round_trip_are_stable():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    digest = schema.digest()
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
    assert FeatureSchema.from_json(schema.to_json()) == schema
    assert FeatureSchema.from_json(schema.to_json()).digest() == digest


def test_no_feature_column_is_a_degree():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    banned = ("degree", "in_strength", "out_strength", "n_partners", "syn_total")
    assert not [c for c in schema.column_names if any(b in c for b in banned)]


def test_schema_resolves_annotation_aliases_or_fails_with_available_columns():
    renamed = _f1_annotations().rename(
        columns={"super_class": "superclass", "nt_type": "neurotransmitter"}
    )
    schema = FeatureSchema.from_f1(renamed, region="CX")
    assert schema.superclass_vocab == ("central", "sensory")

    with pytest.raises(ValueError, match="available columns"):
        FeatureSchema.from_f1(renamed.drop(columns="class"), region="CX")


def test_build_features_preserves_values_from_annotation_aliases():
    aliased = _f1_annotations().rename(
        columns={
            "super_class": "superclass",
            "class": "cell_class",
            "cell_type": "type",
            "nt_type": "neurotransmitter",
        }
    )
    schema = FeatureSchema.from_f1(aliased, region="CX")
    table = build_features(aliased, _graph(), schema, dataset="f1")
    first = table.values[0]
    assert first[table.columns.index("superclass_central")] == 1.0
    assert first[table.columns.index("class_CX")] == 1.0
    assert first[table.columns.index("cell_type")] == schema.type_id("EPG")
    assert first[table.columns.index("nt_acetylcholine")] == 1.0


def test_schema_columns_follow_the_contract_order_and_oov_names():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    assert schema.column_names[:10] == (
        "centroid_x", "centroid_y", "centroid_z", "extent_x", "extent_y",
        "extent_z", "innervation_FB", "innervation_EB", "innervation_PB",
        "innervation_NO",
    )
    assert "superclass_OOV" in schema.column_names
    assert "class_OOV" in schema.column_names
    assert "cell_type" in schema.column_names
    assert "cell_type_id" not in schema.column_names


def test_innervation_fractions_sum_to_one_over_the_region_labels():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    innervation = pd.DataFrame(
        {"FB": [8, 0, 2], "EB": [2, 10, 8], "PB": [0, 0, 0], "NO": [0, 0, 0]},
        index=[1, 2, 3],
    )
    table = build_features(
        _f1_annotations(), _graph(), schema, dataset="f1", innervation=innervation
    )
    frac_cols = [c for c in table.columns if c.startswith("innervation_")]
    assert frac_cols == [
        "innervation_FB", "innervation_EB", "innervation_PB", "innervation_NO"
    ]
    idx = [table.columns.index(c) for c in frac_cols]
    assert np.allclose(table.values[:, idx].sum(axis=1), 1.0)


def test_geometry_is_z_scored_within_the_dataset():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    synapses = pd.DataFrame(
        {
            "root_id": [1, 1, 2, 2, 3, 3],
            "x": [0.0, 2.0, 10.0, 12.0, 100.0, 102.0],
            "y": [0.0] * 6,
            "z": [0.0] * 6,
        }
    )
    table = build_features(
        _f1_annotations(), _graph(), schema, dataset="f1", synapses=synapses
    )
    centroid_x = table.values[:, table.columns.index("centroid_x")]
    assert centroid_x.mean() == pytest.approx(0.0, abs=1e-6)
    assert centroid_x.std() == pytest.approx(1.0, abs=1e-6)


def test_f2_only_neurotransmitters_land_in_the_other_flag():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    f2 = _f1_annotations().assign(nt_type=["histamine", "tyramine", "gaba", "gaba"])
    table = build_features(f2, _graph_for("f2"), schema, dataset="f2")
    other = table.values[:, table.columns.index("nt_other")]
    assert other.tolist() == [1.0, 1.0, 0.0]
    assert np.allclose(table.values[0, table.blocks["nt"]][:6], 0.0)


def test_the_same_schema_gives_f1_and_f2_the_same_columns():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    f1 = build_features(_f1_annotations(), _graph(), schema, dataset="f1")
    f2 = build_features(
        _f1_annotations().assign(cell_type=["unseen"] * 4), _graph_for("f2"), schema,
        dataset="f2",
    )
    assert f1.values.shape[1] == f2.values.shape[1] == schema.n_features
    # Rule 1: a feature that cannot be computed for F2 is not a training feature.
    assert list(f1.columns) == list(f2.columns)


def test_coverage_report_names_what_is_missing():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    sparse = _f1_annotations().assign(cell_type=[None, "EPG", "PEN_a", "R1"])
    table = build_features(sparse, _graph_for("f2"), schema, dataset="f2")
    assert list(table.coverage.columns) == [
        "feature", "n_present", "n_total", "fraction_present"
    ]
    row = table.coverage.set_index("feature").loc["cell_type"]
    assert row["n_present"] == 2
    assert row["n_total"] == 3
    assert row["fraction_present"] == pytest.approx(2 / 3)


def test_perturb_is_deterministic_and_changes_only_one_geometry_row():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    table = build_features(_f1_annotations(), _graph(), schema, dataset="f1")
    first = table.perturb(1)
    second = table.perturb(1)
    assert first is not table
    assert np.array_equal(first.values, second.values)
    geometry = table.blocks["geometry"]
    changed = ~np.isclose(first.values, table.values, rtol=0.0, atol=0.0)
    assert changed[1, geometry].any()
    assert not changed[0].any()
    assert not changed[2].any()
    assert not changed[1, ~np.isin(np.arange(table.values.shape[1]), np.arange(geometry.start, geometry.stop))].any()


def test_innervation_coverage_is_tracked_per_label_when_input_is_partial():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    innervation = pd.DataFrame(
        {"FB": [8, 0, 2], "EB": [2, 10, 8]}, index=[1, 2, 3]
    )
    table = build_features(
        _f1_annotations(), _graph(), schema, dataset="f1", innervation=innervation
    )
    coverage = table.coverage.set_index("feature")
    assert coverage.loc["innervation_FB", "n_present"] == 3
    assert coverage.loc["innervation_EB", "n_present"] == 3
    assert coverage.loc["innervation_PB", "n_present"] == 0
    assert coverage.loc["innervation_NO", "n_present"] == 0


def test_build_features_cli_writes_root_ids_schema_and_coverage(tmp_path):
    from scripts.build_features import main

    graph_dir = tmp_path / "graph"
    _graph().write(graph_dir)
    annotations = tmp_path / "f1.parquet"
    _f1_annotations().to_parquet(annotations, index=False)
    out = tmp_path / "out"

    main([
        "--graph", str(graph_dir), "--annotations", str(annotations),
        "--dataset", "f1", "--out", str(out),
    ])

    features = pd.read_parquet(out / "features.parquet")
    assert list(features["root_id"]) == [1, 2, 3]
    assert (out / "schema.json").exists()
    assert (out / "coverage.csv").exists()
    written = FeatureSchema.from_json((out / "schema.json").read_text())
    assert written.digest() == FeatureSchema.from_f1(_f1_annotations(), "CX").digest()


def test_build_features_cli_f2_requires_f1_and_maps_unknown_to_oov(tmp_path):
    from scripts.build_features import main

    graph_dir = tmp_path / "graph"
    _graph_for("f2").write(graph_dir)
    f1_path = tmp_path / "f1.parquet"
    _f1_annotations().to_parquet(f1_path, index=False)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(FeatureSchema.from_f1(_f1_annotations(), "CX").to_json())
    f2_path = tmp_path / "f2.parquet"
    _f1_annotations().iloc[:3].assign(
        fafb_match_id=[1, 999, 3], root_id=[1, 2, 3], cell_type="banc_only"
    ).to_parquet(f2_path, index=False)

    with pytest.raises(SystemExit, match="2") as error:
        main([
            "--graph", str(graph_dir), "--annotations", str(f2_path),
            "--dataset", "f2", "--schema", str(schema_path),
            "--out", str(tmp_path / "missing-f1"),
        ])
    assert error.value.code == 2

    out = tmp_path / "f2-out"
    main([
        "--graph", str(graph_dir), "--annotations", str(f2_path),
        "--dataset", "f2", "--schema", str(schema_path),
        "--f1-annotations", str(f1_path), "--out", str(out),
    ])
    features = pd.read_parquet(out / "features.parquet")
    assert list(features["root_id"]) == [1, 2, 3]
    assert features.loc[0, "cell_type"] == 1
    assert features.loc[1, "cell_type"] == 0


def test_build_features_cli_aligns_innervation_root_id_alias(tmp_path):
    from scripts.build_features import main

    graph_dir = tmp_path / "graph"
    _graph().write(graph_dir)
    annotations = tmp_path / "f1.parquet"
    _f1_annotations().to_parquet(annotations, index=False)
    innervation = tmp_path / "innervation.parquet"
    pd.DataFrame({"pt_root_id": [2, 3, 1], "FB": [2, 0, 8], "EB": [8, 10, 2]}).to_parquet(innervation, index=False)
    out = tmp_path / "out"

    main(["--graph", str(graph_dir), "--annotations", str(annotations), "--dataset", "f1",
          "--innervation", str(innervation), "--out", str(out)])

    features = pd.read_parquet(out / "features.parquet").set_index("root_id")
    assert features.loc[1, "innervation_FB"] == pytest.approx(0.8)
    assert features.loc[2, "innervation_FB"] == pytest.approx(0.2)
    assert features.loc[3, "innervation_FB"] == pytest.approx(0.0)
    coverage = pd.read_csv(out / "coverage.csv").set_index("feature")
    assert coverage.loc["innervation_FB", "n_present"] == 3


def test_build_features_cli_scans_synapses_and_aggregates_geometry(tmp_path):
    from scripts.build_features import main

    graph_dir = tmp_path / "graph"
    _graph().write(graph_dir)
    annotations = tmp_path / "f1.parquet"
    _f1_annotations().to_parquet(annotations, index=False)
    synapses = tmp_path / "synapses.feather"
    pd.DataFrame({"pt_root_id": [1, 1, 2, 2, 3, 3],
                  "x": [0.0, 2.0, 10.0, 12.0, 20.0, 22.0],
                  "y": [0.0] * 6, "z": [0.0] * 6}).to_feather(synapses)
    out = tmp_path / "out"

    main(["--graph", str(graph_dir), "--annotations", str(annotations), "--dataset", "f1",
          "--synapses", str(synapses), "--out", str(out)])

    features = pd.read_parquet(out / "features.parquet").set_index("root_id")
    assert features.loc[1, "centroid_x"] == pytest.approx(-1.22474487, abs=1e-6)
    assert features.loc[2, "centroid_x"] == pytest.approx(0.0, abs=1e-6)
    assert features.loc[3, "centroid_x"] == pytest.approx(1.22474487, abs=1e-6)
    geometry = ["centroid_x", "centroid_y", "centroid_z", "extent_x", "extent_y", "extent_z"]
    assert features.loc[2, geometry].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert features["geometry_present"].all(), "a neuron at the z-scored origin is located"


def test_build_features_cli_writes_which_neurons_have_geometry(tmp_path):
    from scripts.build_features import main

    graph_dir = tmp_path / "graph"
    _graph().write(graph_dir)
    annotations = tmp_path / "f1.parquet"
    _f1_annotations().to_parquet(annotations, index=False)
    synapses = tmp_path / "synapses.feather"
    pd.DataFrame({"pt_root_id": [1, 1, 2, 2], "x": [0.0, 2.0, 10.0, 12.0],
                  "y": [0.0] * 4, "z": [0.0] * 4}).to_feather(synapses)
    out = tmp_path / "out"

    main(["--graph", str(graph_dir), "--annotations", str(annotations), "--dataset", "f1",
          "--synapses", str(synapses), "--out", str(out)])

    features = pd.read_parquet(out / "features.parquet").set_index("root_id")
    assert features["geometry_present"].to_dict() == {1: True, 2: True, 3: False}


@pytest.mark.parametrize("neuropil_column", ["neuropil", "neuropil_name", "roi", "compartment"])
def test_geometry_excludes_other_region_synapses(neuropil_column):
    schema = FeatureSchema.from_f1(_f1_annotations(), "CX")
    synapses = pd.DataFrame({
        "root_id": [1, 1, 2, 2, 3, 3, 1],
        "x": [0., 2., 10., 14., 20., 26., 1000.],
        "y": [0.] * 7, "z": [0.] * 7,
        neuropil_column: ["FB", "EB", "PB", "NO", "FB", "EB", "AL_L"],
    })
    table = build_features(_f1_annotations(), _graph(), schema, dataset="f1", synapses=synapses)
    # In-region means 1, 12, 23 and SDs 1, 2, 3 each standardize identically.
    for name in ("centroid_x", "extent_x"):
        assert table.values[:, table.columns.index(name)] == pytest.approx([-1.2247448714, 0., 1.2247448714])


def test_streamed_geometry_filters_regions_across_record_batches(tmp_path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from scripts.build_features import stream_synapse_geometry

    frame = pd.DataFrame({
        "pt_root_id": [1, 2, 3, 1, 2, 3, 1],
        "x_nm": [0., 10., 20., 2., 14., 26., 1000.],
        "y_nm": [0.] * 7, "z_nm": [0.] * 7,
        "roi": ["FB", "PB", "FB", "EB", "NO", "EB", "AL_L"],
    })
    arrow = pa.Table.from_pandas(frame)
    path = tmp_path / "multiple-batches.feather"
    with ipc.new_file(path, arrow.schema) as writer:
        for batch in arrow.to_batches(max_chunksize=3):
            writer.write_batch(batch)
    reduced = stream_synapse_geometry(path, _graph())
    assert reduced["root_id"].tolist() == [1, 2, 3]
    assert reduced["centroid_x"].tolist() == pytest.approx([1., 12., 23.])
    assert reduced["extent_x"].tolist() == pytest.approx([1., 2., 3.])


def test_cli_csv_crosswalk_preserves_adjacent_large_match_ids_and_missing(tmp_path):
    from dataclasses import replace
    from scripts.build_features import main

    ids = [72057594037927937, 72057594037927938, 72057594037927939]
    graph = replace(_graph_for("f2"), node_ids=np.array(ids, dtype=np.int64))
    graph.write(tmp_path / "graph")
    f1 = _f1_annotations().iloc[:3].assign(root_id=ids, cell_type=["EPG", "PEN_a", "R1"])
    schema = FeatureSchema.from_f1(f1, "CX")
    (tmp_path / "schema.json").write_text(schema.to_json())
    f1.to_csv(tmp_path / "f1.csv", index=False)
    f2 = f1.assign(fafb_match_id=pd.array([ids[0], ids[1], None], dtype="Int64"), cell_type="banc_only")
    f2.to_csv(tmp_path / "f2.csv", index=False)
    main(["--graph", str(tmp_path / "graph"), "--annotations", str(tmp_path / "f2.csv"),
          "--dataset", "f2", "--schema", str(tmp_path / "schema.json"),
          "--f1-annotations", str(tmp_path / "f1.csv"), "--out", str(tmp_path / "out")])
    features = pd.read_parquet(tmp_path / "out/features.parquet")
    assert features["root_id"].tolist() == ids
    assert features["cell_type"].tolist() == [schema.type_id("EPG"), schema.type_id("PEN_a"), 0.]


def test_duplicate_annotation_root_ids_fail_with_a_named_error():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    duplicated = pd.concat([_f1_annotations().iloc[:3], _f1_annotations().iloc[[0]]])
    with pytest.raises(ValueError, match="annotations contain duplicate root id"):
        build_features(duplicated, _graph(), schema, dataset="f1")


def test_duplicate_preaggregated_synapse_roots_fail_with_a_named_error():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    statistics = pd.DataFrame({
        "root_id": [1, 1, 2],
        "centroid_x": [0., 1., 2.], "centroid_y": [0., 0., 0.],
        "centroid_z": [0., 0., 0.], "extent_x": [0., 0., 0.],
        "extent_y": [0., 0., 0.], "extent_z": [0., 0., 0.],
    })
    with pytest.raises(ValueError, match="preaggregated synapse statistics contain duplicate root id"):
        build_features(_f1_annotations(), _graph(), schema, dataset="f1", synapses=statistics)


def test_duplicate_innervation_roots_fail_with_a_named_error():
    schema = FeatureSchema.from_f1(_f1_annotations(), region="CX")
    innervation = pd.DataFrame({"FB": [8., 2.], "EB": [2., 8.]}, index=[1, 1])
    with pytest.raises(ValueError, match="duplicate root id"):
        build_features(_f1_annotations(), _graph(), schema, dataset="f1", innervation=innervation)


def test_streamed_geometry_equates_to_direct_aggregation(tmp_path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from scripts.build_features import stream_synapse_geometry

    rng = np.random.default_rng(20260818)
    n = 400
    frame = pd.DataFrame({
        "root_id": rng.integers(1, 4, n),
        "x": rng.normal(1e6, 2e3, n), "y": rng.normal(-5e5, 1e2, n),
        "z": rng.normal(0., 5., n),
        "neuropil": rng.choice(["FB", "EB", "AL_L"], n),
    })
    arrow = pa.Table.from_pandas(frame)
    path = tmp_path / "synapses.feather"
    with ipc.new_file(path, arrow.schema) as writer:
        for batch in arrow.to_batches(max_chunksize=37):
            writer.write_batch(batch)

    reduced = stream_synapse_geometry(path, _graph()).set_index("root_id")

    for root in (1, 2, 3):
        own = frame[(frame["root_id"] == root) & frame["neuropil"].isin(("FB", "EB"))]
        assert len(own)
        for axis in ("x", "y", "z"):
            assert reduced.loc[root, f"centroid_{axis}"] == pytest.approx(
                own[axis].mean(), rel=1e-9, abs=1e-6
            )
            assert reduced.loc[root, f"extent_{axis}"] == pytest.approx(
                own[axis].std(ddof=0), rel=1e-9, abs=1e-6
            )


def test_streamed_geometry_accepts_uppercase_feather_extensions(tmp_path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from scripts.build_features import stream_synapse_geometry

    arrow = pa.Table.from_pandas(pd.DataFrame(
        {"root_id": [1, 2, 3], "x": [0., 10., 20.], "y": [0.] * 3, "z": [0.] * 3}
    ))
    path = tmp_path / "SYNAPSES.FEATHER"
    with ipc.new_file(path, arrow.schema) as writer:
        writer.write_batch(arrow.to_batches()[0])

    reduced = stream_synapse_geometry(path, _graph())

    assert reduced["centroid_x"].tolist() == pytest.approx([0., 10., 20.])


@pytest.mark.parametrize("case", ["omitted", "all_null", "zero_total"])
def test_innervation_defaults_each_unusable_row_and_marks_coverage_absent(case):
    schema = FeatureSchema.from_f1(_f1_annotations(), "CX")
    inn = pd.DataFrame({"FB": [8., 2., 0.], "EB": [2., 8., 10.],
                        "PB": [0., 0., 0.], "NO": [0., 0., 0.]}, index=[1, 2, 3])
    if case == "omitted":
        inn = inn.drop(index=2)
    else:
        inn.loc[2] = np.nan if case == "all_null" else 0.
    table = build_features(_f1_annotations(), _graph(), schema, dataset="f1", innervation=inn)
    positions = [table.columns.index("innervation_" + label) for label in schema.labels]
    assert table.values[:, positions] == pytest.approx(np.array([[.8, .2, 0., 0.], [.25] * 4, [0., 1., 0., 0.]]))
    coverage = table.coverage.set_index("feature")
    for label in schema.labels:
        assert coverage.loc["innervation_" + label, "n_present"] == 2


@pytest.mark.parametrize("field,prefix", [("super_class", "superclass"), ("class", "class")])
def test_missing_and_unseen_class_values_use_oov_with_distinct_coverage(field, prefix):
    schema = FeatureSchema.from_f1(_f1_annotations(), "CX")
    annotations = _f1_annotations()
    annotations.loc[0, field] = None
    annotations.loc[1, field] = "unseen"
    table = build_features(annotations, _graph_for("f2"), schema, dataset="f2")
    assert table.values[:, table.columns.index(prefix + "_OOV")].tolist() == [1., 1., 0.]
    assert table.coverage.set_index("feature").loc[field, "n_present"] == 2


@pytest.mark.parametrize("alias", ["nt_{}", "{}_probability"])
def test_nt_fractional_probabilities_take_precedence_over_categorical_values(alias):
    schema = FeatureSchema.from_f1(_f1_annotations(), "CX")
    annotations = _f1_annotations().iloc[:3].copy()
    probability = np.array([[.25, .75, 0., 0., 0., 0.], [0., .4, .6, 0., 0., 0.], [0., 0., 0., .2, .3, .5]])
    for index, label in enumerate(NT_CLASSES):
        annotations[alias.format(label)] = probability[:, index]
    table = build_features(annotations, _graph_for("f2"), schema, dataset="f2")
    assert table.values[:, table.blocks["nt"]] == pytest.approx(np.column_stack([probability, np.zeros(3)]))


def test_nt_probabilities_support_other_classes_without_categorical_input(tmp_path):
    from scripts.build_features import main

    annotations = _f1_annotations().iloc[:3].drop(columns="nt_type")
    for label in NT_CLASSES:
        annotations["nt_" + label] = 0.
    annotations["nt_histamine"] = [1., 0., .25]
    annotations["nt_tyramine"] = [0., 1., 0.]
    annotations["nt_gaba"] = [0., 0., .75]
    annotations.to_parquet(tmp_path / "annotations.parquet")
    _graph().write(tmp_path / "graph")
    main(["--graph", str(tmp_path / "graph"), "--annotations", str(tmp_path / "annotations.parquet"),
          "--dataset", "f1", "--out", str(tmp_path / "out")])
    output = pd.read_parquet(tmp_path / "out/features.parquet")
    assert output["nt_other"].tolist() == [1., 1., .25]
    assert output["nt_gaba"].tolist() == [0., 0., .75]


@pytest.mark.parametrize("case", ["partial_columns", "partial_row", "negative", "nonfinite", "bad_total", "nonnumeric"])
def test_invalid_nt_probability_inputs_fail_loud(case):
    schema = FeatureSchema.from_f1(_f1_annotations(), "CX")
    annotations = _f1_annotations().iloc[:3].copy()
    for label in NT_CLASSES:
        annotations["nt_" + label] = 1. if label == "gaba" else 0.
    if case == "partial_columns":
        annotations = annotations.drop(columns="nt_dopamine")
    elif case == "partial_row":
        annotations.loc[0, "nt_dopamine"] = np.nan
    elif case == "nonnumeric":
        annotations["nt_gaba"] = ["bad", "1", "1"]
    else:
        annotations.loc[0, "nt_gaba"] = {"negative": -1., "nonfinite": np.inf, "bad_total": .5}[case]
    with pytest.raises(ValueError, match="probabilit"):
        build_features(annotations, _graph_for("f2"), schema, dataset="f2")


def test_builder_rejects_schema_for_another_region():
    from dataclasses import replace
    schema = replace(FeatureSchema.from_f1(_f1_annotations(), "CX"), region="AL")
    with pytest.raises(ValueError, match="region"):
        build_features(_f1_annotations(), _graph(), schema, dataset="f1")


def test_cli_crosswalk_retains_small_float_match_ids_from_parquet(tmp_path):
    from scripts.build_features import main

    _graph_for("f2").write(tmp_path / "graph")
    f1 = _f1_annotations()
    f1.to_parquet(tmp_path / "f1.parquet")
    schema = FeatureSchema.from_f1(f1, "CX")
    (tmp_path / "schema.json").write_text(schema.to_json())
    f1.iloc[:3].assign(fafb_match_id=[1., 3., np.nan], cell_type="banc_only").to_parquet(tmp_path / "f2.parquet")
    main(["--graph", str(tmp_path / "graph"), "--annotations", str(tmp_path / "f2.parquet"),
          "--dataset", "f2", "--schema", str(tmp_path / "schema.json"),
          "--f1-annotations", str(tmp_path / "f1.parquet"), "--out", str(tmp_path / "out")])
    features = pd.read_parquet(tmp_path / "out/features.parquet")
    assert features["cell_type"].tolist() == [schema.type_id("EPG"), schema.type_id("PEN_a"), 0.]


def test_cli_csv_crosswalk_normalizes_integral_decimal_match_ids(tmp_path):
    from scripts.build_features import main

    _graph_for("f2").write(tmp_path / "graph")
    f1 = _f1_annotations()
    f1.to_csv(tmp_path / "f1.csv", index=False)
    schema = FeatureSchema.from_f1(f1, "CX")
    (tmp_path / "schema.json").write_text(schema.to_json())
    f2 = f1.iloc[:3].copy()
    f2["fafb_match_id"] = pd.Series(["1.0", "", "3.0"], dtype="string")
    f2["cell_type"] = "banc_only"
    f2.to_csv(tmp_path / "f2.csv", index=False)

    main(["--graph", str(tmp_path / "graph"), "--annotations", str(tmp_path / "f2.csv"),
          "--dataset", "f2", "--schema", str(tmp_path / "schema.json"),
          "--f1-annotations", str(tmp_path / "f1.csv"), "--out", str(tmp_path / "out")])

    features = pd.read_parquet(tmp_path / "out/features.parquet")
    assert features["cell_type"].tolist() == [schema.type_id("EPG"), 0., schema.type_id("PEN_a")]


# --- Pinned F1 input-schema regression tests (PR #24 review) ---
# These frames mirror, column-for-column, the real F1 files so the builder is exercised against the inputs it ships to
# consume, not only against the toy schemas of the tests above.


def _pinned_f1_annotations() -> pd.DataFrame:
    """Columns as in Supplemental_file1_neuron_annotations.tsv (Schlegel v3.1.0)."""
    return pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4],
            "super_class": ["central", "central", "central", "sensory"],
            "cell_class": ["CX", "CX", "CX", "visual"],
            "cell_type": ["EPG", "EPG", "PEN_a", "R1"],
            "side": ["center", "center", "left", "right"],
            "flow": ["intrinsic", "intrinsic", "intrinsic", "afferent"],
            "top_nt": ["acetylcholine", "glutamate", "gaba", "histamine"],
        }
    )


def test_schema_builds_from_the_pinned_f1_annotation_header():
    schema = FeatureSchema.from_f1(_pinned_f1_annotations(), region="CX")
    assert schema.superclass_vocab == ("central", "sensory")
    assert schema.class_vocab == ("CX", "visual")
    table = build_features(_pinned_f1_annotations(), _graph(), schema, dataset="f1")
    assert table.values[0, table.columns.index("nt_acetylcholine")] == 1.0
    assert table.values[1, table.columns.index("nt_glutamate")] == 1.0


def _write_feather(frame: pd.DataFrame, path, *, batch_rows: int = 1000) -> None:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    arrow = pa.Table.from_pandas(frame, preserve_index=False)
    with ipc.new_file(path, arrow.schema) as writer:
        for batch in arrow.to_batches(max_chunksize=batch_rows):
            writer.write_batch(batch)


def _pinned_synapses() -> pd.DataFrame:
    """Columns as in flywire_synapses_783.feather (per-synapse pre/post ends)."""
    return pd.DataFrame(
        {
            "id": [100, 101, 102, 103],
            "pre_pt_root_id": [1, 2, 3, 1],
            "post_pt_root_id": [2, 3, 1, 2],
            "neuropil": ["FB", "EB", "FB", "AL_L"],
            "pre_pt_position_x": [0, 12, 22, 999],
            "pre_pt_position_y": [0, 0, 0, 0],
            "pre_pt_position_z": [0, 0, 0, 0],
            "post_pt_position_x": [10, 20, 2, 999],
            "post_pt_position_y": [0, 0, 0, 0],
            "post_pt_position_z": [0, 0, 0, 0],
        }
    )


def test_streamed_geometry_reads_pinned_pre_post_synapse_endpoints(tmp_path):
    from scripts.build_features import stream_synapse_geometry

    path = tmp_path / "flywire_synapses_783.feather"
    _write_feather(_pinned_synapses(), path, batch_rows=2)
    reduced = stream_synapse_geometry(path, _graph()).set_index("root_id")

    # A neuron's own in-region locations are its pre ends (where it is
    # presynaptic) unioned with its post ends (where it is postsynaptic);
    # the out-of-region synapse 103 (AL_L) drops out. Using the synapse `id`
    # column as the neuron id, or only one endpoint, would miss these.
    assert reduced.loc[1, "centroid_x"] == pytest.approx(1.0)  # {0, 2}
    assert reduced.loc[2, "centroid_x"] == pytest.approx(11.0)  # {12, 10}
    assert reduced.loc[3, "centroid_x"] == pytest.approx(21.0)  # {22, 20}
    for root in (1, 2, 3):
        assert reduced.loc[root, "extent_x"] == pytest.approx(1.0)


def test_cli_combines_pinned_long_form_pre_post_innervation(tmp_path):
    from scripts.build_features import main

    _graph().write(tmp_path / "graph")
    _f1_annotations().to_parquet(tmp_path / "f1.parquet")
    pre = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2, 3],
            "neuropil": ["FB", "EB", "FB", "EB"],
            "count": [3, 1, 2, 4],
        }
    )
    post = pd.DataFrame(
        {
            "post_pt_root_id": [1, 2, 3],
            "neuropil": ["EB", "EB", "FB"],
            "count": [1, 2, 1],
        }
    )
    pre.to_feather(tmp_path / "count_pre.feather")
    post.to_feather(tmp_path / "count_post.feather")
    out = tmp_path / "out"

    main(
        [
            "--graph", str(tmp_path / "graph"),
            "--annotations", str(tmp_path / "f1.parquet"),
            "--dataset", "f1",
            "--innervation",
            str(tmp_path / "count_pre.feather"),
            str(tmp_path / "count_post.feather"),
            "--out", str(out),
        ]
    )

    features = pd.read_parquet(out / "features.parquet").set_index("root_id")
    # Combined FB/EB counts: 1 -> 3/2, 2 -> 2/2, 3 -> 1/4; normalized per neuron.
    assert features.loc[1, "innervation_FB"] == pytest.approx(0.6)
    assert features.loc[1, "innervation_EB"] == pytest.approx(0.4)
    assert features.loc[2, "innervation_FB"] == pytest.approx(0.5)
    assert features.loc[3, "innervation_FB"] == pytest.approx(0.2)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_build_features_rejects_nonfinite_synapse_coordinates(bad):
    schema = FeatureSchema.from_f1(_f1_annotations(), "CX")
    synapses = pd.DataFrame(
        {
            "root_id": [1, 1, 2, 2, 3, 3],
            "x": [0.0, bad, 10.0, 12.0, 20.0, 22.0],
            "y": [0.0] * 6,
            "z": [0.0] * 6,
        }
    )
    with pytest.raises(ValueError, match="finite"):
        build_features(_f1_annotations(), _graph(), schema, dataset="f1", synapses=synapses)


@pytest.mark.parametrize("bad", [-1.0, np.inf])
def test_build_features_rejects_invalid_innervation_counts(bad):
    schema = FeatureSchema.from_f1(_f1_annotations(), "CX")
    innervation = pd.DataFrame(
        {"FB": [8.0, 2.0, bad], "EB": [2.0, 8.0, 10.0], "PB": [0.0] * 3, "NO": [0.0] * 3},
        index=[1, 2, 3],
    )
    with pytest.raises(ValueError, match="innervation"):
        build_features(_f1_annotations(), _graph(), schema, dataset="f1", innervation=innervation)


def test_build_features_rejects_a_graph_from_another_dataset():
    schema = FeatureSchema.from_f1(_f1_annotations(), "CX")
    with pytest.raises(ValueError, match="dataset"):
        build_features(_f1_annotations(), _graph_for("f2"), schema, dataset="f1")


def test_cli_rejects_a_graph_whose_dataset_disagrees_with_the_flag(tmp_path):
    from scripts.build_features import main

    _graph_for("f2").write(tmp_path / "graph")
    _f1_annotations().to_parquet(tmp_path / "ann.parquet")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--graph", str(tmp_path / "graph"),
                "--annotations", str(tmp_path / "ann.parquet"),
                "--dataset", "f1",
                "--out", str(tmp_path / "out"),
            ]
        )
    assert error.value.code == 2
