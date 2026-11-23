"""The suite-input producer covers every graph node, or says which it cannot."""

import numpy as np
import pandas as pd
import pytest

from scripts.build_features import read_table
from scripts.prepare_suite_inputs import build_centroids, build_sides, build_types, main
from src.regions import build_region_graph

# FlyWire root ids are 18 digits, past float64's exact-integer range.
_ROOT_IDS = [720575940633242449, 720575940633242450, 720575940633242451]


def _graph(root_ids: list[int] | None = None):
    ids = _ROOT_IDS if root_ids is None else root_ids
    return build_region_graph(
        pd.DataFrame(
            {
                "pre_pt_root_id": [ids[0], ids[1], ids[2]],
                "post_pt_root_id": [ids[1], ids[2], ids[0]],
                "neuropil": ["FB", "EB", "FB"],
                "syn_count": [9, 9, 9],
            }
        ),
        region="CX",
        dataset="f1",
        threshold=5,
    )


def _synapses(root_ids: list[int]) -> pd.DataFrame:
    """Two single-endpoint positions per neuron, so each mean sits mid-pair."""
    rows = [
        {"pt_root_id": root, "x": 10.0 * i + offset, "y": 0.0, "z": 0.0}
        for i, root in enumerate(root_ids)
        for offset in (0.0, 2.0)
    ]
    return pd.DataFrame(rows)


def test_cli_writes_full_coverage_types_and_centroids(tmp_path):
    graph = _graph()
    graph_dir = tmp_path / "graph"
    graph.write(graph_dir)
    annotations = tmp_path / "annotations.parquet"
    pd.DataFrame(
        {"root_id": _ROOT_IDS[:2], "cell_type": ["EPG", "PEN_a"]}
    ).to_parquet(annotations, index=False)
    synapses = tmp_path / "synapses.feather"
    _synapses(_ROOT_IDS).to_feather(synapses)

    assert main([
        "--graph", str(graph_dir), "--annotations", str(annotations),
        "--synapses", str(synapses), "--out", str(tmp_path / "out"),
    ]) == 0

    types = pd.read_parquet(tmp_path / "out" / "f1_types.parquet")
    assert types["root_id"].dtype == np.dtype("int64")
    assert types["root_id"].tolist() == _ROOT_IDS
    assert types["cell_type"].tolist()[:2] == ["EPG", "PEN_a"]
    assert types["cell_type"].isna().tolist() == [False, False, True]

    centroids = pd.read_parquet(tmp_path / "out" / "f1_centroids.parquet")
    assert list(centroids.columns) == ["root_id", "x", "y", "z"]
    assert centroids["root_id"].tolist() == _ROOT_IDS
    assert centroids["x"].tolist() == [1.0, 11.0, 21.0]

    sides = pd.read_parquet(tmp_path / "out" / "f1_sides.parquet")
    assert sides["root_id"].tolist() == _ROOT_IDS
    assert sides["side"].isna().all()


def test_tsv_annotations_align_on_eighteen_digit_root_ids(tmp_path):
    """The real table is a TSV, read through the shared id-parsing path."""
    annotations = tmp_path / "annotations.tsv"
    pd.DataFrame(
        {"root_id": _ROOT_IDS, "cell_type": ["EPG", "PEN_a", "R1"]}
    ).to_csv(annotations, sep="\t", index=False)

    types = build_types(read_table(annotations), _graph())

    assert types["root_id"].tolist() == _ROOT_IDS
    assert types["cell_type"].tolist() == ["EPG", "PEN_a", "R1"]


def test_annotations_from_another_id_space_fail_loudly():
    annotations = pd.DataFrame({"root_id": [11, 12], "cell_type": ["EPG", "R1"]})
    with pytest.raises(ValueError, match="different id spaces"):
        build_types(annotations, _graph())


def test_a_repeated_root_id_is_rejected_rather_than_resolved(tmp_path):
    annotations = pd.DataFrame(
        {"root_id": [_ROOT_IDS[0], _ROOT_IDS[0]], "cell_type": ["EPG", "PEN_a"]}
    )
    with pytest.raises(ValueError, match="duplicate root id"):
        build_types(annotations, _graph())


def test_types_keep_every_node_when_the_annotation_table_is_short():
    types = build_types(
        pd.DataFrame({"root_id": [_ROOT_IDS[1]], "cell_type": ["EPG"]}), _graph()
    )
    assert types["root_id"].tolist() == _ROOT_IDS
    assert types["cell_type"].notna().tolist() == [False, True, False]


def test_a_node_with_no_synapse_position_is_named_not_dropped(tmp_path):
    synapses = tmp_path / "synapses.feather"
    _synapses(_ROOT_IDS[:2]).to_feather(synapses)
    with pytest.raises(ValueError, match=str(_ROOT_IDS[2])):
        build_centroids(_graph(), synapses)


def test_annotations_without_a_type_column_fail_loudly():
    with pytest.raises(ValueError, match="cell_type"):
        build_types(pd.DataFrame({"root_id": [1], "class": ["CX"]}), _graph())


def test_banc_meta_joins_on_banc_888_id_not_its_later_root_id(tmp_path):
    """BANC meta's root_id is a later id; the graph carries banc_888_id."""
    later = [1, 2, 3]
    meta = pd.DataFrame({
        "root_id": [str(i) for i in later],
        "banc_888_id": [str(i) for i in _ROOT_IDS],
        "side": ["left", "right", "left"],
    })
    path = tmp_path / "banc_888_meta.feather"
    meta.to_feather(path)
    table = read_table(path)
    assert "banc_888_id" not in table.columns
    assert table["root_id"].tolist() == _ROOT_IDS
    assert build_sides(table, _graph())["side"].tolist() == ["L", "R", "L"]


def test_cli_writes_f2_types_in_f1_names_never_raw_ones(tmp_path):
    from dataclasses import replace

    replace(_graph(), dataset="f2").write(tmp_path / "graph")
    f1 = tmp_path / "f1.parquet"
    pd.DataFrame({"root_id": [11, 12], "cell_type": ["EPG", "PEN_a"]}).to_parquet(f1)
    annotations = tmp_path / "annotations.parquet"
    pd.DataFrame({
        "root_id": _ROOT_IDS, "fafb_match": ["11", "12", None], "cell_type": ["a", "b", "c"],
        "side": ["left", "right", "left"], "super_class": "central_brain_intrinsic",
        "cell_class": "central_complex_input_neuron", "flow": "intrinsic",
        "neurotransmitter_predicted": "gaba",
    }).to_parquet(annotations)
    synapses = tmp_path / "synapses.feather"
    _synapses(_ROOT_IDS).to_feather(synapses)
    argv = [
        "--graph", str(tmp_path / "graph"), "--annotations", str(annotations),
        "--synapses", str(synapses), "--out", str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit):
        main(argv)
    assert main([*argv, "--f1-annotations", str(f1)]) == 0
    types = pd.read_parquet(tmp_path / "out" / "f2_types.parquet")
    assert types["root_id"].tolist() == _ROOT_IDS
    assert types["cell_type"].tolist()[:2] == ["EPG", "PEN_a"]
    assert pd.isna(types["cell_type"].iloc[2])
    sides = pd.read_parquet(tmp_path / "out" / "f2_sides.parquet")
    assert sides["side"].tolist() == ["L", "R", "L"]


def test_a_partial_id_match_is_logged(capsys):
    build_sides(pd.DataFrame({"root_id": _ROOT_IDS[:1], "side": ["left"]}), _graph())
    assert "2 of 3 graph nodes have no annotation row" in capsys.readouterr().out


@pytest.mark.parametrize("dataset", ["f1", "f2"])
def test_sides_normalize_annotations_without_losing_root_ids(dataset):
    from dataclasses import replace

    graph = replace(_graph(), dataset=dataset)
    annotations = pd.DataFrame({"root_id": _ROOT_IDS[::-1], "side": ["center", "right", "left"]})
    sides = build_sides(annotations, graph)
    assert sides.root_id.tolist() == _ROOT_IDS
    assert sides.side.iloc[:2].tolist() == ["L", "R"]
    assert pd.isna(sides.side.iloc[2])
    with pytest.raises(ValueError, match="duplicate"):
        build_sides(pd.concat([annotations, annotations.iloc[:1]]), graph)
    with pytest.raises(ValueError, match="different id spaces"):
        build_sides(pd.DataFrame({"root_id": [1], "side": ["left"]}), graph)
