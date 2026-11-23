"""Seams of the Phase 3 union graph (plan 01): raw counts -> union, regional digests, union features."""

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureSchema
from src.regions import RegionGraph, build_region_graph, region_labels
from src.union import CODES, boundary_flags, census, pair_counts, union_graph, union_labels, union_members

B = 10**17  # root ids far outside [0, 2**32), as in F1


@pytest.fixture
def connections():
    rows = [
        (1, 2, "FB", 3), (1, 2, "LH_L", 2),  # 3 + 2 across CX and LH: boundary
        (1, 3, "FB", 6), (1, 3, "LH_L", 1),  # 6 in CX alone: regional
        (2, 3, "LH_L", 4), (2, 3, "LH_R", 4),  # 4 + 4 on two LH labels: regional
        (3, 1, "OCG", 4), (3, 1, "FB", 2), (3, 1, "UNASGD", 3),  # 5+ only with excluded labels
        (4, 5, "AL_L", 2), (4, 5, "FB", 2), (4, 5, "OCG", 1),  # 4 registered + 1 OCG
        (6, 7, "OCG", 5),  # 5 through OCG alone
    ]
    pre, post, neuropil, count = zip(*rows)
    return pd.DataFrame({"pre_pt_root_id": np.add(pre, B), "post_pt_root_id": np.add(post, B),
                         "neuropil": neuropil, "syn_count": count})


def test_split_pair_is_a_boundary_edge_and_excluded_labels_never_count(connections):
    pc = pair_counts(connections)
    graph = union_graph(pc, "UNION12")
    assert graph.node_ids.tolist() == [B + 1, B + 2, B + 3]  # 6->7 (OCG only) and 4->5 (4 registered) absent
    assert graph.edges.tolist() == [[0, 1], [0, 2], [1, 2]] and graph.weights.tolist() == [5, 7, 8]
    assert boundary_flags(graph, pc).tolist() == [True, False, False]
    for code in ("CX", "LH"):
        regional = build_region_graph(connections, region=code, dataset="f1")
        assert (B + 1, B + 2) not in {tuple(regional.node_ids[e]) for e in regional.edges}


def test_a_null_neuropil_label_is_refused_not_counted(connections):
    # A null's category code is -1; indexing with it would count the 2 synapses under the last label.
    nulled = connections.astype({"neuropil": object})
    nulled.loc[1, "neuropil"] = None  # 1->2: FB 3 + null 2 must not become a union edge of 5
    with pytest.raises(ValueError, match="no neuropil label"):
        pair_counts(nulled)


def test_census_counts_on_a_toy_table(connections):
    result = census(pair_counts(connections), ["CX", "LH"])
    assert result["union"] | {"digest": None} == {"name": "CX+LH", "edges": 3, "nodes": 3, "synapses": 20,
                                                  "digest": None}
    assert result["boundary"]["support"] == [
        {"regions": "CX+LH", "edges": 1, "edge_share": 1.0, "synapses": 5, "synapse_share": 1.0}]
    assert result["nodes"] == {"in_any_region_graph": 3, "shared": 1, "shared_share": 1 / 3, "union_only": 0}
    assert result["pairwise_boundary"] == {"CX-LH": 1}
    assert census(pair_counts(connections), CODES)["global"] == {
        "edges": 6, "in_a_region_graph": 2, "boundary": 4, "registered_labels_only": 1,
        "needs_excluded_labels": 3, "region_graph_neurons": 3, "in_two_or_more": 1}


def test_regional_digests_match_build_region_graph_and_never_a_union(connections):
    pc = pair_counts(connections)
    regional = census(pc, CODES)["regional"]
    unions = {union_graph(pc, name).digest() for name in ("UNION12", "UNION11", "CX+LH")}
    for code in ("CX", "LH"):
        digest = build_region_graph(connections, region=code, dataset="f1").digest()
        assert regional[code]["digest"] == digest and digest not in unions
    assert census(pc, ["CX", "LH"])["union"]["digest"] == union_graph(pc, "CX+LH").digest()


@pytest.mark.parametrize("name", ["CX", "LH+CX", "CX+CX", "CX+NOPE"])
def test_union_names_are_two_or_more_sorted_codes(name):
    with pytest.raises(ValueError):
        union_members(name)


def test_union_features_align_with_nodes_and_use_f1_vocabularies(connections, tmp_path):
    from scripts.build_features import main

    graph = union_graph(pair_counts(connections), "CX+LH")
    graph.write(tmp_path / "u")
    ids = [B + 3, B + 1, B + 2, B + 9]  # shuffled, plus a neuron outside the graph
    pd.DataFrame({"root_id": ids, "super_class": ["central", "optic", "central", "sensory"],
                  "class": ["a", "b", "a", "c"], "cell_type": ["t3", "t1", "t2", "t9"],
                  "side": ["left", "right", "left", "left"], "flow": ["intrinsic"] * 4,
                  "nt_type": ["ACH", "GABA", "GLUT", "ACH"]}).to_csv(tmp_path / "ann.tsv", sep="\t", index=False)
    pd.DataFrame({"pre_pt_root_id": [B + 1, B + 1, B + 2, B + 9], "neuropil": ["FB", "LH_L", "LH_R", "FB"],
                  "count": [3, 1, 2, 7]}).to_feather(tmp_path / "pre.feather")
    main(["--graph", str(tmp_path / "u"), "--annotations", str(tmp_path / "ann.tsv"), "--dataset", "f1",
          "--out", str(tmp_path / "u"), "--innervation", str(tmp_path / "pre.feather")])

    schema = FeatureSchema.from_json((tmp_path / "u/schema.json").read_text(encoding="utf-8"))
    features = pd.read_parquet(tmp_path / "u/features.parquet")
    assert schema.region == "CX+LH" and schema.labels == union_labels("CX+LH")
    assert schema.labels == region_labels("CX") + region_labels("LH")
    assert schema.type_vocab == {"t1": 1, "t2": 2, "t3": 3, "t9": 4}  # F1-wide, not just graph nodes
    assert features["root_id"].tolist() == RegionGraph.read(tmp_path / "u").node_ids.tolist()
    assert features["cell_type"].tolist() == [1, 2, 3]
    assert features.loc[0, "innervation_FB"] == pytest.approx(0.75)
    assert features.loc[1, "innervation_LH_R"] == pytest.approx(1.0)


def test_innervation_restricted_to_graph_nodes_keeps_values_and_columns(tmp_path):
    from scripts.build_features import read_innervation

    path = tmp_path / "pre.feather"
    pd.DataFrame({"pre_pt_root_id": [B + 1, B + 1, B + 2, B + 9], "neuropil": ["FB", "EB", "FB", "PB"],
                  "count": [3, 1, 2, 7]}).to_feather(path)
    labels = ("FB", "EB", "PB", "NO")
    full = read_innervation(path, labels)
    restricted = read_innervation(path, labels, root_ids=[B + 1, B + 2])
    pd.testing.assert_frame_equal(restricted, full.loc[[B + 1, B + 2]])
