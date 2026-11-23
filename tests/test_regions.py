"""Tests for the region registry."""

import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.regions import (
    EXCLUDED_BASE,
    REGIONS,
    ColumnMap,
    RegionGraph,
    all_labels,
    build_region_graph,
    expand,
    neuropil_to_region,
    region_labels,
)


def test_registry_has_the_twelve_regions_of_regions_md():
    assert set(REGIONS) == {
        "OL", "LX", "SNP", "INP", "VMNP", "VLNP",
        "CX", "MB", "AL", "LH", "PENP", "GNG",
    }


def test_cx_is_the_four_unpaired_midline_neuropils():
    assert REGIONS["CX"].base_neuropils == ("FB", "EB", "PB", "NO")
    assert region_labels("CX") == ("FB", "EB", "PB", "NO")


def test_paired_neuropils_expand_to_left_and_right():
    assert expand("ME") == ("ME_L", "ME_R")
    assert expand("FB") == ("FB",)


def test_all_79_v783_labels_are_accounted_for():
    labels = all_labels()
    assert len(labels) == len(set(labels)) == 79
    assert "UNASGD" in labels
    assert "OCG" in labels


def test_every_label_maps_to_a_region_or_to_none():
    mapping = neuropil_to_region()
    assert set(mapping) == set(all_labels())
    assert mapping["FB"] == "CX"
    assert mapping["ME_L"] == "OL"
    # OCG and UNASGD are outside the registry by design.
    assert mapping["OCG"] is None
    assert mapping["UNASGD"] is None
    assert set(EXCLUDED_BASE) == {"OCG", "UNASGD"}


def test_regions_do_not_share_neuropils():
    seen: set[str] = set()
    for code in REGIONS:
        labels = set(region_labels(code))
        assert not (labels & seen), f"{code} overlaps an earlier region"
        seen |= labels


def _toy_connections() -> pd.DataFrame:
    """Four neurons, with the region-graph answers hand-computed below."""
    return pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 1, 2, 3, 4, 1],
            "post_pt_root_id": [2, 2, 3, 1, 4, 3, 4],
            "neuropil": ["FB", "EB", "FB", "FB", "FB", "EB", "LAL_L"],
            "syn_count": [3, 3, 4, 9, 2, 5, 50],
        }
    )


def test_edges_use_the_within_region_pair_total():
    graph = build_region_graph(
        _toy_connections(), region="CX", dataset="toy", threshold=5
    )
    pairs = {
        (int(graph.node_ids[s]), int(graph.node_ids[t])): int(w)
        for (s, t), w in zip(graph.edges, graph.weights)
    }
    # 1->2 is 3 (FB) + 3 (EB) = 6: over threshold only once summed.
    assert pairs[(1, 2)] == 6
    # 2->1 is 9 in FB.
    assert pairs[(2, 1)] == 9
    # 4->3 is 5 in EB, exactly at the threshold.
    assert pairs[(4, 3)] == 5
    # 1->3 is 4, under. 3->4 is 2, under. 1->4 lives in LAL, another region.
    assert (1, 3) not in pairs
    assert (3, 4) not in pairs
    assert (1, 4) not in pairs
    assert graph.n_edges == 3


def test_nodes_are_exactly_the_neurons_incident_to_a_region_edge():
    graph = build_region_graph(
        _toy_connections(), region="CX", dataset="toy", threshold=5
    )
    assert graph.node_ids.tolist() == [1, 2, 3, 4]
    assert list(graph.node_ids) == sorted(graph.node_ids)


def test_direction_is_never_symmetrised():
    connections = pd.DataFrame(
        {
            "pre_pt_root_id": [7],
            "post_pt_root_id": [8],
            "neuropil": ["FB"],
            "syn_count": [11],
        }
    )
    graph = build_region_graph(connections, region="CX", dataset="toy", threshold=5)
    assert graph.n_edges == 1
    assert graph.in_strength().tolist() == [0, 11]
    assert graph.out_strength().tolist() == [11, 0]


def _pickle_round_trip(graph):
    return pickle.loads(pickle.dumps(graph))


def test_strengths_are_scanned_once_per_direction(monkeypatch):
    graph = build_region_graph(
        _toy_connections(), region="CX", dataset="toy", threshold=5
    )
    scans: list[int] = []
    real_at = np.add.at

    def counting_at(out, indices, values):
        scans.append(int(indices.shape[0]))
        return real_at(out, indices, values)

    monkeypatch.setattr(np.add, "at", counting_at)
    graph.in_strength()
    graph.in_strength()
    assert len(scans) == 1
    graph.out_strength()
    graph.out_strength()
    assert len(scans) == 2


def test_graph_arrays_and_strengths_are_owned_and_read_only():
    graph = build_region_graph(
        _toy_connections(), region="CX", dataset="toy", threshold=5
    )
    for arr in (
        graph.node_ids,
        graph.edges,
        graph.weights,
        graph.in_strength(),
        graph.out_strength(),
    ):
        assert arr.flags.owndata
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = arr[0]


def test_construction_copies_rather_than_freezing_the_caller_arrays():
    node_ids = np.array([1, 2], dtype=np.int64)
    edges = np.array([[0, 1]], dtype=np.int32)
    weights = np.array([7], dtype=np.int64)
    graph = RegionGraph(
        dataset="toy",
        region="CX",
        threshold=5,
        node_ids=node_ids,
        edges=edges,
        weights=weights,
    )
    assert node_ids.flags.writeable
    assert edges.flags.writeable
    assert weights.flags.writeable
    weights[0] = 99
    assert graph.in_strength().tolist() == [0, 7]


@pytest.mark.parametrize("restore", [copy.deepcopy, _pickle_round_trip])
def test_a_restored_graph_is_refrozen_and_drops_the_strength_cache(restore):
    graph = build_region_graph(
        _toy_connections(), region="CX", dataset="toy", threshold=5
    )
    graph.in_strength()
    restored = restore(graph)
    assert "_in_strength" not in restored.__dict__
    assert not restored.weights.flags.writeable
    with pytest.raises(ValueError):
        restored.weights[0] = 99
    assert restored.in_strength().tolist() == [9, 6, 5, 0]


def test_strengths_are_cached_and_read_only():
    graph = build_region_graph(_toy_connections(), region="CX", dataset="toy")
    assert graph.in_strength() is graph.in_strength()
    assert graph.out_strength() is graph.out_strength()
    with pytest.raises(ValueError):
        graph.in_strength()[0] = 1
    with pytest.raises(ValueError):
        graph.out_strength()[0] = 1


def test_graph_arrays_cannot_change_under_the_strength_cache():
    """A cached strength is only valid while the arrays behind it are stable."""
    graph = build_region_graph(_toy_connections(), region="CX", dataset="toy")
    graph.in_strength(), graph.out_strength()
    for array in (graph.node_ids, graph.edges, graph.weights):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 1


def test_column_aliases_resolve_and_unknown_schemas_fail_loudly():
    renamed = _toy_connections().rename(
        columns={"pre_pt_root_id": "pre", "post_pt_root_id": "post"}
    )
    assert (
        build_region_graph(renamed, region="CX", dataset="toy", threshold=5).n_edges
        == 3
    )
    with pytest.raises(ValueError, match="could not resolve"):
        ColumnMap.resolve(["alpha", "beta", "gamma"])


def test_index_of_round_trips_and_rejects_strangers():
    graph = build_region_graph(
        _toy_connections(), region="CX", dataset="toy", threshold=5
    )
    assert graph.index_of(np.array([4, 1])).tolist() == [3, 0]
    with pytest.raises(KeyError):
        graph.index_of(np.array([999]))


def test_unknown_region_and_bad_threshold_are_rejected():
    with pytest.raises(ValueError, match="unknown region"):
        build_region_graph(_toy_connections(), region="NOPE", dataset="toy")
    with pytest.raises(ValueError, match="threshold"):
        build_region_graph(
            _toy_connections(), region="CX", dataset="toy", threshold=0
        )


def test_a_region_absent_from_the_table_is_an_error_not_an_empty_graph():
    with pytest.raises(ValueError, match="no rows in region"):
        build_region_graph(_toy_connections(), region="MB", dataset="toy")


def test_digest_is_stable_and_content_addressed():
    a = build_region_graph(_toy_connections(), region="CX", dataset="toy", threshold=5)
    b = build_region_graph(_toy_connections(), region="CX", dataset="toy", threshold=5)
    c = build_region_graph(_toy_connections(), region="CX", dataset="toy", threshold=6)
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    graph = build_region_graph(
        _toy_connections(), region="CX", dataset="toy", threshold=5
    )
    graph.write(tmp_path)
    back = RegionGraph.read(tmp_path)
    assert back.digest() == graph.digest()
    assert np.array_equal(back.node_ids, graph.node_ids)
    assert np.array_equal(back.edges, graph.edges)
    assert np.array_equal(back.weights, graph.weights)
