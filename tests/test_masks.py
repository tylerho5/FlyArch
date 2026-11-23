"""Masks are neuron-level; everything a scorer sees is post-mask."""

import numpy as np
import pandas as pd
import pytest

import dataclasses
import hashlib
import json

from src.masks import (
    GAP_TAG,
    MASK_REGIMES,
    MaskInstance,
    VisibleView,
    apply_mask,
    pool_mask,
    random_mask,
    spatial_mask,
    spatial_pool_mask,
    typed_mask,
    typed_pool_mask,
)
from src.regions import build_region_graph


def _graph(n=20, dataset="f2"):
    rng = np.random.default_rng(0)
    pre, post = [], []
    for a in range(n):
        for b in rng.choice(n, size=4, replace=False):
            if a != b:
                pre.append(a + 1)
                post.append(int(b) + 1)
    return build_region_graph(pd.DataFrame({"pre_pt_root_id": pre, "post_pt_root_id": post,
        "neuropil": ["FB"] * len(pre), "syn_count": [9] * len(pre)}),
        region="CX", dataset=dataset, threshold=5)


def test_every_edge_touching_a_masked_neuron_is_hidden():
    graph = _graph(); view = apply_mask(graph, random_mask(graph, fraction=0.2, seed=1))
    masked = set(view.instance.masked.tolist())
    assert not any(s in masked or t in masked for s, t in view.visible_edges)
    assert all(s in masked or t in masked for s, t in view.hidden_edges)


def test_visible_and_hidden_partition_the_graph_exactly():
    graph = _graph(); view = apply_mask(graph, random_mask(graph, fraction=0.2, seed=1))
    assert len(view.visible_edges) + len(view.hidden_edges) == graph.n_edges
    assert view.visible_weights.sum() + view.hidden_weights.sum() == graph.weights.sum()


def test_input_fractions_use_visible_weights_only():
    graph = _graph(); view = apply_mask(graph, random_mask(graph, fraction=0.2, seed=1))
    frac = view.visible_input_fraction()
    for target in range(graph.n_nodes):
        if view.is_masked[target]:
            continue
        incoming = frac[np.asarray(view.visible_edges)[:, 1] == target]
        assert incoming.size == 0 or incoming.sum() == pytest.approx(1.0)


def test_degree_given_total_is_the_one_complete_graph_value():
    graph = _graph(); view = apply_mask(graph, random_mask(graph, fraction=0.2, seed=1))
    total = view.degree_given_total(); complete = graph.in_strength() + graph.out_strength()
    assert np.array_equal(total[view.is_masked], complete[view.is_masked])


def test_random_masks_hit_the_requested_fraction_and_are_seed_stable():
    graph = _graph(n=50); a = random_mask(graph, fraction=0.10, seed=7)
    b = random_mask(graph, fraction=0.10, seed=7); c = random_mask(graph, fraction=0.10, seed=8)
    assert a.n_masked == round(0.10 * graph.n_nodes)
    assert np.array_equal(a.masked, b.masked); assert a.digest() == b.digest() != c.digest()
    assert a.instance_id == "f2-cx-R10-s007"


def test_typed_masks_remove_a_whole_cell_type():
    graph = _graph(); types = pd.Series({int(r): ("EPG" if i < 6 else "PEN_a") for i, r in enumerate(graph.node_ids)})
    instance = typed_mask(graph, types, seed=3); chosen = instance.params["cell_type"]
    expected = graph.index_of(types.index[types == chosen].to_numpy())
    assert sorted(instance.masked.tolist()) == sorted(expected.tolist())
    assert instance.instance_id.startswith("f2-cx-T-s")


def test_spatial_masks_take_a_contiguous_ball():
    graph = _graph(n=40); centroids = np.random.default_rng(0).normal(size=(graph.n_nodes, 3))
    instance = spatial_mask(graph, centroids, seed=2, fraction=0.05)
    centre = centroids[instance.params["centre_node"]]; radius = instance.params["radius"]
    inside = np.linalg.norm(centroids - centre, axis=1) <= radius
    assert sorted(instance.masked.tolist()) == sorted(np.flatnonzero(inside).tolist())
    assert instance.n_masked >= max(1, round(0.05 * graph.n_nodes))


def test_pool_mask_sizes_its_draw_over_the_pool_and_carries_the_gap_tag():
    graph = _graph()
    pool = np.arange(graph.n_nodes, dtype=np.int64)[::2]

    instance = pool_mask(graph, np.random.default_rng(0), fraction=0.2, pool=pool)
    again = pool_mask(graph, np.random.default_rng(0), fraction=0.2, pool=pool)

    assert instance.n_masked == round(0.2 * pool.size)
    assert np.isin(instance.masked, pool).all()
    assert instance.instance_id == (
        f"{graph.dataset}-{graph.region.lower()}-{GAP_TAG}-s{instance.seed:03d}"
    )
    assert instance.instance_id == again.instance_id
    assert np.array_equal(instance.masked, again.masked)


def test_pool_mask_defaults_to_every_node():
    graph = _graph()
    instance = pool_mask(graph, np.random.default_rng(3), fraction=0.5)
    assert instance.n_masked == round(0.5 * graph.n_nodes)
    assert instance.n_masked > 0


def test_apply_mask_takes_the_granted_degree_from_the_named_graph():
    graph = _graph()
    instance = random_mask(graph, fraction=0.2, seed=1)
    pruned = dataclasses.replace(graph, edges=graph.edges[::2], weights=graph.weights[::2])

    view = apply_mask(pruned, instance, complete_graph=graph)

    granted = view.degree_given_total()
    complete = graph.in_strength() + graph.out_strength()
    assert np.array_equal(granted[view.is_masked], complete[view.is_masked])
    assert not granted[~view.is_masked].any()
    # Without the handoff the value is the pruned graph's own, which differs.
    relative = apply_mask(pruned, instance).degree_given_total()
    assert not np.array_equal(relative[view.is_masked], granted[view.is_masked])
    with pytest.raises(ValueError, match="node ids"):
        apply_mask(pruned, instance, complete_graph=_graph(n=21))


def test_both_masked_edges_are_flagged_as_their_own_stratum():
    graph = _graph(); view = apply_mask(graph, random_mask(graph, fraction=0.4, seed=5))
    flags = view.both_masked(); assert flags.shape == (len(view.hidden_edges),)
    for (s, t), flag in zip(view.hidden_edges, flags):
        assert flag == (view.is_masked[s] and view.is_masked[t])


def test_post_mask_view_partitions_edges_and_exposes_only_allowed_graph_values():
    graph = _graph(); view = apply_mask(graph, random_mask(graph, fraction=0.4, seed=5))

    masked = view.is_masked
    assert view.visible_edges.dtype == np.int32
    assert view.hidden_edges.dtype == np.int32
    visible_touch = masked[view.visible_edges].any(axis=1)
    hidden_touch = masked[view.hidden_edges].any(axis=1)
    assert not visible_touch.any()
    assert hidden_touch.all()

    original = {(int(s), int(t), int(w)) for (s, t), w in zip(graph.edges, graph.weights)}
    visible = {(int(s), int(t), int(w)) for (s, t), w in zip(view.visible_edges, view.visible_weights)}
    hidden = {(int(s), int(t), int(w)) for (s, t), w in zip(view.hidden_edges, view.hidden_weights)}
    assert visible.isdisjoint(hidden)
    assert visible | hidden == original

    visible_in = view.visible_in_strength()
    visible_out = view.visible_out_strength()
    expected_in = np.zeros(graph.n_nodes, dtype=np.int64)
    expected_out = np.zeros(graph.n_nodes, dtype=np.int64)
    np.add.at(expected_in, view.visible_edges[:, 1], view.visible_weights)
    np.add.at(expected_out, view.visible_edges[:, 0], view.visible_weights)
    assert np.array_equal(visible_in, expected_in)
    assert np.array_equal(visible_out, expected_out)

    fractions = view.visible_input_fraction()
    for target in range(graph.n_nodes):
        incoming = fractions[view.visible_edges[:, 1] == target]
        assert incoming.size == 0 or incoming.sum() == pytest.approx(1.0)

    degree = view.degree_given_total()
    complete = graph.in_strength() + graph.out_strength()
    assert np.array_equal(degree[masked], complete[masked])
    assert np.all(degree[~masked] == 0)

    both = view.both_masked()
    assert both.shape == (view.hidden_edges.shape[0],)
    assert np.array_equal(both, masked[view.hidden_edges].all(axis=1))


def test_mask_digest_is_sha256_of_id_and_contiguous_mask_bytes():
    graph = _graph(n=20)
    instance = random_mask(graph, fraction=0.10, seed=7)
    expected = hashlib.sha256(
        instance.instance_id.encode() + np.ascontiguousarray(instance.masked, dtype=np.int32).tobytes()
    ).hexdigest()
    assert instance.digest() == expected


def test_generators_use_pinned_negative_rng_derivation(monkeypatch):
    import src.negatives as negatives
    graph = _graph(n=20)
    calls = []
    original = negatives._derive_rng

    def record(master_seed, instance_id):
        calls.append((master_seed, instance_id))
        return original(master_seed, instance_id)

    monkeypatch.setattr(negatives, "_derive_rng", record)
    random_mask(graph, fraction=0.10, seed=7)
    assert calls == [(20260818, "f2-cx-R10-s007")]


def test_regimes_and_round_trip_are_json_safe():
    graph = _graph(n=20)
    instance = spatial_mask(graph, np.arange(60, dtype=float).reshape(20, 3), seed=2)
    assert instance.regime in MASK_REGIMES
    payload = instance.to_dict()
    json.dumps(payload)
    restored = MaskInstance.from_dict(payload)
    assert restored.instance_id == instance.instance_id
    assert restored.params == instance.params
    assert np.array_equal(restored.masked, instance.masked)


def test_spatial_tag_is_pinned_to_s05():
    graph = _graph(n=20)
    instance = spatial_mask(graph, np.arange(60, dtype=float).reshape(20, 3), seed=2)
    assert instance.instance_id == "f2-cx-S05-s002"


def test_mask_instance_rejects_unknown_regime():
    with pytest.raises(ValueError, match="regime"):
        MaskInstance("id", "f2", "CX", "X", 1, {}, np.array([], dtype=np.int32))


@pytest.mark.parametrize("masked", [np.array([[0]], dtype=np.int32), np.array([1, 0], dtype=np.int32), np.array([-1], dtype=np.int32), np.array([0], dtype=np.int64)])
def test_mask_instance_rejects_invalid_masked_arrays(masked):
    with pytest.raises(ValueError, match="masked"):
        MaskInstance("id", "f2", "CX", "R", 1, {}, masked)


def test_from_dict_rejects_unsorted_masked_indices():
    payload = {"instance_id": "id", "dataset": "f2", "region": "CX", "regime": "R", "seed": 1, "params": {}, "masked": [1, 0]}
    with pytest.raises(ValueError, match="masked"):
        MaskInstance.from_dict(payload)


def test_empty_mask_round_trips_through_json_dict():
    instance = MaskInstance("id", "f2", "CX", "R", 1, {}, np.array([], dtype=np.int32))
    restored = MaskInstance.from_dict(instance.to_dict())
    assert restored.masked.dtype == np.int32
    assert restored.masked.size == 0
    assert restored.digest() == instance.digest()


def test_from_dict_rejects_nested_empty_mask():
    payload = {"instance_id": "id", "dataset": "f2", "region": "CX", "regime": "R", "seed": 1, "params": {}, "masked": [[]]}
    with pytest.raises(ValueError, match="masked"):
        MaskInstance.from_dict(payload)


def test_mask_instance_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="unique"):
        MaskInstance("id", "f2", "CX", "R", 1, {}, np.array([0, 1, 1], dtype=np.int32))


def test_from_dict_rejects_duplicate_masked_indices():
    payload = {"instance_id": "id", "dataset": "f2", "region": "CX", "regime": "R",
               "seed": 1, "params": {}, "masked": [0, 1, 1]}
    with pytest.raises(ValueError, match="unique"):
        MaskInstance.from_dict(payload)


def test_mask_instance_masked_array_is_read_only():
    inst = MaskInstance("id", "f2", "CX", "R", 1, {}, np.array([0, 1, 2], dtype=np.int32))
    with pytest.raises(ValueError):
        inst.masked[0] = 99


def test_mask_instance_params_are_immutable():
    inst = MaskInstance("id", "f2", "CX", "R", 1, {"fraction": 0.1}, np.array([0], dtype=np.int32))
    with pytest.raises(TypeError):
        inst.params["fraction"] = 0.2


def test_model_view_withholds_labels_and_the_full_graph():
    graph = _graph(); mv = apply_mask(graph, random_mask(graph, fraction=0.4, seed=5))
    view = mv.model_view()
    assert isinstance(view, VisibleView)
    for leaked in ("graph", "hidden_edges", "hidden_weights", "both_masked", "instance"):
        assert not hasattr(view, leaked)
    assert not view.is_masked[view.visible_edges].any()


def test_model_view_matches_the_masked_view_on_visible_quantities():
    graph = _graph(); mv = apply_mask(graph, random_mask(graph, fraction=0.4, seed=5))
    view = mv.model_view()
    assert view.n_nodes == graph.n_nodes
    assert np.array_equal(view.visible_edges, mv.visible_edges)
    assert np.array_equal(view.visible_weights, mv.visible_weights)
    assert np.array_equal(view.is_masked, mv.is_masked)
    assert np.array_equal(view.visible_in_strength(), mv.visible_in_strength())
    assert np.array_equal(view.visible_out_strength(), mv.visible_out_strength())
    assert np.allclose(view.visible_input_fraction(), mv.visible_input_fraction())
    assert np.array_equal(view.degree_given_total(), mv.degree_given_total())


def test_typed_mask_rejects_duplicate_type_index():
    graph = _graph(n=20); dup = int(graph.node_ids[0])
    types = pd.Series(["EPG", "PEN_a"], index=[dup, dup])
    with pytest.raises(ValueError, match="duplicate"):
        typed_mask(graph, types, seed=1)


def test_typed_mask_accepts_unmatched_null_types():
    graph = _graph(n=20)
    labels = ["EPG" if i < 5 else ("PEN_a" if i < 10 else np.nan) for i in range(20)]
    types = pd.Series(labels, index=[int(r) for r in graph.node_ids])
    instance = typed_mask(graph, types, seed=3)
    chosen = instance.params["cell_type"]
    assert chosen in {"EPG", "PEN_a"}
    values = types.reindex(graph.node_ids).to_numpy()
    unmatched = set(np.flatnonzero(pd.isna(values)).tolist())
    assert not set(instance.masked.tolist()) & unmatched
    assert sorted(instance.masked.tolist()) == sorted(np.flatnonzero(values == chosen).tolist())


def test_typed_mask_rejects_missing_graph_root_ids():
    graph = _graph(n=20)
    ids = [int(r) for r in graph.node_ids]
    types = pd.Series(["EPG"] * (len(ids) - 1), index=ids[:-1])
    with pytest.raises(ValueError, match="missing"):
        typed_mask(graph, types, seed=1)


def test_typed_mask_rejects_all_unmatched_types():
    graph = _graph(n=20)
    types = pd.Series([np.nan] * 20, index=[int(r) for r in graph.node_ids])
    with pytest.raises(ValueError, match="no values"):
        typed_mask(graph, types, seed=1)


@pytest.mark.parametrize("centroids", [np.array(1.0), np.ones(20), np.ones((20, 2))])
def test_spatial_mask_rejects_malformed_centroid_shapes(centroids):
    with pytest.raises(ValueError, match="centroids must have shape"):
        spatial_mask(_graph(), centroids, seed=2)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_spatial_mask_rejects_nonfinite_centroids(bad):
    graph = _graph(n=20)
    centroids = np.arange(60, dtype=float).reshape(20, 3)
    centroids[3, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        spatial_mask(graph, centroids, seed=2)


def test_random_mask_rejects_non_whole_percent_fraction():
    graph = _graph(n=20)
    with pytest.raises(ValueError, match="percent"):
        random_mask(graph, fraction=0.123, seed=1)


def test_spatial_mask_rejects_unregistered_fraction():
    graph = _graph(n=20)
    centroids = np.arange(60, dtype=float).reshape(20, 3)
    with pytest.raises(ValueError, match="fraction"):
        spatial_mask(graph, centroids, seed=2, fraction=0.10)


def test_typed_pool_mask_hides_whole_types_within_the_pool():
    graph = _graph(n=40)
    types = np.repeat(np.arange(10), 4)
    pool = np.arange(graph.n_nodes, dtype=np.int64)[2:]

    instance = typed_pool_mask(
        graph, np.random.default_rng(0), types=types, fraction=0.2, pool=pool
    )
    again = typed_pool_mask(
        graph, np.random.default_rng(0), types=types, fraction=0.2, pool=pool
    )

    hidden_types = np.unique(types[instance.masked])
    assert np.isin([2, 3], pool).all(), "two untyped neurons are in the pool"
    assert 0 not in hidden_types, "untyped neurons are never drawn"
    for cell_type in hidden_types:
        members = pool[types[pool] == cell_type]
        assert np.isin(members, instance.masked).all()
    assert np.isin(instance.masked, pool).all()
    assert abs(instance.n_masked - round(0.2 * pool.size)) <= 2
    assert instance.regime == "T"
    assert GAP_TAG in instance.instance_id
    assert np.array_equal(instance.masked, again.masked)


def test_typed_pool_mask_stops_at_the_size_closer_to_its_target():
    graph = _graph(n=40)
    types = np.array([1] * 6 + [2] * 30 + [0] * 4)
    order = np.random.default_rng(0).permutation([1, 2])
    target = round(0.25 * graph.n_nodes)

    instance = typed_pool_mask(graph, np.random.default_rng(0), types=types, fraction=0.25)

    expected = 6 if order[0] == 1 else 30
    assert (target, instance.n_masked) == (10, expected)
    assert instance.params["types"] == 1


def test_typed_pool_mask_refuses_a_pool_without_types():
    graph = _graph(n=20)
    types = np.zeros(graph.n_nodes, dtype=np.int64)

    with pytest.raises(ValueError, match="typed neurons"):
        typed_pool_mask(graph, np.random.default_rng(0), types=types)


def test_spatial_pool_mask_hides_the_pool_neurons_nearest_its_centre():
    graph = _graph(n=40)
    centroids = np.random.default_rng(1).normal(size=(graph.n_nodes, 3))
    pool = np.arange(graph.n_nodes, dtype=np.int64)[::2]

    instance = spatial_pool_mask(
        graph, np.random.default_rng(0), centroids=centroids, fraction=0.25, pool=pool
    )

    centre = instance.params["centre_node"]
    assert instance.n_masked == round(0.25 * pool.size)
    assert np.isin(instance.masked, pool).all() and centre in instance.masked
    distance = np.linalg.norm(centroids - centroids[centre], axis=1)
    outside = np.setdiff1d(pool, instance.masked)
    assert distance[instance.masked].max() <= distance[outside].min()
    assert instance.regime == "S"
    assert GAP_TAG in instance.instance_id


def test_pool_mask_draws_its_neurons_then_its_id_seed():
    """Training reproducibility rests on this order: the masked neurons, then the id seed."""
    graph = _graph(n=40)
    pool = np.arange(graph.n_nodes, dtype=np.int64)[::2]
    rng = np.random.default_rng(7)
    masked = np.sort(rng.choice(pool, size=round(0.25 * pool.size), replace=False))
    seed = int(rng.integers(1000))

    instance = pool_mask(graph, np.random.default_rng(7), fraction=0.25, pool=pool)

    assert np.array_equal(instance.masked, masked)
    assert instance.seed == seed


def test_spatial_pool_mask_sizes_the_gap_to_the_pool_and_grows_it_among_located():
    graph = _graph(n=40)
    centroids = np.random.default_rng(1).normal(size=(graph.n_nodes, 3))
    located = np.arange(graph.n_nodes) % 4 != 0

    instance = spatial_pool_mask(
        graph, np.random.default_rng(0), centroids=centroids, fraction=0.25, located=located
    )

    assert instance.n_masked == round(0.25 * graph.n_nodes)
    assert located[instance.masked].all()
    with pytest.raises(ValueError, match="located"):
        spatial_pool_mask(
            graph, np.random.default_rng(0), centroids=centroids, fraction=0.9,
            located=located,
        )
