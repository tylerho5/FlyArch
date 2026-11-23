"""Seams of the Phase 3 boundary suites (plan 02): suite -> evaluator rows, exclusions, input contract."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.boundary_suites import (
    BoundarySuite, allowed_columns, boundary_selector, boundary_types, build_boundary_suite, check_inputs,
)
from src.features import FeatureSchema, build_features
from src.masks import MaskInstance, apply_mask
from src.regions import RegionGraph
from src.suites import suite_negatives
from src.union import boundary_flags, pair_counts, support_within, union_graph, union_labels

B = 10**17
NAME = "phase3_boundary_f1_val"


@pytest.fixture(scope="module")
def union():
    """AL+CX+LH: CX-only edges, CX+LH and CX+AL boundary edges, and a typed island with no boundary edge."""
    rng = np.random.default_rng(0)
    rows, support = [], {}
    for pre, post in sorted({(int(a), int(b)) for a, b in rng.integers(0, 300, size=(3000, 2)) if a != b}):
        draw = rng.random()
        if draw < 0.4:
            rows.append((pre, post, "FB", 6))
        elif draw < 0.7:
            rows += [(pre, post, "FB", 3), (pre, post, "LH_L", 2)]
            support[(pre, post)] = {"CX", "LH"}
        else:
            rows += [(pre, post, "FB", 3), (pre, post, "AL_L", 2)]
            support[(pre, post)] = {"AL", "CX"}
    rows += [(300 + k, 300 + (k + 1) % 30, "FB", 6) for k in range(30)]  # the island
    pre, post, neuropil, count = zip(*rows)
    connections = pd.DataFrame({"pre_pt_root_id": np.add(pre, B), "post_pt_root_id": np.add(post, B),
                                "neuropil": neuropil, "syn_count": count})
    pc = pair_counts(connections)
    graph = union_graph(pc, "AL+CX+LH")
    local = graph.node_ids - B
    names = rng.choice([f"t{k}" for k in range(12)], graph.n_nodes)
    types = pd.Series(np.where(local >= 300, "island", names), index=graph.node_ids)
    centroids = rng.normal(size=(graph.n_nodes, 3))
    boundary = boundary_flags(graph, pc)
    block = boundary & support_within(graph, pc, ["CX", "LH"])
    pairs = [tuple(int(x) for x in p) for p in graph.node_ids[graph.edges] - B]
    expected = {
        "boundary": np.array([p in support for p in pairs]),
        "block": np.array([support.get(p) == {"CX", "LH"} for p in pairs]),
    }
    return {"graph": graph, "boundary": boundary, "block": block, "types": types, "centroids": centroids,
            "expected": expected, "pc": pc}


def _build(union, name=NAME, graph=None, **kwargs):
    return build_boundary_suite(graph or union["graph"], union["boundary"], name=name, types=union["types"],
                                centroids=union["centroids"], **kwargs)


def test_flags_match_the_raw_support_sets(union):
    assert np.array_equal(union["boundary"], union["expected"]["boundary"])
    assert np.array_equal(union["block"], union["expected"]["block"])
    assert union["block"].any() and (union["boundary"] & ~union["block"]).any()


def test_selector_rows_are_single_masked_boundary_edges_on_the_evaluators_rows(union):
    graph, boundary, block = union["graph"], union["boundary"], union["block"]
    suite = _build(union)
    assert [i.seed for i in suite.instances] == [5000, 5001, 5002, 5003, 5004]
    assert [i.regime for i in suite.instances] == ["R", "R", "R", "T", "S"]
    selector = boundary_selector(graph, suite, boundary, block)
    negatives = suite_negatives(graph, suite)
    again = suite_negatives(graph, suite)
    assert {k: v.digest() for k, v in negatives.items()} == {k: v.digest() for k, v in again.items()}
    for instance in suite.resolve(graph):
        view = apply_mask(graph, instance)
        row = selector["instances"][instance.instance_id]
        hidden = view.is_masked[graph.edges[:, 0]] | view.is_masked[graph.edges[:, 1]]
        single = boundary[hidden] & ~view.both_masked()
        assert row["rows"] == np.flatnonzero(single).tolist() and row["n_positives"] == len(view.hidden_edges)
        assert row["block_rows"] == np.flatnonzero(single & block[hidden]).tolist()
        # The negative sample's positives are every hidden edge, in the order the selector indexes.
        assert np.array_equal(negatives[instance.instance_id].positives, view.hidden_edges)
    assert any(r["block_rows"] for r in selector["instances"].values())


def test_suite_round_trips_and_refuses_a_regional_name(union, tmp_path):
    graph = union["graph"]
    suite = _build(union)
    suite.write(tmp_path / "s.json")
    again = BoundarySuite.read(tmp_path / "s.json")
    assert again.digest() == suite.digest() and len(again.resolve(graph)) == 5
    with pytest.raises(ValueError, match="unknown boundary suite"):
        dataclasses.replace(again, name="f1_val").validate(graph)


def test_t_draws_only_f1_boundary_types_and_exclusions_are_never_masked(union):
    graph, types = union["graph"], union["types"]
    eligible = boundary_types(graph, union["boundary"], types)
    assert "island" in set(types) and "island" not in eligible
    excluded_roots = graph.node_ids[:5]
    banned = {"t0", "t1"} | set(types.reindex(excluded_roots))
    rules = {"exclude_roots": excluded_roots, "exclude_types": {"t0"}, "mask_exclude_types": ("t1",)}
    for inst in _build(union, **rules).instances:
        assert not np.isin(inst.masked_root_ids, excluded_roots).any()
        assert not types.reindex(inst.masked_root_ids).isin({"t0", "t1"}).any()


@pytest.mark.parametrize("name, seeds", [
    ("phase3_boundary_f2_dev", [s for s in range(6000, 6015) if (s - 6000) % 5 != 3]),
    ("phase3_boundary_f2_confirm", [s for s in range(7000, 7020) if (s - 7000) % 5 != 3]),
])
def test_f2_suites_skip_their_t_slots(union, name, seeds):
    f2 = union_graph(union["pc"], "AL+CX+LH", dataset="f2")
    suite = _build(union, name=name, graph=f2, exclude_roots=union["graph"].node_ids[:5])
    assert [i.seed for i in suite.instances] == seeds
    assert "T" not in {i.regime for i in suite.instances}
    # The draws themselves: no excluded root is masked, and each R instance is the draw its slot's seed gives.
    from src.boundary_suites import _FRACTIONS, _TAGS
    from src.masks import random_mask

    graph, banned = f2, union["graph"].node_ids[:5]
    pool = np.flatnonzero(~np.isin(graph.node_ids, banned))
    first = min(seeds)
    for inst in suite.instances:
        assert not np.isin(inst.masked_root_ids, banned).any()
        tag = _TAGS[(inst.seed - first) % len(_TAGS)]
        if tag in _FRACTIONS:
            want = graph.node_ids[random_mask(graph, fraction=_FRACTIONS[tag], seed=inst.seed, pool=pool).masked]
            assert np.array_equal(inst.masked_root_ids, want)


def test_the_t_draw_excludes_ineligible_and_banned_types(union, monkeypatch):
    # Seed-independent: inspect what the typed draw is told to exclude.
    import src.boundary_suites as module

    graph, types = union["graph"], union["types"]
    excluded_roots = graph.node_ids[:5]
    banned = {"t0", "t1"} | set(types.reindex(excluded_roots))
    full = boundary_types(graph, union["boundary"], types)
    dropped = sorted(full - banned)[0]  # only t_types can make this type undrawable
    eligible = full - {dropped}
    calls, real = [], module.typed_mask
    monkeypatch.setattr(module, "typed_mask", lambda *a, **k: calls.append(set(k["exclude"])) or real(*a, **k))
    _build(union, t_types=eligible, exclude_roots=excluded_roots, exclude_types={"t0"}, mask_exclude_types=("t1",))
    assert len(calls) == 1
    assert (set(types) - eligible) <= calls[0] and dropped in calls[0] and banned <= calls[0]


def test_an_instance_with_no_boundary_row_aborts(union):
    with pytest.raises(ValueError, match="no single-masked boundary row"):
        boundary_selector(union["graph"], _build(union), np.zeros_like(union["boundary"]))


def test_allowlist_refuses_membership_degree_and_boundary_columns(monkeypatch):
    schema = FeatureSchema("CX+LH", ("central",), ("a",), ("left",), ("intrinsic",), {"t1": 1},
                           labels=("FB", "EB", "PB", "NO", "LH_L", "LH_R"))
    columns = ["root_id", "geometry_present", "centroid_x", "innervation_LH_R", "class_a", "cell_type", "nt_gaba"]
    check_inputs(schema, columns)
    for bad in ("member_CX", "boundary", "degree_CX", "innervation_AL_L"):
        with pytest.raises(ValueError, match="allowlist"):
            check_inputs(schema, columns + [bad])
    with pytest.raises(ValueError, match="union's labels"):
        allowed_columns(dataclasses.replace(schema, labels=("FB",)))
    # The documented families and the enforced columns cannot drift: drop a family and the schema is refused.
    import src.boundary_suites as module

    monkeypatch.setattr(module, "INPUT_ALLOWLIST", {k: v for k, v in module.INPUT_ALLOWLIST.items()
                                                    if not k.startswith("nt_")})
    with pytest.raises(ValueError, match="missing from INPUT_ALLOWLIST"):
        allowed_columns(schema)


def test_moving_a_masked_nodes_hidden_partners_changes_no_scorer_input():
    # Node 0 is masked; its 5 synapses go to node 1 in one graph and to node 2 in the other.
    ids = np.add([0, 1, 2, 3], B)
    annotations = pd.DataFrame({"root_id": ids, "super_class": ["central"] * 4, "class": ["a"] * 4,
                                "cell_type": ["t1", "t2", "t2", "t3"], "side": ["left"] * 4,
                                "flow": ["intrinsic"] * 4, "nt_type": ["ACH"] * 4})
    labels = union_labels("CX+LH")
    innervation = pd.DataFrame(np.ones((4, len(labels))), index=pd.Index(ids, name="root_id"), columns=list(labels))
    schema = dataclasses.replace(FeatureSchema.from_f1(annotations, "CX"), region="CX+LH", labels=labels)

    def inputs(partner):
        edges = np.array([[0, partner], [1, 3], [2, 3]], dtype=np.int32)
        graph = RegionGraph("f1", "CX+LH", 5, ids, edges, np.array([5, 6, 7]))
        mask = MaskInstance("i", "f1", "CX+LH", "R", 0, {}, np.array([0], dtype=np.int32))
        table = build_features(annotations, graph, schema, dataset="f1", innervation=innervation)
        check_inputs(schema, table.columns)
        return apply_mask(graph, mask).model_view(), table

    (a, ta), (b, tb) = inputs(1), inputs(2)
    assert np.array_equal(ta.values, tb.values) and ta.columns == tb.columns
    assert np.array_equal(ta.geometry_present, tb.geometry_present)
    assert np.array_equal(a.visible_edges, b.visible_edges) and np.array_equal(a.visible_weights, b.visible_weights)
    assert np.array_equal(a.degree_given_total(), b.degree_given_total()) and a.degree_given_total()[0] == 5


def test_t_floor_and_s05_centres_rest_on_boundary_edges(union):
    from src.boundary_suites import boundary_ends

    graph, boundary, types = union["graph"], union["boundary"], union["types"]
    ends = pd.Series(boundary_ends(graph, boundary), index=types.reindex(graph.node_ids).to_numpy())
    per_type = ends.groupby(level=0).sum()
    floor = int(per_type[per_type > 0].median())
    assert boundary_types(graph, boundary, types, floor=floor) == set(per_type.index[per_type >= floor])
    assert boundary_types(graph, boundary, types, floor=1) == set(per_type.index[per_type > 0])
    suite = _build(union)
    (ball,) = [inst for inst in suite.resolve(graph) if inst.regime == "S"]
    on = boundary_ends(graph, boundary) > 0
    assert on[ball.params["centre_node"]]  # the centre has a boundary edge, drawn by the generator from those
    from src.masks import spatial_mask

    drawn = spatial_mask(graph, union["centroids"], seed=ball.seed, fraction=0.05, pool=np.flatnonzero(on))
    assert ball.params["centre_node"] == drawn.params["centre_node"]
    assert (~on[ball.masked]).any()  # the ball still masks non-boundary neurons around it
    assert len(ball.masked) >= round(0.05 * graph.n_nodes)  # sized over every eligible neuron, not the centres
