"""Seams of the F2 union (plan 01 task 5): threshold calibration and per-region boundary thresholds."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from src.harmonize import calibrate_threshold  # noqa: E402
from src.union import boundary_flags, pair_counts, union_graph  # noqa: E402

B = 10**17


def _connections(scale: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pre, post in sorted({(int(a), int(b)) for a, b in rng.integers(0, 90, size=(1500, 2)) if a != b}):
        fb, lh = int(rng.integers(0, 7)), int(rng.integers(0, 4))
        rows += [(pre, post, "FB", scale * fb), (pre, post, "LH_L", scale * lh)]
    pre, post, neuropil, count = zip(*[r for r in rows if r[3]])
    return pd.DataFrame({"pre_pt_root_id": np.add(pre, B), "post_pt_root_id": np.add(post, B),
                         "neuropil": neuropil, "syn_count": count})


def test_doubled_counts_calibrate_to_double_the_f1_threshold():
    f1 = union_graph(pair_counts(_connections(1)), "CX+LH")
    pc2 = pair_counts(_connections(2))
    types = pd.Series([f"t{k % 6}" for k in range(90)], index=np.add(np.arange(90), B))
    result = calibrate_threshold(f1, types, None, types, region="CX+LH", dataset="f2", candidates=range(1, 16),
                                 graph_at=lambda t: union_graph(pc2, "CX+LH", dataset="f2", threshold=t))
    assert result.chosen == 10  # the F2 graph at 10 is the F1 graph at 5, edge for edge
    row = result.curve.set_index("threshold").loc[10]
    assert row["median_log2_ratio"] == pytest.approx(0.0)


def test_f2_boundary_edges_use_each_regions_own_threshold():
    rows = [(1, 2, "FB", 3), (1, 2, "LH_L", 1),  # union 4: in CX's graph only if CX's threshold is <= 3
            (2, 3, "FB", 2), (2, 3, "LH_L", 2)]  # union 4: in no region graph at 3
    pre, post, neuropil, count = zip(*rows)
    pc = pair_counts(pd.DataFrame({"pre_pt_root_id": np.add(pre, B), "post_pt_root_id": np.add(post, B),
                                   "neuropil": neuropil, "syn_count": count}))
    graph = union_graph(pc, "CX+LH", dataset="f2", threshold=4)
    assert boundary_flags(graph, pc, {"CX": 3, "LH": 3}).tolist() == [False, True]
    assert boundary_flags(graph, pc, {"CX": 4, "LH": 3}).tolist() == [True, True]
    with pytest.raises(ValueError, match="missing"):
        boundary_flags(graph, pc, {"CX": 3})
    with pytest.raises(ValueError, match="missing"):
        boundary_flags(graph, pc)


def test_build_union_builds_f2_only_at_a_committed_threshold():
    from scripts.build_union import F2_THRESHOLDS, main

    assert F2_THRESHOLDS == {"UNION12": 3}  # the 2026-11-07 sweep's choice; UNION11 has none
    with pytest.raises(SystemExit):
        main(["--union", "UNION11", "--dataset", "f2"])
    with pytest.raises(SystemExit):
        main(["--union", "UNION12", "--calibrate"])  # --calibrate is an f2 step


def test_regional_confirm_exclusions_read_frozen_suites_by_digest(tmp_path):
    from src.boundary_suites import regional_confirm_digests, regional_confirm_exclusions
    from src.regions import REGIONS
    from src.suites import Suite, SuiteInstance

    suites = {}
    for k, code in enumerate(sorted(REGIONS)):
        instances = (SuiteInstance(f"f2-{code.lower()}-R01-s2000", "R", 2000, {}, np.array([B + k])),
                     SuiteInstance(f"f2-{code.lower()}-T-s2003", "T", 2003, {"cell_type": f"type{k}"},
                                   np.array([B + 100 + k])))
        suite = Suite("f2_confirm", "f2", code, "0" * 64, 3, 20260818, instances)
        suite.write(tmp_path / "f2" / code.lower() / "f2_confirm.json")
        suites[code] = suite
    registered = {"suites": {code: {"f1_val": "a" * 64, "f2_dev": "b" * 64, "f2_confirm": s.digest()}
                             for code, s in suites.items()}}
    digests = regional_confirm_digests(registered)
    assert digests == {code: s.digest() for code, s in suites.items()}
    roots, types = regional_confirm_exclusions(tmp_path, digests)
    assert roots == {B + k for k in range(12)} | {B + 100 + k for k in range(12)}
    assert types == {f"type{k}" for k in range(12)}
    with pytest.raises(ValueError, match="pending"):
        regional_confirm_digests({"suites": {**registered["suites"], "CX": {}}})
    with pytest.raises(ValueError, match="not the committed"):
        regional_confirm_exclusions(tmp_path, {**digests, "AL": "c" * 64})


def test_every_member_must_rebuild_to_its_frozen_f2_graph():
    # Sol: moving one pair from CX 3 + LH 1 to 2 + 2 keeps the union but changes the boundary; the
    # frozen regional digests catch it, and a member without a frozen digest is refused, not skipped.
    from scripts.build_union import verify_f2_members
    from src.regions import build_region_graph

    connections = _connections(1)
    pc = pair_counts(connections)
    thresholds = {"CX": 3, "LH": 2}
    frozen = {code: build_region_graph(connections, region=code, dataset="f2", threshold=t).digest()
              for code, t in thresholds.items()}
    members = verify_f2_members(pc, ("CX", "LH"), thresholds, frozen)
    assert {code: g.digest() for code, g in members.items()} == frozen
    with pytest.raises(SystemExit, match="rebuild CX"):
        verify_f2_members(pc, ("CX", "LH"), {"CX": 4, "LH": 2}, frozen)
    with pytest.raises(SystemExit, match="no frozen F2 graph"):
        verify_f2_members(pc, ("CX", "LH"), thresholds, {"CX": frozen["CX"]})


def test_a_loaded_f2_union_must_be_at_its_committed_threshold():
    from types import SimpleNamespace

    from scripts.build_union import check_f2_union

    check_f2_union(SimpleNamespace(dataset="f2", region="UNION12", threshold=3), "UNION12")
    for graph in (SimpleNamespace(dataset="f2", region="UNION12", threshold=5),
                  SimpleNamespace(dataset="f1", region="UNION12", threshold=3)):
        with pytest.raises(SystemExit, match="committed threshold"):
            check_f2_union(graph, "UNION12")


def test_confirm_exclusions_come_only_from_the_pinned_dev_suite(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import build_boundary_suites as cli

    pending = {"boundary_suites": {}}
    pinned = {"boundary_suites": {"phase3_boundary_f2_dev": {"suite": "a" * 64, "selector": "b" * 64}}}
    graph = SimpleNamespace(digest=lambda: "g", region="UNION12")
    with pytest.raises(SystemExit, match="not registered"):
        cli.committed_phase3_digest("phase3_boundary_f2_dev", pending)
    assert cli.committed_phase3_digest("phase3_boundary_f2_dev", pinned) == "a" * 64
    for digest, on in (("a" * 64, "other graph"), ("stale", "g")):  # a stale draft, or dev on another graph
        monkeypatch.setattr(cli.BoundarySuite, "read", staticmethod(
            lambda path, d=digest, g=on: SimpleNamespace(digest=lambda: d, graph_digest=g)))
        with pytest.raises(SystemExit, match="not the committed"):
            cli.pinned_dev(tmp_path / "dev.json", graph, pinned)
    monkeypatch.setattr(cli.BoundarySuite, "read", staticmethod(
        lambda path: SimpleNamespace(digest=lambda: "a" * 64, graph_digest="g")))
    assert cli.pinned_dev(tmp_path / "dev.json", graph, pinned).graph_digest == "g"


def test_the_regional_union_adds_member_neurons_the_union_drops_with_no_edge():
    from src.regions import build_region_graph
    from src.union import with_isolated

    rows = [(1, 2, "FB", 5), (2, 3, "LH_L", 4), (3, 4, "LH_L", 1), (5, 1, "FB", 3)]
    pre, post, neuropil, count = zip(*rows)
    connections = pd.DataFrame({"pre_pt_root_id": np.add(pre, B), "post_pt_root_id": np.add(post, B),
                                "neuropil": neuropil, "syn_count": count})
    union = union_graph(pair_counts(connections), "CX+LH", dataset="f2", threshold=4)
    members = [build_region_graph(connections, region=code, dataset="f2", threshold=t).node_ids
               for code, t in (("CX", 3), ("LH", 1))]
    dropped = np.setdiff1d(np.concatenate(members), union.node_ids)
    assert dropped.tolist() == [B + 4, B + 5]  # LH at 1 and CX at 3 keep neurons the union at 4 loses
    regional = with_isolated(union, np.concatenate(members))
    assert np.array_equal(regional.node_ids, np.union1d(union.node_ids, dropped))
    assert np.array_equal(regional.node_ids[regional.edges], union.node_ids[union.edges])
    assert np.array_equal(regional.weights, union.weights)
