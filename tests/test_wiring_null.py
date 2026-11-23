"""Tests for src/wiring_null.py."""

import numpy as np
import pandas as pd
import pytest

from src import wiring_null as wn


def rand_graph(n=60, m=300, seed=1):
    rng = np.random.default_rng(seed)
    keys = np.unique(rng.integers(0, n * n, m))
    src, dst = keys // n, keys % n
    keep = src != dst
    return src[keep], dst[keep], n


def sets(n, seed=0):
    return wn.make_seed_sets(n, np.random.SeedSequence(seed), n_sets=20)


def test_swap_preserves_degrees_no_loops_no_dups():
    src, dst, n = rand_graph()
    s2, d2 = wn.swap_rewire(src, dst, n, np.random.default_rng(2), factor=5)
    assert (np.bincount(s2, minlength=n) == np.bincount(src, minlength=n)).all()
    assert (np.bincount(d2, minlength=n) == np.bincount(dst, minlength=n)).all()
    assert (s2 != d2).all() and len(np.unique(s2 * n + d2)) == len(s2)
    assert (d2 != dst).mean() > 0.3


def test_swap_rejects_loops():
    with pytest.raises(AssertionError):
        wn.swap_rewire(np.array([0, 1]), np.array([0, 2]), 3, np.random.default_rng(0))


def test_null_unreachable_on_complete_digraph():
    n = 3
    s, d = np.array([(a, b) for a in range(n) for b in range(n) if a != b]).T
    with pytest.raises(RuntimeError, match="null unreachable"):
        wn.swap_rewire(s, d, n, np.random.default_rng(0))


def test_accepted_swaps_counted_not_attempts():
    # batch=1: one attempt per loop; reaching factor*|E| ACCEPTED swaps needs at least that many attempts
    src, dst, n = rand_graph(30, 120)

    class Counting:
        calls = 0

        def __init__(self):
            self.rng = np.random.default_rng(0)

        def integers(self, *a, **k):
            Counting.calls += 1
            return self.rng.integers(*a, **k)

    wn.swap_rewire(src, dst, n, Counting(), factor=2, batch=1)
    assert Counting.calls // 2 >= 2 * len(src)


def test_shared_seed_sets_identical_object_and_values():
    src, dst, n = rand_graph()
    w = np.ones(len(src))
    ss = sets(n)
    assert (ss == sets(n)).all()  # fixed seed -> same values
    orig = wn.stats
    real = orig(src, dst, w, n, ss)
    s2, d2 = wn.swap_rewire(src, dst, n, np.random.default_rng(3), factor=2)
    null = orig(s2, d2, w, n, ss)
    assert real["S3_spread3"] > 0 and null["S3_spread3"] > 0
    # S3 on a graph is a pure function of (graph, seed sets): same graph, same sets -> same value
    assert orig(src, dst, w, n, ss)["S3_spread3"] == real["S3_spread3"]


def test_run_draw_uses_given_seed_sets(monkeypatch):
    src, dst, n = rand_graph()
    w = np.ones(len(src))
    ss = sets(n)
    got = []
    orig = wn.stats
    monkeypatch.setattr(wn, "stats", lambda *a, **k: got.append(a[4]) or orig(*a, **k))
    wn.run_draw(src, dst, w, n, ss, 5, 0)
    assert got and all(g is ss for g in got)


def test_draw_rng_is_deterministic_and_shard_independent():
    a = wn.draw_rng(7, 0, 3).integers(0, 10**9, 4)
    assert (a == wn.draw_rng(7, 0, 3).integers(0, 10**9, 4)).all()
    assert (a != wn.draw_rng(7, 0, 4).integers(0, 10**9, 4)).any()


def test_load_graph_drops_loops(tmp_path):
    d = tmp_path / "processed" / "f1" / "xx"
    d.mkdir(parents=True)
    pd.DataFrame({"source": [0, 1, 2, 2], "target": [0, 2, 2, 0], "weight": [5, 1, 9, 2]}).to_parquet(d / "edges.parquet")
    src, dst, w, n, loops = wn.load_graph(tmp_path, "f1", "xx")
    assert loops == 2 and n == 3 and (src != dst).all() and list(w) == [1, 2]


def test_stats_reject_loops():
    with pytest.raises(AssertionError):
        wn.stats(np.array([0]), np.array([0]), np.array([1.0]), 2, sets(2))


def test_s2_triangle_and_path():
    tri = wn.stats(np.array([0, 1, 2]), np.array([1, 2, 0]), np.ones(3), 3, sets(3))
    assert tri["S2_transitivity"] == pytest.approx(1.0)
    path = wn.stats(np.array([0, 1]), np.array([1, 2]), np.ones(2), 3, sets(3))
    assert path["S2_transitivity"] == 0.0


def test_s1_known_graph():
    s = wn.stats(np.array([0, 1, 1]), np.array([1, 0, 2]), np.ones(3), 3, sets(3))
    assert s["S1_reciprocity"] == pytest.approx(2 / 3)


def test_s4_raw_dense_fallback():
    # 3-cycle with weight 2 has eigenvalues 2*cube roots of unity -> radius 2 (not divided by degree)
    s = wn.stats(np.array([0, 1, 2]), np.array([1, 2, 0]), np.full(3, 2.0), 3, sets(3))
    assert s["S4_spectral_radius"] == pytest.approx(2.0)
    assert "S4_spectral_radius_norm" not in s


def test_s4_raises_when_large_and_eigs_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(wn, "eigs", boom)
    monkeypatch.setattr(wn, "DENSE_MAX_N", 2)
    with pytest.raises(RuntimeError):
        wn.spectral_radius(np.array([0, 1, 2]), np.array([1, 2, 0]), np.ones(3), 3)


def test_holm_known_values():
    assert wn.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert wn.holm([0.5, 0.001]) == pytest.approx([0.5, 0.002])


def test_z_and_p_and_saturation():
    null = np.arange(-5, 6, dtype=float)  # mean 0
    z, p, st = wn.z_and_p(10.0, null)
    assert st == "defined" and z > 3 and p == pytest.approx(1 / 12)
    assert wn.z_and_p(1.0, np.ones(50))[2] == "undefined"


def _frame(region, real, null_vals, extra_const=1.0):
    rows = [{"region": region, "kind": "real", "perm": -1, "S1_reciprocity": real, "S2_transitivity": extra_const,
             "S3_spread3": 1.0, "S4_spectral_radius": 1.0}]
    for i, v in enumerate(null_vals):
        rows.append({"region": region, "kind": "N0", "perm": i, "S1_reciprocity": v, "S2_transitivity": extra_const,
                     "S3_spread3": 1.0, "S4_spectral_radius": 1.0})
    return pd.DataFrame(rows)


def test_decide_saturation_excluded_and_replication():
    rng = np.random.default_rng(0)
    null = rng.normal(0, 1, 2000)
    f1 = _frame("A", 6.0, null)
    t = wn.decide(f1, _frame("A", 5.0, null))
    assert (t[t.stat != "S1_reciprocity"].status == "undefined").sum() >= 3
    s1 = t[(t["null"] == "N0") & (t.stat == "S1_reciprocity")].iloc[0]
    assert s1.differs and s1.replicates
    # family is only the single defined test -> Holm adj == raw p
    assert s1.p_holm == pytest.approx(s1.p)
    # opposite sign and weak F2 do not replicate
    assert not wn.decide(f1, _frame("A", -5.0, null)).query("stat == 'S1_reciprocity' and `null` == 'N0'").replicates.iloc[0]
    assert not wn.decide(f1, _frame("A", 1.0, null)).query("stat == 'S1_reciprocity' and `null` == 'N0'").replicates.iloc[0]


def test_derive_seed_is_deterministic_and_distinct_per_region_and_split():
    seeds = {(r, s): wn.derive_seed(20261121, r, s) for r in wn.REGIONS for s in wn.SPLITS}
    assert len(set(seeds.values())) == len(seeds)
    assert wn.derive_seed(20261121, "lx", "f1") == seeds[("lx", "f1")]
    assert wn.derive_seed(1, "lx", "f1") != seeds[("lx", "f1")]


def test_incomplete_b_is_flagged_and_cannot_replicate():
    rng = np.random.default_rng(0)

    def frame(n_null):
        rows = [{"region": "lx", "kind": "real", "perm": -1, **{s: 10.0 for s in wn.STATS}}]
        for k in ("N0", "N1"):
            for i in range(n_null):
                rows.append({"region": "lx", "kind": k, "perm": i,
                             **{s: float(rng.normal()) for s in wn.STATS}})
        return pd.DataFrame(rows)

    full = wn.decide(frame(20), frame(20), expected_b=20)
    assert full[(full["null"] == "N0")].status.eq("defined").all()
    short = wn.decide(frame(19), frame(20), expected_b=20)  # one shard lost in F1
    n0 = short[short["null"] == "N0"]
    assert n0.status.eq("incomplete").all()
    assert not n0.differs.any() and not n0.replicates.any()
