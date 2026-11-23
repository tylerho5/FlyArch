"""Tests for src/rate_model.py. Synthetic graphs only."""

import numpy as np
import pandas as pd
import pytest

from src import rate_model as rm


def rand_graph(n=60, m=400, seed=1, groups=3):
    rng = np.random.default_rng(seed)
    keys = np.unique(rng.integers(0, n * n, m))
    src, dst = keys // n, keys % n
    keep = src != dst
    src, dst = src[keep], dst[keep]
    w = rng.integers(1, 20, len(src)).astype(float)
    grp = rng.integers(0, groups, n).astype(np.int8)
    return src, dst, w, grp, n


def group_matrix(src, dst, grp):
    m = np.zeros((3, 3), int)
    np.add.at(m, (grp[src], grp[dst]), 1)
    return m


def make_graph(seed=3, n=60, m=400):
    src, dst, w, grp, n = rand_graph(n, m, seed)
    grp = grp.copy()
    grp[:12] = rm.GROUP_A
    grp[12:24] = rm.GROUP_E
    grp[24:] = rm.GROUP_I
    sign = np.where(np.random.default_rng(seed).random(n) < 0.3, -1.0, 1.0)
    return rm.Graph(src, dst, w, n, np.flatnonzero(grp == 0), np.flatnonzero(grp == 1), grp, sign, {})


# ---- kernel ---------------------------------------------------------------------------------

def test_swap_invariants():
    src, dst, w, grp, n = rand_graph()
    d2, acc, att, cap = rm.swap_rewire(src, dst, grp, n, np.random.default_rng(2))
    assert not cap and acc == 10 * len(src) and att >= acc
    assert (np.bincount(d2, minlength=n) == np.bincount(dst, minlength=n)).all()  # in-degree
    assert (src != d2).all() and len(np.unique(src * n + d2)) == len(src)  # no loops / multi-edges
    assert (group_matrix(src, d2, grp) == group_matrix(src, dst, grp)).all()  # 3x3 group matrix
    assert (d2 != dst).mean() > 0.3
    # sources and weights are untouched by construction (only dst is returned): out-degree and
    # out-strength per source are therefore unchanged; check the matrix built from them
    out_before = np.bincount(src, weights=w, minlength=n)
    wm = rm.signed_matrix(src, d2, w, n, np.ones(n))
    colsum = np.asarray(wm.sum(0)).ravel()
    assert np.allclose(colsum[out_before > 0], 1.0)


def test_cap_behaviour():
    n = 3  # complete digraph: no swap is ever valid
    s, d = np.array([(a, b) for a in range(n) for b in range(n) if a != b]).T
    d2, acc, att, cap = rm.swap_rewire(s, d, np.zeros(n, np.int8), n, np.random.default_rng(0))
    assert cap and acc == 0 and att == 200 * len(s) and (d2 == d).all()


def test_cap_hit_with_partial_progress():
    src, dst, w, grp, n = rand_graph(n=40, m=150)
    d2, acc, att, cap = rm.swap_rewire(src, dst, grp, n, np.random.default_rng(5), factor=10, cap_factor=1)
    assert att == len(src) and cap and 0 <= acc < 10 * len(src)


def test_group_constraint_blocks_mixed_swaps():
    n = 8
    grp = np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int8)
    src = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    dst = np.array([1, 2, 3, 0, 5, 6, 7, 4])
    d2, acc, att, cap = rm.swap_rewire(src, dst, grp, n, np.random.default_rng(1), factor=20)
    assert (grp[d2] == grp[dst]).all() and (grp[src] == grp[d2]).all()


def test_seed_determinism_and_shard_independence():
    g = make_graph()
    flips = rm.flip_sets(g, 7, "lx", "f2")
    a = rm.run_draw(g, "lx", "f2", 7, 5, flips)
    for i in (0, 1, 9):
        rm.run_draw(g, "lx", "f2", 7, i, flips)
    b = rm.run_draw(g, "lx", "f2", 7, 5, flips)
    for k in ("T", "accepted", "attempts", "T_flip3", "T_amp0.5", "T_amp2.0"):
        assert a[k] == b[k]
    assert rm.run_draw(g, "lx", "f2", 7, 6, flips)["T"] != a["T"]
    assert rm.run_draw(g, "al", "f2", 7, 5, flips)["T"] != a["T"]  # region index enters the entropy


def test_seed_construction_exact():
    r1 = rm.draw_rng(20261122, "lx", "f2", 4).integers(0, 10**9, 3)
    ss = np.random.SeedSequence([20261122, 1, 1], spawn_key=(1, 4))
    assert (np.random.default_rng(ss).integers(0, 10**9, 3) == r1).all()
    f = rm.flip_rng(20261122, "cx", "f1", 2).integers(0, 10**9, 3)
    ss = np.random.SeedSequence([20261122, 2, 0], spawn_key=(2, 2))
    assert (np.random.default_rng(ss).integers(0, 10**9, 3) == f).all()


def test_flip_sets_size_and_support():
    g = make_graph()
    fl = rm.flip_sets(g, 1, "al", "f1")
    assert len(fl) == 10
    srcs = set(np.unique(g.src))
    for f in fl:
        assert len(f) == round(0.10 * len(srcs)) and len(set(f)) == len(f) and set(f) <= srcs
    assert any((fl[0] != fl[k]).any() for k in range(1, 10))


# ---- model ----------------------------------------------------------------------------------

def test_equalised_radius_and_dense_fallback(monkeypatch):
    g = make_graph()
    wm = rm.signed_matrix(g.src, g.dst, g.w, g.n, g.sign)
    out = np.bincount(g.src, weights=g.w, minlength=g.n)
    assert np.isclose(wm[g.dst[0], g.src[0]], g.sign[g.src[0]] * g.w[0] / out[g.src[0]])
    scaled, rho, dense, ok = rm.equalise(wm)
    assert ok and not dense
    assert np.isclose(np.abs(np.linalg.eigvals(scaled.toarray())).max(), 0.9)

    def boom(*a, **k):
        raise RuntimeError("no convergence")
    monkeypatch.setattr(rm, "eigs", boom)
    scaled2, rho2, dense2, ok2 = rm.equalise(wm)
    assert dense2 and ok2 and np.isclose(rho2, rho, rtol=1e-6)
    assert np.isclose(np.abs(np.linalg.eigvals(scaled2.toarray())).max(), 0.9)


def test_undefined_rho_nilpotent():
    wm = rm.signed_matrix(np.array([0, 1]), np.array([1, 2]), np.ones(2), 3, np.ones(3))
    scaled, rho, dense, ok = rm.equalise(wm)
    assert not ok and scaled is None


def test_simulate_hand_computed():
    import scipy.sparse as sp
    wm = sp.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]))  # 0 -> 1 -> 2
    t = rm.simulate(wm, np.array([0]), np.array([2]), amp=1.0, steps=3)
    r1 = 0.2 * np.tanh(1.0)
    r2 = 0.2 * np.tanh(r1)
    assert np.isclose(t, 0.2 * np.tanh(r2))  # r[1]_2 = r[2]_2 = 0, r[3]_2 = .2 tanh(r[2]_1)
    wm1 = sp.csr_matrix((1, 1))
    t1 = rm.simulate(wm1, np.array([0]), np.array([0]), amp=2.0, steps=3)
    a = 0.2 * np.tanh(2.0)
    assert np.isclose(t1, a + 0.8 * a + 0.64 * a)
    wi = sp.csr_matrix(np.array([[0, 0], [-5.0, 0]]))  # inhibitory source clamps at zero
    assert rm.simulate(wi, np.array([0]), np.array([1]), steps=5) == 0.0


# ---- masks and signs ------------------------------------------------------------------------

def feat_frame():
    cols = ["superclass_sensory", "superclass_ascending", "superclass_sensory_ascending",
            "superclass_descending", "superclass_endocrine", "superclass_motor", "class_ALPN", "class_ALON",
            "nt_gaba", "nt_glutamate", "nt_acetylcholine", "nt_dopamine", "nt_serotonin", "nt_octopamine",
            "nt_other"]
    f = pd.DataFrame(0, index=range(8), columns=cols)
    f.loc[0, "superclass_sensory"] = 1
    f.loc[1, "superclass_ascending"] = 1
    f.loc[2, "superclass_sensory_ascending"] = 1
    f.loc[3, "superclass_descending"] = 1
    f.loc[4, "superclass_motor"] = 1
    f.loc[5, "class_ALPN"] = 1
    f.loc[6, "class_ALON"] = 1
    f.loc[3, "nt_gaba"] = 1
    f.loc[4, "nt_glutamate"] = 1
    f.loc[0, "nt_acetylcholine"] = 1
    f.loc[1, "nt_dopamine"] = 1
    f.loc[2, "nt_octopamine"] = 1
    f.loc[5, "nt_serotonin"] = 1
    f.loc[6, "nt_other"] = 1  # node 7: no label
    return f


def test_masks_and_signs():
    f = feat_frame()
    a, e, grp, sign, nd = rm.masks_and_signs(f, "lx")
    assert list(np.flatnonzero(a)) == [0, 1, 2] and list(np.flatnonzero(e)) == [3, 4]
    assert list(grp) == [0, 0, 0, 1, 1, 2, 2, 2]
    assert list(sign) == [1, 1, 1, -1, -1, 1, 1, 1] and nd == 2
    a, e, grp, sign, nd = rm.masks_and_signs(f, "al")  # AL: E = ALPN/ALON; superclass_descending ignored
    assert list(np.flatnonzero(e)) == [5, 6] and list(grp) == [0, 0, 0, 2, 2, 1, 1, 2]
    a, e, *_ = rm.masks_and_signs(f.drop(columns=["superclass_endocrine"]), "cx")  # absent column = 0
    assert list(np.flatnonzero(e)) == [3, 4]
    f.loc[0, "superclass_motor"] = 1
    with pytest.raises(AssertionError):
        rm.masks_and_signs(f, "cx")


def test_build_graph_drops_loops_asserts_dups():
    f = feat_frame()
    e = pd.DataFrame({"source": [0, 1, 3], "target": [3, 1, 4], "weight": [2, 5, 1]})
    g = rm.build_graph(e, f, "lx")
    assert g.meta["n_loops_dropped"] == 1 and len(g.src) == 2
    with pytest.raises(AssertionError):
        rm.build_graph(pd.DataFrame({"source": [0, 0], "target": [3, 3], "weight": [1, 1]}), f, "lx")


def test_eligibility():
    g = make_graph()
    g.meta = {"n_A": 10, "n_E": 10}
    assert rm.eligible(g)
    g.meta = {"n_A": 9, "n_E": 50}
    assert not rm.eligible(g)


# ---- statistics / decision ------------------------------------------------------------------

def test_z_stats_and_saturation():
    null = np.arange(1.0, 101.0)
    s = rm.z_stats(150.0, null)
    assert s["status"] == "defined" and s["p"] == 1 / 101
    assert np.isclose(s["z"], (150 - 50.5) / null.std(ddof=1)) and np.isclose(s["d"], (150 - 50.5) / 50.5)
    assert rm.z_stats(1.0, np.full(100, 5.0))["status"] == "undefined"  # SD = 0
    assert rm.z_stats(1.0, np.zeros(100))["status"] == "undefined"  # dead readout
    assert rm.z_stats(1.0, 1e-14 * np.arange(1, 50))["status"] == "undefined"
    assert rm.z_stats(1.0, 5.0 + 1e-9 * np.arange(100))["status"] == "undefined"  # SD < 1e-6 mean


def test_holm():
    assert np.allclose(rm.holm([0.01, 0.02]), [0.02, 0.02])
    assert np.allclose(rm.holm([0.03]), [0.03])
    assert np.allclose(rm.holm([0.04, 0.01]), [0.04, 0.02])


B, BS = 40, 20


def synth_draws(region, split, mean, sd, b=B, bs=BS, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(b):
        r = {"region": region, "split": split, "draw": i, "T": mean + sd * rng.standard_normal(),
             "rho": 1.0, "dense_fallback": 0, "rho_undefined": 0, "accepted": 1, "attempts": 2, "cap_hit": 0}
        for k in range(rm.K_FLIPS):
            r[f"T_flip{k}"] = mean + sd * rng.standard_normal() if i < bs else np.nan
        for a in ("0.5", "2.0"):
            r[f"T_amp{a}"] = mean + sd * rng.standard_normal() if i < bs else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def real_for(t, flips=None, amps=None):
    return {"T": t, "rho_raw": 1.0, "dense": 0, "undefined": False,
            "T_flip": flips if flips is not None else [t] * rm.K_FLIPS,
            "T_amp": amps if amps is not None else {0.5: t, 2.0: t}}


def mk(region, split, mean, sd, t, seed=0, **kw):
    return rm.cell_from_draws(synth_draws(region, split, mean, sd, seed=seed), real_for(t, **kw), True, b=B, bs=BS)


@pytest.fixture(autouse=True)
def small_b(monkeypatch):
    monkeypatch.setattr(rm, "B_SENS", BS)


def row(res, r, s):
    return res[(res.region == r) & (res.split == s)].iloc[0]


def test_decide_effect_labels_and_cx_secondary():
    cells = {(r, s): mk(r, s, 10, 1, 20 if r != "lx" else 10.0, seed=i)
             for i, (r, s) in enumerate((r, s) for r in rm.REGIONS for s in rm.SPLITS)}
    res = rm.decide(cells)
    al = row(res, "al", "f1")
    assert al.verdict == "effect declared" and al.sign_label == "sign-robust" and al.amp_label == "amplitude-robust"
    assert row(res, "lx", "f1").verdict == "no F1 effect"
    cx = row(res, "cx", "f1")
    assert cx.verdict.startswith("secondary") and cx.secondary


def test_al_needs_f2_strength():
    cells = {("al", "f1"): mk("al", "f1", 10, 1, 20), ("al", "f2"): mk("al", "f2", 10, 1, 12.5, seed=1),
             ("lx", "f1"): mk("lx", "f1", 10, 1, 20, seed=2), ("lx", "f2"): mk("lx", "f2", 10, 1, 12.5, seed=3)}
    assert 2 <= abs(cells[("al", "f2")]["z"]) < 3
    res = rm.decide(cells)
    assert row(res, "al", "f1").verdict == "AL: F1 only, not confirmed" and row(res, "al", "f1").replicated
    assert row(res, "lx", "f1").verdict == "effect declared"


def test_not_replicated_and_incomplete_f2():
    res = rm.decide({("lx", "f1"): mk("lx", "f1", 10, 1, 20), ("lx", "f2"): mk("lx", "f2", 10, 1, 10.0, seed=1)})
    assert row(res, "lx", "f1").verdict == "not replicated"
    short = rm.cell_from_draws(synth_draws("lx", "f2", 10, 1)[:10], real_for(20), True, b=B, bs=BS)
    assert short["status"] == "incomplete"
    res = rm.decide({("lx", "f1"): mk("lx", "f1", 10, 1, 20), ("lx", "f2"): short})
    assert row(res, "lx", "f1").verdict == "not replicated"
    opp = rm.decide({("lx", "f1"): mk("lx", "f1", 10, 1, 20), ("lx", "f2"): mk("lx", "f2", 10, 1, 0.0, seed=1)})
    assert not row(opp, "lx", "f1").replicated  # opposite sign


def test_magnitude_floor():
    cells = {("lx", "f1"): mk("lx", "f1", 1000, 0.001, 1000.1), ("lx", "f2"): mk("lx", "f2", 1000, 0.001, 1000.1)}
    r = row(rm.decide(cells), "lx", "f1")
    assert abs(r.z) >= 3 and abs(r.d) < 0.05 and r.verdict == "no F1 effect"


def test_holm_family_shrinks_and_cx_outside():
    cells = {(r, s): mk(r, s, 10, 1, 20, seed=i)
             for i, (r, s) in enumerate((r, s) for r in rm.REGIONS for s in rm.SPLITS)}
    res = rm.decide(cells)
    expected = rm.holm([cells[("al", "f1")]["p"], cells[("lx", "f1")]["p"]])
    assert np.isclose(row(res, "al", "f1").p_holm, expected[0])
    assert np.isclose(row(res, "cx", "f1").p_holm, row(res, "cx", "f1").p)  # CX never in the family
    # AL incomplete: LX alone in the family, adjusted == raw
    cells[("al", "f1")] = rm.cell_from_draws(synth_draws("al", "f1", 10, 1)[:5], real_for(20), True, b=B, bs=BS)
    res = rm.decide(cells)
    assert row(res, "al", "f1").status == "incomplete"
    assert np.isclose(row(res, "lx", "f1").p_holm, row(res, "lx", "f1").p)
    # holm adjustment really applied with two tests
    d = synth_draws("lx", "f1", 10, 1)
    c_a = rm.cell_from_draws(d, real_for(10.0 + 3.1), True, b=B, bs=BS)
    c_b = rm.cell_from_draws(synth_draws("al", "f1", 10, 1, seed=5), real_for(10.0 + 2.9), True, b=B, bs=BS)
    r2 = rm.decide({("al", "f1"): c_b, ("lx", "f1"): c_a})
    ps = [c_b["p"], c_a["p"]]
    assert np.allclose([row(r2, "al", "f1").p_holm, row(r2, "lx", "f1").p_holm], rm.holm(ps))


def test_status_precedence():
    d = synth_draws("al", "f1", 10, 1)
    kw = dict(b=B, bs=BS)
    assert rm.cell_from_draws(d, real_for(20), False, **kw)["status"] == "excluded"
    assert rm.cell_from_draws(d, dict(real_for(20), undefined=True), True, **kw)["status"] == "undefined"
    d2 = d.copy()
    d2.loc[3, ["cap_hit", "T"]] = [1, np.nan]
    assert rm.cell_from_draws(d2, real_for(20), True, **kw)["status"] == "null unreachable"
    d3 = d.copy()
    d3.loc[3, ["rho_undefined", "T"]] = [1, np.nan]
    c = rm.cell_from_draws(d3, real_for(20), True, **kw)
    assert c["status"] == "undefined" and "rho" in c["reason"]
    d4 = d.copy()
    d4["T"] = 5.0
    assert rm.cell_from_draws(d4, real_for(20), True, **kw)["status"] == "undefined"
    dup = pd.concat([d, d.iloc[:5]])  # duplicate draws across shards are dropped, not double counted
    assert rm.cell_from_draws(dup, real_for(20), True, **kw)["status"] == "defined"
    assert rm.cell_from_draws(d.iloc[0:0], real_for(20), True, **kw)["status"] == "incomplete"


def test_sign_flip_labels():
    d2 = mk("lx", "f2", 10, 1, 20, seed=9)
    flips = [20.0] * 7 + [10.0] * 3
    cells = {("lx", "f1"): mk("lx", "f1", 10, 1, 20, flips=flips), ("lx", "f2"): d2}
    r = row(rm.decide(cells), "lx", "f1")
    assert r.sign_label == "sign-sensitive" and r.n_flip_agree == 7
    assert np.isfinite(r.z_flip_median) and r.z_flip_min <= r.z_flip_max
    cells[("lx", "f1")] = mk("lx", "f1", 10, 1, 20, flips=[20.0] * 8 + [10.0] * 2)
    assert row(rm.decide(cells), "lx", "f1").sign_label == "sign-robust"
    only = {("lx", "f1"): mk("lx", "f1", 10, 1, 11.5, flips=[20.0] * 9 + [10.0])}
    r = row(rm.decide(only), "lx", "f1")
    assert r.verdict == "no F1 effect" and r.sign_label == "sensitivity-only, exploratory"
    rev = {("lx", "f1"): mk("lx", "f1", 10, 1, 20, flips=[0.0] * 10)}  # opposite sign never agrees
    assert row(rm.decide(rev), "lx", "f1").n_flip_agree == 0


def test_amplitude_labels():
    f2 = mk("lx", "f2", 10, 1, 20, seed=9)

    def lab(amps):
        r = row(rm.decide({("lx", "f1"): mk("lx", "f1", 10, 1, 20, amps=amps), ("lx", "f2"): dict(f2)}), "lx", "f1")
        return r.amp_label, r["z_amp0.5"], r["z_amp2.0"]
    assert lab({0.5: 20.0, 2.0: 20.0})[0] == "amplitude-robust"
    assert lab({0.5: 20.0, 2.0: 10.0})[0] == "amplitude-sensitive"  # one amplitude loses the effect
    assert lab({0.5: 0.0, 2.0: 20.0})[0] == "amplitude-sensitive"  # sign flip
    la = lab({0.5: 20.0, 2.0: 20.0})
    assert np.isfinite(la[1]) and np.isfinite(la[2])


def test_amplitude_uses_same_equalised_w():
    g = make_graph()
    ev = rm.evaluate(g, g.dst, [], sens=True)
    wm, *_ = rm.equalise(rm.signed_matrix(g.src, g.dst, g.w, g.n, g.sign))
    assert np.isclose(ev["T"], rm.simulate(wm, g.a_idx, g.e_idx, 1.0))
    assert np.isclose(ev["T_amp"][0.5], rm.simulate(wm, g.a_idx, g.e_idx, 0.5))
    assert np.isclose(ev["T_amp"][2.0], rm.simulate(wm, g.a_idx, g.e_idx, 2.0))


def test_run_cli_resumable_and_shard_independent(tmp_path, monkeypatch):
    from scripts import rate_model_rewiring as cli
    monkeypatch.setattr(cli, "check_frozen", lambda sha: None)
    g = make_graph()
    g.meta = {"n_A": 12, "n_E": 12}
    monkeypatch.setattr(rm, "load_graph", lambda *a, **k: g)
    out = tmp_path / "s.csv"
    base = ["run", "--region", "lx", "--split", "f1", "--seed", "11", "--out", str(out), "--data-root", str(tmp_path)]
    cli.main(base + ["--frozen-sha", "x", "--start", "0", "--end", "3"])
    first = pd.read_csv(out)
    assert list(first.draw) == [0, 1, 2]
    cli.main(base + ["--frozen-sha", "x", "--start", "0", "--end", "5"])
    second = pd.read_csv(out)
    assert list(second.draw) == [0, 1, 2, 3, 4]
    assert (second.iloc[:3]["T"].to_numpy() == first["T"].to_numpy()).all()
    out2 = tmp_path / "s2.csv"
    cli.main(["run", "--frozen-sha", "x", "--region", "lx", "--split", "f1", "--seed", "11", "--out", str(out2), "--start", "3",
              "--end", "5", "--data-root", str(tmp_path)])
    assert np.allclose(pd.read_csv(out2)["T"], second.iloc[3:]["T"])
    assert (second.accepted == 10 * len(g.src)).all()


def test_analyse_cli_runs_end_to_end(tmp_path, monkeypatch, capsys):
    from scripts import rate_model_rewiring as cli
    monkeypatch.setattr(cli, "check_frozen", lambda sha: None)
    g = make_graph()
    g.meta = {"n_A": 12, "n_E": 12}
    monkeypatch.setattr(rm, "load_graph", lambda *a, **k: g)
    shards = []
    for r in rm.REGIONS:
        for s in rm.SPLITS:
            o = tmp_path / f"{r}{s}.csv"
            cli.main(["run", "--frozen-sha", "x", "--region", r, "--split", s, "--n-perm", "6", "--end", "6", "--out", str(o)])
            shards.append(str(o))
    out = tmp_path / "res.csv"
    cli.main(["analyse", "--frozen-sha", "x", "--shards", *shards, "--n-perm", "6", "--out", str(out)])
    res = pd.read_csv(out)
    assert len(res) == 6 and set(res.status) <= {"defined", "undefined"}
    assert "PROBLEMS" in capsys.readouterr().out
    # a missing shard is reported as incomplete, never silently dropped
    cli.main(["analyse", "--frozen-sha", "x", "--shards", *shards[:-1], "--n-perm", "6", "--out", str(out)])
    assert (pd.read_csv(out).query("region == 'cx' and split == 'f2'").status == "incomplete").all()


def test_frozen_guard_refuses_wrong_sha():
    from scripts import rate_model_rewiring as cli
    with pytest.raises(SystemExit, match="refusing to run"):
        cli.check_frozen("0" * 40)


def test_undefined_replicate_keeps_cell_complete():
    bs = 5
    df = pd.DataFrame({"draw": range(bs), "T_flip0": [1.0, 2.0, np.nan, 1.5, 1.2]})
    s = rm._sens_z(df, "T_flip0", 1.0, bs)
    assert s is not None and np.isnan(s["z"])          # replicate undefined, not incomplete
    assert rm._sens_z(df.iloc[:4], "T_flip0", 1.0, bs) is None   # a missing draw is incomplete
    ok = pd.DataFrame({"draw": range(bs), "T_flip0": [1.0, 2.0, 1.1, 1.5, 1.2]})
    assert np.isfinite(rm._sens_z(ok, "T_flip0", 3.0, bs)["z"])
