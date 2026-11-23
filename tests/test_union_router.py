"""Seams of plan 05: the profile router, the linear gate's refusals and folds, and B1 admission."""

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureSchema, FeatureTable
from src.regions import region_labels
from train.union_router import apply_gate, fit_gate, profile_route, region_profile

pytest.importorskip("scipy")


def _z(a):
    return (a - a.mean(-1, keepdims=True)) / a.std(-1, keepdims=True)


def test_profile_router_weights_each_expert_by_the_pairs_innervation():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(2, 3, 5))  # experts A, B; 3 rows; K + 1 = 5
    pairs = rng.integers(0, 4, size=(3, 5, 2))
    profile = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # every neuron in A
    np.testing.assert_allclose(profile_route(scores, pairs, profile), _z(scores[0]))
    np.testing.assert_allclose(profile_route(scores, pairs, np.full((4, 2), 0.5)), _z(scores).mean(axis=0))
    mixed = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    out = profile_route(scores, pairs, mixed)
    w = mixed[pairs[..., 0]] + mixed[pairs[..., 1]]
    w = np.where(w.sum(-1, keepdims=True) > 0, w / np.maximum(w.sum(-1, keepdims=True), 1e-12), 0.5)
    np.testing.assert_allclose(out, (np.moveaxis(_z(scores), 0, -1) * w).sum(-1))


def test_hybrid_hands_a_concentrated_row_to_one_expert():
    rng = np.random.default_rng(1)
    scores, pairs = rng.normal(size=(2, 2, 4)), rng.integers(0, 3, size=(2, 4, 2))
    profile = np.array([[0.05, 0.95], [0.5, 0.5], [0.5, 0.5]])
    out = profile_route(scores, pairs, profile, masked=np.array([0, 1]), home_share=0.9)
    np.testing.assert_allclose(out[0], _z(scores[1])[0])  # masked neuron 0 sits in B
    np.testing.assert_allclose(out[1], profile_route(scores, pairs, profile)[1])
    with pytest.raises(ValueError, match="masked endpoint"):
        profile_route(scores, pairs, profile, home_share=0.9)


def test_region_profile_sums_labels_and_refuses_columns_outside_the_allowlist():
    labels = region_labels("CX") + region_labels("LH")
    schema = FeatureSchema("CX+LH", ("central",), ("a",), ("left",), ("intrinsic",), {"t1": 1}, labels=labels)
    values = np.zeros((2, schema.n_features))
    columns = schema.column_names
    values[0, columns.index("innervation_FB")] = 0.6
    values[0, columns.index("innervation_LH_L")] = 0.4
    table = FeatureTable(values, columns, {}, pd.DataFrame(), None)
    np.testing.assert_allclose(region_profile(table, schema, ["CX", "LH"])[0], [0.6, 0.4])
    np.testing.assert_allclose(region_profile(table, schema, ["CX", "LH"], uniform_rows=[0])[0], [0.5, 0.5])
    bad = FeatureTable(np.zeros((2, 1)), ("member_CX",), {}, pd.DataFrame(), None)
    with pytest.raises(ValueError, match="allowlist"):
        region_profile(bad, schema, ["CX"])


def test_gate_refuses_f2_rows_keeps_roots_in_one_fold_and_learns_the_better_expert():
    rng = np.random.default_rng(2)
    rows, width = 60, 6
    good = np.column_stack([np.full(rows, 3.0), rng.normal(size=(rows, width - 1))])
    bad = rng.normal(size=(rows, width))
    scores = np.stack([good, bad])
    features = np.ones((rows, width, 1))
    groups = np.repeat(np.arange(20), 3)
    with pytest.raises(ValueError, match="F2"):
        fit_gate(scores, features, datasets=["f1"] * (rows - 1) + ["f2"], groups=groups)
    theta, held = fit_gate(scores, features, datasets=["f1"] * rows, groups=groups, l2=1e-3)
    assert len(held) == 5 and theta[0, 0] > theta[0, 1]
    fused = apply_gate(scores, features, theta)
    assert (fused.argmax(axis=1) == 0).mean() > (bad.argmax(axis=1) == 0).mean()


def test_admission_compares_b0_with_the_best_bar_and_its_seed_spread():
    from scripts.router_report import admission

    def row(v):
        return {"all": {"i0": v, "i1": v}}

    report = {"INP expert (3 seeds, §2 form)": row(0.60), "Seed control, leave-one-instance-out weights": row(0.62),
              "Seed control: B0's weights over five seeds (weights fit in-sample)": row(0.70),  # never a bar
              "B0, leave-one-instance-out weights": row(0.58),
              "B0's weights, seed 0 only": row(0.57), "B0's weights, seed 1 only": row(0.59),
              "B0's weights, seed 2 only": row(0.58)}
    admitted, sentence = admission(report)
    assert admitted and "Seed control" in sentence  # gap 0.04 > spread 0.02
    report["B0, leave-one-instance-out weights"] = row(0.61)
    assert not admission(report)[0]  # gap 0.01 <= spread 0.02


def test_complementarity_strata_split_rows_by_visitor_membership():
    from scripts.complementarity_report import strata

    rows = {"i": (np.array([0, 1, 2, 3]), np.array([10, 11, 12, 13]))}
    split = strata(rows, {11, 13})
    assert split["all"]["i"].tolist() == [0, 1, 2, 3]
    assert split["shared"]["i"].tolist() == [1, 3] and split["home only"]["i"].tolist() == [0, 2]


def test_router_report_refuses_caches_from_mixed_commits(tmp_path, monkeypatch):
    # The router reads B0's caches, so it must refuse what B0's report refuses.
    import json

    import pytest

    from scripts import router_report

    for name in ("RegionGraph", "BoundarySuite", "CommittedNegatives"):
        monkeypatch.setattr(getattr(router_report, name), "read", lambda *a, **k: None)
    monkeypatch.setattr(router_report, "read_feature_artifacts", lambda *a: (None, None))
    suites = tmp_path / "f1" / "union12"
    suites.mkdir(parents=True)
    (suites / "s.selector.json").write_text(json.dumps({"instances": {}}))
    entry = {"mode": "union", "suite": "s", "region": "INP", "code": {}, "selector": "x", "scores": "y"}
    manifest = {"a.npz": {**entry, "seed": 0, "commit": "abc"}, "b.npz": {**entry, "seed": 1, "commit": "def"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="different commits"):
        router_report.main(["--suite", "s", "--manifest", str(tmp_path / "manifest.json"),
                            "--suites-root", str(tmp_path), "--data-root", str(tmp_path)])


def test_a_visitor_stands_in_for_one_home_seed(monkeypatch):
    # Question 1 separates visitors from seed averaging, so a visitor is one model, never three.
    from scripts import complementarity_report

    calls = []
    monkeypatch.setattr(complementarity_report, "held_out_fusion", lambda groups: calls.append(groups) or ({}, 0.0))
    home = ["h0", "h1", "h2"]
    visitors = {"VLNP": ["v0", "v1", "v2"], "INP": ["i0", "i1", "i2"]}
    complementarity_report.size_matched(home, visitors, ["VLNP", "INP"])
    assert [{k: len(g) for k, g in c.items()} for c in calls] == [{"home": 3, "VLNP": 1, "INP": 1}] * 3
    assert [c["VLNP"] for c in calls] == [["v0"], ["v1"], ["v2"]]


def test_seed_control_reuses_b0s_fold_weights_and_only_adds_seeds(monkeypatch):
    # The seed control is B0's mixture over five seeds, so its weights are never refit.
    from scripts import b0_report
    from train.union_scoring import save_cache  # noqa: F401  (cache format below matches it)

    calls, real = [], b0_report.loo_b0
    monkeypatch.setattr(b0_report, "loo_b0", lambda groups: calls.append(sorted(map(len, groups.values()))) or real(groups))
    row = {0: [3.0, 2.0, 1.0], 1: [2.0, 3.0, 1.0], 2: [1.0, 2.0, 3.0], 3: [3.0, 2.0, 1.0], 4: [3.0, 1.0, 2.0]}
    caches = {r: {s: {i: (np.arange(1), np.array([v])) for i in ("a", "b", "c")} for s, v in row.items()}
              for r in ("INP", "VLNP")}
    names = [name for name, _ in b0_report.methods(caches, {"all": None})[0]]
    assert b0_report.SEED_CONTROL_LOO in names  # the row plan 05's admission reads as a bar
    assert calls == [[3, 3]]  # one leave-one-out fit, over the gate seeds only


def test_complementarity_fits_and_ranks_on_single_masked_rows_only():
    from scripts.complementarity_report import single_rows

    cache = {"i": (np.array([0, 1, 2, 3]), np.arange(12.0).reshape(4, 3))}
    kept = single_rows(cache, {"i": (np.array([1, 3]), None)})["i"]
    assert kept[0].tolist() == [1, 3] and kept[1][:, 0].tolist() == [3.0, 9.0]


def test_complementarity_refuses_caches_scored_on_different_inputs(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from scripts import complementarity_report as cr
    from src.digests import sha256

    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cr.RegionGraph, "read", lambda *a: None)
    monkeypatch.setattr(cr.Suite, "read", lambda *a: SimpleNamespace(digest=lambda: "s"))
    monkeypatch.setattr(cr, "masked_roots", lambda graph, suite: {})
    monkeypatch.setattr(cr, "load_cache", lambda path: {})
    manifest = {}
    for seed in range(5):
        (tmp_path / f"h{seed}.npz").write_bytes(b"x")
        manifest[f"h{seed}.npz"] = {"mode": "home", "suite": "f1_val", "suite_region": "CX", "region": "CX",
                                    "seed": seed, "commit": "abc", "code": {}, "suite_digest": "s",
                                    "negatives_file": "n1" if seed else "n0", "scores": sha256(tmp_path / f"h{seed}.npz")}
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="different inputs"):
        cr.main(["--region", "CX", "--connections", "x", "--manifest", str(tmp_path / "m.json")])


def test_router_report_refuses_live_inputs_that_differ_from_the_caches(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from scripts import router_report
    from src.boundary_suites import selector_digest
    from src.digests import sha256

    named = lambda d: SimpleNamespace(digest=lambda: d)  # noqa: E731
    monkeypatch.setattr(router_report, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(router_report.RegionGraph, "read", lambda *a: named("g"))
    monkeypatch.setattr(router_report, "read_feature_artifacts", lambda *a: (None, named("sch")))
    monkeypatch.setattr(router_report.BoundarySuite, "read", lambda *a: named("s"))
    monkeypatch.setattr(router_report.CommittedNegatives, "read", lambda *a: None)
    union, suites = tmp_path / "processed" / "f1" / "union12", tmp_path / "f1" / "union12"
    union.mkdir(parents=True), suites.mkdir(parents=True)
    (union / "features.parquet").write_bytes(b"features at scoring time")
    (suites / "s.negatives.json").write_bytes(b"negatives")
    selector = {"instances": {}}
    (suites / "s.selector.json").write_text(json.dumps(selector))
    (tmp_path / "c.npz").write_bytes(b"cache")
    entry = {"mode": "union", "suite": "s", "region": "INP", "seed": 0, "commit": "abc", "code": {},
             "selector": selector_digest(selector), "scores": sha256(tmp_path / "c.npz"), "union_graph": "g",
             "union_schema": "sch", "union_features": sha256(union / "features.parquet"), "suite_digest": "s",
             "negatives_file": sha256(suites / "s.negatives.json")}
    (tmp_path / "m.json").write_text(json.dumps({"c.npz": entry}))
    (union / "features.parquet").write_bytes(b"features edited since")  # the router would read these live
    with pytest.raises(SystemExit, match="union_features"):
        router_report.main(["--suite", "s", "--manifest", str(tmp_path / "m.json"),
                            "--suites-root", str(tmp_path), "--data-root", str(tmp_path)])


def test_gate_refuses_metadata_that_does_not_cover_every_row():
    scores, features = np.zeros((2, 4, 3)), np.zeros((4, 3, 1))
    with pytest.raises(ValueError, match="same rows"):
        fit_gate(scores, features, datasets=["f1"] * 4, groups=[0, 1])
    with pytest.raises(ValueError, match="same rows"):
        fit_gate(scores, features, datasets=[], groups=[0, 1, 2, 3])


def test_each_fold_is_fit_without_its_own_masked_roots(monkeypatch):
    # The held-out loss of a fold must come from a model that never saw that fold's roots.
    import scipy.optimize

    fits, real = [], scipy.optimize.minimize

    def spy(fun, x0, args=(), **kwargs):
        fits.append(np.asarray(args[0]))
        return real(fun, x0, args=args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", spy)
    rows, groups = 40, np.repeat(np.arange(10), 4)
    scores = np.random.default_rng(0).normal(size=(2, rows, 4))
    fit_gate(scores, np.ones((rows, 4, 1)), datasets=["f1"] * rows, groups=groups, folds=5)
    held_out = [np.setdiff1d(np.arange(rows), f) for f in fits[:-1]]  # the last fit is the final one, on every row
    assert np.array_equal(np.sort(np.concatenate(held_out)), np.arange(rows))  # each row held out exactly once
    assert all(not set(groups[f]) & set(groups[h]) for f, h in zip(fits[:-1], held_out))


def test_a_is_never_a_visiting_expert_in_question_1(tmp_path, monkeypatch):
    # Plan 06's regional --system-run caches look like a visitor (mode regional, a gate seed).
    import json
    from types import SimpleNamespace

    from scripts import complementarity_report as cr
    from src.digests import sha256

    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cr.RegionGraph, "read", lambda *a: None)
    monkeypatch.setattr(cr.Suite, "read", lambda *a: SimpleNamespace(digest=lambda: "s"))
    monkeypatch.setattr(cr, "masked_roots", lambda graph, suite: {})
    monkeypatch.setattr(cr, "load_cache", lambda path: {})
    manifest = {}
    common = {"suite": "f1_val", "suite_region": "CX", "commit": "abc", "code": {}, "suite_digest": "s"}
    for name, extra in [*((f"h{s}", {"mode": "home", "region": "CX", "seed": s}) for s in range(5)),
                        *((f"a{s}", {"mode": "regional", "region": "A", "seed": s, "system": "A"}) for s in range(3))]:
        (tmp_path / f"{name}.npz").write_bytes(name.encode())
        manifest[f"{name}.npz"] = {**common, **extra, "scores": sha256(tmp_path / f"{name}.npz")}
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="no visitor"):
        cr.main(["--region", "CX", "--connections", "x", "--manifest", str(tmp_path / "m.json")])
