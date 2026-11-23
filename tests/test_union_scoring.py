"""Seams of plan 03's union scoring: home parity, row subsets, regional masking, fusion, B0 weights."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.evaluate import CommittedNegatives  # noqa: E402
from src.features import FeatureTable  # noqa: E402
from src.masks import apply_mask  # noqa: E402
from src.negatives import DEFAULT_SPEC  # noqa: E402
from src.regions import RegionGraph  # noqa: E402
from src.suites import build_suite, suite_negatives  # noqa: E402
from train import union_scoring  # noqa: E402
from train.model import M1  # noqa: E402
from train.scoring import score_expert  # noqa: E402
from train.union_scoring import (  # noqa: E402
    GEOMETRY, expert_geometry, expert_table, fit_weights, fuse, instance_mrr, load_cache, save_cache, score_cache,
)


@pytest.fixture(scope="module")
def scored(cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids):
    torch.manual_seed(0)
    model = M1(cx_schema, hidden=8, layers=2)
    model.set_trained_types(np.ones(len(cx_schema.type_vocab) + 1, dtype=bool))
    suite = build_suite(cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids)
    samples = suite_negatives(cx_graph_f1, suite)
    negatives = CommittedNegatives(suite.name, DEFAULT_SPEC, {k: v.digest() for k, v in samples.items()})
    return model, suite, samples, negatives


def test_home_mode_matches_the_score_run_path(scored, cx_graph_f1, cx_features_f1, cx_types):
    model, suite, samples, negatives = scored
    cache = score_cache(model, mode="home", graph=cx_graph_f1, table=cx_features_f1, suite=suite,
                        negatives=negatives, samples=samples)
    rows, _ = score_expert(model, cx_graph_f1, cx_features_f1, suite, types=cx_types, negatives=negatives,
                           rung="b")
    single = {inst.instance_id: np.flatnonzero(~apply_mask(cx_graph_f1, inst).both_masked())
              for inst in suite.resolve(cx_graph_f1)}
    mine = instance_mrr(cache, single)
    for row in rows.itertuples():
        assert mine[row.mask_instance] == pytest.approx(row.mrr, abs=1e-12)


def test_scoring_a_row_subset_gives_those_rows_scores(scored, cx_graph_f1, cx_features_f1):
    model, suite, samples, negatives = scored
    full = score_cache(model, mode="home", graph=cx_graph_f1, table=cx_features_f1, suite=suite,
                       negatives=negatives, samples=samples)
    picked = {iid: rows[::3] for iid, (rows, _) in full.items()}
    selector = {"suite_digest": suite.digest(), "instances": {iid: {"rows": r.tolist()} for iid, r in picked.items()}}
    subset = score_cache(model, mode="union", graph=cx_graph_f1, table=cx_features_f1, suite=suite,
                         negatives=negatives, selector=selector, samples=samples)
    for iid, (rows, scores) in subset.items():
        assert np.array_equal(rows, picked[iid])
        np.testing.assert_allclose(scores, full[iid][1][rows], rtol=1e-6, atol=1e-6)


def test_regional_mode_hides_every_union_edge_of_a_masked_root(scored, cx_graph_f1, cx_features_f1, monkeypatch):
    model, suite, samples, negatives = scored
    graph = cx_graph_f1
    # The "union" adds an edge from every node to its successor: some join a masked root
    # to a partner the regional graph never links it to.
    extra = np.stack([np.arange(graph.n_nodes - 1), np.arange(1, graph.n_nodes)], axis=1).astype(np.int32)
    edges = np.unique(np.concatenate([graph.edges, extra]), axis=0)
    union = RegionGraph("f1", "CX+LH", 5, graph.node_ids, edges, np.full(len(edges), 5))
    seen = []
    real = union_scoring._predict
    monkeypatch.setattr(union_scoring, "_predict", lambda m, view, t, p, w: seen.append(view) or real(m, view, t, p, w))
    score_cache(model, mode="regional", graph=graph, table=cx_features_f1, suite=suite, negatives=negatives,
                union_graph=union, union_table=cx_features_f1, samples=samples)
    for view, inst in zip(seen, suite.resolve(graph), strict=True):
        flags = np.zeros(graph.n_nodes, dtype=bool)
        flags[inst.masked] = True
        assert not (flags[view.visible_edges[:, 0]] | flags[view.visible_edges[:, 1]]).any()
        total = np.zeros(graph.n_nodes, dtype=np.int64)
        np.add.at(total, edges.ravel(), 5)
        assert np.array_equal(view.degree_given_total()[flags], total[flags])
    # A union missing half the region's neurons must refuse, not index the last node with -1.
    half = graph.n_nodes // 2
    inside = edges[(edges < half).all(axis=1)]
    partial = RegionGraph("f1", "CX+LH", 5, graph.node_ids[:half], inside, np.full(len(inside), 5))
    with pytest.raises(ValueError, match="not in the union graph"):
        score_cache(model, mode="regional", graph=graph, table=cx_features_f1, suite=suite, negatives=negatives,
                    union_graph=partial, union_table=cx_features_f1, samples=samples)


def test_fusion_combines_seeds_before_ranking_and_respects_columns():
    rng = np.random.default_rng(1)
    caches = [{"i": (np.arange(4), rng.normal(size=(4, 6)))} for _ in range(3)]
    fused = fuse({"A": caches[:2], "B": caches[2:]}, {"A": 1.0, "B": 0.5})["i"][1]
    z = [(c["i"][1] - c["i"][1].mean(1, keepdims=True)) / c["i"][1].std(1, keepdims=True) for c in caches]
    np.testing.assert_allclose(fused, (z[0] + z[1]) / 2 + 0.5 * z[2])
    perm = rng.permutation(6)
    permuted = [{"i": (c["i"][0], c["i"][1][:, perm])} for c in caches]
    np.testing.assert_allclose(fuse({"A": permuted[:2], "B": permuted[2:]}, {"A": 1.0, "B": 0.5})["i"][1], fused[:, perm])
    constant = {"i": (np.arange(2), np.ones((2, 6)))}
    assert not fuse({"C": [constant]})["i"][1].any()
    with pytest.raises(ValueError, match="different rows"):
        fuse({"A": [caches[0], {"i": (np.arange(1, 5), caches[1]["i"][1])}]})


def test_weights_never_use_held_out_instances_and_pick_the_informative_expert():
    good = {f"i{k}": (np.arange(5), np.column_stack([np.full(5, 9.0), np.zeros((5, 4))])) for k in range(3)}
    bad = {f"i{k}": (np.arange(5), np.column_stack([np.zeros(5), np.full((5, 4), 9.0)])) for k in range(3)}
    weights = fit_weights({"good": [good], "bad": [bad]}, ["i0", "i1"])
    assert weights["bad"] == 0.0 and weights["good"] > 0
    flipped = {**bad, "i2": good["i2"]}  # the held-out instance contradicts the fitted ones
    assert fit_weights({"good": [good], "bad": [flipped]}, ["i0", "i1"]) == weights


def test_expert_table_renormalises_innervation_over_the_experts_labels(cx_schema):
    union = dataclasses.replace(cx_schema, region="CX+LH", labels=(*cx_schema.labels, "LH_L"))
    values = np.zeros((2, union.n_features))
    inn = {c: k for k, c in enumerate(union.column_names) if c.startswith("innervation_")}
    first = f"innervation_{cx_schema.labels[0]}"
    values[0, inn[first]], values[0, inn["innervation_LH_L"]] = 0.25, 0.75
    values[1, inn["innervation_LH_L"]] = 1.0  # nothing on the expert's labels: uniform
    table = FeatureTable(values, union.column_names, {}, pd.DataFrame(), None)
    out = expert_table(table, union, cx_schema, geometry=(np.zeros((2, 6)), np.zeros(2, dtype=bool)))
    cols = [cx_schema.column_names.index(f"innervation_{label}") for label in cx_schema.labels]
    assert out.values[0, cols].sum() == pytest.approx(1.0)
    assert out.values[0, cx_schema.column_names.index(first)] == pytest.approx(1.0)
    np.testing.assert_allclose(out.values[1, cols], 1 / len(cols))


def test_expert_geometry_on_the_union_is_the_regions_own_geometry(tmp_path):
    # A region neuron must read, through the union, the geometry its expert trained on.
    pa = pytest.importorskip("pyarrow")
    from pyarrow import feather

    from scripts.build_features import stream_synapse_geometry
    from src.features import FeatureSchema, build_features
    from src.regions import region_labels

    rng = np.random.default_rng(3)
    big = 10**17
    ids = np.add(np.arange(12), big)
    n = 4000
    pre, post = rng.integers(0, 12, n), rng.integers(0, 12, n)
    labels = rng.choice(["FB", "EB", "PB", "LH_L", "LH_R"], n)
    xyz = {f"{end}_{a}": rng.normal(loc=k * 50 + (labels == "LH_L") * 400, scale=30 + k * 5, size=n)
           for k, a in enumerate("xyz") for end in ("pre", "post")}
    feather.write_feather(pa.table({"pre_root_id": ids[pre], "post_root_id": ids[post], "neuropil": labels, **xyz}),
                          tmp_path / "syn.feather")
    region_ids = ids[:7]  # the CX graph's nodes; the union adds five neurons
    cx = RegionGraph("f1", "CX", 5, region_ids, np.array([[0, 1], [1, 2]], dtype=np.int32), np.array([5, 5]))
    union = RegionGraph("f1", "CX+LH", 5, ids, np.array([[0, 1], [1, 2]], dtype=np.int32), np.array([5, 5]))
    annotations = pd.DataFrame({"root_id": ids, "super_class": "central", "class": "a", "cell_type": "t1",
                                "side": "left", "flow": "intrinsic", "nt_type": "ACH"})
    schema = FeatureSchema.from_f1(annotations, "CX")
    regional = build_features(annotations[:7], cx, schema, dataset="f1",
                              synapses=stream_synapse_geometry(tmp_path / "syn.feather", cx))
    raw = stream_synapse_geometry(tmp_path / "syn.feather", union, labels=region_labels("CX"))
    values, present = expert_geometry(raw, union.node_ids, region_ids)
    cols = [schema.column_names.index(c) for c in GEOMETRY]
    np.testing.assert_allclose(values[:7], regional.values[:, cols], atol=1e-9)
    assert np.array_equal(present[:7], regional.geometry_present)
    # The union's own frame (all union labels, z-scored union-wide) is not the expert's.
    union_raw = stream_synapse_geometry(tmp_path / "syn.feather", union, labels=region_labels("CX") + region_labels("LH"))
    union_frame, _ = expert_geometry(union_raw, union.node_ids, union.node_ids)
    assert not np.allclose(union_frame[:7], regional.values[:, cols], atol=0.05)


def test_masked_innervation_uniform_reaches_the_scorer_for_masked_nodes_only(
        scored, cx_graph_f1, cx_features_f1, monkeypatch):
    model, suite, samples, negatives = scored
    selector = {"suite_digest": suite.digest(),
                "instances": {inst.instance_id: {"rows": [0]} for inst in suite.resolve(cx_graph_f1)}}
    seen = []
    real = union_scoring._predict
    monkeypatch.setattr(union_scoring, "_predict", lambda m, v, t, p, w: seen.append((v, t)) or real(m, v, t, p, w))
    score_cache(model, mode="union", graph=cx_graph_f1, table=cx_features_f1, suite=suite,
                negatives=negatives, selector=selector, samples=samples, masked_uniform=True)
    cols = [k for k, c in enumerate(cx_features_f1.columns) if c.startswith("innervation_")]
    assert cols
    for view, table in seen:
        masked = np.flatnonzero(view.is_masked)
        assert masked.size  # a boundary suite masks roots, so there is always something to ablate
        np.testing.assert_allclose(table.values[masked][:, cols], 1.0 / len(cols))
        keep = np.ones(len(table.values), dtype=bool)
        keep[masked] = False
        np.testing.assert_array_equal(table.values[keep], cx_features_f1.values[keep])
    with pytest.raises(ValueError, match="masks nothing"):
        score_cache(model, mode="home", graph=cx_graph_f1, table=cx_features_f1, suite=suite,
                    negatives=negatives, samples=samples, masked_uniform=True)


def test_boundary_baselines_score_an_experts_rows_through_uniform_innervation(scored, cx_graph_f1, cx_features_f1):
    from scripts.score_union_baselines import score_rows
    from train.baselines import BASELINES

    model, suite, samples, negatives = scored
    full = score_cache(model, mode="home", graph=cx_graph_f1, table=cx_features_f1, suite=suite,
                       negatives=negatives, samples=samples)
    selector = {"suite_digest": suite.digest(),
                "instances": {iid: {"rows": rows[::3].tolist()} for iid, (rows, _) in full.items()}}
    common = dict(graph=cx_graph_f1, table=cx_features_f1, suite=suite, negatives=negatives,
                  selector=selector, samples=samples)
    expert = score_cache(model, mode="union", masked_uniform=True, **common)
    seen = []
    configuration = BASELINES["configuration"]()
    cache = score_rows(lambda v, t, p: seen.append((v, t)) or configuration.score(v, t, p), uniform=True, **common)
    assert cache.keys() == expert.keys()
    for iid, (rows, scores) in cache.items():
        assert np.array_equal(rows, expert[iid][0]) and scores.shape == expert[iid][1].shape
    cols = [k for k, c in enumerate(cx_features_f1.columns) if c.startswith("innervation_")]
    for view, table in seen:
        np.testing.assert_allclose(table.values[np.flatnonzero(view.is_masked)][:, cols], 1.0 / len(cols))
    assert set(instance_mrr(cache)) == set(cache)


def test_baseline_manifest_keys_are_posix_paths_under_the_repo(tmp_path, monkeypatch):
    from scripts.score_union_baselines import manifest_key

    monkeypatch.chdir(tmp_path)
    root = tmp_path.resolve()
    assert manifest_key(Path("results") / "union" / "x.npz", root) == "results/union/x.npz"  # a relative --out
    assert manifest_key(root / "results" / "x.npz", root) == "results/x.npz"
    outside = tmp_path.parent / "elsewhere.npz"
    assert manifest_key(outside, root) == outside.resolve().as_posix()


def test_cache_round_trip(tmp_path):
    cache = {"f1-union12-R01-s5000": (np.array([3, 7]), np.arange(12, dtype=float).reshape(2, 6))}
    save_cache(tmp_path / "c.npz", cache)
    back = load_cache(tmp_path / "c.npz")
    assert np.array_equal(back["f1-union12-R01-s5000"][0], [3, 7])
    np.testing.assert_allclose(back["f1-union12-R01-s5000"][1], cache["f1-union12-R01-s5000"][1])


def test_b0_report_runs_end_to_end_on_synthetic_caches(tmp_path, monkeypatch):
    import json

    from scripts import b0_report

    rng = np.random.default_rng(2)
    instances = ["f1-union12-R01-s5000", "f1-union12-T-s5003"]
    from src.boundary_suites import selector_digest
    from src.digests import sha256

    selector = {"instances": {iid: {"rows": list(range(6)), "n_rows": 6, "block_rows": [0, 2]} for iid in instances}}
    manifest = {}
    for region in ("INP", "VLNP"):
        for seed in range(5):
            cache = {iid: (np.arange(6), rng.normal(size=(6, 101))) for iid in instances}
            path = tmp_path / f"{region}-{seed}.npz"
            save_cache(path, cache)
            manifest[path.name] = {"mode": "union", "suite": "s", "region": region, "seed": seed, "commit": "abc",
                                   "code": {}, "selector": selector_digest(selector), "scores": sha256(path)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "f1" / "union12").mkdir(parents=True)
    (tmp_path / "f1" / "union12" / "s.selector.json").write_text(json.dumps(selector))
    monkeypatch.setattr(b0_report, "REPO_ROOT", tmp_path)
    b0_report.main(["--suite", "s", "--manifest", str(tmp_path / "manifest.json"), "--suites-root", str(tmp_path),
                    "--out", str(tmp_path / "report.md")])
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    for name in ("INP expert (3 seeds, §2 form)", "VLNP 5-seed ensemble", "B0, leave-one-instance-out weights",
                 "Seed control: B0's weights over five seeds", "B0's weights, seed 2 only"):
        assert name in text
    assert "the mean of the 2 instances' MRRs, over 12 metric rows (4 in the" in text
    assert "Those rows are not independent replicates" in text


def _b0_group(tmp_path, manifest, *, suite, dataset, instances, rows, code,
               registry="results/experts-clean.csv", masked_innervation="uniform"):
    import json

    from src.boundary_suites import selector_digest
    from src.digests import sha256

    selector = {"instances": {iid: {"rows": [0, 1], "n_rows": 2, "block_rows": [0]} for iid in instances}}
    selector_path = tmp_path / dataset / "union12" / f"{suite}.selector.json"
    selector_path.parent.mkdir(parents=True, exist_ok=True)
    selector_path.write_text(json.dumps(selector), encoding="utf-8")
    for region, seeds in rows.items():
        for seed, scores in seeds.items():
            cache = {iid: (np.arange(2), np.array([scores, scores])) for iid in instances}
            path = tmp_path / f"{dataset}-{suite}-{region}-{seed}.npz"
            save_cache(path, cache)
            manifest[path.name] = {
                "mode": "union", "suite": suite, "region": region, "seed": seed, "dataset": dataset,
                "registry": registry, "masked_innervation": masked_innervation,
                "commit": "abc", "code": code, "selector": selector_digest(selector), "scores": sha256(path),
                "union_graph": f"graph-{dataset}", "union_schema": "schema",
                "union_features": f"features-{dataset}", "suite_digest": "suite", "negatives_file": "neg",
                "cohort": f"cohort-{region}", "expert_schema": f"schema-{region}",
                "expert_geometry": f"geometry-{dataset}-{region}",
            }


def _b0_args(tmp_path, *, dataset="f2", suite="s"):
    return ["--suite", suite, "--dataset", dataset, "--manifest", str(tmp_path / "manifest.json"),
            "--suites-root", str(tmp_path), "--registry", str(tmp_path / "results" / "experts-clean.csv"),
            "--masked-innervation", "uniform", "--out", str(tmp_path / "report.md")]


def test_b0_report_reads_one_dataset_at_a_time(tmp_path, monkeypatch):
    """A cache records the dataset it was scored on; a report reads only the one it names."""
    import json

    from scripts import b0_report

    manifest, code, first = {}, {"score_union.py": "same"}, [3.0, 2.0, 1.0]
    rows = {"INP": {s: first for s in range(3)}, "VLNP": {s: first for s in range(3)}}
    _b0_group(tmp_path, manifest, suite=b0_report.WEIGHTS_SUITE, dataset="f1",
              instances=["f1-i0", "f1-i1"], rows=rows, code=code)
    _b0_group(tmp_path, manifest, suite="s", dataset="f2",
              instances=["f2-i0", "f2-i1"], rows=rows, code=code)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "experts-clean.csv").write_text("", encoding="utf-8")
    monkeypatch.setattr(b0_report, "REPO_ROOT", tmp_path)
    b0_report.main(_b0_args(tmp_path))
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "(F2, cross-dataset)" in text
    assert "whose edges no expert saw" in text
    with pytest.raises(SystemExit):  # the F2 caches are not F1's
        b0_report.main(_b0_args(tmp_path, dataset="f1"))


def test_b0_report_on_f2_uses_f1_val_weights(tmp_path, monkeypatch):
    import json

    from scripts import b0_report

    manifest, code, first, last = {}, {"score_union.py": "same"}, [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]
    f1_rows = {"INP": {s: first for s in range(3)}, "VLNP": {s: first for s in range(3)}}
    f2_rows = {"INP": {s: last for s in range(3)}, "VLNP": {s: first for s in range(3)}}
    _b0_group(tmp_path, manifest, suite=b0_report.WEIGHTS_SUITE, dataset="f1",
              instances=["f1-i0", "f1-i1"], rows=f1_rows, code=code)
    _b0_group(tmp_path, manifest, suite="s", dataset="f2",
              instances=["f2-i0", "f2-i1"], rows=f2_rows, code=code)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "experts-clean.csv").write_text("", encoding="utf-8")
    monkeypatch.setattr(b0_report, "REPO_ROOT", tmp_path)
    b0_report.main(_b0_args(tmp_path))
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    rows = {line.split(" | ")[0].lstrip("| "): line.split(" | ")[1]
            for line in text.splitlines() if line.startswith("| ")}
    assert rows[b0_report.B0_FROZEN] == "0.5000"
    assert "leave-one-instance-out" not in text
    assert "in-sample" not in text
    assert "B0's weights (fit on `phase3_boundary_f1_val` (F1-val) only): INP 1, VLNP 1." in text


def test_b0_report_on_f2_refuses_missing_weight_caches(tmp_path, monkeypatch):
    import json

    from scripts import b0_report

    manifest, first = {}, [3.0, 2.0, 1.0]
    rows = {"INP": {s: first for s in range(3)}, "VLNP": {s: first for s in range(3)}}
    _b0_group(tmp_path, manifest, suite="s", dataset="f2",
              instances=["f2-i0", "f2-i1"], rows=rows, code={})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "experts-clean.csv").write_text("", encoding="utf-8")
    monkeypatch.setattr(b0_report, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="no phase3_boundary_f1_val caches"):
        b0_report.main(_b0_args(tmp_path))


def test_b0_report_on_f2_refuses_other_weight_cohort(tmp_path, monkeypatch):
    import json

    from scripts import b0_report

    manifest, code, first = {}, {"score_union.py": "same"}, [3.0, 2.0, 1.0]
    rows = {"INP": {s: first for s in range(3)}, "VLNP": {s: first for s in range(3)}}
    _b0_group(tmp_path, manifest, suite=b0_report.WEIGHTS_SUITE, dataset="f1",
              instances=["f1-i0", "f1-i1"], rows=rows, code=code)
    _b0_group(tmp_path, manifest, suite="s", dataset="f2",
              instances=["f2-i0", "f2-i1"], rows=rows, code=code)
    for entry in manifest.values():
        if entry["suite"] == b0_report.WEIGHTS_SUITE and entry["region"] == "INP":
            entry["cohort"] = "other"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "experts-clean.csv").write_text("", encoding="utf-8")
    monkeypatch.setattr(b0_report, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="another cohort"):
        b0_report.main(_b0_args(tmp_path))


def test_score_union_reads_the_dataset_it_names(tmp_path, paired_graph_f2, paired_annotations):
    """The CLI's dataset picks the processed and suites trees."""
    from conftest import _synthetic_centroids, _write_committed_negatives

    from scripts import score_union
    from src.features import FeatureSchema, build_features

    graph = paired_graph_f2
    schema = FeatureSchema.from_f1(paired_annotations, "AL")
    folder = tmp_path / "processed" / "f2" / "al"
    folder.mkdir(parents=True)
    graph.write(folder)
    table = build_features(paired_annotations, graph, schema, dataset="f2")
    frame = pd.DataFrame(table.values, columns=table.columns)
    frame.insert(0, "root_id", graph.node_ids)
    frame.to_parquet(folder / "features.parquet", index=False)
    (folder / "schema.json").write_text(schema.to_json())
    table.coverage.to_csv(folder / "coverage.csv", index=False)

    types = paired_annotations.set_index("root_id")["cell_type"].reindex(graph.node_ids)
    full = build_suite(graph, name="f2_dev", types=types, centroids=_synthetic_centroids(graph.n_nodes))
    suite = dataclasses.replace(full, instances=full.instances[:1])
    suites = tmp_path / "suites" / "f2" / "al"
    suites.mkdir(parents=True)
    suite.write(suites / "f2_dev.json")
    _write_committed_negatives(tmp_path / "suites", suite, suite_negatives(graph, suite))

    read, _, _, suite_read, _, path = score_union.region_inputs("AL", "f2_dev", tmp_path, tmp_path / "suites", "f2")
    assert read.dataset == "f2"
    assert suite_read.name == "f2_dev"
    assert path == suites / "f2_dev.json"
    with pytest.raises(OSError):  # nothing was written under f1
        score_union.region_inputs("AL", "f2_dev", tmp_path, tmp_path / "suites", "f1")


@pytest.mark.parametrize("mode, extra, error, missing", [
    ("union", [], OSError, "union12"),
    ("regional", ["--region", "AL"], SystemExit, "regional is missing"),  # A's regional rows read U+
])
def test_score_union_allows_a_on_f2(tmp_path, mode, extra, error, missing):
    from scripts import score_union

    with pytest.raises(error, match=missing):  # the missing F2 union inputs, not the old dataset refusal
        score_union.main(["--mode", mode, "--dataset", "f2", "--suite", "s", "--system-run", "r",
                          "--data-root", str(tmp_path), "--suites-root", str(tmp_path / "suites"),
                          "--out", str(tmp_path), *extra])


def test_train_inputs_reads_the_f1_union_folder(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import score_union

    seen = {}

    def read(folder):
        seen["folder"] = folder
        return SimpleNamespace(digest=lambda: "g")

    monkeypatch.setattr(score_union, "RegionGraph", SimpleNamespace(read=read))
    monkeypatch.setattr(score_union, "read_feature_artifacts",
                        lambda folder, graph: (SimpleNamespace(columns=("x",)),
                                               SimpleNamespace(digest=lambda: "s", column_names=("x",))))
    monkeypatch.setattr(score_union, "check_inputs", lambda *a: None)
    graph, _, schema = score_union.train_inputs("union12", tmp_path)
    assert seen["folder"] == tmp_path / "processed" / "f1" / "union12"
    assert graph.digest() == "g" and schema.digest() == "s"


def test_score_union_refuses_unlabelled_confirm_reads(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import score_union

    args = SimpleNamespace(suite="phase3_boundary_f2_confirm", mode="union", region=None, milestone=None)
    monkeypatch.setattr(score_union, "phase2_verdict", lambda *a: "PASS")
    for blank in (None, "  "):
        args.milestone = blank
        with pytest.raises(SystemExit, match="requires --milestone"):
            score_union.phase3_confirm_preflight(args, tmp_path)
    args.milestone = "m1"
    assert score_union.phase3_confirm_preflight(args, tmp_path) == {"milestone": "m1"}
    with pytest.raises(SystemExit, match="confirm suites only"):
        score_union.phase3_confirm_preflight(SimpleNamespace(**{**vars(args), "suite": "f1_val"}), tmp_path)
    args.suite, args.mode, args.region = "f2_confirm", "regional", ["AL"]
    monkeypatch.setattr(score_union, "phase2_verdict", lambda *a: None)
    with pytest.raises(SystemExit, match="no accepted Phase 2 verdict"):
        score_union.phase3_confirm_preflight(args, tmp_path)
    monkeypatch.setattr(score_union, "phase2_verdict", lambda *a: "PASS")
    assert score_union.phase3_confirm_preflight(args, tmp_path) == {"milestone": "m1"}
    with pytest.raises(SystemExit, match="requires --milestone"):
        score_union.main(["--mode", "union", "--dataset", "f2", "--suite", "phase3_boundary_f2_confirm"])
    from scripts import score_union_baselines
    with pytest.raises(SystemExit, match="requires --milestone"):  # before any input opens
        score_union_baselines.main(["--suite", "phase3_boundary_f2_confirm", "--data-root", str(tmp_path)])


def test_b0_report_refuses_mixed_code_another_selector_or_a_changed_cache(tmp_path):
    from scripts.b0_report import check_provenance
    from src.boundary_suites import selector_digest
    from src.digests import sha256

    selector = {"suite_digest": "s", "instances": {"i": {"rows": [1, 2]}}}
    entries = {}
    for k in range(2):
        save_cache(tmp_path / f"c{k}.npz", {"i": (np.array([1, 2]), np.zeros((2, 3)))})
        entries[f"c{k}.npz"] = {"commit": "abc", "code": {"x.py": "1"}, "selector": selector_digest(selector),
                                "scores": sha256(tmp_path / f"c{k}.npz"), "region": "CX", "union_features": "u",
                                "cohort": "c", "expert_geometry": "g"}
    check_provenance(entries, selector, tmp_path)
    # Valid hashes, different inputs: a seed rescored on new geometry, or on another union table.
    for key in ("expert_geometry", "cohort", "union_features"):
        with pytest.raises(SystemExit, match="different inputs"):
            check_provenance({**entries, "c1.npz": {**entries["c1.npz"], key: "new"}}, selector, tmp_path)
    other = {**entries["c1.npz"], "region": "LH", "cohort": "lh", "expert_geometry": "lh"}
    check_provenance({**entries, "c1.npz": other}, selector, tmp_path)  # regions may differ from each other
    with pytest.raises(SystemExit, match="different commits"):
        check_provenance({**entries, "c1.npz": {**entries["c1.npz"], "commit": "def"}}, selector, tmp_path)
    with pytest.raises(SystemExit, match="another selector"):
        check_provenance(entries, {**selector, "instances": {"i": {"rows": [1]}}}, tmp_path)
    save_cache(tmp_path / "c1.npz", {"i": (np.array([1, 2]), np.ones((2, 3)))})
    with pytest.raises(SystemExit, match="changed on disk"):
        check_provenance(entries, selector, tmp_path)


def test_b0_report_numbers_match_hand_computed_mrrs(tmp_path, monkeypatch):
    # Seeds 0-2 rank the positive 1st, 2nd, 3rd; seeds 3-4 rank it 1st. Both regions are identical.
    import json

    from scripts import b0_report
    from src.boundary_suites import selector_digest
    from src.digests import sha256

    by_seed = {0: [3.0, 2.0, 1.0], 1: [2.0, 3.0, 1.0], 2: [1.0, 2.0, 3.0], 3: [3.0, 2.0, 1.0], 4: [3.0, 2.0, 1.0]}
    instances = ["f1-union12-R01-s5000", "f1-union12-T-s5003"]
    selector = {"instances": {iid: {"rows": [0, 1], "n_rows": 2, "block_rows": [0] if "R01" in iid else []}
                              for iid in instances}}
    manifest = {}
    for region in ("INP", "VLNP"):
        for seed, row in by_seed.items():
            path = tmp_path / f"{region}-{seed}.npz"
            save_cache(path, {iid: (np.arange(2), np.array([row, row])) for iid in instances})
            manifest[path.name] = {"mode": "union", "suite": "s", "region": region, "seed": seed, "commit": "abc",
                                   "code": {}, "selector": selector_digest(selector), "scores": sha256(path)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "f1" / "union12").mkdir(parents=True)
    (tmp_path / "f1" / "union12" / "s.selector.json").write_text(json.dumps(selector))
    monkeypatch.setattr(b0_report, "REPO_ROOT", tmp_path)
    b0_report.main(["--suite", "s", "--manifest", str(tmp_path / "manifest.json"), "--suites-root", str(tmp_path),
                    "--out", str(tmp_path / "report.md")])
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Block rows average only the 1 instances with block rows; `f1-union12-T-s5003` have none" in text
    rows = {line.split(" | ")[0].lstrip("| "): line.split(" | ")[1]
            for line in text.splitlines() if line.startswith("| ")}
    assert rows["INP expert (3 seeds, §2 form)"] == f"{(1 + 1 / 2 + 1 / 3) / 3:.4f}"  # mean of per-seed MRRs
    assert rows["INP 1-seed ensemble"] == "1.0000"
    assert rows["INP 3-seed ensemble"] == "0.5000"  # z-scores [0, 1.22, -1.22]: the positive ranks 2nd
    assert rows["INP 5-seed ensemble"] == "1.0000"
    assert rows["B0, equal weights"] == "0.5000"  # two identical regions: B0 is the 3-seed ensemble
    assert rows["B0's weights, seed 2 only (weights fit in-sample)"] == f"{1 / 3:.4f}"


def test_a_report_reads_one_cohort_not_two(tmp_path, monkeypatch):
    # Two cohorts can share a manifest; naming one registry must exclude the other's caches.
    import json

    from scripts import b0_report
    from src.boundary_suites import selector_digest
    from src.digests import sha256

    instances = ["f1-union12-R01-s5000", "f1-union12-T-s5003"]  # two, so leave-one-instance-out has a fold
    selector = {"instances": {iid: {"rows": [0], "n_rows": 1, "block_rows": [0]} for iid in instances}}
    manifest = {}
    for registry in ("results/experts.csv", "results/experts-clean.csv"):
        for region in ("INP", "VLNP"):
            for seed in (0, 1, 2):
                path = tmp_path / f"{registry.split('/')[-1]}-{region}-{seed}.npz"
                save_cache(path, {iid: (np.arange(1), np.array([[3.0, 2.0, 1.0]])) for iid in instances})
                manifest[path.name] = {"mode": "union", "suite": "s", "region": region, "seed": seed, "commit": "abc",
                                       "code": {}, "selector": selector_digest(selector), "scores": sha256(path),
                                       "registry": registry}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "f1" / "union12").mkdir(parents=True)
    (tmp_path / "f1" / "union12" / "s.selector.json").write_text(json.dumps(selector))
    monkeypatch.setattr(b0_report, "REPO_ROOT", tmp_path)
    read = ["--suite", "s", "--manifest", str(tmp_path / "manifest.json"),
            "--suites-root", str(tmp_path), "--out", str(tmp_path / "report.md")]
    b0_report.main([*read, "--registry", str(tmp_path / "results/experts-clean.csv")])
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert text.count("INP expert (3 seeds, §2 form)") == 1
    # The default names the other cohort, which has the same shape: still one table, not two.
    b0_report.main(read)
    assert (tmp_path / "report.md").read_text(encoding="utf-8").count("B0, equal weights") == 1


def test_b0_report_refuses_a_region_missing_a_gate_seed():
    # Dropping VLNP would quietly change B0's mixture; the report must refuse instead.
    from scripts import b0_report

    with pytest.raises(SystemExit, match=r"VLNP \(missing \[1\]\)"):
        b0_report.methods({"INP": {0: {}, 1: {}, 2: {}}, "VLNP": {0: {}, 2: {}, 3: {}}}, {"all": None})


def test_a_is_reported_beside_b0_and_never_mixed_into_it(tmp_path, monkeypatch):
    # Plan 06: A's caches get their own rows; B0 and its weights must not move when they appear.
    import json

    from scripts import b0_report
    from src.boundary_suites import selector_digest
    from src.digests import sha256

    by_seed = {0: [3.0, 2.0, 1.0], 1: [2.0, 3.0, 1.0], 2: [1.0, 2.0, 3.0], 3: [3.0, 2.0, 1.0], 4: [3.0, 2.0, 1.0]}
    instances = ["f1-union12-R01-s5000", "f1-union12-T-s5003"]
    selector = {"instances": {iid: {"rows": [0, 1], "n_rows": 2, "block_rows": [0]} for iid in instances}}

    def entry(path, region, seed, **extra):
        return {"mode": "union", "suite": "s", "region": region, "seed": seed, "commit": "abc", "code": {},
                "selector": selector_digest(selector), "scores": sha256(path), **extra}

    def report(with_a, a_code=None):
        manifest = {}
        for region in ("INP", "VLNP"):
            for seed, row in by_seed.items():
                path = tmp_path / f"{region}-{seed}.npz"
                save_cache(path, {iid: (np.arange(2), np.array([row, row])) for iid in instances})
                manifest[path.name] = entry(path, region, seed)
        if with_a:  # A's seeds rank the positive 1st, 3rd, 2nd: §2 mean 0.6111, score ensemble 0.5
            for seed, row in enumerate(([3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [2.0, 3.0, 1.0])):
                path = tmp_path / f"A-{seed}.npz"
                save_cache(path, {iid: (np.arange(2), np.array([row] * 2)) for iid in instances})
                # scored later (another commit) is fine; other scoring code is not
                manifest[path.name] = entry(path, "A", seed, system="A", commit="def", code=a_code or {})
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        b0_report.main(["--suite", "s", "--manifest", str(tmp_path / "manifest.json"), "--suites-root", str(tmp_path),
                        "--out", str(tmp_path / "report.md")])
        return {line.split(" | ")[0].lstrip("| "): line.split(" | ")[1]
                for line in (tmp_path / "report.md").read_text(encoding="utf-8").splitlines() if line.startswith("| ")}

    (tmp_path / "f1" / "union12").mkdir(parents=True)
    (tmp_path / "f1" / "union12" / "s.selector.json").write_text(json.dumps(selector))
    monkeypatch.setattr(b0_report, "REPO_ROOT", tmp_path)
    without, with_a = report(False), report(True)
    assert with_a["A (3 seeds, §2 form)"] == f"{(1 + 1 / 3 + 1 / 2) / 3:.4f}"  # mean of seed MRRs
    assert with_a["A 3-seed ensemble"] == "0.5000"  # z-scores [0, .41, -.41]: the positive ranks 2nd
    assert {k: v for k, v in with_a.items() if not k.startswith("A ")} == without  # B0 untouched
    with pytest.raises(SystemExit, match="rescore the experts"):
        report(True, a_code={"scripts/score_union.py": "changed"})
    report(True)
    path = tmp_path / "A-1.npz"  # A scored on fewer rows than the experts: refused, not compared
    save_cache(path, {iid: (np.arange(1), np.array([[3.0, 2.0, 1.0]])) for iid in instances})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest[path.name]["scores"] = sha256(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="exactly the selector's rows"):
        b0_report.main(["--suite", "s", "--manifest", str(tmp_path / "manifest.json"),
                        "--suites-root", str(tmp_path)])


def test_a_system_run_must_have_trained_on_the_union_and_selected_on_its_boundary_suite(tmp_path, monkeypatch):
    # A run without the boundary F1-val suite (digest "none") trained on its roots, so its F1-val score is exposed.
    from types import SimpleNamespace

    from scripts import score_union

    trained = {"region": "UNION12", "seed": 0, "config": {}, "graph_digest": "g", "schema_digest": "sc",
               "feature_digest": "f", "suite_digest": "val"}
    manifests = {"f1-union12-s000-x": trained, "f1-al-s000-y": {**trained, "region": "AL"},
                 "f1-union12-s000-noval": {**trained, "suite_digest": "none"},
                 "f1-union12-s000-feat": {**trained, "feature_digest": "other"},
                 "f1-union12-s000-f2graph": {**trained, "graph_digest": "g2"}}
    monkeypatch.setattr(score_union, "_read_manifest", lambda run_dir: manifests[run_dir.name])
    monkeypatch.setattr(score_union, "cohort_digest", lambda manifest: "c")
    monkeypatch.setattr(score_union, "_feature_digest", lambda table, extras: "f")
    union = {"graph": SimpleNamespace(digest=lambda: "g"), "schema": SimpleNamespace(digest=lambda: "sc"),
             "table": None, "val_suite_digest": "val"}
    row = score_union.system_rows(tmp_path, ["f1-union12-s000-x"], "UNION12", **union)[0]
    assert row["region"] == "A"
    assert row["train"] == {"dataset": "f1", "graph_digest": "g", "schema_digest": "sc",
                            "feature_digest": "f", "suite_digest": "val"}
    for run, message in (("f1-al-s000-y", "not a union system"), ("f1-union12-s000-noval", "suite_digest"),
                         ("f1-union12-s000-feat", "feature_digest"), ("f1-union12-s000-f2graph", "graph_digest")):
        with pytest.raises(SystemExit, match=message):
            score_union.system_rows(tmp_path, [run], "UNION12", **union)
