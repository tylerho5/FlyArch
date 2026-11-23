"""The Phase 2 pipeline on the synthetic paired region, one stage per test.

Each Phase 2 plan adds its stage as a new test function and leaves the others
alone: data build, training, scoring, gate and reports meet here.
"""

import contextlib
import io
import json
import re
import dataclasses
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")


def _config_args(config, inputs, runs):
    processed, suite, suites_dir = inputs
    return ["--config", str(config), "--processed-dir", str(processed), "--suite", str(suite),
            "--suites-dir", str(suites_dir), "--runs-dir", str(runs), "--seed", "0",
            "--device", "cpu", "--epochs", "1", "--hidden", "8"]


def test_config_stage_trains_a_region_from_its_config(paired_inputs, tmp_path, capsys):
    from scripts.train_m1 import main

    runs = tmp_path / "runs"
    assert main(_config_args("train/configs/al.toml", paired_inputs, runs)) == 0

    (run_dir,) = runs.iterdir()
    assert run_dir.name.startswith("f1-al-s000-")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["region"] == "AL"
    assert (manifest["config"]["layers"], manifest["config"]["sampling"]) == (2, "full")
    assert manifest["config"]["pair_batch"] == 2048


def test_config_stage_refuses_another_regions_config(paired_inputs, tmp_path, capsys):
    from scripts.train_m1 import main

    runs = tmp_path / "runs"
    assert main(_config_args("train/configs/lh.toml", paired_inputs, runs)) == 2
    assert not runs.exists()
    assert "is for LH" in capsys.readouterr().err


def test_scoring_stage_writes_rows_and_the_gate_accepts_uppercase_region(
    paired_graph_f2, paired_annotations, tmp_path, monkeypatch
):
    import json
    import torch

    from src.features import FeatureSchema, build_features
    from src.masks import apply_mask, random_mask
    from src.negatives import DEFAULT_SPEC
    from src.suites import Suite, SuiteInstance, view_negatives
    from src.gate import PreregConfig, evaluate_gate
    from train.cohorts import cohort_digest
    from train.model import M1
    from train.region_config import region_paths
    from scripts import score_run
    from scripts.scoring_common import load_result_tables

    data_root = tmp_path / "data"
    paths = region_paths("AL", "f2", data_root)
    graph = paired_graph_f2
    schema = FeatureSchema.from_f1(paired_annotations, "AL")
    table = build_features(paired_annotations, graph, schema, dataset="f2")
    graph.write(paths.processed)
    frame = pd.DataFrame(table.values, columns=table.columns)
    frame.insert(0, "root_id", graph.node_ids)
    frame.to_parquet(paths.processed / "features.parquet", index=False)
    (paths.processed / "schema.json").write_text(schema.to_json(), encoding="utf-8")
    table.coverage.to_csv(paths.processed / "coverage.csv", index=False)
    types = paired_annotations.set_index("root_id")["cell_type"].reindex(graph.node_ids)
    paths.interim.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"root_id": graph.node_ids, "cell_type": types.to_numpy()}).to_parquet(
        paths.interim / "f2_types.parquet", index=False
    )

    instance = random_mask(graph, fraction=0.05, seed=71)
    suite = Suite(
        "f2_dev", "f2", "AL", graph.digest(), graph.threshold, DEFAULT_SPEC.master_seed,
        (SuiteInstance(instance.instance_id, instance.regime, instance.seed,
                       dict(instance.params), graph.node_ids[instance.masked]),),
    )
    sample = view_negatives(apply_mask(graph, instance))
    suite_dir = paths.suites
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_path = suite_dir / "f2_dev.json"
    suite.write(suite_path)
    (suite_dir / "f2_dev.negatives.json").write_text(json.dumps({
        "name": "f2_dev",
        "spec": {"k": DEFAULT_SPEC.k, "alpha": DEFAULT_SPEC.alpha,
                 "master_seed": DEFAULT_SPEC.master_seed},
        "digests": {instance.instance_id: sample.digest()},
    }), encoding="utf-8")

    run_dir = tmp_path / "runs" / "toy-run"
    run_dir.mkdir(parents=True)
    config = {"hidden": 8, "layers": 2, "sibling_features": False,
              "encoder_dropout": 0.0, "decoder_dropout": 0.0}
    torch.save({"model": M1(schema, **config).state_dict()}, run_dir / "best.pt")
    manifest = {
        "run_id": run_dir.name, "region": "AL", "seed": 0,
        "schema_digest": schema.digest(), "device": "cpu", "precision": "fp32",
        "config": {**config, "epochs": 1},
        "code_digest": "code", "graph_digest": "f1graph", "feature_digest": "features",
        "suite_digest": "f1suite",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert cohort_digest(manifest)
    monkeypatch.setattr(score_run, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(score_run, "scoring_commit", lambda *a: "c" * 40)
    assert score_run.main([
        "--config", "train/configs/al.toml", "--run", str(run_dir),
        "--suite", str(suite_path), "--data-root", str(data_root),
    ]) == 0
    result_dir = tmp_path / "repo" / "results" / "al"
    scored, positives = load_result_tables(result_dir)
    assert scored.iloc[0].region == "AL" and scored.iloc[0].model == "gnn_a4"
    assert scored.iloc[0].cohort == cohort_digest(manifest)
    assert {"root_id", "reciprocal_rank", "both_masked"}.issubset(positives.columns)

    # The complete synthetic gate table proves row production reaches the pinned gate.
    rows = []
    for name, value in (("configuration", 0.1), ("logreg", 0.2), ("xgboost", 0.15)):
        for instance_id in ("DEV00", "DEV01"):
            for seed in range(3):
                rows.append({"dataset": "f2", "region": "AL", "split": "f2_dev",
                             "degree_setting": "degree_given", "model": name,
                             "mask_instance": instance_id, "mask_regime": "c",
                             "seed": seed, "mrr": value})
    for index in range(20):
        instance_id = f"CONF{index:02}"
        for seed in range(3):
            rows.append({"dataset": "f2", "region": "AL", "split": "f2_confirm",
                         "degree_setting": "degree_given", "model": "gnn_a4",
                         "mask_instance": instance_id, "mask_regime": "c",
                         "seed": seed, "mrr": 0.2 * 2**0.5})
            for name, value in (("configuration", 0.1), ("logreg", 0.2), ("xgboost", 0.15)):
                rows.append({"dataset": "f2", "region": "AL", "split": "f2_confirm",
                             "degree_setting": "degree_given", "model": name,
                             "mask_instance": instance_id, "mask_regime": "c",
                             "seed": seed, "mrr": value})
    assert evaluate_gate(pd.DataFrame(rows), PreregConfig(delta=0.1, region="al")).passed


def test_real_baseline_cli_rows_reach_the_pinned_gate_contract(
    cx_graph_f1, cx_graph, cx_annotations, tmp_path, monkeypatch,
):
    import torch

    from conftest import _synthetic_centroids, _write_committed_negatives
    from scripts import run_baselines, score_run
    from scripts.scoring_common import (
        _validate_score_positive_pair,
        file_digest,
        load_result_tables,
        load_scoring_inputs,
    )
    from src.features import FeatureSchema, build_features
    from src.gate import PreregConfig, evaluate_gate
    from src.suites import build_suite, suite_negatives
    from train import baselines, train as training
    from train.cohorts import cohort_digest
    from train.model import M1
    from train.region_config import load_region_config, region_paths

    paired_graph_f1, paired_graph_f2, paired_annotations = cx_graph_f1, cx_graph, cx_annotations
    region = "CX"
    data_root = tmp_path / "data"
    schema = FeatureSchema.from_f1(paired_annotations, region)
    types = paired_annotations.set_index("root_id")["cell_type"]
    sides = paired_annotations.set_index("root_id")["side"]
    centroids = _synthetic_centroids(paired_graph_f1.n_nodes)

    def write_dataset(graph, dataset):
        paths = region_paths(region, dataset, data_root)
        graph.write(paths.processed)
        table = build_features(paired_annotations, graph, schema, dataset=dataset)
        frame = pd.DataFrame(table.values, columns=table.columns)
        frame.insert(0, "root_id", graph.node_ids)
        frame.to_parquet(paths.processed / "features.parquet", index=False)
        (paths.processed / "schema.json").write_text(schema.to_json(), encoding="utf-8")
        table.coverage.to_csv(paths.processed / "coverage.csv", index=False)
        paths.interim.mkdir(parents=True)
        aligned_types = types.reindex(graph.node_ids)
        aligned_sides = sides.reindex(graph.node_ids)
        pd.DataFrame({"root_id": graph.node_ids, "cell_type": aligned_types}).to_parquet(
            paths.interim / f"{dataset}_types.parquet", index=False
        )
        pd.DataFrame({"root_id": graph.node_ids, "side": aligned_sides}).to_parquet(
            paths.interim / f"{dataset}_sides.parquet", index=False
        )
        return paths, table, aligned_types

    f1_paths, f1_table, f1_types = write_dataset(paired_graph_f1, "f1")
    f2_paths, _, f2_types = write_dataset(paired_graph_f2, "f2")
    f1_suite = build_suite(
        paired_graph_f1, name="f1_val", types=f1_types, centroids=centroids
    )
    full_dev_suite = build_suite(
        paired_graph_f2, name="f2_dev", types=f2_types, centroids=centroids
    )
    dev_suite = dataclasses.replace(full_dev_suite, instances=full_dev_suite.instances[:2])
    confirm_suite = build_suite(
        paired_graph_f2, name="f2_confirm", types=f2_types, centroids=centroids
    )
    for paths, suite, graph in (
        (f1_paths, f1_suite, paired_graph_f1),
        (f2_paths, dev_suite, paired_graph_f2),
        (f2_paths, confirm_suite, paired_graph_f2),
    ):
        suite.write(paths.suites / f"{suite.name}.json")
        _write_committed_negatives(data_root / "suites", suite, suite_negatives(graph, suite))

    config = load_region_config("train/configs/cx.toml")
    current_f1 = load_scoring_inputs(config, f1_paths.suites / "f1_val.json", data_root)
    current_f1_graph, current_f1_table, current_f1_schema, _, _, current_f1_suite, _ = current_f1

    runs_dir = data_root / "runs"
    manifests = []
    for seed in range(3):
        run_id = f"f1-cx-s{seed:03d}-gate-seam"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        config = {"hidden": 8, "layers": 2, "sibling_features": False,
                  "encoder_dropout": 0.0, "decoder_dropout": 0.0}
        manifest = {
            "run_id": run_id, "region": region, "seed": seed, "device": "cpu",
            "precision": "fp32", "config": {**config, "epochs": 1},
            "code_digest": "training-code", "graph_digest": current_f1_graph.digest(),
            "feature_digest": training._feature_digest(current_f1_table, extras=()),
            "schema_digest": current_f1_schema.digest(),
            "suite_digest": current_f1_suite.digest(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        torch.save({"model": M1(current_f1_schema, **config).state_dict()}, run_dir / "best.pt")
        manifests.append(manifest)
    cohort = cohort_digest(manifests[0])

    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    delta_path = tmp_path / "results" / "cx" / "delta.json"
    delta_path.parent.mkdir(parents=True)
    tag = f"delta-cx-{cohort}"
    delta_path.write_text(json.dumps({
        "region": region, "s": 0.0, "delta": 0.0,
        "run_ids": [manifest["run_id"] for manifest in manifests],
        "cohort": cohort, "tag": tag,
    }) + "\n", encoding="utf-8", newline="\n")
    git("add", "results/cx/delta.json")
    git("commit", "-qm", "record synthetic calibration fixture")
    git("tag", tag)
    (tmp_path / "descendant.txt").write_text("later fixture commit\n", encoding="utf-8")
    git("add", "descendant.txt")
    git("commit", "-qm", "advance fixture history")

    monkeypatch.setattr(run_baselines, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(score_run, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(baselines, "_TARGET_ENCODING_POSITIVES", 5)
    monkeypatch.setattr(baselines.LogRegBaseline, "grid", ({"C": 1.0},))
    monkeypatch.setattr(baselines.XGBoostBaseline, "grid", ({
        "max_depth": 3, "n_estimators": 5, "learning_rate": 0.1, "subsample": 1.0,
    },))
    common = ["--config", "train/configs/cx.toml", "--runs-dir", str(runs_dir),
              "--cohort", cohort, "--data-root", str(data_root)]
    assert run_baselines.main([
        *common, "--suite", str(f2_paths.suites / "f2_dev.json"),
    ]) == 0
    assert run_baselines.main([
        *common, "--suite", str(f2_paths.suites / "f2_confirm.json"),
        "--milestone", "gate-seam",
    ]) == 0
    expert_args = ["--config", "train/configs/cx.toml", "--data-root", str(data_root),
                   "--suite", str(f2_paths.suites / "f2_confirm.json"),
                   "--milestone", "gate-seam"]
    for manifest in manifests:
        expert_args.extend(["--run", str(runs_dir / manifest["run_id"])])
    assert score_run.main(expert_args) == 0

    rows, positives = load_result_tables(tmp_path / "results" / "cx")
    _validate_score_positive_pair(rows, positives)
    tag_commit = subprocess.run(
        ["git", "rev-parse", f"{tag}^{{commit}}"], cwd=tmp_path, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    for frame in (rows, positives):
        confirm = frame[frame.split == "f2_confirm"]
        assert set(confirm.calibration_tag_commit) == {tag_commit}
        assert set(confirm.calibration_delta_digest) == {file_digest(delta_path)}
    verdict = evaluate_gate(rows, PreregConfig(delta=0.0, region=region))
    assert verdict.n_instances == 20


def test_side_stage_freezes_and_scores_hemisphere_holdouts(
    paired_graph_f2, paired_annotations, tmp_path,
):
    from scripts.build_suites import main
    from scripts.prepare_suite_inputs import build_sides
    from src.evaluate import (
        CommittedNegatives,
        build_scoring_task,
        constant_scorer,
        score_instance,
        strata_frame,
    )
    from src.masks import apply_mask
    from src.suites import Suite, suite_negatives

    graph = paired_graph_f2
    graph.write(tmp_path / "graph")
    sides_path = tmp_path / "f2_sides.parquet"
    sides = build_sides(paired_annotations, graph)
    sides.to_parquet(sides_path, index=False)
    assert main([
        "--graph", str(tmp_path / "graph"), "--name", "f2_side",
        "--sides", str(sides_path), "--out", str(tmp_path / "suites"),
    ]) == 0
    suite = Suite.read(tmp_path / "suites/f2_side.json")
    committed = CommittedNegatives.read(tmp_path / "suites/f2_side.negatives.json")
    samples = suite_negatives(graph, suite)
    results = [score_instance(build_scoring_task(
        apply_mask(graph, mask), samples[mask.instance_id], suite=suite, negatives=committed,
        types=paired_annotations.set_index("root_id")["cell_type"],
        soma_sides=sides.set_index("root_id")["side"], trained_types={"PA1", "PA2", "PA3", "PS1"},
    ), constant_scorer(0.5)) for mask in suite.resolve(graph)]
    report = strata_frame(results, model="gnn_a4", seed=0, regime="c")
    assert set(report.split) == {"f2_side"}
    assert {"contra", "untrained"} <= set(report.loc[report.axis == "siblings", "stratum"])


def test_side_stage_cannot_be_a_training_gap_regime():
    from train.train import TrainConfig

    with pytest.raises(ValueError, match="gap_regimes"):
        TrainConfig(epochs=1, gap_regimes=("H",))


# A toy AL in both datasets' raw formats: eight types, twenty copies per side
# (one hemisphere alone leaves K=100 negative partners), and one untyped
# neuron. F2 rewires the same neurons and adds what the BANC reader must drop
# or report.
_TOY_TYPES = tuple(f"T{i}" for i in range(1, 9))
_F1_BASE, _F2_BASE = 720575940000000000, 720575941000000000


# The toy OL adds a lamina type (L1, wired in LA and ME in F1) and a medulla
# type (M9) whose only partner is L1.
_OL_TYPES = (*_TOY_TYPES, "L1", "M9")


def _toy_neurons(types: tuple[str, ...] = _TOY_TYPES) -> pd.DataFrame:
    rows = [(t, side) for t in types for side in ("left", "right") for _ in range(20)]
    rows.append((None, "left"))
    frame = pd.DataFrame(rows, columns=["cell_type", "side"])
    rng = np.random.default_rng(3)
    frame["x"], frame["y"], frame["z"] = rng.normal(size=(3, len(frame))) * 1000
    frame["f1_id"] = _F1_BASE + np.arange(len(frame))
    frame["f2_id"] = _F2_BASE + np.arange(len(frame))
    return frame


def _toy_edges(neurons: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    types = neurons["cell_type"].fillna("").to_numpy()
    same = (types[:, None] == types[None, :]) & (types[:, None] != "")
    connect = np.random.default_rng(seed).random(same.shape) < np.where(same, 0.3, 0.04)
    np.fill_diagonal(connect, False)
    return np.nonzero(connect)


def _ol_edges(neurons: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``_toy_edges`` with L1 wired to itself in LA and M9 wired to L1 alone; also the LA mask."""
    types = neurons["cell_type"].fillna("").to_numpy()
    lamina, partner = types == "L1", types == "M9"
    chance = np.where(types[:, None] == types[None, :], 0.3, 0.04)
    only_lamina = (lamina[:, None] & partner[None, :]) | (partner[:, None] & lamina[None, :])
    chance = np.where(partner[:, None] | partner[None, :], np.where(only_lamina, 0.3, 0.0), chance)
    connect = np.random.default_rng(seed).random(chance.shape) < chance
    np.fill_diagonal(connect, False)
    pre, post = np.nonzero(connect)
    return pre, post, lamina[pre] & lamina[post]


def _toy_config(path: Path, code: str, f2_threshold: int | None) -> Path:
    """``code``'s committed config at ``path``, with ``f2_threshold`` set or removed."""
    text = re.sub(r"^f2_threshold = \d+\n", "", Path(f"train/configs/{code}.toml").read_text(), flags=re.M)
    if f2_threshold is not None:
        text = text.replace("f1_threshold = 5", f"f1_threshold = 5\nf2_threshold = {f2_threshold}")
    path.write_text(text)
    return path


def _write_toy_raw(raw: Path, region: str = "AL") -> pd.DataFrame:
    """The toy region's raw files; OL's F1 has LA rows and its BANC files have none."""
    ol = region == "OL"
    neurons = _toy_neurons(_OL_TYPES if ol else _TOY_TYPES)
    n = len(neurons)
    side = neurons["side"].to_numpy()
    xyz = neurons[["x", "y", "z"]].to_numpy()
    flow = np.where(np.arange(n) % 3, "intrinsic", "afferent")
    nt = np.where(np.arange(n) % 2, "acetylcholine", "gaba")

    base = "ME" if ol else "AL"
    if ol:
        pre, post, in_lamina = _ol_edges(neurons, 11)
    else:
        pre, post = _toy_edges(neurons, 11)
        in_lamina = np.zeros(len(pre), dtype=bool)
    f1 = neurons["f1_id"].to_numpy()
    label = np.char.add(np.where(in_lamina, "LA", base), np.where(side[pre] == "right", "_R", "_L"))
    pd.DataFrame({"pre_pt_root_id": f1[pre], "post_pt_root_id": f1[post],
                  "neuropil": label, "syn_count": 9}).to_feather(raw / "proofread_connections_783.feather")
    synapses = {"pre_pt_root_id": f1[pre], "post_pt_root_id": f1[post], "neuropil": label}
    for role, ends in (("pre", pre), ("post", post)):
        for j, axis in enumerate("xyz"):
            synapses[f"{role}_pt_position_{axis}"] = xyz[ends, j]
        counts = pd.DataFrame({"root_id": f1[ends], "neuropil": label}).value_counts()
        counts.rename("count").reset_index().to_feather(raw / f"per_neuron_neuropil_count_{role}_783.feather")
    pd.DataFrame(synapses).to_feather(raw / "flywire_synapses_783.feather")
    pd.DataFrame({
        "root_id": f1, "super_class": "central",
        "cell_class": np.where(np.arange(n) % 2, "ALPN", "ALLN"), "cell_type": neurons["cell_type"],
        "side": side, "flow": flow, "top_nt": nt,
        "supertype": np.where(np.arange(n) % 3, "st_a", "st_b"),
        "ito_lee_hemilineage": np.where(np.arange(n) % 4, "hemi_1", "hemi_2"),
    }).to_csv(raw / "Supplemental_file1_neuron_annotations.tsv", sep="\t", index=False)

    banc = raw / "banc"
    banc.mkdir()
    # Two extra segments wired to neuron 0, one never proofread and one glia,
    # then an autapse and a row outside the registry.
    stray, glia = _F2_BASE + n, _F2_BASE + n + 1
    if ol:
        pre, post, in_lamina = _ol_edges(neurons, 12)
        pre, post = pre[~in_lamina], post[~in_lamina]
    else:
        pre, post = _toy_edges(neurons, 12)
    f2 = neurons["f2_id"].to_numpy()
    sources = np.concatenate([f2[pre], [stray, f2[0], glia, f2[1], f2[2]]])
    targets = np.concatenate([f2[post], [f2[0], stray, f2[0], f2[1], f2[3]]])
    pd.DataFrame({
        "pre_root_888": sources.astype(str), "post_root_888": targets.astype(str),
        "neuropil": [base] * (len(sources) - 1) + ["ANm"],
        "side": np.concatenate([side[pre], ["left"] * 5]),
        "count": np.full(len(sources), 9, dtype="int32"),
    }).to_feather(banc / "banc_888_edgelist_neuropil_v2.feather")
    # BANC's axes: F1 (x, y, z) = (-X, -Z, +Y).
    banc_xyz = {"x": -xyz[:, 0], "y": xyz[:, 2], "z": -xyz[:, 1]}
    synapses = {"pre_root_id": f2[pre].astype(str), "post_root_id": f2[post].astype(str),
                "neuropil": base, "side": side[pre]}
    for role, ends in (("pre", pre), ("post", post)):
        for axis in "xyz":
            synapses[f"{role}_{axis}"] = banc_xyz[axis][ends]
    pd.DataFrame(synapses).to_parquet(banc / "banc_888_synapses_v2_enriched.parquet")
    pd.concat([
        pd.DataFrame({"root_id": f2[ends].astype(str), "side": side[pre], "prepost": role})
        for role, ends in (("pre", pre), ("post", post))
    ]).value_counts().rename("count").reset_index().assign(neuropil=base).to_parquet(
        banc / "banc_888_synapses_per_neuron_neuropil_side_v2.parquet"
    )
    cell_class = np.where(np.arange(n) % 2, "antennal_lobe_projection_neuron", "antennal_lobe_local_neuron")
    cell_class[5] = "ventral_nerve_cord_dorsal"
    match = f1.astype(str).astype(object)
    match[7] = None
    # BANC's own names differ from F1's, except neuron 7, unmatched but named
    # by its F1 type as OL's columnar types are, and neuron 40, a T2 matched
    # to its twin but named T1.
    names = [f"banc_{t}" for t in neurons["cell_type"]]
    names[7] = names[40] = neurons.loc[7, "cell_type"]
    if ol:
        # Neuron 40 is matched to a T2 but BANC calls it L1; neuron 9 is
        # unmatched and called L1, so its own name types it.
        names[40] = names[9] = "L1"
        match[9] = None
    ids = np.concatenate([f2, [stray, glia]])
    pd.DataFrame({
        "banc_888_id": ids.astype(str), "root_id": (ids + 7).astype(str),
        "proofread": ["TRUE"] * n + ["FALSE", "TRUE"], "roughly_proofread": "FALSE",
        "status": [None] * n + [None, "GLIA,NOT_A_NEURON"],
        "super_class": ["central_brain_intrinsic"] * n + [None, "glia"],
        "cell_class": [*cell_class, None, None],
        "cell_type": names + [None, None],
        "side": [*side, None, None], "flow": [*flow, None, None], "fafb_match": [*match, None, None],
        "neurotransmitter_predicted": [*nt, None, None],
    }).to_feather(banc / "banc_888_meta.feather")
    return neurons


@pytest.fixture(scope="module")
def toy_build(tmp_path_factory):
    """The toy AL built for F1, then F2 at ``f2_threshold = 5``, with the F2 build's output."""
    from scripts.build_region import main

    root = tmp_path_factory.mktemp("build") / "data"
    (root / "raw").mkdir(parents=True)
    neurons = _write_toy_raw(root / "raw")
    config = root.parent / "al.toml"
    _toy_config(config, "al", 5)
    assert main(["--config", str(config), "--dataset", "f1", "--data-root", str(root)]) == 0
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main(["--config", str(config), "--dataset", "f2", "--data-root", str(root),
                     "--steps", "extract", "features", "suite-inputs", "suites", "--report"]) == 0
    return root, neurons, out.getvalue()


def test_build_stage_f1_build_trains_from_the_config(toy_build, tmp_path, monkeypatch):
    from scripts.train_m1 import main

    root, _, _ = toy_build
    monkeypatch.setenv("FLYARCH_DATA_ROOT", str(root))
    runs = tmp_path / "runs"
    assert main(["--config", "train/configs/al.toml", "--runs-dir", str(runs), "--seed", "0",
                 "--device", "cpu", "--epochs", "1", "--hidden", "8"]) == 0
    (run_dir,) = runs.iterdir()
    assert json.loads((run_dir / "manifest.json").read_text())["region"] == "AL"


def test_build_stage_extras_step_is_optional_and_writes_f1_extras(tmp_path):
    from scripts.build_region import main

    root = tmp_path / "data"
    (root / "raw").mkdir(parents=True)
    _write_toy_raw(root / "raw")
    config = tmp_path / "al.toml"
    config.write_text(Path("train/configs/al.toml").read_text())
    base = ["--config", str(config), "--dataset", "f1", "--data-root", str(root)]
    extras = root / "processed/f1/al/extras.npz"

    assert main([*base, "--steps", "extract", "features"]) == 0
    assert not extras.exists()
    assert main(base) == 0
    assert extras.is_file()


def test_build_stage_extras_step_skips_the_datasets_it_cannot_align(toy_build, capsys):
    from scripts.build_region import main

    root, _, _ = toy_build
    assert main(["--config", str(root.parent / "al.toml"), "--dataset", "f2",
                 "--data-root", str(root), "--steps", "extras"]) == 0

    assert "extras skipped" in capsys.readouterr().out
    assert not (root / "processed/f2/al/extras.npz").exists()


def test_build_stage_f2_suites_validate_against_their_graph(toy_build):
    from src.regions import RegionGraph
    from src.suites import Suite, suite_negatives

    root, neurons, output = toy_build
    graph = RegionGraph.read(root / "processed/f2/al")
    assert (graph.dataset, graph.threshold) == ("f2", 5)
    # The stray and glia segments, the autapse and the ANm row never reach the graph.
    assert set(graph.node_ids) <= set(neurons["f2_id"])
    assert "--- f2/AL pre-build report ---" in output

    t_draws = set()
    for name in ("f2_dev", "f2_confirm", "f2_side"):
        suite = Suite.read(root / f"suites/f2/al/{name}.json")
        suite.validate(graph)
        committed = json.loads((root / f"suites/f2/al/{name}.negatives.json").read_text())["digests"]
        assert {iid: s.digest() for iid, s in suite_negatives(graph, suite).items()} == committed
        drawn = {inst.instance_id: inst.params["cell_type"] for inst in suite.instances if inst.regime == "T"}
        assert set(drawn.values()) <= set(_TOY_TYPES) and bool(drawn) == (name != "f2_side")
        if drawn:
            # Unmatched neuron 7 takes its own name, so a T1 draw masks it; neuron
            # 40 keeps its match's T2 and stays visible.
            record = json.loads((root / f"suites/f2/al/{name}.same_name.json").read_text())
            assert {iid: row["same_name_unmasked"] for iid, row in record["instances"].items()} == {
                iid: int(cell_type == "T1") for iid, cell_type in drawn.items()
            }
            for inst in suite.instances:
                if inst.regime == "T" and inst.params["cell_type"] == "T1":
                    assert neurons.loc[7, "f2_id"] in set(inst.masked_root_ids.tolist())
            t_draws.update(drawn.values())
    assert "T1" in t_draws
    assert "\tsame_name_unmasked" in output
    types = pd.read_parquet(root / "interim/f2/al/f2_types.parquet").set_index("root_id")["cell_type"]
    assert types[neurons.loc[7, "f2_id"]] == "T1" and types[neurons.loc[40, "f2_id"]] == "T2"
    assert set(types.dropna()) <= set(_TOY_TYPES)


def test_build_stage_f2_refuses_without_a_threshold_and_calibrate_runs(toy_build, tmp_path, capsys):
    from scripts.build_region import main

    root, _, _ = toy_build
    config = _toy_config(tmp_path / "al_unfrozen.toml", "al", None)
    with pytest.raises(SystemExit) as refused:
        main(["--config", str(config), "--dataset", "f2",
              "--data-root", str(tmp_path), "--raw-dir", str(tmp_path / "absent")])
    assert refused.value.code == 2
    assert "no f2_threshold" in capsys.readouterr().err
    assert not (tmp_path / "processed").exists()

    assert main(["--config", str(config), "--dataset", "f2", "--data-root", str(root),
                 "--calibrate"]) == 0
    assert "chosen f2_threshold for AL: 9" in capsys.readouterr().out


def test_build_stage_f2_annotations_encode_like_their_f1_twins(toy_build):
    root, neurons, output = toy_build
    f1 = pd.read_parquet(root / "processed/f1/al/features.parquet").set_index("root_id")
    f2 = pd.read_parquet(root / "processed/f2/al/features.parquet").set_index("root_id")
    shared = neurons[neurons["f1_id"].isin(f1.index) & neurons["f2_id"].isin(f2.index)]
    assert len(shared) == len(f1) == len(f2)
    twins = shared.drop(index=[5, 7])
    one_hot = [c for c in f1.columns if c.startswith(("superclass_", "class_", "side_", "flow_", "nt_"))]
    np.testing.assert_array_equal(f1.loc[twins["f1_id"], one_hot].to_numpy(),
                                  f2.loc[twins["f2_id"], one_hot].to_numpy())
    geometry = ["centroid_x", "centroid_y", "centroid_z"]
    np.testing.assert_allclose(f1.loc[shared["f1_id"], geometry].to_numpy(),
                               f2.loc[shared["f2_id"], geometry].to_numpy(), atol=1e-9)
    assert f2.loc[neurons.loc[5, "f2_id"], "class_OOV"] == 1
    assert ("f2 class: 1 graph node(s) carry a value with no F1 counterpart: "
            "{'ventral_nerve_cord_dorsal': 1}") in output


@pytest.fixture(scope="module")
def toy_ol_build(tmp_path_factory):
    """The toy OL (the config's lamina rule, f2_threshold 5) built for F1, then F2, with the F2 output."""
    from scripts.build_region import main

    root = tmp_path_factory.mktemp("build_ol") / "data"
    (root / "raw").mkdir(parents=True)
    neurons = _write_toy_raw(root / "raw", "OL")
    config = root.parent / "ol.toml"
    _toy_config(config, "ol", 5)
    assert main(["--config", str(config), "--dataset", "f1", "--data-root", str(root)]) == 0
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main(["--config", str(config), "--dataset", "f2", "--data-root", str(root)]) == 0
    return root, config, neurons, out.getvalue()


def _ids(neurons, dataset, cell_type):
    return set(neurons.loc[neurons["cell_type"] == cell_type, f"{dataset}_id"].tolist())


def _f2_lamina(neurons):
    # Neuron 9 is unmatched and named L1, so its own name types it L1; neuron 40
    # is matched to a T2 and only BANC names it L1, so it stays eligible.
    return _ids(neurons, "f2", "L1") | {int(neurons.loc[9, "f2_id"])}


def test_ol_builds_keep_every_neuron_and_suites_never_mask_the_lamina_types(toy_ol_build):
    import hashlib

    from src.regions import RegionGraph
    from src.suites import Suite, suite_negatives

    root, _, neurons, output = toy_ol_build
    folder = root / "processed/f2/ol"
    record = json.loads((folder / "excluded_types.json").read_text())
    assert (record["rule"], record["min_fraction"], record["types"]) == ("lamina_fraction", 0.01, ["L1"])
    names = ["Supplemental_file1_neuron_annotations.tsv", "per_neuron_neuropil_count_pre_783.feather",
             "per_neuron_neuropil_count_post_783.feather", "banc/banc_888_meta.feather"]
    assert record["inputs"] == {
        Path(name).name: hashlib.sha256((root / "raw" / name).read_bytes()).hexdigest() for name in names
    }
    f1_record = json.loads((root / "processed/f1/ol/excluded_types.json").read_text())
    assert f1_record["types"] == ["L1"] and "banc_888_meta.feather" not in f1_record["inputs"]

    graph = RegionGraph.read(folder)
    lamina, nodes = _f2_lamina(neurons), set(graph.node_ids.tolist())
    assert lamina <= nodes and int(neurons.loc[40, "f2_id"]) in nodes
    assert "1 F1 type(s) are never masked: L1" in output
    assert f"{len(lamina)} of {graph.n_nodes} graph neuron(s) carry one" in output
    assert "1 F2 neuron(s) carry an excluded BANC name but crosswalk to a kept type" in output

    for name in ("f2_dev", "f2_confirm", "f2_side"):
        suite = Suite.read(root / f"suites/f2/ol/{name}.json")
        suite.validate(graph)
        assert suite.mask_exclude_types == ("L1",)
        assert not {r for i in suite.instances for r in i.masked_root_ids.tolist()} & lamina
        committed = json.loads((root / f"suites/f2/ol/{name}.negatives.json").read_text())["digests"]
        assert {iid: s.digest() for iid, s in suite_negatives(graph, suite).items()} == committed
    f1_val = Suite.read(root / "suites/f1/ol/f1_val.json")
    assert f1_val.mask_exclude_types == ("L1",)
    assert not {r for i in f1_val.instances for r in i.masked_root_ids.tolist()} & _ids(neurons, "f1", "L1")


def test_a_region_without_mask_exclude_writes_no_exclusion_and_keeps_its_suite_json(toy_build):
    root, _, _ = toy_build
    assert not (root / "processed/f2/al/excluded_types.json").exists()
    for path in (root / "suites/f2/al").glob("f2_*.json"):
        assert "mask_exclude_types" not in path.read_text()


def test_ol_calibrate_leaves_the_excluded_types_out_of_the_comparison(toy_ol_build, monkeypatch, capsys):
    from scripts import build_region

    root, config, _, _ = toy_ol_build
    seen = {}
    real = build_region.calibrate_threshold

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(build_region, "calibrate_threshold", spy)
    assert build_region.main(["--config", str(config), "--dataset", "f2", "--data-root", str(root),
                              "--calibrate"]) == 0
    assert seen["exclude_types"] == ("L1",)
    assert "chosen f2_threshold for OL" in capsys.readouterr().out


def test_suites_refuse_a_missing_or_stale_exclusion_record(toy_ol_build, tmp_path):
    import shutil

    from scripts.build_region import main

    built, config, _, _ = toy_ol_build
    root = tmp_path / "data"
    shutil.copytree(built, root)
    args = ["--config", str(config), "--dataset", "f2", "--data-root", str(root), "--steps", "suites"]
    path = root / "processed/f2/ol/excluded_types.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "min_fraction": 1.0}))
    with pytest.raises(SystemExit, match="rerun with --steps extract"):
        main(args)
    path.unlink()
    with pytest.raises(SystemExit, match="is missing"):
        main(args)


def test_f3_annotations_encode_like_their_f1_twins():
    from src.features import FeatureSchema, build_features
    from src.harmonize import normalize_f3_annotations, type_crosswalk
    from src.regions import build_region_graph

    f1 = pd.DataFrame({
        "root_id": [1, 2], "super_class": ["central", "sensory"], "cell_class": ["ALPN", "olfactory"],
        "cell_type": ["T1", "T2"], "side": ["left", "right"], "flow": ["intrinsic", "afferent"],
        "top_nt": ["acetylcholine", "gaba"],
    })
    # Twins of the F1 rows: the sensory neuron has no soma side, so its root side stands in.
    raw = pd.DataFrame({
        "bodyId": [11, 12], "status": "Traced", "superclass": ["cb_intrinsic", "cb_sensory"],
        "class": ["ALPN", "olfactory"], "flywireType": ["T1", None], "type": ["T1", "T2"],
        "somaSide": ["L", None], "rootSide": [None, "R"], "dimorphism": [None, "male-specific"],
        "predicted_nt": ["acetylcholine", "gaba"],
    })
    annotations, unmapped = normalize_f3_annotations(raw)
    assert unmapped == {}
    annotations["cell_type"] = type_crosswalk(
        annotations, dataset="f3", f1_types=f1["cell_type"]).to_numpy()

    schema = FeatureSchema.from_f1(f1, "AL")
    def encode(table, ids, dataset):
        edges = pd.DataFrame({"pre_pt_root_id": ids, "post_pt_root_id": ids[::-1],
                              "neuropil": "AL_L", "syn_count": 9})
        graph = build_region_graph(edges, region="AL", dataset=dataset, threshold=5)
        built = build_features(table, graph, schema, dataset=dataset)
        return pd.DataFrame(built.values, index=graph.node_ids, columns=built.columns)

    f1_features = encode(f1, [1, 2], "f1")
    f3_features = encode(annotations, [11, 12], "f3")
    types = [c for c in f1_features.columns if c.startswith(("superclass_", "class_", "side_", "flow_", "nt_", "cell_type"))]
    np.testing.assert_array_equal(f1_features.loc[[1, 2], types].to_numpy(),
                                  f3_features.loc[[11, 12], types].to_numpy())

def test_fetch_stage_translates_rois_into_the_region_graphs():
    from scripts.fetch_f3 import EXCLUDED_ROIS, JANELIA_ROI_TO_LABEL, SPLIT_BY_HEMISPHERE, translate_rois
    from src.regions import EXCLUDED_BASE, all_labels, build_region_graph

    roi_conn = pd.DataFrame({
        "bodyId_pre": [1, 1, 2, 3, 1, 1],
        "bodyId_post": [2, 2, 3, 1, 2, 2],
        "roi": ["FB", "FB", "EB", "AL(R)", "ANm", "IB"],
        "weight": [3, 4, 9, 6, 8, 5],
        "side": [None, None, None, None, None, "L"],
    })
    out = translate_rois(roi_conn)
    assert sorted(out["neuropil"]) == ["AL_R", "EB", "FB", "IB_L"]
    assert out.set_index("neuropil").loc["FB", "syn_count"] == 7

    # The old fetch_f3_cx.py mapping, built from the input and not from translate_rois.
    cx_rows = roi_conn[roi_conn["roi"].isin(["FB", "EB", "PB", "NO"])]
    old = pd.DataFrame({"pre_pt_root_id": cx_rows["bodyId_pre"], "post_pt_root_id": cx_rows["bodyId_post"],
                        "neuropil": cx_rows["roi"], "syn_count": cx_rows["weight"]})
    cx = build_region_graph(out, region="CX", dataset="f3", threshold=5)
    assert cx.digest() == build_region_graph(old, region="CX", dataset="f3", threshold=5).digest()
    assert cx.weights.tolist() == [7, 9]
    assert build_region_graph(out, region="AL", dataset="f3", threshold=5).weights.tolist() == [6]

    with pytest.raises(ValueError, match="NEW_ROI"):
        translate_rois(roi_conn.assign(roi="NEW_ROI"))
    with pytest.raises(ValueError, match="side"):
        translate_rois(roi_conn.assign(roi="IB", side=None))

    registry = set(all_labels()) - EXCLUDED_BASE - {"IB_L", "IB_R"}
    assert registry <= set(JANELIA_ROI_TO_LABEL.values())
    primaries = (
        [f"{b}({s})" for b in "AB AL AME AOTU ATL AVLP BU CA CAN CRE EPA FLA GOR ICL IPS LA LAL LH LO LOP ME "
         "PED PLP PVLP SCL SIP SLP SMP SPS VES WED a'L aL b'L bL gL".split() for s in "LR"]
        + "EB FB GNG IB NO PB PRW SAD <unspecified> CentralBrain-unspecified CV CV-unspecified "
          "VNC-unspecified Optic-unspecified(L) Optic-unspecified(R) NotPrimary".split()
    )
    known = set(JANELIA_ROI_TO_LABEL) | EXCLUDED_ROIS | SPLIT_BY_HEMISPHERE
    assert not set(primaries) - known

_F3_BASE = 720575942000000000


@pytest.fixture(scope="module")
def toy_f3_build(toy_build):
    """The toy AL's F1 build plus F3 raw files as ``fetch_f3.py`` writes them, built at ``f3_threshold = 5``."""
    from scripts.build_region import main

    root, neurons, _ = toy_build
    f3_dir = root / "raw" / "f3"
    f3_dir.mkdir()
    f3 = _F3_BASE + np.arange(len(neurons))
    side, xyz = neurons["side"].to_numpy(), neurons[["x", "y", "z"]].to_numpy()
    pre, post = _toy_edges(neurons, 13)
    label = np.where(side[pre] == "right", "AL_R", "AL_L")
    pd.DataFrame({"pre_pt_root_id": f3[pre], "post_pt_root_id": f3[post],
                  "neuropil": label, "syn_count": 9}).to_parquet(f3_dir / "connections.parquet")
    ends = np.concatenate([pre, post])
    pd.DataFrame({"root_id": f3[ends], "x": xyz[ends, 0].astype("int32"), "y": xyz[ends, 1].astype("int32"),
                  "z": xyz[ends, 2].astype("int32"), "neuropil": np.concatenate([label, label])}
                 ).to_parquet(f3_dir / "synapses.parquet")
    pd.concat([pd.DataFrame({"root_id": f3[e], "neuropil": label, "kind": kind})
               for kind, e in (("pre", pre), ("post", post))]).value_counts().rename("count").reset_index(
    ).to_parquet(f3_dir / "innervation.parquet")
    n = len(neurons)
    pd.DataFrame({
        "bodyId": f3, "status": "Traced", "superclass": np.where(np.arange(n) % 3, "cb_intrinsic", "cb_sensory"),
        "class": np.where(np.arange(n) % 2, "ALPN", "ALLN"), "flywireType": neurons["cell_type"],
        "type": neurons["cell_type"].map(lambda t: f"f3_{t}"), "somaSide": np.where(side == "right", "R", "L"),
        "rootSide": None, "dimorphism": None, "predicted_nt": np.where(np.arange(n) % 2, "acetylcholine", "gaba"),
    }).to_parquet(f3_dir / "annotations.parquet")
    config = root.parent / "al_f3.toml"
    config.write_text(re.sub(r"(?m)^f3_threshold = \d+$", "f3_threshold = 5", Path("train/configs/al.toml").read_text()))
    assert main(["--config", str(config), "--dataset", "f3", "--data-root", str(root)]) == 0
    return root, config, neurons


def test_build_stage_f3_eval_validates_and_scores_under_rung_x(toy_f3_build):
    from scripts.scoring_common import load_scoring_inputs, rung_for_suite
    from src.suites import suite_negatives
    from train.model import M1
    from train.region_config import load_region_config, region_paths
    from train.scoring import score_expert

    root, config_path, neurons = toy_f3_build
    config = load_region_config(config_path)
    paths = region_paths("AL", "f3", root)
    assert sorted(p.name for p in paths.suites.glob("f3_*.json") if p.name.count(".") == 1) == ["f3_eval.json"]
    assert (paths.interim / "f3_sides.parquet").exists()

    graph, table, schema, types, _, suite, negatives = load_scoring_inputs(config, paths.suites / "f3_eval.json", root)
    assert (graph.dataset, len(suite.instances)) == ("f3", 15)
    assert all(i.instance_id.startswith("f3-al-") for i in suite.instances)
    assert set(types.dropna()) <= set(_TOY_TYPES) and any(i.regime == "T" for i in suite.instances)
    assert {iid: s.digest() for iid, s in suite_negatives(graph, suite).items()} == negatives.digests

    rung = rung_for_suite("f3_eval")
    rows, positives = score_expert(M1(schema, hidden=8), graph, table, suite, types=types,
                                   negatives=negatives, rung=rung)
    assert (rung, set(rows.mask_regime), set(rows.dataset)) == ("x", {"x"}, {"f3"})
    assert len(rows) == 15 and len(positives)


def test_build_stage_f3_refuses_without_a_threshold_before_reading_data(tmp_path, capsys):
    from scripts.build_region import main

    config = tmp_path / "al_no_f3.toml"
    config.write_text(re.sub(r"(?m)^f3_threshold = \d+\n", "", Path("train/configs/al.toml").read_text()))
    for flags in ([], ["--steps", "extract"]):
        with pytest.raises(SystemExit) as refused:
            main(["--config", str(config), "--dataset", "f3", "--data-root", str(tmp_path),
                  "--raw-dir", str(tmp_path / "absent"), *flags])
        assert refused.value.code == 2
        assert "no f3_threshold" in capsys.readouterr().err
    assert not (tmp_path / "processed").exists()


def test_reencode_maps_by_name_scores_and_matches_the_home_ablation(paired_graph_f2, paired_annotations):
    from src.features import FeatureSchema, build_features
    from src.masks import apply_mask, random_mask
    from src.negatives import DEFAULT_SPEC
    from src.evaluate import CommittedNegatives
    from src.suites import Suite, SuiteInstance, view_negatives
    from src.transfer import ablate_innervation, masked_oov_fraction, reencode_features
    from train.model import M1
    from train.scoring import score_expert

    graph_i = paired_graph_f2
    graph_j = dataclasses.replace(graph_i, region="LH")
    ann_j = paired_annotations.copy()
    ann_j["super_class"] = np.where(ann_j.index % 2, "sensory", "central")
    ann_i = paired_annotations[paired_annotations.cell_type != "PA4"]
    schema_i = FeatureSchema.from_f1(ann_i, "AL")
    schema_j = FeatureSchema.from_f1(ann_j, "LH")
    assert "PA4" in schema_j.type_vocab and "PA4" not in schema_i.type_vocab
    t_j = build_features(ann_j, graph_j, schema_j, dataset="f2")
    t_i = build_features(paired_annotations, graph_i, schema_i, dataset="f2")

    moved = reencode_features(t_j, schema_j, schema_i)
    assert moved.columns == schema_i.column_names
    ct = moved.values[:, moved.columns.index("cell_type")]
    types = ann_j.set_index("root_id")["cell_type"].reindex(graph_j.node_ids).to_numpy()
    assert (ct[types == "PA4"] == 0).all() and (ct[types == "PA1"] == schema_i.type_vocab["PA1"]).all()
    oov = moved.values[:, moved.columns.index("superclass_OOV")]
    sensory = ann_j.set_index("root_id")["super_class"].reindex(graph_j.node_ids).to_numpy() == "sensory"
    assert (oov == sensory).all()

    np.testing.assert_array_equal(reencode_features(t_i, schema_i, schema_i).values,
                                  ablate_innervation(t_i, schema_i).values)

    instance = random_mask(graph_i, fraction=0.05, seed=71)
    suite = Suite("f2_dev", "f2", "AL", graph_i.digest(), graph_i.threshold, DEFAULT_SPEC.master_seed,
                  (SuiteInstance(instance.instance_id, instance.regime, instance.seed,
                                 dict(instance.params), graph_i.node_ids[instance.masked]),))
    negatives = CommittedNegatives("f2_dev", DEFAULT_SPEC,
                                   {instance.instance_id: view_negatives(apply_mask(graph_i, instance)).digest()})
    model = M1(schema_i, hidden=8, layers=2, sibling_features=False, encoder_dropout=0.0, decoder_dropout=0.0)
    node_types = ann_j.set_index("root_id")["cell_type"].reindex(graph_i.node_ids)
    rows, _ = score_expert(model, graph_i, moved, suite, types=node_types, negatives=negatives, rung="c")
    assert np.isfinite(rows["mrr"]).all()

    model.set_trained_types(np.array([v for n, v in schema_i.type_vocab.items() if n != "PA1"]))
    masked = ann_j.set_index("root_id")["cell_type"].reindex(suite.instances[0].masked_root_ids)
    expected = masked.isin(["PA4", "PA1"]).mean()
    assert masked_oov_fraction(model, moved, graph_i, suite) == {instance.instance_id: expected}


def test_load_result_tables_refuses_transfer_rows(tmp_path):
    from scripts.scoring_common import load_result_tables

    scored = tmp_path / "scored"
    scored.mkdir()
    pd.DataFrame({"expert_region": ["AL"]}).to_parquet(scored / "a.scores.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(scored / "a.positives.parquet")
    with pytest.raises(ValueError, match="transfer-matrix"):
        load_result_tables(tmp_path)


def test_transfer_home_rows_equal_the_experts_own_scoring(
    paired_graph_f2, paired_annotations, tmp_path, monkeypatch
):
    import torch

    from src.features import FeatureSchema, build_features
    from src.gate import REQUIRED_COLUMNS
    from src.masks import apply_mask, random_mask
    from src.negatives import DEFAULT_SPEC
    from src.suites import Suite, SuiteInstance, view_negatives
    from train.cohorts import cohort_digest
    from train.model import M1
    from train.region_config import region_paths
    from scripts import score_run, transfer_matrix
    from scripts.scoring_common import load_result_tables

    data_root = tmp_path / "data"
    paths = region_paths("AL", "f2", data_root)
    graph = paired_graph_f2
    schema = FeatureSchema.from_f1(paired_annotations, "AL")
    table = build_features(paired_annotations, graph, schema, dataset="f2")
    graph.write(paths.processed)
    frame = pd.DataFrame(table.values, columns=table.columns)
    frame.insert(0, "root_id", graph.node_ids)
    frame.to_parquet(paths.processed / "features.parquet", index=False)
    (paths.processed / "schema.json").write_text(schema.to_json(), encoding="utf-8")
    table.coverage.to_csv(paths.processed / "coverage.csv", index=False)
    f1_processed = region_paths("AL", "f1", data_root).processed
    f1_processed.mkdir(parents=True)
    (f1_processed / "schema.json").write_text(schema.to_json(), encoding="utf-8")
    types = paired_annotations.set_index("root_id")["cell_type"].reindex(graph.node_ids)
    paths.interim.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"root_id": graph.node_ids, "cell_type": types.to_numpy()}).to_parquet(
        paths.interim / "f2_types.parquet", index=False
    )
    instance = random_mask(graph, fraction=0.05, seed=71)
    suite = Suite(
        "f2_dev", "f2", "AL", graph.digest(), graph.threshold, DEFAULT_SPEC.master_seed,
        (SuiteInstance(instance.instance_id, instance.regime, instance.seed,
                       dict(instance.params), graph.node_ids[instance.masked]),),
    )
    sample = view_negatives(apply_mask(graph, instance))
    paths.suites.mkdir(parents=True, exist_ok=True)
    suite_path = paths.suites / "f2_dev.json"
    suite.write(suite_path)
    (paths.suites / "f2_dev.negatives.json").write_text(json.dumps({
        "name": "f2_dev",
        "spec": {"k": DEFAULT_SPEC.k, "alpha": DEFAULT_SPEC.alpha,
                 "master_seed": DEFAULT_SPEC.master_seed},
        "digests": {instance.instance_id: sample.digest()},
    }), encoding="utf-8")

    config = {"hidden": 8, "layers": 2, "sibling_features": False,
              "encoder_dropout": 0.0, "decoder_dropout": 0.0}
    monkeypatch.setattr(score_run, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(score_run, "scoring_commit", lambda *a: "c" * 40)
    registry = []
    for seed in (0, 1, 2):
        run_dir = tmp_path / "runs" / f"toy-run-{seed}"
        run_dir.mkdir(parents=True)
        torch.manual_seed(seed)
        torch.save({"model": M1(schema, **config).state_dict()}, run_dir / "best.pt")
        manifest = {
            "run_id": run_dir.name, "region": "AL", "seed": seed,
            "schema_digest": schema.digest(), "device": "cpu", "precision": "fp32",
            "config": {**config, "epochs": 1},
            "code_digest": "code", "graph_digest": "f1graph", "feature_digest": "features",
            "suite_digest": "f1suite",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        registry.append({"region": "AL", "gate_seed": "true", "run_id": run_dir.name,
                         "seed": seed, "cohort": cohort_digest(manifest)})
        assert score_run.main([
            "--config", "train/configs/al.toml", "--run", str(run_dir),
            "--suite", str(suite_path), "--data-root", str(data_root),
        ]) == 0
    registry_path = tmp_path / "experts.csv"
    pd.DataFrame(registry).to_csv(registry_path, index=False)

    # The champion baseline needs stored baseline rows this toy region does not have.
    monkeypatch.setattr(transfer_matrix, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(transfer_matrix, "scoring_commit", lambda *a: "c" * 40)
    monkeypatch.setattr(transfer_matrix, "champion_baseline_rows", lambda *a: pd.DataFrame())
    out = tmp_path / "transfer" / "f2_dev.csv"
    assert transfer_matrix.main([
        "--registry", str(registry_path), "--regions", "AL", "--split", "f2_dev",
        "--out", str(out), "--data-root", str(data_root), "--runs-dir", str(tmp_path / "runs"),
    ]) == 0
    result = pd.read_csv(out)
    assert set(result.variant) == {"home", "ablated", "ablated_geo"}
    assert set(result.model) == {"gnn_a4@AL"}
    assert result.masked_oov_frac.between(0, 1).all()
    columns = [c for c in REQUIRED_COLUMNS if c != "model"] + ["mrr", "n_positives", "run_id", "cohort"]
    key = ["run_id", "mask_instance", "degree_setting"]
    home = result[result.variant == "home"][columns].sort_values(key).reset_index(drop=True)
    stored, _ = load_result_tables(tmp_path / "repo" / "results" / "al")
    pd.testing.assert_frame_equal(
        home, stored[columns].sort_values(key).reset_index(drop=True), check_dtype=False
    )

    # A stale baseline row fails before any expert is scored.
    def stale(*a):
        raise ValueError("stale baseline")

    monkeypatch.setattr(transfer_matrix, "champion_baseline_rows", stale)
    monkeypatch.setattr(transfer_matrix, "score_expert", stale)
    assert transfer_matrix.main([
        "--registry", str(registry_path), "--regions", "AL", "--split", "f2_dev",
        "--out", str(tmp_path / "again.csv"), "--data-root", str(data_root),
        "--runs-dir", str(tmp_path / "runs"),
    ]) == 2


# (expert, target, variant) -> mean MRR; LH has ablated 0 and home == baseline.
_TRANSFER_CELLS = {
    ("AL", "AL", "home"): .4, ("AL", "AL", "ablated"): .2, ("AL", "AL", "ablated_geo"): .1,
    ("AL", "LH", "transfer"): .15, ("AL", "LH", "transfer_geo"): .05,
    ("LH", "LH", "home"): .3, ("LH", "LH", "ablated"): 0.0, ("LH", "LH", "ablated_geo"): .12,
    ("LH", "AL", "transfer"): .24, ("LH", "AL", "transfer_geo"): .08,
    ("AL", "AL", "baseline"): .2, ("LH", "LH", "baseline"): .3,
}


def _transfer_rows(drop_seed=False):
    rows = []
    for (i, j, variant), v in _TRANSFER_CELLS.items():
        for inst, di in (("m1", -.01), ("m2", .01)):
            for seed, ds in enumerate((-.03, 0, .03)):
                row = dict(expert_region=i, region=j, variant=variant, mask_instance=inst,
                           seed=seed, degree_setting="degree_given", mrr=v + di + ds,
                           masked_oov_frac=.5 if variant.startswith("transfer") else 0.,
                           split="f2_dev", dataset="f2",
                           model="logreg" if variant == "baseline" else f"gnn_a4@{i}")
                rows += [row, dict(row, degree_setting="degree_free", mrr=.99)]
    df = pd.DataFrame(rows)
    if drop_seed:
        gone = ((df.seed == 2) & (df.mask_instance == "m2") & (df.variant == "home")
                & (df.expert_region == "AL") & (df.degree_setting == "degree_given"))
        df = df[~gone]
    return df


def _matrix(rows):
    return pd.DataFrame(rows, index=["AL", "LH"], columns=["AL", "LH"], dtype=float)


def test_transfer_report_matrices_page_and_refusal(tmp_path):
    from scripts.transfer_report import compute, main

    out = compute(_transfer_rows(), "f2_dev")
    nan = np.nan
    pd.testing.assert_frame_equal(out["raw"], _matrix([[.4, .15], [.24, .3]]))
    pd.testing.assert_frame_equal(out["home_norm"], _matrix([[1, .5], [.6, 1]]))
    pd.testing.assert_frame_equal(out["ablated_norm"], _matrix([[2, nan], [1.2, nan]]))
    pd.testing.assert_frame_equal(out["base_norm"], _matrix([[1, nan], [.2, nan]]))
    pd.testing.assert_frame_equal(out["geo"], _matrix([[.1, .05], [.08, .12]]))
    assert list(out["baselines"]["mrr"]) == pytest.approx([.2, .3])
    pd.testing.assert_frame_equal(out["oov"], _matrix([[0, .5], [.5, 0]]))

    src = tmp_path / "in"
    src.mkdir()
    _transfer_rows().to_csv(src / "f2_dev.csv", index=False)
    page = tmp_path / "docs" / "transfer-matrix.md"
    assert main(["--in", str(src), "--out", str(page)]) == 0
    assert "undefined" in page.read_text(encoding="utf-8")
    assert (tmp_path / "docs" / "transfer-matrix-f2_dev.svg").read_text().startswith("<svg")

    _transfer_rows(drop_seed=True).to_csv(src / "f2_dev.csv", index=False)
    assert main(["--in", str(src), "--out", str(tmp_path / "other.md")]) == 2

    # F2-dev rows filed as the F1-val control must not be labelled as either.
    (src / "f2_dev.csv").unlink()
    _transfer_rows().to_csv(src / "f1_val.csv", index=False)
    assert main(["--in", str(src), "--out", str(tmp_path / "mislabelled.md")]) == 2
