import json
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.masks import apply_mask, random_mask
from src.negatives import DEFAULT_SPEC
from src.suites import Suite, SuiteInstance, view_negatives
from train.checkpoints import load_expert
from train.cohorts import cohort_digest
from train.model import M1
from train.scoring import score_expert

_RUN_FIELDS = {
    "seed": 0, "device": "cpu", "precision": "fp32", "graph_digest": "g",
    "feature_digest": "f", "suite_digest": "u",
}


@pytest.mark.parametrize("structural", [
    {},
    {"layer_norm": True},
    {"layer_norm": True, "residual": True, "branch_norm": True},
])
def test_load_expert_rebuilds_sibling_dropout_model_and_rejects_wrong_schema(
    tmp_path, cx_schema, structural
):
    schema = cx_schema
    config = {
        "hidden": 8, "layers": 2, "sibling_features": True,
        "encoder_dropout": 0.2, "decoder_dropout": 0.3, **structural,
    }
    original = M1(schema, **config)
    original.set_trained_types(np.ones(len(schema.type_vocab) + 1, dtype=bool))
    run = tmp_path / "run"
    run.mkdir()
    torch.save({"model": original.state_dict()}, run / "best.pt")
    (run / "manifest.json").write_text(json.dumps({
        **_RUN_FIELDS, "run_id": run.name, "schema_digest": schema.digest(),
        "config": {**config, "epochs": 1},
    }))

    loaded = load_expert(run, schema)
    assert loaded.sibling_features
    assert loaded.encoder.dropout.p == pytest.approx(0.2)
    assert loaded.decoder.dropout == pytest.approx(0.3)
    assert loaded.encoder.residual == structural.get("residual", False)
    assert loaded.encoder.branch_norm == structural.get("branch_norm", False)
    assert (loaded.encoder.norms is not None) == structural.get("layer_norm", False)
    assert all(torch.equal(a, b) for a, b in zip(
        original.state_dict().values(), loaded.state_dict().values(), strict=True
    ))
    with pytest.raises(ValueError, match="schema digest"):
        load_expert(run, dataclasses.replace(schema, class_vocab=(*schema.class_vocab, "OTHER")))


def test_load_expert_uses_legacy_plain_model_defaults(tmp_path, cx_schema):
    schema = cx_schema
    original = M1(schema, hidden=8, layers=2, sibling_features=False)
    run = tmp_path / "legacy"
    run.mkdir()
    torch.save({"model": original.state_dict()}, run / "best.pt")
    (run / "manifest.json").write_text(json.dumps({
        **_RUN_FIELDS, "run_id": run.name, "schema_digest": schema.digest(),
        "config": {"epochs": 1, "hidden": 8},
    }))
    loaded = load_expert(run, schema)
    assert not loaded.sibling_features
    assert isinstance(loaded.encoder.dropout, torch.nn.Identity)
    assert loaded.decoder.dropout == 0


def test_load_expert_wraps_corrupt_checkpoint_and_non_object_manifest(tmp_path, cx_schema):
    run = tmp_path / "bad"
    run.mkdir()
    (run / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        load_expert(run, cx_schema)
    (run / "manifest.json").write_text(json.dumps({
        **_RUN_FIELDS, "run_id": run.name, "schema_digest": cx_schema.digest(),
        "config": {"epochs": 1, "hidden": 8},
    }))
    (run / "best.pt").write_bytes(b"broken checkpoint")
    with pytest.raises(ValueError, match="best.pt is unreadable"):
        load_expert(run, cx_schema)


def test_cohort_digest_uses_plan07_scientific_inputs_only():
    manifest = {
        "config": {"epochs": 10, "hidden": 16}, "code_digest": "code",
        "graph_digest": "graph", "feature_digest": "features",
        "schema_digest": "schema", "suite_digest": "suite",
        "seed": 1, "run_id": "run-a",
    }
    changed_run = {**manifest, "seed": 2, "run_id": "run-b"}
    assert cohort_digest(manifest) == cohort_digest(changed_run)
    assert cohort_digest(manifest) != cohort_digest({**manifest, "suite_digest": "other"})


def test_f1_val_check_uses_f2_dev_champion_and_selection_instances_only():
    from scripts.check_f1_val import compare_f1_val

    rows = []
    for model, dev in (("configuration", .1), ("logreg", .4), ("xgboost", .3)):
        for instance in ("D0", "D1"):
            for seed in range(3):
                rows.append({"dataset": "f2", "region": "AL", "split": "f2_dev",
                             "degree_setting": "degree_given", "model": model,
                             "mask_instance": instance, "mask_regime": "c", "seed": seed,
                             "mrr": dev})
    for model, value in (("configuration", .1), ("logreg", .3), ("xgboost", .5)):
        for instance, instance_value in (("R0", value), ("R1", value), ("R2", value),
                                         ("R3", value), ("T0", 1.0 if model == "logreg" else value)):
            for seed in range(3):
                rows.append({"dataset": "f1", "region": "AL", "split": "f1_val",
                             "degree_setting": "degree_given", "model": model,
                             "mask_instance": instance, "mask_regime": "b", "seed": seed,
                             "mrr": instance_value})
    history = {"val_selection_instances": ["R0", "R1", "R2", "R3"]}
    result = compare_f1_val(
        pd.DataFrame(rows), history,
        {**{iid: .4 for iid in history["val_selection_instances"]}, "T0": 0.01}, "AL",
    )
    assert result["champion"] == "logreg"
    assert result["baseline_mrr"] == pytest.approx(.3)
    assert result["expert_mrr"] == pytest.approx(.4)
    assert result["passed"]


def test_score_expert_emits_gate_and_positive_rows(
    cx_graph, cx_features, cx_schema, cx_types
):
    instance = random_mask(cx_graph, fraction=0.05, seed=71)
    suite = Suite(
        name="f2_dev", dataset="f2", region=cx_graph.region,
        graph_digest=cx_graph.digest(), threshold=cx_graph.threshold,
        master_seed=DEFAULT_SPEC.master_seed,
        instances=(SuiteInstance(
            instance.instance_id, instance.regime, instance.seed, dict(instance.params),
            cx_graph.node_ids[instance.masked],
        ),),
    )
    sample = view_negatives(apply_mask(cx_graph, instance))
    negatives = __import__("src.evaluate", fromlist=["CommittedNegatives"]).CommittedNegatives(
        "f2_dev", DEFAULT_SPEC, {instance.instance_id: sample.digest()}
    )
    model = M1(cx_schema, hidden=8)
    rows, positives = score_expert(
        model, cx_graph, cx_features, suite, types=cx_types, negatives=negatives, rung="c"
    )
    assert list(rows.region.unique()) == ["CX"]
    assert rows.iloc[0].model == "gnn_a4"
    assert {"run_id", "cohort", "mask_instance"}.issubset(rows.columns)
    assert {"root_id", "reciprocal_rank", "both_masked", "run_id", "cohort"}.issubset(
        positives.columns
    )


def test_trained_sibling_dropout_checkpoint_reloads_and_reproduces_f1_history(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path,
):
    from conftest import _write_committed_negatives
    from src.evaluate import CommittedNegatives
    from src.suites import build_suite, suite_negatives
    from train.train import TrainConfig, fit

    suite = build_suite(cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids)
    samples = suite_negatives(cx_graph_f1, suite)
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, samples)
    config = TrainConfig(
        epochs=2, hidden=8, patience=2, run_dir=tmp_path / "runs",
        suites_dir=suites_dir, sibling_features=True, encoder_dropout=0.1,
        decoder_dropout=0.1,
    )
    result = fit(
        cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema
    )
    run_dir = Path(config.run_dir) / result.manifest["run_id"]
    expert = load_expert(run_dir, cx_schema)
    negatives = CommittedNegatives.read(
        suites_dir / "f1" / "cx" / "f1_val.negatives.json"
    )
    rows, _ = score_expert(
        expert, cx_graph_f1, cx_features_f1, suite, types=cx_types,
        negatives=negatives, rung="b", run_id=run_dir.name,
    )
    best_epoch = int(result.manifest["best_epoch"])
    history_index = result.history["val_mrr_epoch"].index(best_epoch)
    expected = {
        instance: values[history_index]
        for instance, values in result.history["val_mrr_by_instance"].items()
    }
    assert expert.sibling_features
    assert expert.encoder.dropout.p == pytest.approx(0.1)
    assert expert.decoder.dropout == pytest.approx(0.1)
    assert rows.set_index("mask_instance").mrr.to_dict() == pytest.approx(expected)


def test_f1_check_selects_one_complete_cohort_from_shared_runs(
    tmp_path, monkeypatch, capsys,
):
    from scripts import check_f1_val

    runs_dir = tmp_path / "runs"
    rows = []
    cohorts = {}
    run_ids = {}
    for label, hidden in (("first", 8), ("second", 16)):
        manifests = []
        for seed in range(3):
            run_id = f"f1-al-s{seed:03d}-{label}"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True)
            manifest = {
                "run_id": run_id, "region": "AL", "seed": seed, "best_epoch": 0,

                "device": "cpu", "precision": "fp32", "config": {"epochs": 1, "hidden": hidden}, "code_digest": "code",
                "graph_digest": "graph", "feature_digest": "features",
                "schema_digest": "schema", "suite_digest": "suite",
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            torch.save({"history": {
                "val_mrr_epoch": [0], "val_mrr_by_instance": {"I0": [0.9]},
                "val_selection_instances": ["I0"],
            }}, run_dir / "best.pt")
            manifests.append(manifest)
        cohort = cohort_digest(manifests[0])
        cohorts[label] = cohort
        run_ids[label] = {manifest["run_id"] for manifest in manifests}
        for model, dev_mrr in (("configuration", 0.1), ("logreg", 0.3), ("xgboost", 0.2)):
            for instance in ("D0", "D1"):
                for seed in range(3):
                    rows.append({
                        "dataset": "f2", "region": "AL", "split": "f2_dev",
                        "degree_setting": "degree_given", "model": model,
                        "mask_instance": instance, "mask_regime": "c", "seed": seed,
                        "mrr": dev_mrr, "cohort": cohort,
                    })
            for seed in range(3):
                rows.append({
                    "dataset": "f1", "region": "AL", "split": "f1_val",
                    "degree_setting": "degree_given", "model": model,
                    "mask_instance": "I0", "mask_regime": "b", "seed": seed,
                    "mrr": 0.1, "cohort": cohort,
                })
    bad = runs_dir / "malformed"
    bad.mkdir()
    (bad / "manifest.json").write_text("[]", encoding="utf-8")
    results = tmp_path / "results" / "al"
    results.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(results / "scores.csv", index=False)
    monkeypatch.setattr(check_f1_val, "REPO_ROOT", tmp_path)
    from conftest import _bypass_result_verification

    _bypass_result_verification(monkeypatch, pd.DataFrame(rows))

    common = ["--config", "train/configs/al.toml", "--runs-dir", str(runs_dir)]
    assert check_f1_val.main([*common, "--cohort", cohorts["second"]]) == 0
    output = capsys.readouterr().out
    assert all(run_id in output for run_id in run_ids["second"])
    assert all(run_id not in output for run_id in run_ids["first"])
    assert check_f1_val.main(common) == 2
    assert "expected one cohort" in capsys.readouterr().err


def _f1_check_cli_fixture(tmp_path, expert_values):
    runs_dir = tmp_path / "runs"
    manifests = []
    for seed, value in enumerate(expert_values):
        run_id = f"f1-al-s{seed:03d}-aggregate"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": run_id, "region": "AL", "seed": seed, "best_epoch": 0,

            "device": "cpu", "precision": "fp32", "config": {"epochs": 1, "hidden": 8}, "code_digest": "code",
            "graph_digest": "graph", "feature_digest": "features",
            "schema_digest": "schema", "suite_digest": "suite",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        torch.save({"history": None if value is None else {
            "val_mrr_epoch": [0], "val_mrr_by_instance": {"I0": [value]},
            "val_selection_instances": ["I0"],
        }}, run_dir / "best.pt")
        manifests.append(manifest)
    cohort = cohort_digest(manifests[0])
    rows = []
    for model, dev_mrr in (("configuration", 0.1), ("logreg", 0.3), ("xgboost", 0.2)):
        for instance in ("D0", "D1"):
            for seed in range(3):
                rows.append({"dataset": "f2", "region": "AL", "split": "f2_dev",
                             "degree_setting": "degree_given", "model": model,
                             "mask_instance": instance, "mask_regime": "c", "seed": seed,
                             "mrr": dev_mrr, "cohort": cohort})
        for seed in range(3):
            rows.append({"dataset": "f1", "region": "AL", "split": "f1_val",
                         "degree_setting": "degree_given", "model": model,
                         "mask_instance": "I0", "mask_regime": "b", "seed": seed,
                         "mrr": 0.4, "cohort": cohort})
    results = tmp_path / "results" / "al"
    results.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(results / "scores.csv", index=False)
    return runs_dir, cohort


@pytest.mark.parametrize("base, offset, accepted", [
    (0.6, 1 / (13 * 14 * 565), True),  # a rank-13 flip on a 565-positive instance: the bound's edge
    (0.6, 1 / (12 * 13 * 565), False),  # a top-12 flip; also refused only because rtol=0
])
def test_f1_val_rows_may_differ_from_training_history_by_a_low_rank_flip(tmp_path, base, offset, accepted):
    from scripts import scoring_common

    runs_dir, cohort = _f1_check_cli_fixture(tmp_path, (base,))
    run_dir = runs_dir / "f1-al-s000-aggregate"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = pd.DataFrame([{
        "model": "gnn_a4", "seed": 0, "split": "f1_val", "mask_instance": "I0", "mrr": base + offset,
        "run_id": run_dir.name, "checkpoint_digest": scoring_common.file_digest(run_dir / "best.pt"),
    }])
    if accepted:
        scoring_common.validate_expert_provenance(rows, [(run_dir, manifest)], cohort)
    else:
        with pytest.raises(ValueError, match="do not match best.pt history"):
            scoring_common.validate_expert_provenance(rows, [(run_dir, manifest)], cohort)


def test_f1_check_uses_three_seed_expert_aggregate(tmp_path, monkeypatch, capsys):
    from scripts import check_f1_val

    runs_dir, cohort = _f1_check_cli_fixture(tmp_path, (0.6, 0.6, 0.2))
    monkeypatch.setattr(check_f1_val, "REPO_ROOT", tmp_path)
    _bypass_f1_result_validation(check_f1_val, monkeypatch, tmp_path)
    assert check_f1_val.main([
        "--config", "train/configs/al.toml", "--runs-dir", str(runs_dir),
        "--cohort", cohort,
    ]) == 0
    output = capsys.readouterr().out
    assert "cohort expert mean=0.466667" in output
    assert "check=PASS" in output


def test_f1_check_ignores_scored_calibration_only_seeds(tmp_path, monkeypatch, capsys):
    from scripts import check_f1_val, scoring_common

    validate_expert = scoring_common.validate_expert_provenance
    values = (0.6, 0.6, 0.2)
    runs_dir, cohort = _f1_check_cli_fixture(tmp_path, values)
    scores = tmp_path / "results" / "al" / "scores.csv"
    rows = pd.read_csv(scores)
    expert = []
    for seed in range(5):
        run_dir = runs_dir / f"f1-al-s{seed:03d}-aggregate"
        expert.append({
            "dataset": "f1", "region": "AL", "split": "f1_val",
            "degree_setting": "degree_given", "model": "gnn_a4", "mask_instance": "I0",
            "mask_regime": "b", "seed": seed, "mrr": values[seed] if seed < 3 else 0.5,
            "cohort": cohort, "run_id": run_dir.name,
            "checkpoint_digest": scoring_common.file_digest(run_dir / "best.pt")
            if seed < 3 else "calibration",
        })
    pd.concat([rows, pd.DataFrame(expert)], ignore_index=True).to_csv(scores, index=False)
    monkeypatch.setattr(check_f1_val, "REPO_ROOT", tmp_path)
    _bypass_f1_result_validation(check_f1_val, monkeypatch, tmp_path)
    monkeypatch.setattr(scoring_common, "validate_expert_provenance", validate_expert)
    assert check_f1_val.main([
        "--config", "train/configs/al.toml", "--runs-dir", str(runs_dir),
        "--cohort", cohort,
    ]) == 0, capsys.readouterr().err


def test_f1_check_returns_exit_2_for_null_history(tmp_path, monkeypatch, capsys):
    from scripts import check_f1_val

    runs_dir, cohort = _f1_check_cli_fixture(tmp_path, (None, 0.6, 0.6))
    monkeypatch.setattr(check_f1_val, "REPO_ROOT", tmp_path)
    _bypass_f1_result_validation(check_f1_val, monkeypatch, tmp_path)
    assert check_f1_val.main([
        "--config", "train/configs/al.toml", "--runs-dir", str(runs_dir),
        "--cohort", cohort,
    ]) == 2
    assert "no per-instance F1-val history" in capsys.readouterr().err


def _bypass_f1_result_validation(module, monkeypatch, tmp_path):
    from conftest import _bypass_result_verification

    _bypass_result_verification(
        monkeypatch, pd.read_csv(tmp_path / "results" / "al" / "scores.csv")
    )


def test_f1_check_refuses_scores_without_paired_positive_ranks(tmp_path, monkeypatch, capsys):
    from scripts import check_f1_val

    runs_dir, cohort = _f1_check_cli_fixture(tmp_path, (0.6, 0.6, 0.6))
    parts = tmp_path / "results" / "al" / "scored"
    parts.mkdir()
    pd.read_csv(tmp_path / "results" / "al" / "scores.csv").to_parquet(
        parts / "call.scores.parquet", index=False
    )
    monkeypatch.setattr(check_f1_val, "REPO_ROOT", tmp_path)
    assert check_f1_val.main([
        "--config", "train/configs/al.toml", "--runs-dir", str(runs_dir),
        "--cohort", cohort,
    ]) == 2
    assert "stored scoring results are unreadable" in capsys.readouterr().err
