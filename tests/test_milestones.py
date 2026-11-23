"""The deterministic milestone export over completed M1 run directories."""

import csv
import json
import shutil
import subprocess

import pytest

torch = pytest.importorskip("torch")

from src.run_control import RunStopped
from src.negatives import DEFAULT_SPEC
from src.suites import build_suite, suite_negatives
from train import milestones, train as training
from train.milestones import export_milestones
from train.milestones import EXPERT_COLUMNS, export_experts
from train.train import TrainConfig, fit, validation_mrr

#: The committed CSV's columns, in write order.
COLUMNS = [
    "run_id",
    "seed",
    "graph_digest",
    "feature_digest",
    "schema_digest",
    "suite_digest",
    "config_json",
    "best_epoch",
    "best_val_mrr",
    "selection_instances",
    "completion",
    "device",
    "precision",
]


@pytest.fixture(scope="module")
def committed_suite(tmp_path_factory, cx_graph_f1, cx_types, cx_centroids):
    """One committed F1-val suite and negative digests, shared by this module."""
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path_factory.mktemp("suites")
    path = suites_dir / "f1" / "cx" / "f1_val.negatives.json"
    path.parent.mkdir(parents=True)
    negatives = suite_negatives(cx_graph_f1, suite, spec=DEFAULT_SPEC)
    path.write_text(
        json.dumps(
            {
                "name": suite.name,
                "spec": {
                    "k": DEFAULT_SPEC.k,
                    "alpha": DEFAULT_SPEC.alpha,
                    "master_seed": DEFAULT_SPEC.master_seed,
                },
                "digests": {key: sample.digest() for key, sample in negatives.items()},
            }
        )
    )
    return suite, suites_dir


@pytest.fixture
def completed_run(tmp_path, cx_graph_f1, cx_features_f1, cx_schema, committed_suite):
    """A real completed fit, returned as ``(run_dir, suite, result)``."""
    suite, suites_dir = committed_suite

    def _run(*, seed=0, epochs=3, patience=3, eval_every=1, name="runs", **overrides):
        config = TrainConfig(
            epochs=epochs,
            hidden=8,
            patience=patience,
            eval_every=eval_every,
            **overrides,
            run_dir=tmp_path / name,
            suites_dir=suites_dir,
        )
        result = fit(
            cx_graph_f1,
            cx_features_f1,
            suite,
            seed=seed,
            config=config,
            schema=cx_schema,
        )
        return tmp_path / name / result.manifest["run_id"], suite, result

    return _run


def _rows(output):
    with output.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_writes_the_run_identity_and_selection(
    completed_run, tmp_path, cx_graph_f1
):
    run_dir, suite, result = completed_run(seed=0)
    output = tmp_path / "results" / "m1-milestones.csv"

    export_milestones([run_dir], output)

    (row,) = _rows(output)
    assert list(row) == COLUMNS
    assert row["run_id"] == run_dir.name
    assert row["seed"] == "0"
    assert row["graph_digest"] == cx_graph_f1.digest()
    assert row["feature_digest"] == result.manifest["feature_digest"]
    assert row["schema_digest"] == result.manifest["schema_digest"]
    assert row["suite_digest"] == suite.digest()
    assert row["config_json"] == json.dumps(
        result.manifest["config"], sort_keys=True, separators=(",", ":")
    )
    assert row["best_epoch"] == str(result.best_epoch)
    assert row["best_val_mrr"] == repr(max(validation_mrr(result.history)[1]))
    assert row["selection_instances"] == " ".join(
        i.instance_id for i in suite.instances if i.regime != "T"
    )
    assert row["completion"] == "epoch_budget"
    assert row["device"] == "cpu"
    assert row["precision"] == "fp32"


def _registry_runs(completed_run, tmp_path, region="AL"):
    source, _, _ = completed_run(seed=0, name=f"registry-source-{region}")
    output = []
    for seed in range(5):
        run_id = f"f1-{region.lower()}-s{seed:03d}-registry{seed}"
        destination = tmp_path / run_id
        shutil.copytree(source, destination)
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["run_id"], manifest["seed"], manifest["region"] = run_id, seed, region
        manifest_path.write_text(json.dumps(manifest))
        output.append(destination)
    return output


def test_expert_registry_groups_hardware_by_region_and_marks_gate_seeds(
    completed_run, tmp_path
):
    al_runs = _registry_runs(completed_run, tmp_path, "AL")
    lh_runs = _registry_runs(completed_run, tmp_path, "LH")
    for run in al_runs:
        _set_backend(run, "cuda:0", "fp32")
    for run in lh_runs:
        _set_backend(run, "mps", "fp32")
    output = tmp_path / "experts.csv"

    export_experts([*al_runs, *lh_runs], output)

    rows = _rows(output)
    assert list(rows[0]) == list(EXPERT_COLUMNS)
    assert {row["backend"] for row in rows if row["region"] == "AL"} == {"cuda/fp32"}
    assert {row["backend"] for row in rows if row["region"] == "LH"} == {"mps/fp32"}
    assert {row["gate_seed"] for row in rows if row["seed"] in {"0", "1", "2"}} == {"true"}
    assert {row["gate_seed"] for row in rows if row["seed"] in {"3", "4"}} == {"false"}
    assert len({row["cohort"] for row in rows if row["region"] == "AL"}) == 1


def test_expert_registry_infers_region_for_legacy_cx_and_keeps_history_export(
    completed_run, tmp_path
):
    runs = _registry_runs(completed_run, tmp_path, "CX")
    for run in runs:
        path = run / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.pop("region")
        path.write_text(json.dumps(manifest))
    output = tmp_path / "experts.csv"

    export_experts(runs, output)

    assert {row["region"] for row in _rows(output)} == {"CX"}
    history = tmp_path / "m1-milestones.csv"
    export_milestones([runs[0]], history)
    assert list(_rows(history)[0]) == COLUMNS


def test_expert_registry_rejects_manifest_region_that_disagrees_with_run_id(
    completed_run, tmp_path,
):
    run_dir, _, _ = completed_run(seed=0, name="wrong-region")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["region"] = "AL"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest region.*disagrees with run ID"):
        milestones._row(run_dir)


@pytest.mark.parametrize("field", ["config", "code_digest", "graph_digest", "feature_digest",
                                    "schema_digest", "suite_digest"])
def test_expert_registry_rejects_mixed_region_cohorts(completed_run, tmp_path, field):
    runs = _registry_runs(completed_run, tmp_path)
    path = runs[-1] / "manifest.json"
    manifest = json.loads(path.read_text())
    if field == "config":
        manifest[field]["hidden"] += 1
    else:
        manifest[field] += "changed"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="cohort|scientific config"):
        export_experts(runs, tmp_path / "experts.csv")


@pytest.mark.parametrize("seeds", [[0, 1, 2, 3, 3], [0, 1, 2, 3]])
def test_expert_registry_requires_exact_distinct_calibration_seeds(
    completed_run, tmp_path, seeds
):
    runs = _registry_runs(completed_run, tmp_path)
    for run, seed in zip(runs, seeds):
        path = run / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["seed"] = seed
        path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="calibration seed|expected calibration seeds"):
        export_experts(runs[:len(seeds)], tmp_path / "experts.csv")


def test_export_marks_an_early_stopped_run_as_complete(completed_run, tmp_path):
    run_dir, _, result = completed_run(seed=0, epochs=12, patience=1, lr_schedule="constant")
    assert len(result.history["loss"]) < 12, "the fixture run must stop early"
    output = tmp_path / "m1-milestones.csv"

    export_milestones([run_dir], output)

    (row,) = _rows(output)
    assert row["completion"] == "early_stopped"
    assert row["best_epoch"] == str(result.best_epoch)


def test_export_selects_through_the_scored_epochs_of_a_strided_run(
    completed_run, committed_suite, tmp_path, monkeypatch
):
    """Epoch 3 is the second score, so indexing scores by epoch would miss it."""
    suite, _ = committed_suite
    scores = iter([0.1, 0.5, 0.2])

    def fake_scores(*args, **kwargs):
        value = next(scores)
        return {instance.instance_id: value for instance in suite.instances}

    monkeypatch.setattr(training, "_validation_scores", fake_scores)
    run_dir, _, result = completed_run(seed=0, epochs=7, patience=7, eval_every=3)
    assert result.history["val_mrr_epoch"] == [0, 3, 6]
    assert result.best_epoch == 3
    output = tmp_path / "m1-milestones.csv"

    export_milestones([run_dir], output)

    (row,) = _rows(output)
    assert row["best_epoch"] == "3"
    assert row["best_val_mrr"] == repr(0.5)


def test_export_reads_a_history_scored_every_epoch_without_an_epoch_list(
    completed_run, tmp_path
):
    """Checkpoints written before the stride carry one mean F1-val score per epoch."""
    run_dir, _, result = completed_run(seed=0)
    _, means = validation_mrr(result.history)
    for name in ("last.pt", "best.pt"):
        checkpoint = torch.load(run_dir / name, map_location="cpu", weights_only=True)
        history = checkpoint["history"]
        del history["val_mrr_epoch"], history["val_mrr_by_instance"]
        history["val_mrr"] = means
        torch.save(checkpoint, run_dir / name)
    output = tmp_path / "m1-milestones.csv"

    export_milestones([run_dir], output)

    (row,) = _rows(output)
    assert row["best_val_mrr"] == repr(max(means))
    assert row["selection_instances"] == "all"


@pytest.mark.parametrize("corrupt", [
    lambda ids: [],
    lambda ids: ["no-such-instance"],
    lambda ids: ids[0],
    lambda ids: [ids],
    lambda ids: ids + ids[:1],
], ids=["empty", "unknown", "string", "nested", "duplicate"])
def test_export_rejects_selection_instances_outside_the_history(completed_run, tmp_path, corrupt):
    run_dir, _, _ = completed_run(seed=0)
    for name in ("last.pt", "best.pt"):
        checkpoint = torch.load(run_dir / name, map_location="cpu", weights_only=True)
        history = checkpoint["history"]
        history["val_selection_instances"] = corrupt(history["val_selection_instances"])
        torch.save(checkpoint, run_dir / name)

    with pytest.raises(ValueError, match="selection instances"):
        export_milestones([run_dir], tmp_path / "m1-milestones.csv")


def test_export_names_the_instances_each_runs_score_averages(completed_run, tmp_path):
    """A history from before T was left out selects on every instance, and its row says so."""
    new_dir, suite, new = completed_run(seed=0)
    legacy_dir, _, legacy = completed_run(seed=1, epochs=1, patience=1)
    for name in ("last.pt", "best.pt"):
        checkpoint = torch.load(legacy_dir / name, map_location="cpu", weights_only=True)
        del checkpoint["history"]["val_selection_instances"]
        torch.save(checkpoint, legacy_dir / name)
    del legacy.history["val_selection_instances"]
    output = tmp_path / "m1-milestones.csv"

    export_milestones([new_dir, legacy_dir], output)

    rows = {row["run_id"]: row for row in _rows(output)}
    all_ids = " ".join(i.instance_id for i in suite.instances)
    assert rows[legacy_dir.name]["selection_instances"] == all_ids
    assert rows[new_dir.name]["selection_instances"] != all_ids
    assert rows[legacy_dir.name]["best_val_mrr"] == repr(max(validation_mrr(legacy.history)[1]))
    assert rows[new_dir.name]["best_val_mrr"] == repr(max(validation_mrr(new.history)[1]))


@pytest.mark.parametrize(
    "scored",
    [(0, 1, 2), [0, "1", 2], [0, 2, 1], [0, 1, 3]],
    ids=["tuple", "not_int", "unsorted", "past_last_epoch"],
)
def test_export_rejects_malformed_scored_epochs(completed_run, tmp_path, scored):
    run_dir, _, _ = completed_run(seed=0)
    checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    checkpoint["history"]["val_mrr_epoch"] = scored
    torch.save(checkpoint, run_dir / "last.pt")

    with pytest.raises(ValueError, match=f"{run_dir}: F1-val scored epochs"):
        export_milestones([run_dir], tmp_path / "m1-milestones.csv")


def test_export_orders_rows_by_run_id_regardless_of_argument_order(
    completed_run, tmp_path
):
    first, _, _ = completed_run(seed=0, name="runs-a")
    second, _, _ = completed_run(seed=1, name="runs-b")
    ordered = tmp_path / "ordered.csv"
    reversed_order = tmp_path / "reversed.csv"

    export_milestones([first, second], ordered)
    export_milestones([second, first], reversed_order)

    assert ordered.read_text() == reversed_order.read_text()
    assert [row["run_id"] for row in _rows(ordered)] == sorted(
        [first.name, second.name]
    )


def test_rerunning_the_export_replaces_the_output(completed_run, tmp_path):
    run_dir, _, _ = completed_run(seed=0)
    output = tmp_path / "m1-milestones.csv"

    export_milestones([run_dir], output)
    once = output.read_text()
    export_milestones([run_dir], output)

    assert output.read_text() == once
    assert [row["run_id"] for row in _rows(output)] == [run_dir.name]


def test_export_rejects_an_interrupted_run(
    tmp_path, cx_graph_f1, cx_features_f1, cx_schema, committed_suite, monkeypatch
):
    suite, suites_dir = committed_suite
    config = TrainConfig(
        epochs=5, hidden=8, patience=5, run_dir=tmp_path, suites_dir=suites_dir,
        steps_per_epoch=1,
    )
    real_loss = training._epoch_loss
    calls = {"n": 0}

    def preempted(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(training, "_epoch_loss", preempted)
    with pytest.raises(RunStopped):
        fit(
            cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema
        )
    monkeypatch.undo()

    (run_dir,) = [path for path in tmp_path.iterdir() if path.is_dir()]
    output = tmp_path / "m1-milestones.csv"
    with pytest.raises(ValueError, match="interrupted"):
        export_milestones([run_dir], output)
    assert not output.exists()


def test_export_rejects_a_manifest_whose_best_epoch_is_not_the_history_best(
    completed_run, tmp_path
):
    run_dir, _, result = completed_run(seed=0)
    scored, values = validation_mrr(result.history)
    wrong = next(epoch for epoch, value in zip(scored, values) if value < max(values))

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["best_epoch"] = wrong
    manifest_path.write_text(json.dumps(manifest))
    for name in ("last.pt", "best.pt"):
        checkpoint = torch.load(run_dir / name, map_location="cpu", weights_only=True)
        checkpoint["best_epoch"] = wrong
        torch.save(checkpoint, run_dir / name)

    with pytest.raises(ValueError, match="history's best"):
        export_milestones([run_dir], tmp_path / "m1-milestones.csv")


def test_export_rejects_a_best_checkpoint_that_disagrees_with_the_run(
    completed_run, tmp_path
):
    run_dir, _, result = completed_run(seed=0)
    epochs = result.manifest["config"]["epochs"]
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    checkpoint["best_epoch"] = (result.best_epoch + 1) % epochs
    torch.save(checkpoint, run_dir / "best.pt")

    with pytest.raises(ValueError, match="disagree"):
        export_milestones([run_dir], tmp_path / "m1-milestones.csv")


def test_export_rejects_a_directory_that_is_not_a_run(tmp_path):
    (tmp_path / "not-a-run").mkdir()

    with pytest.raises(ValueError, match="manifest"):
        export_milestones([tmp_path / "not-a-run"], tmp_path / "m1-milestones.csv")


def test_export_rejects_a_run_without_an_f1_val_history(
    tmp_path, cx_graph_f1, cx_features_f1
):
    config = TrainConfig(epochs=2, hidden=8, patience=2, run_dir=tmp_path)
    result = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)
    run_dir = tmp_path / result.manifest["run_id"]

    with pytest.raises(ValueError, match="F1-val"):
        export_milestones([run_dir], tmp_path / "m1-milestones.csv")


def _set_backend(run_dir, device, precision):
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["device"], manifest["precision"] = device, precision
    path.write_text(json.dumps(manifest))


def test_export_rejects_runs_from_different_backends(completed_run, tmp_path):
    first, _, _ = completed_run(seed=0, name="a")
    second, _, _ = completed_run(seed=1, name="b")
    _set_backend(first, "cuda:0", "fp32")
    _set_backend(second, "mps", "fp32")
    output = tmp_path / "m1-milestones.csv"

    with pytest.raises(ValueError, match="cuda/fp32, mps/fp32"):
        export_milestones([first, second], output)
    assert not output.exists()


def test_export_rejects_a_legacy_bf16_run_beside_fp32(completed_run, tmp_path):
    """CUDA runs recorded before training moved to fp32 still carry bf16."""
    first, _, _ = completed_run(seed=0, name="a")
    second, _, _ = completed_run(seed=1, name="b")
    _set_backend(first, "cuda:0", "bf16")
    _set_backend(second, "cuda:0", "fp32")
    output = tmp_path / "m1-milestones.csv"

    with pytest.raises(ValueError, match="cuda/bf16, cuda/fp32"):
        export_milestones([first, second], output)
    assert not output.exists()


def _set_fused_adamw(run_dir, fused):
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    runtime = manifest["invocations"][0]["runtime"]
    if fused is None:
        runtime.pop("fused_adamw", None)
    else:
        runtime["fused_adamw"] = fused
    path.write_text(json.dumps(manifest))


def test_export_rejects_runs_with_and_without_fused_adamw(completed_run, tmp_path):
    first, _, _ = completed_run(seed=0, name="a")
    second, _, _ = completed_run(seed=1, name="b")
    _set_backend(first, "cuda:0", "fp32")
    _set_backend(second, "cuda:0", "fp32")
    _set_fused_adamw(first, None)
    _set_fused_adamw(second, True)
    output = tmp_path / "m1-milestones.csv"

    with pytest.raises(ValueError, match="cuda/fp32, cuda/fp32/fused-adamw"):
        export_milestones([first, second], output)
    assert not output.exists()


def test_export_reads_a_run_without_the_adamw_flag_as_unfused(completed_run, tmp_path):
    first, _, _ = completed_run(seed=0, name="a")
    second, _, _ = completed_run(seed=1, name="b")
    _set_fused_adamw(first, None)
    _set_fused_adamw(second, False)
    output = tmp_path / "m1-milestones.csv"

    export_milestones([first, second], output)

    assert len(_rows(output)) == 2


def test_export_accepts_runs_on_different_cuda_ordinals(completed_run, tmp_path):
    first, _, _ = completed_run(seed=0, name="a")
    second, _, _ = completed_run(seed=1, name="b")
    _set_backend(first, "cuda:0", "fp32")
    _set_backend(second, "cuda:1", "fp32")
    output = tmp_path / "m1-milestones.csv"

    export_milestones([first, second], output)

    assert [row["device"] for row in _rows(output)] == ["cuda:0", "cuda:1"]


def test_export_rejects_a_duplicate_run_id(completed_run, tmp_path):
    run_dir, _, _ = completed_run(seed=0)

    with pytest.raises(ValueError, match="duplicate"):
        export_milestones([run_dir, run_dir], tmp_path / "m1-milestones.csv")


def test_export_never_invokes_a_subprocess(completed_run, tmp_path, monkeypatch):
    """Staging the CSV is the operator's deliberate act, not the exporter's."""
    run_dir, _, _ = completed_run(seed=0)
    output = tmp_path / "m1-milestones.csv"

    def refuse(*args, **kwargs):
        raise AssertionError("the exporter must not invoke git or any subprocess")

    monkeypatch.setattr(subprocess, "run", refuse)

    export_milestones([run_dir], output)

    assert output.is_file()


def test_export_loads_checkpoints_without_unpickling_objects(
    completed_run, tmp_path, monkeypatch
):
    """A checkpoint written by another process must not execute code on export."""
    run_dir, _, _ = completed_run(seed=0)
    real_load = torch.load
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(milestones.torch, "load", spy)

    export_milestones([run_dir], tmp_path / "m1-milestones.csv")

    assert seen == [True, True]


def test_cli_exports_the_requested_output(completed_run, tmp_path, capsys):
    from scripts.export_m1_milestones import main

    run_dir, _, _ = completed_run(seed=0)
    output = tmp_path / "results" / "m1-milestones.csv"

    assert main([str(run_dir), "--out", str(output)]) == 0

    assert [row["run_id"] for row in _rows(output)] == [run_dir.name]
    assert str(output) in capsys.readouterr().out


def test_cli_reports_a_rejected_run(tmp_path, capsys):
    from scripts.export_m1_milestones import main

    bad = tmp_path / "not-a-run"
    bad.mkdir()
    output = tmp_path / "m1-milestones.csv"

    assert main([str(bad), "--out", str(output)]) == 2

    assert "milestone export failed:" in capsys.readouterr().err
    assert not output.exists()
