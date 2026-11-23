import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import write_training_inputs as _write_inputs
from src.suites import Suite


def _args(processed, suite, suites_dir, runs):
    return ["--processed-dir", str(processed), "--suite", str(suite),
            "--suites-dir", str(suites_dir), "--runs-dir", str(runs),
            "--seed", "3", "--epochs", "1", "--device", "cpu",
            "--hidden", "8", "--patience", "1"]


def test_cli_runs_and_resumes_same_f1_cx_training_input(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids, capsys
):
    from scripts.train_m1 import main

    processed, suite, suites_dir = _write_inputs(
        tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids
    )
    runs = tmp_path / "runs"
    args = _args(processed, suite, suites_dir, runs)

    assert main(args) == 0
    first = capsys.readouterr().out
    assert "resumed: False" in first
    assert re.search(r"^best_val_mrr: \d\.\d{4}$", first, re.M)
    assert re.search(r"^final_val_mrr: \d\.\d{4}$", first, re.M)
    run_id = next(runs.iterdir()).name
    assert (runs / run_id / "last.pt").is_file()
    assert (runs / run_id / "best.pt").is_file()

    assert main(args) == 0
    second = capsys.readouterr().out
    assert f"run_id: {run_id}" in second
    assert "resumed: True" in second


def test_cli_runs_the_f1_val_preflight_once_per_invocation(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids,
    monkeypatch, capsys
):
    import train.train as training
    from scripts.train_m1 import main

    processed, suite, suites_dir = _write_inputs(
        tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids
    )
    calls = {"n": 0}
    real = training._validation_negatives

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "_validation_negatives", counted)
    args = _args(processed, suite, suites_dir, tmp_path / "runs")

    assert main(args) == 0
    assert calls["n"] == 1
    assert main(args) == 0
    assert "resumed: True" in capsys.readouterr().out
    assert calls["n"] == 2


@pytest.mark.parametrize("failure", ["f2_graph", "wrong_suite", "missing_features", "bad_negatives"])
def test_cli_rejects_invalid_inputs_before_creating_run_directory(
    tmp_path, cx_graph, cx_graph_f1, cx_annotations, cx_schema, cx_types,
    cx_centroids, failure, capsys
):
    from scripts.train_m1 import main

    processed, suite_path, suites_dir = _write_inputs(
        tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids
    )
    if failure == "f2_graph":
        cx_graph.write(processed)
    elif failure == "wrong_suite":
        suite = Suite.read(suite_path)
        Suite(**{**suite.__dict__, "name": "f2_dev"}).write(suite_path)
    elif failure == "missing_features":
        (processed / "features.parquet").unlink()
    else:
        negatives_path = suites_dir / "f1" / "cx" / "f1_val.negatives.json"
        payload = json.loads(negatives_path.read_text())
        payload["digests"] = {key: "bad" for key in payload["digests"]}
        negatives_path.write_text(json.dumps(payload))

    runs = tmp_path / "runs"
    assert main(_args(processed, suite_path, suites_dir, runs)) == 2
    assert not runs.exists()
    assert "M1 training failed:" in capsys.readouterr().err


def test_cli_rejects_a_bad_config_before_reading_inputs(tmp_path, capsys, monkeypatch):
    import scripts.train_m1 as train_m1

    def read(*args, **kwargs):
        pytest.fail("read the graph before checking the config")

    monkeypatch.setattr(train_m1.RegionGraph, "read", read)
    runs = tmp_path / "runs"
    args = _args(tmp_path / "processed", tmp_path / "suite.json", tmp_path, runs)
    assert train_m1.main([*args, "--eval-every", "0"]) == 2
    assert "eval_every must be >= 1" in capsys.readouterr().err
    assert not runs.exists()


def test_cli_reports_runtime_error_without_traceback(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids,
    capsys, monkeypatch
):
    import scripts.train_m1 as train_m1

    processed, suite, suites_dir = _write_inputs(
        tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids
    )

    def fail_with_cuda_error(*args, **kwargs):
        raise RuntimeError("TensorBoard is required for CUDA training")

    monkeypatch.setattr(train_m1, "fit", fail_with_cuda_error)
    runs = tmp_path / "runs"
    assert train_m1.main(_args(processed, suite, suites_dir, runs)) == 2
    error = capsys.readouterr().err
    assert "M1 training failed: TensorBoard is required for CUDA training" in error
    assert "Traceback" not in error


def test_cli_reports_the_run_id_of_a_run_that_failed_after_starting(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids,
    capsys, monkeypatch
):
    import scripts.train_m1 as train_m1

    args = _cli_inputs(tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids)
    run_dir = tmp_path / "runs" / "abc123"

    def fit(*args, on_run, **kwargs):
        on_run(run_dir)
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(train_m1, "fit", fit)

    assert train_m1.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["run_id: abc123", f"run_path: {run_dir}"]
    assert "M1 training failed: CUDA out of memory" in captured.err


def test_cli_flushes_the_run_id_while_a_fresh_or_resumed_run_trains(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids, monkeypatch
):
    import io

    import scripts.train_m1 as train_m1

    args = _cli_inputs(tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids)
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="utf-8"))
    real_fit = train_m1.fit
    seen = []

    def fit(*args, on_run, **kwargs):
        def observed(run_dir):
            on_run(run_dir)
            seen.append((run_dir.name, raw.getvalue().decode()))
        return real_fit(*args, on_run=observed, **kwargs)

    monkeypatch.setattr(train_m1, "fit", fit)

    assert train_m1.main(args) == 0
    assert train_m1.main(args) == 0
    sys.stdout.flush()
    assert "resumed: True" in raw.getvalue().decode()
    assert len(seen) == 2
    for run_id, out_so_far in seen:
        assert out_so_far.splitlines()[-1] == f"run_id: {run_id}"


def test_cli_requires_explicit_device_and_shows_full_help(capsys):
    from scripts.train_m1 import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--processed-dir", "--suite", "--suites-dir", "--runs-dir",
                 "--seed", "--epochs", "--device", "--hidden", "--lr",
                 "--patience", "--gap-fraction",
                 "--aux-weight", "--eval-every", "--warmup-epochs",
                 "--lr-schedule", "--gap-regimes"):
        assert flag in help_text
    with pytest.raises(SystemExit) as exc:
        main(["--processed-dir", "p", "--suite", "s", "--suites-dir", "d",
              "--runs-dir", "r", "--seed", "1", "--epochs", "1"])
    assert exc.value.code == 2


def test_cli_accepts_every_training_device():
    from scripts.train_m1 import parse_args
    from train.train import DEVICES

    for device in DEVICES:
        args = parse_args([
            "--processed-dir", "p", "--suite", "s", "--suites-dir", "d",
            "--runs-dir", "r", "--seed", "1", "--epochs", "1", "--device", device,
        ])
        assert args.device == device


def test_cli_passes_the_schedule_flags_to_the_config():
    from scripts.train_m1 import parse_args
    from train.train import TrainConfig

    base = ["--processed-dir", "p", "--suite", "s", "--suites-dir", "d",
            "--runs-dir", "r", "--seed", "1", "--epochs", "1", "--device", "cpu"]
    defaults = parse_args(base)
    assert (defaults.eval_every, defaults.warmup_epochs, defaults.lr_schedule) == (
        TrainConfig.eval_every, TrainConfig.warmup_epochs, TrainConfig.lr_schedule
    )
    args = parse_args(
        base + ["--eval-every", "5", "--warmup-epochs", "20", "--lr-schedule", "cosine"]
    )
    assert (args.eval_every, args.warmup_epochs, args.lr_schedule) == (5, 20, "cosine")
    with pytest.raises(SystemExit):
        parse_args(base + ["--lr-schedule", "linear"])


def test_cli_defaults_and_disables_the_model_and_schedule_knobs():
    from scripts.train_m1 import parse_args
    from train.train import TrainConfig

    base = ["--processed-dir", "p", "--suite", "s", "--suites-dir", "d",
            "--runs-dir", "r", "--seed", "1", "--epochs", "1", "--device", "cpu"]
    knobs = ("sibling_features", "steps_per_epoch", "encoder_dropout", "decoder_dropout")
    defaults = parse_args(base)
    assert [getattr(defaults, knob) for knob in knobs] == [
        getattr(TrainConfig, knob) for knob in knobs
    ] == [True, 8, 0.1, 0.1]
    earlier = parse_args(base + [
        "--no-sibling-features", "--steps-per-epoch", "1",
        "--encoder-dropout", "0", "--decoder-dropout", "0",
    ])
    assert [getattr(earlier, knob) for knob in knobs] == [False, 1, 0.0, 0.0]


def test_cli_builds_the_run_config_from_the_schedule_flags(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids, capsys
):
    from scripts.train_m1 import main

    processed, suite, suites_dir = _write_inputs(
        tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids
    )
    runs = tmp_path / "runs"
    args = _args(processed, suite, suites_dir, runs) + [
        "--epochs", "3", "--eval-every", "3", "--warmup-epochs", "2",
        "--lr-schedule", "cosine",
    ]

    assert main(args) == 0

    (run_dir,) = runs.iterdir()
    config = json.loads((run_dir / "manifest.json").read_text())["config"]
    assert (config["eval_every"], config["warmup_epochs"], config["lr_schedule"]) == (
        3, 2, "cosine"
    )


def test_cli_passes_the_gap_regimes_to_the_config():
    from scripts.train_m1 import parse_args
    from train.train import TrainConfig

    base = ["--processed-dir", "p", "--suite", "s", "--suites-dir", "d",
            "--runs-dir", "r", "--seed", "1", "--epochs", "1", "--device", "cpu"]
    assert tuple(parse_args(base).gap_regimes) == TrainConfig.gap_regimes
    args = parse_args(base + ["--gap-regimes", "R", "T", "S"])
    assert args.gap_regimes == ["R", "T", "S"]
    with pytest.raises(SystemExit):
        parse_args(base + ["--gap-regimes", "X"])


def test_cli_builds_the_run_config_from_the_gap_regimes(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids, capsys
):
    from scripts.train_m1 import main

    processed, suite, suites_dir = _write_inputs(
        tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids
    )
    runs = tmp_path / "runs"
    args = _args(processed, suite, suites_dir, runs) + ["--gap-regimes", "R", "T"]

    assert main(args) == 0

    (run_dir,) = runs.iterdir()
    config = json.loads((run_dir / "manifest.json").read_text())["config"]
    assert config["gap_regimes"] == ["R", "T"]


def test_cli_test_run_suffixes_the_run_directory(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids, capsys
):
    from scripts.train_m1 import main

    processed, suite, suites_dir = _write_inputs(
        tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids
    )
    runs = tmp_path / "runs"

    assert main(_args(processed, suite, suites_dir, runs) + ["--test-run"]) == 0

    (run_dir,) = runs.iterdir()
    assert run_dir.name.endswith("-test")
    assert f"run_id: {run_dir.name}" in capsys.readouterr().out


def _stop(status, tmp_path, *, signum=None, reason=None, resume_epoch=None):
    from src.run_control import RunStopped

    stop = RunStopped(status, signum=signum, reason=reason)
    stop.resume_epoch = resume_epoch
    stop.run_dir = tmp_path / "runs" / "abc123"
    return stop


def _raise_from_fit(monkeypatch, exc):
    import scripts.train_m1 as train_m1

    def fit(*args, **kwargs):
        raise exc

    monkeypatch.setattr(train_m1, "fit", fit)
    return train_m1


def _cli_inputs(tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids):
    processed, suite, suites_dir = _write_inputs(
        tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids
    )
    return _args(processed, suite, suites_dir, tmp_path / "runs")


def test_cli_prints_a_signal_pause_and_exits_128_plus_the_signal(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids,
    capsys, monkeypatch
):
    import shlex
    import signal
    import sys

    args = _cli_inputs(tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids)
    stop = _stop("paused", tmp_path, signum=signal.SIGTERM, resume_epoch=7)
    train_m1 = _raise_from_fit(monkeypatch, stop)
    monkeypatch.setattr(sys, "argv", ["queue_driver.py", "--other", "flags"])

    assert train_m1.main(args) == 128 + signal.SIGTERM
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "run_id: abc123",
        f"run_path: {stop.run_dir}",
        "status: paused",
        "resume_epoch: 7",
    ]
    assert lines[4] == "resume: " + shlex.join([sys.executable, train_m1.__file__, *args])
    assert len(lines) == 5


def test_cli_quotes_the_resume_command_for_windows(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids,
    capsys, monkeypatch
):
    import subprocess
    import sys
    import types

    args = _cli_inputs(tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids)
    train_m1 = _raise_from_fit(monkeypatch, _stop("paused", tmp_path, resume_epoch=2))
    argv = [*args, "--runs-dir", "C:\\my runs"]
    monkeypatch.setattr(train_m1, "os", types.SimpleNamespace(name="nt", environ={}))

    train_m1.main(argv)
    out = capsys.readouterr().out
    command = [sys.executable, train_m1.__file__, *argv]
    assert "resume: " + subprocess.list2cmdline(command) in out.splitlines()


def test_cli_exits_3_for_a_requested_pause_and_prints_the_reason(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids,
    capsys, monkeypatch
):
    args = _cli_inputs(tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids)
    train_m1 = _raise_from_fit(
        monkeypatch, _stop("paused", tmp_path, reason="gpu needed", resume_epoch=4)
    )

    assert train_m1.main(args) == 3
    lines = capsys.readouterr().out.splitlines()
    assert lines[:5] == [
        "run_id: abc123",
        f"run_path: {tmp_path / 'runs' / 'abc123'}",
        "status: paused",
        "resume_epoch: 4",
        "reason: gpu needed",
    ]
    assert lines[5].startswith("resume: ")


def test_cli_exits_4_for_a_cancel_and_prints_the_reason(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids,
    capsys, monkeypatch
):
    args = _cli_inputs(tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids)
    train_m1 = _raise_from_fit(
        monkeypatch, _stop("cancelled", tmp_path, reason="bad config")
    )

    assert train_m1.main(args) == 4
    assert capsys.readouterr().out.splitlines() == [
        "run_id: abc123",
        f"run_path: {tmp_path / 'runs' / 'abc123'}",
        "status: cancelled",
        "reason: bad config",
    ]


@pytest.mark.parametrize("name", ["RunLocked", "RunCancelled"])
def test_cli_reports_a_held_lock_or_cancelled_run_as_a_failure(
    name, tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids,
    capsys, monkeypatch
):
    import src.run_control
    import train.train

    args = _cli_inputs(tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids)
    cls = getattr(src.run_control, name, None) or getattr(train.train, name)
    train_m1 = _raise_from_fit(monkeypatch, cls("run is locked"))

    assert train_m1.main(args) == 2
    error = capsys.readouterr().err
    assert "M1 training failed: run is locked" in error
    assert "Traceback" not in error


def test_cli_import_leaves_scipy_stats_unloaded():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    probe = "import sys, scripts.train_m1; print('scipy.stats' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=root, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


_DOCUMENTED_CX = [
    "--processed-dir", "data/processed/f1/cx", "--suite", "data/suites/f1/cx/f1_val.json",
    "--suites-dir", "data/suites", "--runs-dir", "data/runs",
    "--seed", "0", "--epochs", "600", "--device", "cpu",
]


def test_the_cx_config_builds_the_documented_cx_command(monkeypatch):
    from scripts.train_m1 import parse_args
    from train.region_config import config_from_args

    monkeypatch.delenv("FLYARCH_DATA_ROOT", raising=False)
    from_file = parse_args(["--config", "train/configs/cx.toml", "--runs-dir", "data/runs",
                            "--seed", "0", "--device", "cpu"])
    from_flags = parse_args(_DOCUMENTED_CX)
    assert config_from_args(from_file) == config_from_args(from_flags)
    assert (from_file.processed_dir, from_file.suite) == (from_flags.processed_dir, from_flags.suite)


def _config_file(tmp_path, train):
    path = tmp_path / "al.toml"
    path.write_text('region = "AL"\n[train]\nepochs = 4\n' + train)
    return path


@pytest.mark.parametrize("flags, field, expected", [
    ([], "sibling_features", False),
    (["--sibling-features"], "sibling_features", True),
    ([], "gap_regimes", ["R", "T"]),
    (["--gap-regimes", "S"], "gap_regimes", ["S"]),
    ([], "hidden", 16),
    (["--hidden", "32"], "hidden", 32),
    ([], "epochs", 4),
    (["--epochs", "2"], "epochs", 2),
    ([], "lr", 1e-3),
])
def test_a_flag_beats_the_file_and_the_file_beats_the_default(tmp_path, flags, field, expected):
    from scripts.train_m1 import parse_args
    from train.region_config import config_from_args

    path = _config_file(
        tmp_path, 'sibling_features = false\ngap_regimes = ["R", "T"]\nhidden = 16\n'
    )
    args = parse_args(["--config", str(path), "--runs-dir", "r", "--seed", "0",
                       "--device", "cpu", *flags])
    value = getattr(config_from_args(args), field)
    assert (list(value) if isinstance(value, tuple) else value) == expected


def test_without_a_config_the_input_flags_are_required(capsys):
    from scripts.train_m1 import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--processed-dir", "p", "--suite", "s", "--suites-dir", "d",
                    "--runs-dir", "r", "--seed", "1", "--device", "cpu"])
    assert "--epochs" in capsys.readouterr().err


def test_every_scalar_flag_names_a_train_config_field():
    from scripts._train_cli import _TRAIN_FLAGS
    from train.train import TrainConfig

    names = [name for _, name, _ in _TRAIN_FLAGS]
    assert len(names) == len(set(names))
    assert set(names) <= set(TrainConfig.__dataclass_fields__)


def test_dev_training_provenance_is_explicitly_unavailable(monkeypatch):
    import train.train as training

    monkeypatch.setattr(training, "_git_commit", lambda: pytest.fail("dev must not probe Git"))
    assert training._source_provenance({"FLYARCH_REQUIRE_METADATA": "0"}) == {
        "git": {"commit": None, "dirty": None}, "image_identifier": None,
        "metadata_unavailable": ["git", "image_identifier"],
    }


def test_direct_host_training_keeps_git_fallback(monkeypatch):
    import train.train as training

    monkeypatch.setattr(training, "_git_commit", lambda: "a" * 40)
    assert training._source_provenance({})["git"] == {"commit": "a" * 40, "dirty": None}


def test_lightweight_parser_choices_match_training_choices():
    from scripts import _train_cli
    from src.masks import MASK_REGIMES
    from train.train import DEVICES, LR_SCHEDULES

    actions = {action.dest: action for action in _train_cli.build_argparser()._actions}
    assert actions["device"].choices == DEVICES
    assert actions["lr_schedule"].choices == LR_SCHEDULES
    assert actions["gap_regimes"].choices == MASK_REGIMES


@pytest.mark.parametrize("seed_args", [
    ["--seed", "4"], ["--seed", "4", "--seeds", "1", "2", "3"],
    ["--seeds", "1", "1", "3"], ["--seeds", "-1", "2"], ["--seeds"],
])
def test_the_sweep_parser_refuses_single_repeated_or_negative_seeds(seed_args):
    from scripts._train_cli import build_sweep_argparser

    with pytest.raises(SystemExit):
        build_sweep_argparser().parse_args([
            "--config", "train/configs/cx.toml", "--runs-dir", "r", "--device", "cuda", *seed_args,
        ])


@pytest.mark.parametrize("seeds", [["7"], ["0", "1"], ["0", "1", "2", "3", "4"]])
def test_the_sweep_parser_takes_any_count_of_distinct_seeds(seeds):
    from scripts._train_cli import build_sweep_argparser

    args = build_sweep_argparser().parse_args(
        ["--runs-dir", "r", "--device", "cuda", "--seeds", *seeds])
    assert args.seeds == [int(seed) for seed in seeds]


_IMPORT_WITHOUT_TRAINING_STACK = """
import sys


class NoTrainingStack:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "src", "train"}:
            raise ImportError(f"{name} needs the training stack")
        return None


sys.meta_path.insert(0, NoTrainingStack())
import scripts._train_cli as cli

print(cli.build_sweep_argparser().parse_args(
    ["--runs-dir", "r", "--device", "cpu", "--seeds", "1", "2", "3"]).seeds)
"""


def test_the_shared_cli_imports_without_the_training_stack():
    root = Path(__file__).resolve().parent.parent
    done = subprocess.run([sys.executable, "-c", _IMPORT_WITHOUT_TRAINING_STACK], cwd=root,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[1, 2, 3]"
def _al_args(tmp_path, *flags, train="hidden = 16\ngap_regimes = [\"R\", \"T\"]\n"):
    return ["--config", str(_config_file(tmp_path, train)), "--runs-dir", "r",
            "--seed", "0", "--device", "cpu", *flags]


def test_overrides_lists_flags_that_differ_from_the_file(tmp_path):
    from scripts.train_m1 import parse_args

    args = parse_args(_al_args(tmp_path, "--hidden", "32", "--gap-regimes", "S", "--lr", "0.01"))
    assert args.overrides == {"hidden": 32, "gap_regimes": ["S"], "lr": 0.01}


def test_a_flag_equal_to_the_file_value_is_not_an_override(tmp_path):
    from scripts.train_m1 import parse_args

    args = parse_args(_al_args(tmp_path, "--hidden", "16", "--epochs", "4",
                               "--gap-regimes", "R", "T"))
    assert args.overrides == {}
    assert parse_args(_al_args(tmp_path)).overrides == {}


def test_overrides_without_config_compares_to_defaults_and_skips_epochs():
    from scripts.train_m1 import parse_args
    from train.train import TrainConfig

    base = ["--processed-dir", "p", "--suite", "s", "--suites-dir", "d",
            "--runs-dir", "r", "--seed", "1", "--device", "cpu", "--epochs", "3"]
    default_hidden = TrainConfig.__dataclass_fields__["hidden"].default
    args = parse_args([*base, "--hidden", str(default_hidden + 1)])
    assert args.overrides == {"hidden": default_hidden + 1}
    assert parse_args([*base, "--hidden", str(default_hidden)]).overrides == {}


def test_sibling_features_flag_is_an_override_only_when_it_differs(tmp_path):
    from scripts.train_m1 import parse_args

    train = "sibling_features = true\n"
    assert parse_args(_al_args(tmp_path, "--sibling-features", train=train)).overrides == {}
    assert parse_args(_al_args(tmp_path, "--no-sibling-features", train=train)).overrides == {
        "sibling_features": False
    }


def test_title_is_cleaned_in_parse_args(tmp_path):
    from scripts.train_m1 import parse_args

    assert parse_args(_al_args(tmp_path, "--title", "  a\n\tb  ")).title == "a b"
    assert parse_args(_al_args(tmp_path)).title is None


def _stub_fit(monkeypatch):
    import scripts.train_m1 as train_m1

    seen = {}

    def fit(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(train_m1, "fit", fit)
    return seen


def test_main_warns_without_title(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids,
    monkeypatch, capsys
):
    from src.run_labels import TITLE_WARNING

    _stub_fit(monkeypatch)
    from scripts.train_m1 import main

    main(_cli_inputs(tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids))
    assert capsys.readouterr().err.startswith(TITLE_WARNING + "\n")


def test_main_is_quiet_with_title_or_test_run(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids,
    monkeypatch, capsys
):
    from src.run_labels import TITLE_WARNING

    _stub_fit(monkeypatch)
    from scripts.train_m1 import main

    args = _cli_inputs(tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids)
    main([*args, "--title", "x"])
    assert TITLE_WARNING not in capsys.readouterr().err
    main([*args, "--test-run"])
    assert TITLE_WARNING not in capsys.readouterr().err


def test_parse_args_never_warns(tmp_path, capsys):
    from scripts.train_m1 import parse_args

    parse_args(_al_args(tmp_path))
    assert capsys.readouterr().err == ""


def test_main_passes_the_labels_to_fit(
    tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids, monkeypatch
):
    from scripts.train_m1 import main
    from src.run_labels import RunLabels

    seen = _stub_fit(monkeypatch)
    args = _cli_inputs(tmp_path, cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids)
    main([*args, "--title", "Wide hidden", "--lr", "0.5"])
    assert seen["labels"] == RunLabels(
        title="Wide hidden", overrides={"hidden": 8, "patience": 1, "lr": 0.5}
    )
