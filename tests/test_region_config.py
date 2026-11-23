import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from scripts._train_cli import build_argparser, build_sweep_argparser  # noqa: E402
from src.regions import DEFAULT_THRESHOLD, REGIONS  # noqa: E402
from train.region_config import (  # noqa: E402
    CONFIG_DIR,
    TRAINING_ORDER,
    RegionConfig,
    check_region,
    config_from_args,
    load_region_config,
    region_paths,
    resolve_args,
)
from train.train import TrainConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_every_committed_config_loads_and_covers_the_registry():
    configs = {path.stem: load_region_config(path) for path in CONFIG_DIR.glob("*.toml")}
    assert all(config.region.lower() == stem for stem, config in configs.items())
    assert {config.region for config in configs.values()} == set(REGIONS)


def test_transfer_report_copies_the_training_order():
    from scripts.transfer_report import TRAINING_ORDER as REPORT_ORDER

    assert REPORT_ORDER == TRAINING_ORDER


@pytest.mark.parametrize("train, message", [
    ({"epochs": 1, "run_dir": "x"}, "cannot set"),
    ({"epochs": 1, "suites_dir": "x"}, "cannot set"),
    ({"epochs": 1, "hiden": 8}, "not TrainConfig fields"),
    ({"hidden": 8}, "must set epochs"),
    ({"epochs": 1, "sampling": "random"}, "sampling"),
])
def test_the_train_table_is_checked(train, message):
    with pytest.raises(ValueError, match=message):
        RegionConfig(region="AL", train=train)


def test_calibration_seeds_hold_the_gate_seeds():
    with pytest.raises(ValueError, match="missing"):
        RegionConfig(region="AL", seeds=(0, 1, 9), train={"epochs": 1})
    with pytest.raises(ValueError, match="5 distinct"):
        RegionConfig(region="AL", calibration_seeds=(0, 1, 2), train={"epochs": 1})
    with pytest.raises(ValueError, match="5 distinct"):
        RegionConfig(region="AL", calibration_seeds=(0, 1, 2, 3, 4, 4), train={"epochs": 1})


def test_unknown_region_and_top_level_keys_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown region"):
        RegionConfig(region="XX", train={"epochs": 1})
    path = tmp_path / "al.toml"
    path.write_text('region = "AL"\nthreshold = 5\n[train]\nepochs = 1\n')
    with pytest.raises(ValueError, match="unknown keys"):
        load_region_config(path)


def test_region_paths_follow_the_data_root(monkeypatch):
    monkeypatch.delenv("FLYARCH_DATA_ROOT", raising=False)
    paths = region_paths("AL", "f2")
    assert paths.processed == Path("data/processed/f2/al")
    assert paths.interim == Path("data/interim/f2/al")
    assert paths.suites == Path("data/suites/f2/al")
    assert paths.suites_root == Path("data/suites")
    assert paths.results == Path("results/al")
    monkeypatch.setenv("FLYARCH_DATA_ROOT", "/mnt/flyarch")
    assert region_paths("AL", "f1").processed == Path("/mnt/flyarch/processed/f1/al")
    assert region_paths("AL", "f1", "elsewhere").processed == Path("elsewhere/processed/f1/al")


def test_resolver_defaults_match_the_training_config():
    parser = build_argparser()
    args = resolve_args(parser.parse_args([
        "--processed-dir", "p", "--suite", "s", "--suites-dir", "d",
        "--runs-dir", "r", "--seed", "1", "--epochs", "1", "--device", "cpu",
    ]), parser)
    assert config_from_args(args) == TrainConfig(epochs=1, run_dir=Path("r"), suites_dir=Path("d"))


def test_a_config_sweep_resolves_at_the_same_boundary(monkeypatch):
    monkeypatch.setenv("FLYARCH_DATA_ROOT", "/mnt/flyarch")
    parser = build_sweep_argparser()
    args = resolve_args(parser.parse_args([
        "--config", str(CONFIG_DIR / "cx.toml"), "--runs-dir", "data/runs",
        "--device", "cuda", "--seeds", "0", "1", "2",
    ]), parser)

    assert args.seeds == [0, 1, 2]
    assert args.seed is None
    assert (args.processed_dir, args.suite, args.suites_dir) == (
        Path("/mnt/flyarch/processed/f1/cx"), Path("/mnt/flyarch/suites/f1/cx/f1_val.json"),
        Path("/mnt/flyarch/suites"),
    )
    assert args.region_config == load_region_config(CONFIG_DIR / "cx.toml")
    assert config_from_args(args).epochs == args.region_config.train["epochs"]


def test_a_sweep_without_seeds_uses_the_configs_calibration_seeds():
    parser = build_sweep_argparser()
    argv = ["--runs-dir", "data/runs", "--device", "cuda"]
    args = resolve_args(parser.parse_args([*argv, "--config", str(CONFIG_DIR / "cx.toml")]), parser)
    assert args.seeds == list(load_region_config(CONFIG_DIR / "cx.toml").calibration_seeds)
    with pytest.raises(SystemExit):
        resolve_args(parser.parse_args([*argv, "--epochs", "1", "--processed-dir", "p",
                                        "--suite", "s", "--suites-dir", "d"]), parser)


@pytest.mark.parametrize("builder, seed_args", [
    (build_argparser, ["--seed", "1"]), (build_sweep_argparser, ["--seeds", "0", "1", "2"]),
])
@pytest.mark.parametrize("use_config, message", [(False, "--epochs"), (True, "--config")])
def test_resolution_refusals_use_the_callers_parser(tmp_path, monkeypatch, builder, seed_args,
                                                   use_config, message):
    parser = builder()
    argv = ["--runs-dir", "r", "--device", "cpu", *seed_args]
    if use_config:
        argv += ["--config", str(tmp_path / "absent.toml")]

    def error(text):
        raise ValueError(text)

    monkeypatch.setattr(parser, "error", error)
    with pytest.raises(ValueError, match=message):
        resolve_args(parser.parse_args(argv), parser)


def test_check_region_refuses_a_graph_from_another_region():
    args = SimpleNamespace(config=Path("train/configs/al.toml"), processed_dir=Path("p"),
                           region_config=RegionConfig(region="AL", train={"epochs": 1}))
    other = SimpleNamespace(region="CX", threshold=DEFAULT_THRESHOLD)

    check_region(SimpleNamespace(region_config=None), other)
    check_region(args, SimpleNamespace(region="AL", threshold=DEFAULT_THRESHOLD))
    with pytest.raises(ValueError, match="is for AL"):
        check_region(args, other)
def test_ol_config_loads_the_mask_exclude_table_read_only():
    rule = load_region_config(CONFIG_DIR / "ol.toml").mask_exclude
    assert dict(rule) == {"rule": "lamina_fraction", "min_fraction": 0.01}
    with pytest.raises(TypeError):
        rule["min_fraction"] = 1.0
    assert all(
        load_region_config(path).mask_exclude is None
        for path in CONFIG_DIR.glob("*.toml") if path.stem != "ol"
    )


@pytest.mark.parametrize("region, mask_exclude, message", [
    ("OL", {"rule": "other", "min_fraction": 0.5}, "unknown rule.*lamina_fraction"),
    ("OL", {"rule": "lamina_fraction"}, "exactly rule and min_fraction"),
    ("OL", {"min_fraction": 0.5}, "exactly rule and min_fraction"),
    ("OL", {"rule": "lamina_fraction", "min_fraction": 0.5, "extra": 1}, "exactly rule and min_fraction"),
    ("OL", {"rule": "lamina_fraction", "min_fraction": 0}, "number in"),
    ("OL", {"rule": "lamina_fraction", "min_fraction": 1.5}, "number in"),
    ("OL", {"rule": "lamina_fraction", "min_fraction": True}, "number in"),
    ("OL", {"rule": "lamina_fraction", "min_fraction": "0.5"}, "number in"),
    ("AL", {"rule": "lamina_fraction", "min_fraction": 0.5}, "OL only"),
])
def test_the_mask_exclude_table_is_checked(region, mask_exclude, message):
    with pytest.raises(ValueError, match=message):
        RegionConfig(region=region, mask_exclude=mask_exclude, train={"epochs": 1})


def test_mask_exclude_is_optional_and_accepts_the_upper_bound(tmp_path):
    assert RegionConfig(region="OL", train={"epochs": 1}).mask_exclude is None
    path = tmp_path / "ol.toml"
    path.write_text('region = "OL"\n[mask_exclude]\nrule = "lamina_fraction"\n'
                    "min_fraction = 1\n[train]\nepochs = 1\n")
    assert load_region_config(path).mask_exclude["min_fraction"] == 1
