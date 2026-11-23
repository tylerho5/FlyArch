import numpy as np
import pytest

from scripts import phase3_components as pc
from src.digests import sha256
from train.union_scoring import instance_mrr, save_cache

RANK1 = [[3.0, 1.0, 0.0, -1.0]] * 2  # positive first: rank 1 on both rows, MRR 1.0
RANK3 = [[0.0, 3.0, 1.0, -1.0]] * 2  # positive below two negatives: rank 3 on both rows, MRR 1/3
SUITE = pc.SUITE


def one(scores):
    return {"i1": (np.array([0, 1]), np.array(scores, dtype=np.float64))}


def write(root, rel, scores):
    save_cache(root / rel, one(scores))
    return rel, sha256(root / rel)


def test_seed_mean_averages_per_instance_and_refuses_mismatched_instances():
    assert pc.seed_mean([{"a": 0.2, "b": 1.0}, {"a": 0.4, "b": 0.5}]) == {"a": pytest.approx(0.3), "b": 0.75}
    with pytest.raises(ValueError, match="different instances"):
        pc.seed_mean([{"a": 1.0}, {"b": 1.0}])
    with pytest.raises(ValueError, match="no seeds"):
        pc.seed_mean([])


def test_gate_runs_reads_only_the_registry_gate_seeds(tmp_path):
    path = tmp_path / "reg.csv"
    path.write_text("region,gate_seed,run_id,seed\nAL,true,r0,0\nAL,true,r1,1\nAL,false,r3,3\n", encoding="utf-8")
    assert pc.gate_runs(path) == {"AL": {0: "r0", 1: "r1"}}


def test_require_milestone_and_unchanged(tmp_path):
    rel, digest = write(tmp_path, "a.npz", RANK1)
    entries = {rel: {"milestone": "gate-1", "scores": digest}}
    pc.require_milestone(entries, "gate-1", "x")
    pc.require_unchanged(entries, tmp_path, "x")
    with pytest.raises(ValueError, match="not scored under milestone 'gate-2'"):
        pc.require_milestone(entries, "gate-2", "x")
    with pytest.raises(ValueError, match="changed on disk"):
        pc.require_unchanged({rel: {"scores": "0" * 64}}, tmp_path, "x")
    with pytest.raises(ValueError, match="missing"):
        pc.require_unchanged({"nope.npz": {"scores": digest}}, tmp_path, "x")


def baseline_manifest(root, milestone="gate-1", families=pc.FAMILIES, seeds=(11, 22, 33)):
    manifest = {}
    for family in families:
        for seed in seeds:
            rel, digest = write(root, f"union/union12/{SUITE}/baseline-{family}-s{seed}.npz", RANK3)
            manifest[rel] = {"mode": "baselines", "suite": SUITE, "dataset": "f2", "masked_innervation": "uniform",
                             "region": f"baseline:{family}", "seed": seed, "milestone": milestone, "scores": digest}
    return manifest


def test_baseline_maps_select_by_manifest_and_require_three_seeds_and_the_milestone(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "check_provenance", lambda *a, **k: None)  # b0_report's own, tested elsewhere
    monkeypatch.setattr(pc, "check_rows", lambda *a, **k: None)
    manifest = baseline_manifest(tmp_path)
    maps = pc.baseline_maps(manifest, {}, tmp_path, "gate-1")
    assert set(maps) == {f"baseline:{f}" for f in pc.FAMILIES}
    assert maps["baseline:logreg"]["i1"] == pytest.approx(1 / 3)
    stray = dict(manifest)  # a hand-copied file with no manifest entry is ignored, not read
    save_cache(tmp_path / f"union/union12/{SUITE}/baseline-logreg-s99.npz", one(RANK1))
    assert pc.baseline_maps(stray, {}, tmp_path, "gate-1") == maps
    with pytest.raises(ValueError, match="not scored under milestone 'gate-2'"):
        pc.baseline_maps(manifest, {}, tmp_path, "gate-2")
    short = {k: v for k, v in manifest.items() if not k.endswith("xgboost-s33.npz")}
    with pytest.raises(ValueError, match="baseline:xgboost needs exactly three seed entries"):
        pc.baseline_maps(short, {}, tmp_path, "gate-1")
    first = next(iter(manifest))
    changed = {**manifest, first: {**manifest[first], "scores": "0" * 64}}
    with pytest.raises(ValueError, match="changed on disk"):
        pc.baseline_maps(changed, {}, tmp_path, "gate-1")


def regional_manifest(root, region="AL", milestone="gate-1", graph="U+", system="A"):
    runs = ["run-s000", "run-s001", "run-s002"]
    manifest = {}
    for run, scores in zip(runs, (RANK1, RANK3, RANK3)):
        rel, digest = write(root, f"regional/{region.lower()}/f2_confirm/{run}.npz", scores)
        manifest[rel] = {"mode": "regional", "suite": "f2_confirm", "dataset": "f2", "suite_region": region,
                         "system": system, "run_id": run, "masked_innervation": "uniform",
                         "milestone": milestone, "scores": digest, "union_graph": graph}
    return runs, manifest


def test_regional_system_means_the_gate_runs_from_manifest_entries(tmp_path):
    runs, manifest = regional_manifest(tmp_path)
    mine, graphs = pc.regional_system(manifest, tmp_path, "AL", runs, "gate-1")
    assert mine == {"i1": pytest.approx((1.0 + 1 / 3 + 1 / 3) / 3)} and graphs == {"U+"}
    assert instance_mrr(one(RANK1)) == {"i1": 1.0}
    with pytest.raises(ValueError, match="not scored under milestone 'gate-2'"):
        pc.regional_system(manifest, tmp_path, "AL", runs, "gate-2")
    with pytest.raises(ValueError, match="expected one manifest entry for A's run run-s003"):
        pc.regional_system(manifest, tmp_path, "AL", runs + ["run-s003"], "gate-1")
    with pytest.raises(ValueError, match="expected one manifest entry"):
        pc.regional_system(manifest, tmp_path, "CX", runs, "gate-1")
    _, other = regional_manifest(tmp_path, system="B")  # entries from another system are not A's
    with pytest.raises(ValueError, match="expected one manifest entry"):
        pc.regional_system(other, tmp_path, "AL", runs, "gate-1")


def test_main_exits_2_when_inputs_are_missing(tmp_path, capsys):
    code = pc.main(["--milestone", "gate-1", "--manifest", str(tmp_path / "none.json"),
                    "--out", str(tmp_path / "out.json")])
    assert code == 2 and "phase3_components failed" in capsys.readouterr().err
