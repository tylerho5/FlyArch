"""Seams of the Phase 3 calibration writer: its refusals, and the record check."""

import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from scripts import phase3_calibrate  # noqa: E402
from src.integration_gate import calibrate_delta_p3  # noqa: E402
from src.digests import sha256  # noqa: E402
from train.cohorts import cohort_digest  # noqa: E402
from train.train import _feature_digest  # noqa: E402
from train.union_scoring import instance_mrr, load_cache, save_cache  # noqa: E402

RUNS = [f"f1-union12-s00{k}-76efea1e" for k in range(5)]
INSTANCES = [f"f1-union12-R0{k}-s500{k}" for k in range(1, 6)]
TABLE = SimpleNamespace(values=np.array([[1.0, 2.0], [3.0, 4.0]]), columns=["a", "b"],
                        geometry_present=None, extras=None)
DIGESTS = {"graph_digest": "graph", "schema_digest": "schema", "suite_digest": "suite"}


def _manifest(run_id, **overrides):
    return {"run_id": run_id, "seed": RUNS.index(run_id), "region": "UNION12", "device": "cpu",
            "precision": "float32", "config": {"epochs": 1}, "code_digest": "code",
            "feature_digest": _feature_digest(TABLE, extras=()), **DIGESTS, **overrides}


def _cache(seed):
    rng = np.random.default_rng(seed)
    return {iid: (np.arange(4), np.column_stack([np.ones(4), rng.uniform(0.5, 2.0, (4, 3))]))
            for iid in INSTANCES}


def _cache_path(workspace, run_id):
    return phase3_calibrate.cache_path(workspace.scores_root, "UNION12", "phase3_boundary_f1_val", run_id)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A repo root, five run manifests and their score caches, and the score manifest over them."""
    runs_dir, scores_root = tmp_path / "data" / "runs", tmp_path / "results" / "phase3" / "scores"
    entries, keys = {}, {}
    for run_id in RUNS:
        (runs_dir / run_id).mkdir(parents=True)
        (runs_dir / run_id / "manifest.json").write_text(json.dumps(_manifest(run_id)), encoding="utf-8")
    (tmp_path / "results" / "phase3").mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)  # the calibration freeze reads tags
    monkeypatch.setattr(phase3_calibrate, "union_inputs", lambda *a, **k: (TABLE, DIGESTS))
    scope = SimpleNamespace(root=tmp_path, runs_dir=runs_dir, scores_root=scores_root, entries=entries, keys=keys)
    for run_id in RUNS:
        cache = _cache_path(scope, run_id)
        save_cache(cache, _cache(RUNS.index(run_id)))
        keys[run_id] = phase3_calibrate.registry_label(cache, tmp_path)
        entries[keys[run_id]] = {"cohort": cohort_digest(_manifest(run_id)), "system": "A",
                                 "masked_innervation": "uniform", "registry": "results/experts.csv",
                                 "scores": sha256(cache)}
    scope.save_entries = lambda: (tmp_path / "results" / "phase3" / "manifest.json").write_text(
        json.dumps(entries), encoding="utf-8")
    scope.save_entries()
    s, delta = calibrate_delta_p3([instance_mrr(load_cache(_cache_path(scope, run_id))) for run_id in RUNS])
    cohort = cohort_digest(_manifest(RUNS[0]))
    scope.record = {"region": "PHASE3", "s": s, "delta": delta, "run_ids": RUNS, "cohort": cohort,
                    "system": "A", "tag": f"delta-phase3-{cohort[:8]}"}
    return scope


def _argv(workspace, mode):
    return [mode, "--repo-root", str(workspace.root), "--scores-root", str(workspace.scores_root),
            "--runs-dir", str(workspace.runs_dir), "--registry", "results/experts.csv",
            "--masked-innervation", "uniform", *[f"--system-run={run}" for run in RUNS]]


def _write_record(workspace, **overrides):
    path = workspace.root / "results" / "phase3" / "delta.json"
    path.write_text(json.dumps({**workspace.record, **overrides}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_calibrate_refuses_runs_from_two_cohorts(workspace, capsys):
    manifest = _manifest(RUNS[4], code_digest="other")
    (workspace.runs_dir / RUNS[4] / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    workspace.entries[workspace.keys[RUNS[4]]]["cohort"] = cohort_digest(manifest)
    workspace.save_entries()
    assert phase3_calibrate.main(_argv(workspace, "calibrate")) == 2
    assert "not one cohort" in capsys.readouterr().err
    assert not (workspace.root / "results" / "phase3" / "delta.json").exists()


def test_calibrate_refuses_to_overwrite_a_record(workspace, capsys):
    _write_record(workspace)
    assert phase3_calibrate.main(_argv(workspace, "calibrate")) == 2
    assert "already exists" in capsys.readouterr().err


def test_check_accepts_the_record_and_refuses_a_tampered_s(workspace, capsys):
    _write_record(workspace)
    assert phase3_calibrate.main(_argv(workspace, "check")) == 0
    assert f"s={workspace.record['s']!r}" in capsys.readouterr().out
    _write_record(workspace, s=0.5)
    assert phase3_calibrate.main(_argv(workspace, "check")) == 2
    assert "recorded s=0.5" in capsys.readouterr().err
