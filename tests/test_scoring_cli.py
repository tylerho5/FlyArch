import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from scripts.scoring_common import (
    _validate_score_positive_pair, _validate_scoring_identity, append_results,
    baseline_fit_digest,
    confirm_preflight, file_digest, manifest_region, phase2_verdict,
    inspect_suite_identity, load_result_tables, resolve_cohort_manifests,
    scoring_commit, scoring_input_identity,
    validate_baseline_provenance,
    validate_expert_provenance, validate_f1_cohort_inputs, validate_scored_suite,
)


def _suite_input(data_root, name, *, internal=None, dataset="f2", region="AL",
                 mask_exclude_types=()):
    from src.suites import Suite

    path = data_root / "suites" / dataset / region.lower() / f"{name}.json"
    Suite(name=internal or name, dataset=dataset, region=region, graph_digest="graph",
          threshold=5, master_seed=20260818, instances=(),
          mask_exclude_types=mask_exclude_types).write(path)
    return path


def _frozen_suite(data_root, name, instances, *, dataset="f2", region="AL"):
    import numpy as np
    from src.suites import Suite, SuiteInstance

    path = data_root / "suites" / dataset / region.lower() / f"{name}.json"
    Suite(
        name=name, dataset=dataset, region=region, graph_digest="graph", threshold=5,
        master_seed=20260818,
        instances=tuple(
            SuiteInstance(instance_id=instance, regime="random", seed=index, params={},
                          masked_root_ids=np.asarray([index + 1], dtype=np.int64))
            for index, instance in enumerate(instances)
        ),
    ).write(path)
    path.with_name(f"{name}.negatives.json").write_text(json.dumps({
        "name": name,
        "spec": {"k": 100, "alpha": 1.0, "master_seed": 20260818},
        "digests": {instance: f"{index + 1:064x}" for index, instance in enumerate(instances)},
    }), encoding="utf-8")
    return path


def _git(path, *args):
    result = subprocess.run(["git", *args], cwd=path, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().strip()


def _tagged_repo(tmp_path, *, cohort="abc", tag="delta-al-abc"):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    delta_path = repo / "results" / "al" / "delta.json"
    delta_path.parent.mkdir(parents=True)
    delta_path.write_bytes(json.dumps({"region": "AL", "s": 0.1, "delta": 0.2,
                                       "cohort": cohort, "tag": tag}).encode() + b"\n")
    _git(repo, "add", "results/al/delta.json")
    _git(repo, "commit", "-qm", "calibrate")
    _git(repo, "tag", tag)
    (repo / "later.txt").write_text("descendant\n")
    _git(repo, "add", "later.txt")
    _git(repo, "commit", "-qm", "later")
    return repo, delta_path


def test_confirm_preflight_accepts_unchanged_tag_blob_on_later_descendant(tmp_path):
    repo, delta = _tagged_repo(tmp_path)
    result = confirm_preflight(
        region="AL", milestone="review-1", delta_path=delta,
        expected_cohort="abc", repo_root=repo,
    )
    assert result["tag"] == "delta-al-abc"
    assert result["calibration_tag_commit"] == _git(repo, "rev-parse", "delta-al-abc^{commit}")
    assert result["calibration_delta_digest"] == file_digest(delta)


@pytest.mark.parametrize("failure", ["cohort", "tag", "blob", "ancestor"])
def test_confirm_preflight_refuses_unregistered_or_changed_calibration(tmp_path, failure):
    repo, delta = _tagged_repo(tmp_path)
    cohort = "different" if failure == "cohort" else "abc"
    if failure == "tag":
        document = json.loads(delta.read_text())
        document["tag"] = "missing-tag"
        delta.write_text(json.dumps(document) + "\n")
        _git(repo, "add", "results/al/delta.json")
        _git(repo, "commit", "-qm", "edited tag field")
    elif failure == "blob":
        delta.write_text(json.dumps({"region": "AL", "s": 0.1, "cohort": "abc",
                                    "tag": "delta-al-abc", "delta": 0.2}) + "\n")
        _git(repo, "add", "results/al/delta.json")
        _git(repo, "commit", "-qm", "changed delta")
    elif failure == "ancestor":
        _git(repo, "checkout", "--orphan", "unrelated")
        (repo / "unrelated.txt").write_text("other history\n")
        _git(repo, "add", "unrelated.txt")
        _git(repo, "commit", "-qm", "unrelated")
    with pytest.raises(ValueError):
        confirm_preflight(
            region="AL", milestone="review-1", delta_path=delta,
            expected_cohort=cohort, repo_root=repo,
        )


@pytest.mark.parametrize("ref", ["HEAD", "branch", "branch-named-like-tag", "-q"])
def test_confirm_preflight_accepts_only_region_calibration_tags(tmp_path, ref):
    repo, delta = _tagged_repo(tmp_path)
    tag = {"branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
           "branch-named-like-tag": "delta-al-branch"}.get(ref, ref)
    if ref == "branch-named-like-tag":
        _git(repo, "branch", tag)
    document = json.loads(delta.read_text())
    document["tag"] = tag
    delta.write_bytes(json.dumps(document).encode() + b"\n")
    _git(repo, "add", "results/al/delta.json")
    _git(repo, "commit", "-qm", "point delta at a non-tag ref")
    if ref == "branch-named-like-tag":
        _git(repo, "branch", "-f", tag, "HEAD")
    with pytest.raises(ValueError, match="tag"):
        confirm_preflight(region="AL", milestone="review-1", delta_path=delta, expected_cohort="abc", repo_root=repo)


def test_confirm_preflight_refuses_a_second_region_calibration_tag(tmp_path):
    repo, delta = _tagged_repo(tmp_path)
    _git(repo, "tag", "delta-al-second")
    with pytest.raises(ValueError, match="one calibration tag"):
        confirm_preflight(region="AL", milestone="review-1", delta_path=delta, expected_cohort="abc", repo_root=repo)


def test_confirm_preflight_rejects_byte_different_working_delta(tmp_path):
    repo, delta = _tagged_repo(tmp_path)
    delta.write_text('{ "cohort": "abc", "tag": "delta-al-abc" }\n')
    with pytest.raises(ValueError, match="working delta.json differs"):
        confirm_preflight(region="AL", milestone="review-1", delta_path=delta, expected_cohort="abc", repo_root=repo)


@pytest.mark.parametrize("milestone", [None, "", "   "])
def test_confirm_preflight_refuses_blank_milestone_before_the_delta(tmp_path, monkeypatch, milestone):
    from scripts import scoring_common

    monkeypatch.setattr(scoring_common, "verify_tagged_delta", lambda **k: pytest.fail("delta was read"))
    with pytest.raises(ValueError, match="requires --milestone"):
        confirm_preflight(region="AL", milestone=milestone, delta_path=tmp_path / "delta.json",
                          expected_cohort="abc", repo_root=tmp_path)


_COHORT = "c" * 64


def _evidence_repo(tmp_path, *, listed=True):
    """A tagless repo whose release evidence pins AL's delta.json to an original tag commit it lacks."""
    from conftest import evidence_entry, write_release_evidence

    repo, delta = tmp_path / "repo", tmp_path / "repo" / "results" / "al" / "delta.json"
    delta.parent.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    delta.write_bytes(json.dumps({"region": "AL", "s": 0.1, "delta": 0.2, "cohort": _COHORT,
                                  "tag": "delta-al-cccccccc"}).encode() + b"\n")
    write_release_evidence(repo, [evidence_entry(delta, repo)] if listed else [])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "release")
    return repo, delta


def test_tagless_replay_returns_the_original_calibration_stamp(tmp_path):
    repo, delta = _evidence_repo(tmp_path)
    result = confirm_preflight(region="AL", milestone="gate-1", delta_path=delta,
                               expected_cohort=_COHORT, repo_root=repo)
    assert result["calibration_tag_commit"] == "1" * 40
    assert result["calibration_delta_digest"] == file_digest(delta)
    assert result["calibration_source"] == "release evidence"


@pytest.mark.parametrize("failure, message", [
    ("delta", "bytes differ"), ("evidence", "bytes differ"), ("uncommitted evidence", "release-evidence"),
    ("cohort", "cohort does not match"), ("unlisted", "calibration tag"),
])
def test_tagless_replay_refuses_changed_delta_or_evidence_and_wrong_cohort(tmp_path, failure, message):
    repo, delta = _evidence_repo(tmp_path, listed=failure != "unlisted")
    evidence = repo / "results" / "release-evidence.json"
    if failure == "delta":
        delta.write_bytes(delta.read_bytes().replace(b'"s": 0.1', b'"s":  0.1'))
    elif failure in ("evidence", "uncommitted evidence"):
        document = json.loads(evidence.read_text())
        document["calibrations"][0]["delta_sha256"] = "0" * 64
        evidence.write_text(json.dumps(document))
    if failure in ("delta", "evidence"):
        _git(repo, "commit", "-qam", f"edit {failure}")
    with pytest.raises(ValueError, match=message):
        confirm_preflight(region="AL", milestone="gate-1", delta_path=delta,
                          expected_cohort="d" * 64 if failure == "cohort" else _COHORT, repo_root=repo)


@pytest.mark.parametrize("tags", [["source"], ["moved"], ["source", "second"], ["renamed"]])
def test_a_local_tag_is_checked_strictly_and_never_falls_back_to_the_evidence(tmp_path, tags):
    from conftest import evidence_entry, write_release_evidence

    repo, delta = _evidence_repo(tmp_path, listed=False)
    source = _git(repo, "rev-parse", "HEAD")
    write_release_evidence(repo, [evidence_entry(delta, repo, source_tag_commit=source)])
    _git(repo, "commit", "-qam", "publish evidence")
    names = {"source": ("delta-al-cccccccc", source), "moved": ("delta-al-cccccccc", "HEAD"),
             "second": ("delta-al-second", source), "renamed": ("delta-al-other", source)}
    for name, target in (names[tag] for tag in tags):
        _git(repo, "tag", name, target)
    check = dict(region="AL", milestone="gate-1", delta_path=delta, expected_cohort=_COHORT, repo_root=repo)
    if tags == ["source"]:
        assert confirm_preflight(**check)["calibration_source"] == "tag"
        return
    with pytest.raises(ValueError, match="tag"):
        confirm_preflight(**check)


@pytest.mark.parametrize("writer", ["run_gate", "phase3_calibrate"])
@pytest.mark.parametrize("freeze", ["tag", "evidence"])
def test_a_tag_or_published_calibration_freezes_both_calibration_writers(
    tmp_path, monkeypatch, capsys, writer, freeze,
):
    from conftest import write_release_evidence
    from scripts import phase3_calibrate, run_gate

    region = "AL" if writer == "run_gate" else "PHASE3"
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    write_release_evidence(repo, [{
        "region": region, "path": f"results/{region.lower()}/delta.json", "cohort": _COHORT,
        "tag": f"delta-{region.lower()}-cccccccc", "source_tag_commit": "1" * 40, "delta_sha256": "2" * 64,
    }] if freeze == "evidence" else [])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "release")
    if freeze == "tag":
        _git(repo, "tag", f"delta-{region.lower()}-cccccccc")
    if writer == "run_gate":
        monkeypatch.setattr(run_gate, "REPO_ROOT", repo)
        status = run_gate.main(["--config", "train/configs/al.toml", "calibrate"])
    else:
        status = phase3_calibrate.main(["calibrate", "--repo-root", str(repo),
                                        *[f"--system-run=r{index}" for index in range(5)]])
    assert status == 2 and "calibration is frozen" in capsys.readouterr().err
    assert not (repo / "results" / region.lower() / "delta.json").exists()


def _frames(cohort):
    base = {
        "dataset": "f2", "region": "AL", "split": "f2_dev",
        "degree_setting": "degree_given", "model": "gnn_a4",
        "mask_instance": "I00", "mask_regime": "c", "seed": 0,
        "run_id": "run", "cohort": cohort, "negatives_digest": "d" * 64,
    }
    rows = pd.DataFrame([{**base, "mrr": 1.0, "n_positives": 1}])
    positives = pd.DataFrame([{**base, "positive_index": 0, "root_id": 3,
                               "reciprocal_rank": 1.0, "both_masked": False}])
    return rows, positives


def test_result_store_allows_two_cohorts_and_refuses_reruns(tmp_path):
    first, first_positive = _frames("cohort-a")
    second, second_positive = _frames("cohort-b")
    append_results(tmp_path, first, first_positive)
    append_results(tmp_path, second, second_positive)
    rows, positives = load_result_tables(tmp_path)
    assert set(rows.cohort) == set(positives.cohort) == {"cohort-a", "cohort-b"}
    with pytest.raises(ValueError, match="already exist"):
        append_results(tmp_path, first, first_positive)


def test_result_store_ignores_a_call_interrupted_before_its_scores_publish(
    tmp_path, monkeypatch,
):
    from scripts import scoring_common as common

    rows, positives = _frames("cohort-a")
    replace = common.replace

    def fail_scores(src, dst):
        if Path(dst).name.endswith(".scores.parquet"):
            raise OSError("injected")
        replace(src, dst)

    monkeypatch.setattr(common, "replace", fail_scores)
    with pytest.raises(OSError, match="injected"):
        append_results(tmp_path, rows, positives)
    with pytest.raises(ValueError, match="no scoring results"):
        load_result_tables(tmp_path)
    monkeypatch.setattr(common, "replace", replace)
    append_results(tmp_path, rows, positives)
    stored, ranks = load_result_tables(tmp_path)
    assert len(stored) == len(ranks) == 1


def test_stored_scores_that_drift_from_their_ranks_are_refused_on_read(tmp_path):
    rows, positives = _frames("cohort-a")
    append_results(tmp_path, rows, positives)
    path = next((tmp_path / "scored").glob("*.scores.parquet"))
    stored = pd.read_parquet(path)
    stored["mrr"] = stored["mrr"] + 9e-6  # inside the F1-val history tolerance, outside the exact pair check
    stored.to_parquet(path)
    with pytest.raises(ValueError, match="MRR/count"):
        load_result_tables(tmp_path)


def test_result_pair_accepts_multiple_ranks_and_rejects_changed_aggregate_or_negatives():
    rows, positives = _frames("cohort-a")
    rows.loc[0, ["mrr", "n_positives"]] = [0.75, 2]
    positives = pd.concat([
        positives,
        positives.assign(positive_index=1, root_id=4, reciprocal_rank=0.5),
    ], ignore_index=True)
    _validate_score_positive_pair(rows, positives)
    with pytest.raises(ValueError, match="MRR/count"):
        _validate_score_positive_pair(rows.assign(mrr=0.7), positives)
    other_rows = rows.assign(model="logreg", run_id="baseline-logreg-s0",
                             negatives_digest="e" * 64)
    other_positives = positives.assign(model="logreg", run_id="baseline-logreg-s0",
                                       negatives_digest="e" * 64)
    with pytest.raises(ValueError, match="one nonempty negatives digest"):
        _validate_score_positive_pair(
            pd.concat([rows, other_rows], ignore_index=True),
            pd.concat([positives, other_positives], ignore_index=True),
        )


@pytest.mark.parametrize("change, message", [
    ({"negatives_digest": None}, "no matching positive ranks"),
    ({"negatives_digest": "e" * 64}, "different negatives digests"),
    ({"both_masked": True}, "MRR/count"),
    ({"reciprocal_rank": 0.5}, "MRR/count"),
])
def test_result_pair_compares_each_row_with_its_positive_ranks_without_row_loops(
    monkeypatch, change, message,
):
    rows, positives = _frames("cohort-a")
    rows = pd.concat([rows, rows.assign(mask_instance="I01")], ignore_index=True)
    positives = pd.concat([positives, positives.assign(mask_instance="I01")],
                          ignore_index=True)
    monkeypatch.setattr(pd.DataFrame, "iterrows", lambda self: pytest.fail("row loop"))
    _validate_score_positive_pair(rows, positives)
    positives.loc[1, list(change)] = list(change.values())
    with pytest.raises(ValueError, match=message):
        _validate_score_positive_pair(rows, positives)


def test_scored_suite_provenance_rejects_changed_suite_or_incomplete_coverage(
    tmp_path, monkeypatch,
):
    from scripts import scoring_common
    from src.evaluate import CommittedNegatives
    from src.suites import Suite
    from train.region_config import RegionConfig

    data_root = tmp_path / "data"
    suite_path = _frozen_suite(data_root, "f1_val", ["I0", "I1"], dataset="f1")
    suite = Suite.read(suite_path)
    negative_file = suite_path.with_name("f1_val.negatives.json")
    negatives = CommittedNegatives.read(negative_file)
    identity = {name: name for name in (
        "scoring_graph_digest", "scoring_feature_digest", "scoring_schema_digest",
        "scoring_types_digest", "scoring_sides_digest",
    )}
    monkeypatch.setattr(scoring_common, "load_scoring_inputs", lambda *a, **k: (
        object(), object(), object(), object(), None, suite, negatives,
    ))
    monkeypatch.setattr(scoring_common, "scoring_input_identity", lambda *a, **k: identity)
    monkeypatch.setattr(scoring_common, "_is_ancestor", lambda *a: True)
    rows = pd.DataFrame([
        {**identity, "scoring_commit": "c" * 40, "dataset": "f1", "region": "AL", "split": "f1_val", "model": "gnn_a4",
         "seed": 0, "run_id": "run", "mask_instance": instance,
         "suite_digest": suite.digest(), "negative_file_digest": file_digest(negative_file),
         "negatives_digest": f"{index + 1:064x}"}
        for index, instance in enumerate(("I0", "I1"))
    ])
    config = RegionConfig("AL", train={"epochs": 1})
    validate_scored_suite(rows, config, "f1_val", data_root)
    with pytest.raises(ValueError, match="frozen suite"):
        validate_scored_suite(rows.assign(suite_digest="changed"), config, "f1_val", data_root)
    with pytest.raises(ValueError, match="incomplete instance coverage"):
        validate_scored_suite(rows.iloc[:1], config, "f1_val", data_root)


def test_expert_provenance_rejects_fake_run_and_changed_f1_history(tmp_path):
    import torch
    from train.cohorts import cohort_digest

    run = tmp_path / "f1-al-s000-test"
    run.mkdir()
    manifest = {
        "run_id": run.name, "seed": 0, "best_epoch": 0,
        "device": "cpu", "precision": "fp32", "config": {"epochs": 1},
        "code_digest": "legacy-code", "graph_digest": "g", "feature_digest": "f",
        "schema_digest": "s", "suite_digest": "u",
    }
    torch.save({"history": {
        "val_mrr_epoch": [0], "val_mrr_by_instance": {"I0": [0.4], "I1": [0.5]},
    }}, run / "best.pt")
    rows = pd.DataFrame([
        {"model": "gnn_a4", "split": "f1_val", "seed": 0, "run_id": run.name,
         "mask_instance": instance, "mrr": value,
         "checkpoint_digest": file_digest(run / "best.pt")}
        for instance, value in (("I0", 0.4), ("I1", 0.5))
    ])
    cohort = cohort_digest(manifest)
    validate_expert_provenance(rows, [(run, manifest)], cohort)
    with pytest.raises(ValueError, match="do not identify run"):
        validate_expert_provenance(rows.assign(run_id="invented"), [(run, manifest)], cohort)
    changed = rows.copy()
    changed.loc[changed.mask_instance == "I1", "mrr"] = 0.9
    with pytest.raises(ValueError, match="do not match best.pt history"):
        validate_expert_provenance(changed, [(run, manifest)], cohort)


def test_legacy_cohort_checks_scientific_inputs_without_requiring_current_code_digest(
    tmp_path, monkeypatch,
):
    from types import SimpleNamespace
    from scripts import scoring_common
    from train import train as training
    from train.region_config import RegionConfig

    data_root = tmp_path / "data"
    negatives = data_root / "suites" / "f1" / "al" / "f1_val.negatives.json"
    negatives.parent.mkdir(parents=True)
    negatives.write_text("frozen\n", encoding="utf-8")
    graph = SimpleNamespace(digest=lambda: "graph")
    schema = SimpleNamespace(digest=lambda: "schema")
    suite = SimpleNamespace(digest=lambda: "suite")
    monkeypatch.setattr(scoring_common, "load_scoring_inputs", lambda *a, **k: (
        graph, object(), schema, None, None, suite, object()
    ))
    monkeypatch.setattr(training, "_feature_digest", lambda table, **_: "feature")
    manifest = {
        "run_id": "f1-al-s000-legacy", "seed": 0,
        "device": "cpu", "precision": "fp32", "config": {"epochs": 1},
        "code_digest": "old-training-code", "graph_digest": "graph",
        "feature_digest": "feature", "schema_digest": "schema", "suite_digest": "suite",
    }
    identity = validate_f1_cohort_inputs(
        RegionConfig("AL", train={"epochs": 1}), [manifest], data_root
    )
    assert identity["graph_digest"] == "graph"
    with pytest.raises(ValueError, match="graph_digest"):
        validate_f1_cohort_inputs(
            RegionConfig("AL", train={"epochs": 1}),
            [{**manifest, "graph_digest": "rebuilt"}], data_root,
        )
    assert manifest_region({**manifest, "run_id": "f1-cx-s000-legacy"}) == "CX"


def test_phase2_verdict_reads_the_registered_verdicts_at_head(tmp_path):
    repo = tmp_path / "verdicts"
    (repo / "results").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    registered = repo / "results" / "registered.json"
    registered.write_text(json.dumps({"phase2_verdicts": {"CX": "PASS", "LH": "FAIL", "SNP": "pending"}}),
                          encoding="utf-8")
    _git(repo, "add", "results/registered.json")
    _git(repo, "commit", "-qm", "record verdicts")
    assert phase2_verdict("CX", repo) == "PASS"
    assert phase2_verdict("lh", repo) == "FAIL"
    assert phase2_verdict("AL", repo) is None
    assert phase2_verdict("SNP", repo) is None
    registered.write_text(json.dumps({"phase2_verdicts": {"AL": "PASS"}}), encoding="utf-8")
    assert phase2_verdict("AL", repo) is None


@pytest.mark.parametrize("field", ["graph_digest", "feature_digest", "schema_digest",
                                  "suite_digest", "negative_file_digest"])
def test_baseline_gate_refuses_changed_dev_tuning_inputs(field):
    identity = {key: key for key in ("graph_digest", "feature_digest", "schema_digest",
                                   "suite_digest", "negative_file_digest")}
    fitted = baseline_fit_digest("cohort", identity, "logreg", 0, identity)
    rows = pd.DataFrame([{
        "model": "logreg", "split": split, "seed": 0, "run_id": "baseline-logreg-s0",
        "baseline_fit_inputs_digest": fitted,
    } for split in ("f2_dev", "f2_confirm")])
    validate_baseline_provenance(rows, "cohort", identity, identity)
    with pytest.raises(ValueError, match="fit inputs"):
        validate_baseline_provenance(rows, "cohort", identity, {**identity, field: "changed"})


def test_baseline_fit_views_are_pinned_independent_of_region_config():
    from train.scoring import baseline_fit_views

    drawn = []

    class Context:
        def draw_fit_view(self, rng, fraction):
            drawn.append((int(rng.integers(2**32)), fraction))
            return len(drawn)

    assert baseline_fit_views(Context()) == [1, 2, 3]
    assert [fraction for _, fraction in drawn] == [0.05] * 3
    assert len({value for value, _ in drawn}) == 3
    first = list(drawn)
    drawn.clear()
    baseline_fit_views(Context())
    assert drawn == first


_IDENTITY = {name: name for name in (
    "scoring_graph_digest", "scoring_feature_digest", "scoring_schema_digest",
    "scoring_types_digest", "scoring_sides_digest",
)}


@pytest.mark.parametrize("model", ["gnn_a4", "logreg"])
def test_scoring_identity_rejects_changed_inputs_for_expert_and_baseline(
    tmp_path, model,
):
    repo = _code_repo(tmp_path)
    rows = pd.DataFrame([{**_IDENTITY, "model": model, "scoring_commit": scoring_commit(repo)}])
    positives = rows.copy()
    _validate_scoring_identity(rows, positives, _IDENTITY, repo)
    with pytest.raises(ValueError, match="scoring_feature_digest"):
        _validate_scoring_identity(
            rows, positives, {**_IDENTITY, "scoring_feature_digest": "changed"}, repo
        )


def test_scoring_identity_rejects_score_positive_metadata_disagreement(tmp_path):
    repo = _code_repo(tmp_path)
    identity = {**_IDENTITY, "scoring_commit": scoring_commit(repo)}
    with pytest.raises(ValueError, match="positive ranks"):
        _validate_scoring_identity(
            pd.DataFrame([identity]),
            pd.DataFrame([{**identity, "scoring_feature_digest": "other"}]),
            _IDENTITY, repo,
        )


def _code_repo(tmp_path):
    repo = tmp_path / "code-repo"
    (repo / "src").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "src" / "evaluate.py").write_text("v1\n")
    _git(repo, "add", "src")
    _git(repo, "commit", "-qm", "code")
    return repo


def test_scored_rows_stay_valid_after_later_code_commits(tmp_path):
    repo = _code_repo(tmp_path)
    commit = scoring_commit(repo)
    assert commit == _git(repo, "rev-parse", "HEAD")
    (repo / "src" / "evaluate.py").write_text("v2\n")
    with pytest.raises(ValueError, match="before scoring"):
        scoring_commit(repo)
    _git(repo, "commit", "-qam", "later code")
    rows = pd.DataFrame([{**_IDENTITY, "scoring_commit": commit}])
    _validate_scoring_identity(rows, rows.copy(), _IDENTITY, repo)
    with pytest.raises(ValueError, match="not an ancestor"):
        _validate_scoring_identity(
            rows.assign(scoring_commit="0" * 40), rows, _IDENTITY, repo
        )


@pytest.mark.parametrize("case", ["unlisted", "pinned", "changed bytes", "shared commit"])
def test_original_score_files_skip_ancestry_only_by_pinned_bytes(tmp_path, case):
    from conftest import write_release_evidence

    repo = _code_repo(tmp_path)
    results = repo / "results" / "al"
    original = "f" * 40  # a development-repository commit, absent here
    for cohort in ("cohort-a", "cohort-b") if case == "shared commit" else ("cohort-a",):
        rows, positives = _frames(cohort)
        append_results(results, rows.assign(**_IDENTITY, scoring_commit=original),
                       positives.assign(**_IDENTITY, scoring_commit=original))
    files = sorted((results / "scored").glob("*.parquet"))
    pinned = files if case in ("pinned", "changed bytes") else files[:2]
    write_release_evidence(repo, scored_files=[] if case == "unlisted" else [
        {"path": path.relative_to(repo).as_posix(),
         "sha256": "0" * 64 if case == "changed bytes" else file_digest(path)} for path in pinned])
    _git(repo, "add", "results/release-evidence.json")
    _git(repo, "commit", "-qm", "publish evidence")
    rows, positives = load_result_tables(results, repo)
    if case == "pinned":
        _validate_scoring_identity(rows, positives, _IDENTITY, repo)
        return
    with pytest.raises(ValueError, match="not an ancestor"):
        _validate_scoring_identity(rows, positives, _IDENTITY, repo)


def test_scoring_input_identity_tracks_actual_feature_values(
    tmp_path, monkeypatch, cx_graph, cx_features, cx_schema,
):
    from types import SimpleNamespace
    from scripts import scoring_common
    from train.region_config import RegionConfig

    paths = SimpleNamespace(interim=tmp_path)
    (tmp_path / "f2_types.parquet").write_bytes(b"types")
    monkeypatch.setattr(scoring_common, "inspect_suite_identity", lambda *a: ("f2_dev", paths))
    suite = SimpleNamespace()
    inputs = (cx_graph, cx_features, cx_schema, object(), None, suite, object())
    before = scoring_input_identity(
        RegionConfig("CX", train={"epochs": 1}), tmp_path / "f2_dev.json",
        inputs=inputs,
    )
    changed = scoring_input_identity(
        RegionConfig("CX", train={"epochs": 1}), tmp_path / "f2_dev.json",
        inputs=(cx_graph, cx_features.perturb(0), cx_schema, object(), None, suite, object()),
    )
    assert changed["scoring_feature_digest"] != before["scoring_feature_digest"]


@pytest.mark.parametrize("variant", ["duplicate", "reordered"])
def test_suite_identity_fails_closed_for_noncanonical_metadata(tmp_path, variant):
    from train.region_config import RegionConfig

    data_root = tmp_path / "data"
    path = _suite_input(data_root, "f2_dev")
    document = path.read_text(encoding="utf-8")
    if variant == "duplicate":
        document = document.replace('"dataset":', '"name": "f2_dev", "dataset":', 1)
    else:
        parsed = json.loads(document)
        document = json.dumps({"dataset": parsed.pop("dataset"), **parsed})
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError, match="suite is unreadable|canonical Suite.write"):
        inspect_suite_identity(RegionConfig("AL", train={"epochs": 1}), path, data_root)


def test_suite_identity_accepts_a_suite_with_mask_exclusions(tmp_path):
    from train.region_config import RegionConfig

    data_root = tmp_path / "data"
    path = _suite_input(data_root, "f1_val", dataset="f1", region="OL",
                        mask_exclude_types=("L1",))
    name, _ = inspect_suite_identity(RegionConfig("OL", train={"epochs": 1}), path, data_root)
    assert name == "f1_val"


def test_confirm_cli_refuses_before_attempting_suite_input(tmp_path, monkeypatch, capsys):
    from scripts import score_run

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_dir.name, "region": "AL", "seed": 0,

        "device": "cpu", "precision": "fp32", "config": {"epochs": 1}, "code_digest": "c", "graph_digest": "g",
        "feature_digest": "f", "schema_digest": "s", "suite_digest": "u",
    }))
    monkeypatch.setattr(score_run, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(score_run, "load_scoring_inputs", lambda *a, **k: pytest.fail("suite was opened"))
    suite_path = _suite_input(tmp_path / "data", "f2_confirm")
    code = score_run.main([
        "--config", "train/configs/al.toml", "--run", str(run_dir),
        "--suite", str(suite_path), "--data-root", str(tmp_path / "data"),
    ])
    assert code == 2
    assert "requires --milestone" in capsys.readouterr().err
    assert not (tmp_path / "results" / "al").exists()


def test_score_run_returns_exit_2_for_non_integer_manifest_seed(
    tmp_path, monkeypatch, capsys,
):
    from scripts import score_run

    data_root = tmp_path / "data"
    suite = _suite_input(data_root, "f2_dev")
    run = tmp_path / "f1-al-s000-invalid"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "run_id": run.name, "region": "AL", "seed": None,
        "device": "cpu", "precision": "fp32", "config": {"epochs": 1},
        "code_digest": "c", "graph_digest": "g", "feature_digest": "f",
        "schema_digest": "s", "suite_digest": "u",
    }), encoding="utf-8")
    monkeypatch.setattr(score_run, "REPO_ROOT", tmp_path)
    assert score_run.main([
        "--config", "train/configs/al.toml", "--run", str(run),
        "--suite", str(suite), "--data-root", str(data_root),
    ]) == 2
    assert "integer seed" in capsys.readouterr().err
    assert not (tmp_path / "results" / "al").exists()


def test_confirm_cli_refuses_non_gate_seed_before_suite_input(tmp_path, monkeypatch, capsys):
    from scripts import score_run

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": run_dir.name, "region": "AL", "seed": 42,

        "device": "cpu", "precision": "fp32", "config": {"epochs": 1}, "code_digest": "c", "graph_digest": "g",
        "feature_digest": "f", "schema_digest": "s", "suite_digest": "u",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(score_run, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(score_run, "load_scoring_inputs", lambda *a, **k: pytest.fail("suite was opened"))
    suite_path = _suite_input(tmp_path / "data", "f2_confirm")
    code = score_run.main([
        "--config", "train/configs/al.toml", "--run", str(run_dir),
        "--suite", str(suite_path), "--data-root", str(tmp_path / "data"),
        "--milestone", "review-1",
    ])
    assert code == 2
    assert "non-gate seed 42" in capsys.readouterr().err
    assert not (tmp_path / "results" / "al").exists()


def test_score_run_returns_exit_2_for_corrupt_checkpoint(
    tmp_path, monkeypatch, capsys, cx_schema,
):
    from types import SimpleNamespace
    from scripts import score_run

    run_dir = tmp_path / "f1-al-s000-bad"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_dir.name, "region": "AL", "seed": 0,

        "device": "cpu", "precision": "fp32", "config": {"epochs": 1, "hidden": 8}, "code_digest": "c", "graph_digest": "g",
        "feature_digest": "f", "schema_digest": cx_schema.digest(),
        "suite_digest": "u",
    }), encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"broken checkpoint")
    monkeypatch.setattr(
        score_run, "inspect_suite_identity",
        lambda *a, **k: ("f2_dev", SimpleNamespace(results=Path("results/al"))),
    )
    monkeypatch.setattr(score_run, "load_scoring_inputs", lambda *a, **k: (
        object(), object(), cx_schema, None, None,
        SimpleNamespace(name="f2_dev"), object(),
    ))
    monkeypatch.setattr(score_run, "scoring_input_identity", lambda *a, **k: {})
    monkeypatch.setattr(score_run, "scoring_commit", lambda *a: "c" * 40)
    assert score_run.main([
        "--config", "train/configs/al.toml", "--run", str(run_dir),
        "--suite", str(tmp_path / "f2_dev.json"),
    ]) == 2
    assert "best.pt is unreadable or incompatible" in capsys.readouterr().err


def test_baseline_confirm_guard_runs_before_suite_input(tmp_path, monkeypatch, capsys):
    from scripts import run_baselines
    from train.cohorts import cohort_digest

    runs_dir = tmp_path / "runs" / "expert"
    runs_dir.mkdir(parents=True)
    manifest = {
        "run_id": "expert", "region": "AL", "seed": 0,

        "device": "cpu", "precision": "fp32", "config": {"epochs": 1}, "code_digest": "c", "graph_digest": "g",
        "feature_digest": "f", "schema_digest": "s", "suite_digest": "u",
    }
    (runs_dir / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(run_baselines, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_baselines, "load_scoring_inputs", lambda *a, **k: pytest.fail("suite was opened"))
    suite_path = _suite_input(tmp_path / "data", "f2_confirm")
    code = run_baselines.main([
        "--config", "train/configs/al.toml", "--runs-dir", str(tmp_path / "runs"),
        "--cohort", cohort_digest(manifest), "--suite", str(suite_path),
        "--data-root", str(tmp_path / "data"),
    ])
    assert code == 2
    assert "requires --milestone" in capsys.readouterr().err
    assert not (tmp_path / "results" / "al").exists()


@pytest.mark.parametrize("command", ["expert", "baseline"])
def test_confirm_alias_copy_is_rejected_by_both_scoring_clis(tmp_path, monkeypatch, capsys, command):
    from scripts import run_baselines, score_run
    from train.cohorts import cohort_digest

    data_root = tmp_path / "data"
    alias = _suite_input(data_root, "f2_dev", internal="f2_confirm")
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "expert"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_dir.name, "region": "AL", "seed": 42 if command == "expert" else 0,

        "device": "cpu", "precision": "fp32", "config": {"epochs": 1}, "code_digest": "c", "graph_digest": "g",
        "feature_digest": "f", "schema_digest": "s", "suite_digest": "u",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if command == "expert":
        monkeypatch.setattr(score_run, "REPO_ROOT", tmp_path)
        main = score_run.main
        args = ["--config", "train/configs/al.toml", "--run", str(run_dir),
                "--suite", str(alias), "--data-root", str(data_root)]
    else:
        monkeypatch.setattr(run_baselines, "REPO_ROOT", tmp_path)
        main = run_baselines.main
        args = ["--config", "train/configs/al.toml", "--runs-dir", str(runs_dir),
                "--cohort", cohort_digest(manifest), "--suite", str(alias),
                "--data-root", str(data_root)]
    assert main(args) == 2
    assert "name/dataset/region" in capsys.readouterr().err


def test_run_baselines_accepts_explicit_cohort_with_one_matching_manifest(tmp_path):
    from scripts.scoring_common import resolve_cohort_manifests
    from train.cohorts import cohort_digest
    from train.region_config import RegionConfig

    config = RegionConfig("AL", train={"epochs": 1})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "run", "region": "AL", "seed": 0,
        "device": "cpu", "precision": "fp32", "config": {"epochs": 1, "hidden": 8},
        "code_digest": "c", "graph_digest": "g", "feature_digest": "f",
        "schema_digest": "s", "suite_digest": "u",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cohort = cohort_digest(manifest)
    assert resolve_cohort_manifests(tmp_path, config, cohort)[0] == cohort
    with pytest.raises(ValueError, match="no matching region manifest"):
        resolve_cohort_manifests(tmp_path, config, "a" * 64)
    with pytest.raises(ValueError, match="requires configured seed manifests"):
        resolve_cohort_manifests(tmp_path, config, cohort, require_complete=True)


def test_run_baselines_records_actual_dev_fit_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import run_baselines
    from train.region_config import RegionConfig

    config = RegionConfig("AL", train={"epochs": 1})
    manifest = {"seed": 0, "config": {"gap_fraction": 0.07}}
    views = []
    paths = SimpleNamespace(suites=tmp_path / "suites", results=Path("results/al"))
    dev_suite = SimpleNamespace(resolve=lambda graph: ["dev-view"], digest=lambda: "dev-suite")
    score_suite = SimpleNamespace(name="f2_dev", digest=lambda: "score-suite")
    dev_inputs = (object(), object(), object(), object(), None, dev_suite, object())
    score_inputs = (object(), object(), object(), object(), None, score_suite, object())
    loaded = iter((dev_inputs, score_inputs))
    dev_identity = {
        "graph_digest": "dev-graph", "feature_digest": "dev-features",
        "schema_digest": "dev-schema", "suite_digest": "dev-suite",
        "negative_file_digest": "dev-negatives",
    }
    f1_identity = {
        "inputs": (object(), object(), None, None, None, object(), None),
        "graph_digest": "f1-graph", "feature_digest": "f1-features",
        "schema_digest": "f1-schema", "suite_digest": "f1-suite",
        "negative_file_digest": "f1-negatives",
    }
    captured = {}

    class Context:
        def __init__(self, *args):
            pass

        def draw_fit_view(self, rng, fraction):
            views.append(fraction)
            return "fit-view"

    class Baseline:
        seeds = (7, 8, 9)

        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

    def score(*args, seed, run_id, cohort, model_name, **kwargs):
        values = {"model": model_name, "seed": seed, "run_id": run_id, "cohort": cohort}
        return pd.DataFrame([values]), pd.DataFrame([values])

    def identify(*args, inputs=None, **kwargs):
        assert inputs is dev_inputs
        return dev_identity

    monkeypatch.setattr(run_baselines, "load_region_config", lambda path: config)
    monkeypatch.setattr(run_baselines, "inspect_suite_identity", lambda *a: ("f2_dev", paths))
    monkeypatch.setattr(run_baselines, "resolve_cohort_manifests",
                        lambda *a, **k: ("cohort", [(tmp_path / "run", manifest)]))
    monkeypatch.setattr(run_baselines, "validate_f1_cohort_inputs", lambda *a: f1_identity)
    monkeypatch.setattr(run_baselines, "load_scoring_inputs", lambda *a: next(loaded))
    monkeypatch.setattr(run_baselines, "dev_fit_identity", identify)
    score_identity = {
        "scoring_graph_digest": "score-graph", "scoring_feature_digest": "score-features",
        "scoring_schema_digest": "score-schema", "scoring_types_digest": "score-types",
        "scoring_sides_digest": "absent",
    }
    monkeypatch.setattr(run_baselines, "scoring_input_identity", lambda *a, **k: score_identity)
    monkeypatch.setattr(run_baselines, "scoring_commit", lambda *a: "c" * 40)
    monkeypatch.setattr(run_baselines, "F1TrainingContext", Context)
    monkeypatch.setattr(run_baselines, "apply_mask", lambda graph, view: view)
    monkeypatch.setattr(run_baselines, "BASELINES", {"logreg": Baseline})
    monkeypatch.setattr(run_baselines, "score_baseline", score)
    monkeypatch.setattr(run_baselines, "file_digest", lambda path: "scored-negatives")
    monkeypatch.setattr(run_baselines, "append_results",
                        lambda path, rows, positives: captured.update(rows=rows, positives=positives))
    monkeypatch.setattr(run_baselines, "REPO_ROOT", tmp_path)

    assert run_baselines.main([
        "--config", "ignored.toml", "--suite", str(tmp_path / "f2_dev.json"),
    ]) == 0
    assert views == [0.05] * 3
    expected = {
        baseline_fit_digest("cohort", f1_identity, "logreg", seed, dev_identity)
        for seed in (7, 8, 9)
    }
    assert set(captured["rows"].baseline_fit_inputs_digest) == expected
    assert set(captured["positives"].baseline_fit_inputs_digest) == expected
    assert set(captured["rows"].scoring_commit) == {"c" * 40}
    for field, value in score_identity.items():
        assert set(captured["rows"][field]) == {value}
        assert set(captured["positives"][field]) == {value}


@pytest.mark.parametrize("command", ["expert", "baseline"])
def test_scoring_clis_return_exit_2_for_malformed_config(monkeypatch, capsys, command):
    from scripts import run_baselines, score_run

    module = score_run if command == "expert" else run_baselines
    monkeypatch.setattr(module, "load_region_config",
                        lambda path: (_ for _ in ()).throw(TypeError("malformed config")))
    args = ["--config", "bad.toml", "--suite", "f2_dev.json"]
    args += ["--run", "run"] if command == "expert" else []
    assert module.main(args) == 2
    assert "malformed config" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["baselines", "check"])
def test_cli_runs_dir_defaults_under_flyarch_data_root(tmp_path, monkeypatch, capsys, command):
    from scripts import check_f1_val, run_baselines, scoring_common

    module = run_baselines if command == "baselines" else check_f1_val
    seen = {}

    def resolve(runs_dir, *args, **kwargs):
        seen["runs_dir"] = runs_dir
        raise ValueError("stop")

    monkeypatch.setenv("FLYARCH_DATA_ROOT", str(tmp_path / "env-data"))
    monkeypatch.setattr(run_baselines if command == "baselines" else scoring_common,
                        "resolve_cohort_manifests", resolve)
    if command == "baselines":
        monkeypatch.setattr(module, "inspect_suite_identity", lambda *a: ("f2_dev", None))
    assert module.main(["--config", "train/configs/al.toml", "--suite", "f2_dev.json"]
                       if command == "baselines" else ["--config", "train/configs/al.toml"]) == 2
    assert seen["runs_dir"] == tmp_path / "env-data" / "runs"


def test_cohort_resolution_rejects_wrong_region_legacy_run_id(tmp_path):
    from train.cohorts import cohort_digest
    from train.region_config import RegionConfig

    run_dir = tmp_path / "f1-cx-s000-legacy"
    run_dir.mkdir()
    manifest = {
        "run_id": run_dir.name, "seed": 0,
        "device": "cpu", "precision": "fp32", "config": {"epochs": 1}, "code_digest": "old",
        "graph_digest": "g", "feature_digest": "f", "schema_digest": "s",
        "suite_digest": "u",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="no matching region manifest"):
        resolve_cohort_manifests(
            tmp_path, RegionConfig("AL", train={"epochs": 1}), cohort_digest(manifest)
        )


@pytest.mark.parametrize("stamp", ["current", "stale"])
def test_run_gate_passes_only_requested_cohort_to_gate(tmp_path, monkeypatch, capsys, stamp):
    from types import SimpleNamespace

    from conftest import _bypass_result_verification
    from scripts import run_gate, scoring_common

    results_dir = tmp_path / "results" / "al"
    results_dir.mkdir(parents=True)
    rows = []
    for cohort, region in (("selected", "AL"), ("other", "CX")):
        for split in ("f2_dev", "f2_confirm"):
            rows.append({
                "dataset": "f2", "region": region, "split": split,
                "degree_setting": "degree_given", "model": "gnn_a4",
                "mask_instance": f"{cohort}-{split}", "mask_regime": "c",
                "seed": 0, "mrr": 0.5, "run_id": cohort, "cohort": cohort,
                "milestone": "review-1" if split == "f2_confirm" else None,
                "calibration_tag_commit": "c" * 40 if stamp == "current" else "0" * 40,
                "calibration_delta_digest": "d" * 64,
            })
    rows.append({**rows[0], "seed": 4, "run_id": "calibration-only"})
    raw = pd.DataFrame(rows)
    seen = {}

    def evaluate(frame, prereg):
        seen["cohorts"] = set(frame.cohort) if "cohort" in frame else set()
        seen["seeds"] = set(frame.seed)
        return SimpleNamespace(passed=True, summary=lambda: "PASS")

    monkeypatch.setattr(run_gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_gate, "verify_tagged_delta", lambda **kwargs: {
        "delta": 0.2, "calibration_tag_commit": "c" * 40, "calibration_delta_digest": "d" * 64,
    })
    _bypass_result_verification(monkeypatch, raw)
    monkeypatch.setattr(scoring_common, "resolve_cohort_manifests", lambda *a, **k: (
        "selected", [(tmp_path / "selected", {"seed": 0})]
    ))
    monkeypatch.setattr(run_gate, "evaluate_gate", evaluate)
    status = run_gate.main(["--config", "train/configs/al.toml", "gate", "--cohort", "selected"])
    if stamp == "stale":
        assert status == 2 and "current tagged delta" in capsys.readouterr().err
        return
    assert status == 0
    assert seen["cohorts"] == set()
    assert seen["seeds"] == {0}
    output = capsys.readouterr().out
    assert "gate=PASS" in output
    assert f"delta=0.20 ({'c' * 40})" in output


@pytest.mark.parametrize("confirm_scored", [False, True])
def test_run_gate_calibrates_from_exactly_five_f1_val_seeds(tmp_path, monkeypatch, confirm_scored):
    from conftest import _bypass_result_verification
    from scripts import run_gate, scoring_common

    results_dir = tmp_path / "results" / "al"
    results_dir.mkdir(parents=True)
    rows = []
    for index in range(5):
        for seed in range(5):
            rows.append({
                "dataset": "f1", "region": "AL", "split": "f1_val",
                "degree_setting": "degree_given", "model": "gnn_a4",
                "mask_instance": f"F1-{index:02}", "mask_regime": "b",
                "seed": seed, "mrr": 0.4 + 0.005 * seed + 0.01 * index,
                "run_id": f"al-s{seed}", "cohort": "cohort-a",
            })
    if confirm_scored:
        rows.append({**rows[0], "dataset": "f2", "split": "f2_confirm", "cohort": "cohort-old"})
    raw = pd.DataFrame(rows)
    monkeypatch.setattr(run_gate, "REPO_ROOT", tmp_path)
    _bypass_result_verification(monkeypatch, raw)
    monkeypatch.setattr(run_gate, "load_result_tables", lambda path: (raw, raw.copy()))
    monkeypatch.setenv("FLYARCH_DATA_ROOT", str(tmp_path / "env-data"))
    requested = {}
    def resolve(*args, **kwargs):
        requested["runs_dir"] = args[0]
        requested["cohort"] = args[2]
        return "cohort-a", [
            (tmp_path / f"al-s{seed}", {"seed": seed}) for seed in range(5)
        ]
    monkeypatch.setattr(scoring_common, "resolve_cohort_manifests", resolve)
    monkeypatch.setattr(run_gate, "refuse_recalibration",
                        lambda region, root: requested.update(frozen_check=region))
    status = run_gate.main([
        "--config", "train/configs/al.toml", "calibrate", "--cohort", "cohort-a"
    ])
    if confirm_scored:
        assert status == 2 and not (results_dir / "delta.json").exists()
        return
    assert status == 0
    assert requested["runs_dir"] == tmp_path / "env-data" / "runs"
    assert requested["cohort"] == "cohort-a" and requested["frozen_check"] == "AL"
    assert b"\r\n" not in (results_dir / "delta.json").read_bytes()
    delta = json.loads((results_dir / "delta.json").read_text(encoding="utf-8"))
    assert delta["cohort"] == "cohort-a"
    assert delta["run_ids"] == [f"al-s{seed}" for seed in range(5)]


def test_run_gate_refuses_to_overwrite_frozen_delta(tmp_path, monkeypatch):
    from scripts import run_gate

    results_dir = tmp_path / "results" / "al"
    results_dir.mkdir(parents=True)
    rows = []
    for index in range(5):
        for seed in range(5):
            rows.append({"dataset": "f1", "region": "AL", "split": "f1_val",
                         "degree_setting": "degree_given", "model": "gnn_a4",
                         "mask_instance": f"F1-{index:02}", "mask_regime": "b",
                         "seed": seed, "mrr": 0.4 + 0.005 * seed + 0.01 * index,
                         "run_id": f"al-s{seed}", "cohort": "cohort-a"})
    pd.DataFrame(rows).to_csv(results_dir / "scores.csv", index=False)
    frozen = b'{"region":"AL","delta":0.1,"cohort":"cohort-a","tag":"delta-al-old"}\n'
    (results_dir / "delta.json").write_bytes(frozen)
    monkeypatch.setattr(run_gate, "REPO_ROOT", tmp_path)
    assert run_gate.main(["--config", "train/configs/al.toml", "calibrate"]) == 2
    assert (results_dir / "delta.json").read_bytes() == frozen
