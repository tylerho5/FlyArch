import pandas as pd
import pytest

from scripts.phase3_reference_mrr import reference_mrr


def rows(**over):
    base = [{"split": "f2_confirm", "model": "gnn_a4", "run_id": r, "mask_instance": i, "mrr": 0.5 + 0.1 * k}
            for k, r in enumerate(["a", "b", "c"]) for i in ("i1", "i2")]
    base += [{"split": "f2_confirm", "model": "xgboost", "run_id": "x", "mask_instance": "i1", "mrr": 0.1},
             {"split": "f2_dev", "model": "gnn_a4", "run_id": "a", "mask_instance": "i9", "mrr": 0.9}]
    return pd.DataFrame(base)


def test_reference_keeps_only_the_gnn_confirm_rows_by_run_and_instance():
    ref = reference_mrr(rows(), "gnn_a4", 3)
    assert set(ref) == {"a", "b", "c"} and set(ref["a"]) == {"i1", "i2"}
    assert ref["c"]["i1"] == pytest.approx(0.7)


def test_reference_refuses_a_missing_seed_or_a_repeated_instance():
    with pytest.raises(ValueError, match="expected 3 gate runs"):
        reference_mrr(rows().query("run_id != 'c'"), "gnn_a4", 3)
    dup = pd.concat([rows(), rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="more than one row"):
        reference_mrr(dup, "gnn_a4", 3)


def _confirm(**over):
    frame = pd.DataFrame([{"milestone": "gate-1", "calibration_tag_commit": "c1", "calibration_delta_digest": "d1"}] * 2)
    return frame.assign(**over)


def test_confirm_checks_mirror_run_gate():
    from scripts.phase3_reference_mrr import check_confirm_rows

    doc = {"calibration_tag_commit": "c1", "calibration_delta_digest": "d1"}
    assert check_confirm_rows(_confirm(), _confirm(), doc) == "gate-1"
    with pytest.raises(ValueError, match="one nonempty milestone"):
        check_confirm_rows(_confirm(milestone=" "), _confirm(), doc)
    with pytest.raises(ValueError, match="current tagged delta"):
        check_confirm_rows(_confirm(calibration_tag_commit="old"), _confirm(), doc)
    with pytest.raises(ValueError, match="current tagged delta"):
        check_confirm_rows(_confirm(), _confirm(calibration_delta_digest="old"), doc)


def test_registry_cohort_reads_the_gate_rows():
    from scripts.phase3_reference_mrr import registry_cohort

    assert registry_cohort("AL").startswith("94a5924d")
    with pytest.raises(ValueError, match="expected exactly one"):
        registry_cohort("NOPE")


def test_cli_runs_the_confirm_checks_and_the_registry_cohort_check(monkeypatch, tmp_path, capsys):
    """main() must call check_confirm_rows and registry_cohort, not just define them."""
    from scripts import phase3_reference_mrr as mod

    good = rows().assign(milestone="gate-1", calibration_tag_commit="c1", calibration_delta_digest="d1")
    doc = {"calibration_tag_commit": "c1", "calibration_delta_digest": "d1"}
    monkeypatch.setattr(mod, "verify_tagged_delta", lambda **kw: doc)
    monkeypatch.setattr(mod, "load_verified_results", lambda *a, **kw: (None, None, good, good))
    monkeypatch.setattr(mod, "check_confirm_rows",
                        lambda *a, **kw: (_ for _ in ()).throw(ValueError("stamp refused")))
    argv = ["--config", "train/configs/al.toml", "--cohort", "x", "--data-root", str(tmp_path)]
    assert mod.main(argv) == 2 and "stamp refused" in capsys.readouterr().err

    monkeypatch.undo()
    monkeypatch.setattr(mod, "verify_tagged_delta", lambda **kw: doc)
    monkeypatch.setattr(mod, "load_verified_results", lambda *a, **kw: (None, None, good, good))
    monkeypatch.setattr(mod, "check_confirm_rows", lambda *a, **kw: "gate-1")
    assert mod.main(argv) == 2 and "registry's cohort" in capsys.readouterr().err
