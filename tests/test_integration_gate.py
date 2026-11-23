"""Seams of plan 07: the conjunction, its refusals, and the statistics' unit."""

import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from src.integration_gate import (  # noqa: E402
    boundary_comparison, calibrate_delta_p3, integrated_claim, regional_guard, required_opponents,
)

IDS = [f"i{k:02d}" for k in range(20)]
EXPERTS = ["AL", "CX"]


def _mrr(base, lift=0.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return {i: float(base * 2 ** (lift + rng.normal(scale=noise))) for i in IDS}


def _components(regional_lift=0.0, boundary_lift=0.3, system_is_b0=False):
    regional = [regional_guard(r, _mrr(0.6, regional_lift, seed=k), [_mrr(0.6, seed=10 + k + s) for s in range(3)],
                               delta=0.05) for k, r in enumerate(EXPERTS)]
    boundary = [boundary_comparison(name, _mrr(0.5, boundary_lift, seed=20 + k), _mrr(0.5, seed=40 + k), 0.05, n_instances=20)
                for k, name in enumerate(sorted(required_opponents(EXPERTS, system_is_b0=system_is_b0)))]
    return regional, boundary


def test_a_matching_system_passes_and_one_failing_component_fails_the_claim():
    regional, boundary = _components()
    assert integrated_claim(regional, boundary, gated_regions=EXPERTS, experts=EXPERTS, system_is_b0=False)[0]
    worse = regional_guard("CX", _mrr(0.6, -0.3, seed=1), [_mrr(0.6, seed=11 + s) for s in range(3)], delta=0.05)
    passed, failures = integrated_claim([regional[0], worse], boundary, gated_regions=EXPERTS, experts=EXPERTS,
                                        system_is_b0=False)
    assert not passed and failures[0].startswith("regional CX")
    flat = boundary_comparison("B0", _mrr(0.5, 0.0, seed=3), _mrr(0.5, seed=4), 0.05, n_instances=20)
    others = [c for c in boundary if c.name != "B0"]
    assert not integrated_claim(regional, [*others, flat], gated_regions=EXPERTS, experts=EXPERTS,
                                system_is_b0=False)[0]


def test_non_inferiority_tolerates_noise_within_delta_but_zero_margin_does_not():
    system, ref = _mrr(0.6, -0.02, seed=5), [_mrr(0.6, seed=6 + s) for s in range(3)]
    assert regional_guard("AL", system, ref, delta=0.1).passed
    assert not regional_guard("AL", system, ref, delta=None).passed  # no delta: the margin is 0


def test_refusals():
    regional, boundary = _components()
    with pytest.raises(ValueError, match="missing"):
        integrated_claim(regional, boundary[1:], gated_regions=EXPERTS, experts=EXPERTS, system_is_b0=False)
    with pytest.raises(ValueError, match="regional guards must be exactly"):
        integrated_claim(regional[:1], boundary, gated_regions=EXPERTS, experts=EXPERTS, system_is_b0=False)
    with pytest.raises(ValueError, match="extra"):  # a failing guard for an ungated region must not decide
        integrated_claim(regional, boundary, gated_regions=EXPERTS[:1], experts=EXPERTS, system_is_b0=False)
    with pytest.raises(ValueError, match="wrong list"):
        integrated_claim(regional, [*boundary, regional[0]], gated_regions=EXPERTS, experts=EXPERTS,
                         system_is_b0=False)
    with pytest.raises(ValueError, match="exactly 3 seeds"):
        regional_guard("AL", _mrr(0.6), [_mrr(0.6), _mrr(0.6)], delta=0.05)
    with pytest.raises(ValueError, match="exactly 20 instances"):
        boundary_comparison("B0", {"i00": 0.5}, {"i00": 0.4}, 0.05, n_instances=20)
    with pytest.raises(ValueError, match="calibrated"):
        boundary_comparison("B0", _mrr(0.5), _mrr(0.5), None)


def test_b0_as_the_system_is_not_its_own_opponent():
    assert "B0" not in required_opponents(EXPERTS, system_is_b0=True)
    assert {"B0", "seed_control", "baseline:xgboost", "expert:CX"} <= required_opponents(EXPERTS, system_is_b0=False)
    regional, boundary = _components(system_is_b0=True)
    assert integrated_claim(regional, boundary, gated_regions=EXPERTS, experts=EXPERTS, system_is_b0=True)[0]


def test_the_statistic_is_per_instance_and_uses_the_three_seed_mean():
    system = {i: 0.5 for i in IDS}
    seeds = [{i: v for i in IDS} for v in (0.4, 0.5, 0.6)]
    guard = regional_guard("AL", system, seeds, delta=0.0)
    assert guard.r_bar == pytest.approx(0.0) and guard.ci_low == pytest.approx(0.0)  # the seeds' mean is 0.5


def test_delta_p3_follows_section_3():
    rng = np.random.default_rng(7)
    ids = [f"v{k}" for k in range(5)]
    realizations = [{i: float(0.5 * 2 ** rng.normal(scale=0.02)) for i in ids} for _ in range(5)]
    s, delta = calibrate_delta_p3(realizations)
    logs = np.log2([[r[i] for r in realizations] for i in ids])
    assert s == pytest.approx(np.median(logs.std(axis=1, ddof=1))) and delta == round(2 * s, 2)
    with pytest.raises(ValueError, match="exactly 5 realizations"):
        calibrate_delta_p3(realizations[:4])


def _git(path, *args):
    import subprocess

    result = subprocess.run(["git", *args], cwd=path, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().strip()


def _gate_repo(tmp_path, *, delta=True, tagged=True):
    """A repo with a two-region registry and optionally a tagged delta_P3."""
    repo = tmp_path / "repo"
    (repo / "results" / "phase3").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "results" / "experts.csv").write_text(REGISTRY, encoding="utf-8")
    (repo / "results" / "experts-clean.csv").write_text(CLEAN_REGISTRY, encoding="utf-8")
    if delta:
        (repo / "results" / "phase3" / "delta.json").write_bytes(json.dumps(
            {"region": "PHASE3", "system": "B1", "s": 0.025, "delta": 0.05, "cohort": "sys", "tag": "delta-phase3-sys"}).encode() + b"\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "setup")
    if delta and tagged:
        _git(repo, "tag", "delta-phase3-sys")
    return repo


def _runs(region):
    return [f"f1-{region.lower()}-s00{s}" for s in range(3)]


def _clean_runs(region):
    return [f"f1-{region.lower()}-c00{s}" for s in range(3)]


#: The Phase 2 registry: the cohorts each regional guard is scored against.
REGISTRY = "region,gate_seed,run_id,seed,cohort\n" + "".join(
    f"{r},{'true' if s < 3 else 'false'},f1-{r.lower()}-s00{s},{s},c{r.lower()}\n" for r in EXPERTS for s in range(5))
#: The clean cohort, the primary boundary opponent (registered 2026-11-07): other runs, other cohorts.
CLEAN_REGISTRY = "region,gate_seed,run_id,seed,cohort\n" + "".join(
    f"{r},{'true' if s < 3 else 'false'},f1-{r.lower()}-c00{s},{s},clean-{r.lower()}\n"
    for r in EXPERTS for s in range(5))


def _spec(tmp_path, *, opponents_clean=True, **overrides):
    regional = {r: {"system": _mrr(0.6, 0.1, seed=k),
                    "reference": {run: _mrr(0.6, seed=10 + k + s) for s, run in enumerate(_runs(r))}}
                for k, r in enumerate(EXPERTS)}
    def f2(scores):  # the boundary confirm suite holds 16 instances
        return {i: v for i, v in scores.items() if i in IDS[:16]}

    expert_runs = _clean_runs if opponents_clean else _runs
    boundary = {name: ({run: f2(_mrr(0.5, seed=40 + k + s)) for s, run in enumerate(expert_runs(name[7:]))}
                       if name.startswith("expert:") else f2(_mrr(0.5, seed=40 + k)))
                for k, name in enumerate(sorted(required_opponents(EXPERTS, system_is_b0=False)))}
    spec = {"milestone": "m1", "system": "B1", "system_cohort": "sys", "regional": regional,
            "boundary_system": f2(_mrr(0.5, 0.3, seed=20)), "boundary": boundary}
    spec.update(overrides)
    path = tmp_path / "c.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _run(repo, components, *extra):
    from scripts.phase3_gate import main

    return main(["--components", str(components), "--repo-root", str(repo), *extra])


def test_gate_runs_on_a_milestone_with_a_tagged_delta(tmp_path):
    assert _run(_gate_repo(tmp_path), _spec(tmp_path)) == 0


def test_gate_report_names_both_registries_and_labels_a_sensitivity_run(tmp_path):
    repo = _gate_repo(tmp_path)
    out = tmp_path / "gate.md"
    assert _run(repo, _spec(tmp_path), "--out", str(out)) == 0
    text = out.read_text(encoding="utf-8")
    reference_blob = _git(repo, "rev-parse", "HEAD:results/experts.csv")
    opponents_blob = _git(repo, "rev-parse", "HEAD:results/experts-clean.csv")
    assert f"`results/experts.csv` ({reference_blob})" in text
    assert f"`results/experts-clean.csv` ({opponents_blob}, primary)" in text
    sensitivity = tmp_path / "sensitivity.md"
    assert _run(repo, _spec(tmp_path, opponents_clean=False), "--opponents-registry", "results/experts.csv",
                "--out", str(sensitivity)) == 0
    assert "labelled sensitivity comparison" in sensitivity.read_text(encoding="utf-8")


def test_gate_refuses_an_opponents_registry_with_different_regions(tmp_path):
    repo = _gate_repo(tmp_path)
    (repo / "results" / "experts-clean.csv").write_text(CLEAN_REGISTRY.split("CX")[0], encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "drop CX")
    with pytest.raises(SystemExit, match="regions differ"):
        _run(repo, _spec(tmp_path))


@pytest.mark.parametrize("failure, message", [
    ("blank milestone", "nonempty milestone"), ("no delta", "calibrate delta_P3"), ("untagged", "calibration tag"),
])
def test_gate_needs_a_milestone_and_a_frozen_delta_p3(tmp_path, failure, message):
    repo = _gate_repo(tmp_path, delta=failure != "no delta", tagged=failure != "untagged")
    with pytest.raises(SystemExit, match=message):
        _run(repo, _spec(tmp_path, **({"milestone": "  "} if failure == "blank milestone" else {})))


def test_gate_takes_regions_and_opponents_from_the_registry_not_the_components(tmp_path):
    repo = _gate_repo(tmp_path)
    (repo / "results" / "experts.csv").write_text(REGISTRY.split("CX")[0])  # uncommitted: ignored
    good = json.loads(_spec(tmp_path).read_text())
    dropped_region = {"regional": {"AL": good["regional"]["AL"]}}  # CX failed its guard, so it was left out
    dropped_expert = {"boundary": {k: v for k, v in good["boundary"].items() if k != "expert:CX"}}
    typed_delta = {"regional": {**good["regional"], "AL": {**good["regional"]["AL"], "delta": 9.0}}}
    renamed = {"system": "B0"}  # B0 would drop itself as an opponent; the tagged record says B1
    for override, message in ((dropped_region, "gated regions"), (dropped_expert, "opponent set"),
                              (typed_delta, "may not carry a delta"), (renamed, "calibrated for 'B1'")):
        with pytest.raises(SystemExit, match=message):
            _run(repo, _spec(tmp_path, **override))


def test_the_guard_reads_the_phase_2_registry_and_the_opponents_read_the_clean_one(tmp_path):
    """The two roles are separate: swapping either registry's runs for the other's is refused."""
    repo, good = _gate_repo(tmp_path), json.loads(_spec(tmp_path).read_text())
    guard = {**good["regional"]["AL"],
             "reference": {run: v for run, v in zip(_clean_runs("AL"), good["regional"]["AL"]["reference"].values())}}
    swapped_guard = {"regional": {**good["regional"], "AL": guard}}
    swapped_opponent = {"boundary": {**good["boundary"], "expert:CX": {
        run: v for run, v in zip(_runs("CX"), good["boundary"]["expert:CX"].values())}}}
    for override in (swapped_guard, swapped_opponent):
        with pytest.raises(SystemExit, match="registered gate runs"):
            _run(repo, _spec(tmp_path, **override))
    # Both are read from HEAD: an uncommitted edit to the clean registry changes no opponent.
    (repo / "results" / "experts-clean.csv").write_text(CLEAN_REGISTRY.split("CX")[0], encoding="utf-8")
    assert _run(repo, _spec(tmp_path)) == 0


@pytest.mark.parametrize("case, message", [("untagged", "calibration tag"), ("deleted", "unreadable")])
def test_a_regional_delta_without_a_tag_or_its_published_file_is_refused(tmp_path, case, message):
    from conftest import evidence_entry, write_release_evidence

    repo = _gate_repo(tmp_path)
    delta = repo / "results" / "al" / "delta.json"
    delta.parent.mkdir()
    delta.write_bytes(json.dumps(
        {"region": "AL", "s": 0.1, "delta": 0.2, "cohort": "cal" if case == "untagged" else "c" * 64,
         "tag": "delta-al-cccccccc"}).encode() + b"\n")
    if case == "deleted":  # published in the evidence file, then removed: never margin 0
        write_release_evidence(repo, [evidence_entry(delta, repo)])
        delta.unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "regional delta")
    with pytest.raises(SystemExit, match=message):
        _run(repo, _spec(tmp_path))


def test_one_weak_seed_cannot_stand_in_for_the_registered_three(tmp_path):
    repo, good = _gate_repo(tmp_path), json.loads(_spec(tmp_path).read_text())
    weak = good["regional"]["AL"]["reference"]["f1-al-s000"]
    repeated = {"regional": {**good["regional"], "AL": {**good["regional"]["AL"],
                                                        "reference": {"a": weak, "b": weak, "c": weak}}}}
    swapped = {k: v for k, v in good["boundary"]["expert:CX"].items() if k != "f1-cx-s000"}
    expert = {"boundary": {**good["boundary"], "expert:CX": {"f1-cx-s003": weak, **swapped}}}  # a non-gate seed
    for override in (repeated, expert):
        with pytest.raises(SystemExit, match="registered gate runs"):
            _run(repo, _spec(tmp_path, **override))


def test_invalid_raw_mrrs_are_refused_before_averaging():
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        boundary_comparison("B0", {i: 2.0 for i in IDS}, _mrr(0.5), 0.05, n_instances=20)
    seeds = [{i: -0.5 for i in IDS}, {i: 0.5 for i in IDS}, {i: 0.5 for i in IDS}]
    with pytest.raises(ValueError, match="reference seed 0"):  # the mean of these three looks fine
        regional_guard("AL", {i: 0.5 for i in IDS}, seeds, delta=0.05)


def test_the_ci_lower_bound_decides_even_when_the_mean_and_sign_test_pass():
    # Every r_i is positive (sign p ~ 1e-6) and the mean, 0.107, clears delta_P3 = 0.05, but three
    # outliers carry it: the bootstrap's lower bound falls below 0.05, so the comparison fails.
    ratios = [0.02] * 17 + [0.6] * 3
    system = {i: 0.5 * 2 ** r for i, r in zip(IDS, ratios)}
    c = boundary_comparison("B0", system, {i: 0.5 for i in IDS}, 0.05, n_instances=20)
    assert c.r_bar > 0.05 and c.p_sign < 0.01 and c.ci_low < 0.05 and not c.passed


def test_the_boundary_comparison_reads_the_sixteen_instance_f2_suite_by_default():
    ids16 = IDS[:16]
    sys16, ref16 = {i: 0.6 for i in ids16}, {i: 0.5 for i in ids16}
    assert boundary_comparison("B0", sys16, ref16, 0.05).kind == "boundary"
    with pytest.raises(ValueError, match="exactly 16 instances"):
        boundary_comparison("B0", {i: 0.6 for i in IDS}, {i: 0.5 for i in IDS}, 0.05)


def test_the_shipped_components_reproduce_the_registered_phase_3_fail(capsys):
    """The registered Phase 3 gate-1 verdict, replayed from tracked inputs and the release evidence."""
    from scripts.phase3_gate import REPO_ROOT, main

    if not (REPO_ROOT / ".git").exists():
        pytest.skip("needs a Git checkout of the repository")
    assert main(["--components", str(REPO_ROOT / "results" / "phase3" / "confirm-gate-1.json")]) == 1
    cells = [line.split("|") for line in capsys.readouterr().out.splitlines()
             if line.startswith(("| regional", "| boundary"))]
    verdicts = {kind: [c[7].strip() for c in cells if c[1].strip() == kind] for kind in ("regional", "boundary")}
    assert verdicts == {"regional": ["no"] * 12, "boundary": ["yes"] * 17}
