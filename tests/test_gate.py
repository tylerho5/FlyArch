"""The gate, checked against synthetic result tables with known verdicts.

This is the file that decides whether the pre-registered criterion is worth
anything. A gate that only ever says PASS is not a gate, so most of what
follows constructs results that *must* be rejected:

* a null effect,
* a real but scientifically trivial win, below the noise floor,
* a mean carried by three outlier instances.

The last two are the registered synthetic pair: each condition
alone would wave one of them through, and the pair correctly rejects both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.gate import (
    CALIBRATION_SEEDS,
    REQUIRED_COLUMNS,
    PreregConfig,
    calibrate_delta,
    evaluate_gate,
    instance_log_ratios,
    select_champion_baseline,
)

BASELINES = ("configuration", "logreg", "xgboost")
BASE_MRR = 0.10

#: Five seeds whose log2 MRR is -3.1/-3.1/-3.0/-2.9/-2.9: sample SD exactly 0.1.
KNOWN_SPREAD = [2**-3.1, 2**-3.1, 2**-3.0, 2**-2.9, 2**-2.9]


def build_results(
    instance_ratios: dict[str, float],
    *,
    n_seeds: int = 3,
    region: str = "cx",
    mask_regime: str = "R",
    confirm_baseline_mrr: dict[str, float] | None = None,
    dev_baseline_mrr: dict[str, float] | None = None,
    gnn_val_by_seed: dict[str, list[float]] | None = None,
    drop_baseline_for: str | None = None,
) -> pd.DataFrame:
    """Assemble a results table with a known per-instance log2 ratio.

    ``instance_ratios`` maps a mask instance to the log2 ratio the gate must
    recover for it. F2-dev carries the baselines the champion is selected
    from; F1-val carries the ``gnn_a4`` seeds ``calibrate_delta`` measures.
    """
    confirm_baseline_mrr = confirm_baseline_mrr or dict.fromkeys(BASELINES, BASE_MRR)
    dev_baseline_mrr = dev_baseline_mrr or {
        "configuration": 0.07,
        "logreg": 0.08,
        "xgboost": 0.09,
    }
    gnn_val_by_seed = gnn_val_by_seed or {
        f"VAL{i:02d}": KNOWN_SPREAD for i in range(5)
    }

    rows: list[dict] = []

    for instance, ratio in sorted(instance_ratios.items()):
        common = {
            "dataset": "f2",
            "region": region,
            "split": "f2_confirm",
            "degree_setting": "degree_given",
            "mask_instance": instance,
            "mask_regime": mask_regime,
        }
        for seed in range(n_seeds):
            rows.append(
                {
                    **common,
                    "model": "gnn_a4",
                    "seed": seed,
                    "mrr": BASE_MRR * 2.0**ratio,
                }
            )
        if instance == drop_baseline_for:
            continue
        for name, value in confirm_baseline_mrr.items():
            for seed in range(n_seeds):
                rows.append({**common, "model": name, "seed": seed, "mrr": value})

    for name, value in dev_baseline_mrr.items():
        for instance in ("DEV00", "DEV01"):
            for seed in range(3):  # dev stays well-formed; n_seeds is the confirm knob
                rows.append(
                    {
                        "dataset": "f2",
                        "region": region,
                        "split": "f2_dev",
                        "degree_setting": "degree_given",
                        "mask_instance": instance,
                        "mask_regime": mask_regime,
                        "model": name,
                        "seed": seed,
                        "mrr": value,
                    }
                )

    for instance, values in gnn_val_by_seed.items():
        for seed, value in enumerate(values):
            rows.append(
                {
                    "dataset": "f1",
                    "region": region,
                    "split": "f1_val",
                    "degree_setting": "degree_given",
                    "mask_instance": instance,
                    "mask_regime": mask_regime,
                    "model": "gnn_a4",
                    "seed": seed,
                    "mrr": value,
                }
            )

    return pd.DataFrame(rows)


def spread(base: float, n: int = 20, jitter: float = 0.02) -> dict[str, float]:
    """``n`` mask instances centred on ``base`` with deterministic jitter."""
    rng = np.random.default_rng(4)
    return {f"I{i:02d}": float(base + rng.normal(0.0, jitter)) for i in range(n)}


class TestContract:
    def test_region_code_is_case_insensitive_end_to_end(self):
        results = build_results({f"I{i:02d}": 0.5 for i in range(20)}, region="CX")
        prereg = PreregConfig(delta=0.25, region="cx")
        assert prereg.region == "CX"
        assert evaluate_gate(results, prereg).passed

    def test_required_columns_are_the_pinned_tuple_in_order(self):
        """`05-eval-harness` emits these; reordering silently breaks it."""
        assert REQUIRED_COLUMNS == (
            "dataset",
            "region",
            "split",
            "degree_setting",
            "model",
            "mask_instance",
            "mask_regime",
            "seed",
        )

    def test_calibration_seed_count_is_five(self):
        assert CALIBRATION_SEEDS == 5


class TestVerdicts:
    def test_clear_win_passes(self):
        results = build_results(spread(0.45))
        verdict = evaluate_gate(results, PreregConfig(delta=0.10))

        assert verdict.passed
        assert verdict.r_bar == pytest.approx(0.45, abs=0.02)
        assert verdict.ci_low > 0.10
        assert verdict.p_sign < 0.05
        assert verdict.n_instances == 20

    def test_null_effect_fails(self):
        results = build_results(spread(0.0, jitter=0.05))
        verdict = evaluate_gate(results, PreregConfig(delta=0.10))

        assert not verdict.passed
        assert verdict.ci_low < 0.10
        assert any("condition 1 failed" in r for r in verdict.reasons)

    def test_tiny_consistent_win_fails_on_effect_size(self):
        """The test that proves delta has teeth.

        Every instance improves, so the sign test is emphatic and a
        CI-excludes-zero criterion would sail through. But a 0.7 percent gain
        is not a finding, and delta is what says so.
        """
        results = build_results(spread(0.01, jitter=0.001))
        verdict = evaluate_gate(results, PreregConfig(delta=0.10))

        assert verdict.n_instances_positive == verdict.n_instances
        assert verdict.p_sign < 0.05, "sign test alone would have passed this"
        assert verdict.ci_low > 0.0, "a CI-excludes-zero gate would have passed this"
        assert not verdict.passed
        assert any("condition 1 failed" in r for r in verdict.reasons)

    def test_outlier_driven_mean_fails_on_sign_test(self):
        """Three spectacular instances must not carry seventeen mediocre ones."""
        ratios = {f"BIG{i}": 3.0 for i in range(3)}
        ratios.update({f"SMALL{i:02d}": -0.05 for i in range(17)})
        results = build_results(ratios)
        verdict = evaluate_gate(results, PreregConfig(delta=0.10))

        assert verdict.r_bar > 0.10, "the naive mean is positive and sizeable"
        assert verdict.n_instances_positive == 3
        assert verdict.p_sign > 0.05
        assert not verdict.passed
        assert any("condition 2 failed" in r for r in verdict.reasons)


class TestChampionSelection:
    def test_champion_comes_from_dev_not_confirm(self):
        """The dev champion and the confirm champion must be different baselines.

        logreg is the strongest baseline on dev, xgboost on confirm, so a gate
        that read the confirm block to choose the opponent would pick xgboost.
        Selecting on the confirm statistic is forbidden; asserting
        logreg proves the choice is made on dev, not merely that some baseline
        happens to win both.
        """
        results = build_results(
            spread(0.30),
            dev_baseline_mrr={"configuration": 0.04, "logreg": 0.15, "xgboost": 0.05},
            confirm_baseline_mrr={
                "configuration": 0.02,
                "logreg": 0.10,
                "xgboost": 0.20,
            },
        )
        prereg = PreregConfig(delta=0.05)

        assert select_champion_baseline(results, prereg) == "logreg"
        assert evaluate_gate(results, prereg).champion_baseline == "logreg"

    def test_baseline_absent_from_dev_is_refused(self):
        results = build_results(spread(0.3))
        absent = (results["split"] == "f2_dev") & (results["model"] == "xgboost")
        results = results[~absent]
        with pytest.raises(ValueError, match="have no f2_dev results"):
            select_champion_baseline(results, PreregConfig(delta=0.05))


class TestGuards:
    def test_delta_must_be_calibrated_first(self):
        results = build_results(spread(0.45))
        with pytest.raises(ValueError, match="delta.json"):
            evaluate_gate(results, PreregConfig())

    def test_too_few_confirm_instances_is_refused(self):
        results = build_results(spread(0.45, n=5))
        with pytest.raises(ValueError, match="20"):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_too_many_confirm_instances_is_refused(self):
        results = build_results(spread(0.45, n=25))
        with pytest.raises(ValueError, match="mask instances, got 25"):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_too_few_seeds_is_refused(self):
        results = build_results(spread(0.45), n_seeds=1)
        with pytest.raises(ValueError, match="exactly the pre-registered"):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_too_many_seeds_is_refused(self):
        results = build_results(spread(0.45), n_seeds=5)
        with pytest.raises(ValueError, match="exactly the pre-registered"):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_too_few_champion_seeds_is_refused(self):
        results = build_results(spread(0.45))
        keep = ~(
            (results["split"] == "f2_confirm")
            & (results["model"] == "xgboost")
            & (results["seed"] > 0)
        )
        with pytest.raises(ValueError, match="exactly.*champion seeds"):
            evaluate_gate(results[keep], PreregConfig(delta=0.10))

    def test_mixed_mask_regime_slice_is_refused(self):
        results = build_results(spread(0.45))
        flip = (results["split"] == "f2_confirm") & results["mask_instance"].isin(
            [f"I{i:02d}" for i in range(10)]
        )
        results.loc[flip, "mask_regime"] = "T"
        with pytest.raises(ValueError, match="mask_regime"):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_unpaired_instance_is_refused(self):
        results = build_results(spread(0.45), drop_baseline_for="I03")
        with pytest.raises(ValueError, match="must be paired"):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_missing_columns_are_named(self):
        results = build_results(spread(0.45)).drop(columns=["mask_instance"])
        with pytest.raises(
            ValueError, match=r"missing column\(s\): \['mask_instance'\]"
        ):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_region_without_confirm_rows_is_refused(self):
        """A region with dev rows but no confirm rows must not score empty."""
        results = build_results(spread(0.45))
        dev_only = results[results["split"] == "f2_dev"].copy()
        dev_only["region"] = "ol"
        results = pd.concat([results, dev_only], ignore_index=True)
        prereg = PreregConfig(delta=0.10, region="ol")

        with pytest.raises(ValueError, match="no f2_confirm rows"):
            evaluate_gate(results, prereg)

    def test_gnn_cannot_also_be_a_baseline(self):
        with pytest.raises(ValueError, match="both the GNN and a baseline"):
            PreregConfig(gnn_model="logreg", baseline_models=("logreg",))


class TestDeltaCalibration:
    def test_recovers_a_known_seed_spread(self):
        """log2 values -3.1/-3.1/-3.0/-2.9/-2.9 have sample sd = 0.1 exactly."""
        results = build_results(spread(0.3))
        s, delta = calibrate_delta(results, PreregConfig())

        assert s == pytest.approx(0.1, abs=1e-9)
        assert delta == pytest.approx(0.20)

    def test_uses_f1_val_only(self):
        """A huge confirm effect must not influence the noise floor."""
        quiet = {
            f"VAL{i:02d}": [0.10, 0.10001, 0.09999, 0.10, 0.10001] for i in range(5)
        }
        results = build_results(spread(2.0), gnn_val_by_seed=quiet)
        s, delta = calibrate_delta(results, PreregConfig())

        assert s < 0.01
        assert delta == pytest.approx(0.0)

    def test_too_few_calibration_seeds_names_the_instance(self):
        vals = {f"VAL{i:02d}": KNOWN_SPREAD for i in range(5)}
        vals["VAL00"] = KNOWN_SPREAD[:4]
        results = build_results(spread(0.3), gnn_val_by_seed=vals)
        with pytest.raises(ValueError, match=r"VAL00.*4"):
            calibrate_delta(results, PreregConfig())

    def test_too_many_calibration_seeds_names_the_instance(self):
        vals = {f"VAL{i:02d}": KNOWN_SPREAD for i in range(5)}
        vals["VAL00"] = [*KNOWN_SPREAD, 0.1]
        results = build_results(spread(0.3), gnn_val_by_seed=vals)
        with pytest.raises(ValueError, match=r"VAL00.*6"):
            calibrate_delta(results, PreregConfig())

    def test_the_three_seed_gate_fixture_is_refused(self):
        """delta is defined over five seeds; a floor measured from three is a
        different number, so ``n_seeds_required`` deliberately does not apply."""
        results = build_results(
            spread(0.3), gnn_val_by_seed={"VAL00": KNOWN_SPREAD[:3]}
        )
        with pytest.raises(ValueError, match=r"VAL00.*3"):
            calibrate_delta(results, PreregConfig())


class TestAuditHardening:
    """Hostile-input regressions: a corrupted table must never reach a verdict.

    Each case is a way a caller could hand the gate data that is not the
    pre-registered shape and, without these guards, still get a PASS.
    """

    def test_nonfinite_confirm_metric_is_refused(self):
        for bad in (np.inf, np.nan):
            results = build_results(spread(0.45))
            first = results.index[
                (results["split"] == "f2_confirm") & (results["model"] == "gnn_a4")
            ][0]
            results.loc[first, "mrr"] = bad
            with pytest.raises(ValueError, match="non-finite"):
                evaluate_gate(results, PreregConfig(delta=0.10))

    def test_out_of_range_confirm_metric_is_refused(self):
        results = build_results(spread(0.45))
        first = results.index[
            (results["split"] == "f2_confirm") & (results["model"] == "gnn_a4")
        ][0]
        results.loc[first, "mrr"] = 1.5
        with pytest.raises(ValueError, match=r"outside \(0, 1\]"):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_f1_rows_wearing_a_confirm_label_are_refused(self):
        """A cross-individual gate must not score F1 rows as if they were F2."""
        results = build_results(spread(0.45))
        results.loc[results["split"] == "f2_confirm", "dataset"] = "f1"
        with pytest.raises(ValueError, match="dataset"):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_weakened_pins_are_refused(self):
        results = build_results(spread(0.45))
        with pytest.raises(ValueError, match="pins"):
            evaluate_gate(
                results,
                PreregConfig(
                    delta=0.1, n_instances_required=1, n_seeds_required=1
                ),
            )

    @pytest.mark.parametrize("n_bootstrap", [1_000, 20_000])
    def test_bootstrap_count_is_pinned(self, n_bootstrap):
        results = build_results(spread(0.45))
        with pytest.raises(ValueError, match="pins"):
            evaluate_gate(results, PreregConfig(delta=0.1, n_bootstrap=n_bootstrap))

    def test_near_degenerate_ci_is_order_invariant(self):
        """Near-equal ratios below delta must FAIL, and the verdict must not
        depend on which instance sorts first."""
        lo, hi = 0.1499995, 0.1500005
        hi_first = {"I00": hi, **{f"I{i:02d}": lo for i in range(1, 20)}}
        hi_last = {**{f"I{i:02d}": lo for i in range(19)}, "I19": hi}
        prereg = PreregConfig(delta=0.15)

        first = evaluate_gate(build_results(hi_first), prereg)
        last = evaluate_gate(build_results(hi_last), prereg)

        assert first.ci_method != "degenerate"
        assert not first.passed
        assert first.passed == last.passed

    def test_duplicate_confirm_seed_row_is_refused(self):
        """A repeated (instance, seed) passes the unique-seed count but
        double-weights that seed in the mean."""
        results = build_results(spread(0.45))
        dup = results[
            (results["split"] == "f2_confirm")
            & (results["model"] == "gnn_a4")
            & (results["mask_instance"] == "I00")
            & (results["seed"] == 0)
        ]
        results = pd.concat([results, dup], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_duplicate_calibration_seed_row_is_refused(self):
        results = build_results(spread(0.3))
        dup = results[
            (results["split"] == "f1_val")
            & (results["mask_instance"] == "VAL00")
            & (results["seed"] == 0)
        ]
        results = pd.concat([results, dup], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            calibrate_delta(results, PreregConfig())

    def test_sparse_baseline_coverage_is_refused(self):
        """One baseline evaluated on fewer dev instances than the others could
        be crowned champion on an easier subset."""
        results = build_results(spread(0.45))
        sparse = (
            (results["split"] == "f2_dev")
            & (results["model"] == "xgboost")
            & (results["mask_instance"] == "DEV01")
        )
        results = results[~sparse]
        with pytest.raises(ValueError, match="coverage"):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_uneven_baseline_seed_depth_is_refused(self):
        """Baselines must share seed depth, not just the instance set."""
        results = build_results(spread(0.45))
        thin = (
            (results["split"] == "f2_dev")
            & (results["model"] == "logreg")
            & (results["mask_instance"] == "DEV00")
            & (results["seed"] > 0)
        )
        results = results[~thin]
        with pytest.raises(ValueError, match="seeds"):
            evaluate_gate(results, PreregConfig(delta=0.10))

    def test_calibration_needs_exactly_five_instances(self):
        vals = {f"VAL{i:02d}": KNOWN_SPREAD for i in range(4)}
        results = build_results(spread(0.3), gnn_val_by_seed=vals)
        with pytest.raises(ValueError, match="got 4"):
            calibrate_delta(results, PreregConfig())


class TestReporting:
    def test_ratios_are_recovered_per_instance(self):
        ratios = {"A": 0.5, "B": -0.25, "C": 1.0}
        table = instance_log_ratios(build_results(ratios), PreregConfig(delta=0.1))

        assert table["mask_instance"].tolist() == ["A", "B", "C"]
        assert table["r_i"].tolist() == pytest.approx([0.5, -0.25, 1.0])
        assert list(table.columns) == ["mask_instance", "r_i"]

    def test_summary_states_the_verdict_and_its_inputs(self):
        verdict = evaluate_gate(
            build_results(spread(0.45)), PreregConfig(delta=0.10)
        )
        text = verdict.summary()

        assert text.startswith("GATE: PASS")
        assert "champion baseline" in text
        assert "delta (noise floor)" in text
        assert "instances positive" in text
