"""The pre-registered decision gate, as an executable function.

The criterion is registered in prose. This module is the same criterion in
code, and it is the only thing allowed to decide whether a region's model
passes. It runs once per region, on that region's F2-confirm mask suite,
after ``delta`` has been frozen.

    PASS iff, on the F2-confirm suite / degree-given / MRR, with the
    committed degree-corrected negatives, for ``gnn_a4`` against the
    baseline champion chosen on F2-dev:

      1. the lower bound of the 95 percent BCa bootstrap CI on mean(r_i)
         exceeds delta, and
      2. a one-sided sign test on r_i > 0 over the 20 mask instances is
         significant at p < 0.05.

where ``r_i = log2(MRR_gnn(i) / MRR_champion(i))`` and each MRR is the mean
over the 3 pre-registered seeds. The resampling unit is the mask instance:
one level of aggregation (seeds within an instance), not two.

Three deliberate choices, all of which exist to stop the gate from being
gameable after the fact:

* **The comparison is a log ratio.** Mask instances differ in difficulty by
  orders of magnitude, so an absolute MRR difference is not commensurable
  across them. ``r = 0.32`` means a 1.25x gain whatever the instance's scale.
* **The champion baseline is chosen on F2-dev.** Taking the best of three
  baselines on the confirm block is selection on the test statistic and
  biases the opponent downward. F2-dev rather than F1-val because the GNN
  iterates on F2-dev: a champion chosen on F1-val alone is handicapped under
  dataset shift, which biases the gate toward the GNN.
* **Both conditions must hold.** The CI bound is the effect-size test; the
  sign test stops a mean carried by two or three outlier instances.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

__all__ = [
    "CALIBRATION_INSTANCES",
    "CALIBRATION_SEEDS",
    "REQUIRED_COLUMNS",
    "GateVerdict",
    "PreregConfig",
    "calibrate_delta",
    "evaluate_gate",
    "instance_log_ratios",
    "select_champion_baseline",
]

#: Columns every results table must carry. Provenance columns (config hash,
#: torch/CUDA version, GPU name, TF32 flag) are required by the run logger but
#: are not read here. This tuple is a contract with the eval harness, which
#: emits exactly these: do not reorder or rename. ``mask_regime`` carries the
#: A/B/C ladder value, not the R/T/S mask regime, which lives in
#: the ``mask_instance`` id.
REQUIRED_COLUMNS = (
    "dataset",
    "region",
    "split",
    "degree_setting",
    "model",
    "mask_instance",
    "mask_regime",
    "seed",
)

#: Seeds per F1-val instance for the ``delta`` calibration.
#: Distinct from the 3-seed mean the gated comparison uses.
CALIBRATION_SEEDS = 5

#: F1-val mask instances the ``delta`` calibration runs over, per region.
#: The suite is exactly five fixed instances; a table with a different count is
#: not that suite, so :func:`calibrate_delta` refuses it, mirroring the confirm side's exact 20.
CALIBRATION_INSTANCES = 5

#: The only fields a real run may vary: ``delta`` is the calibration output and
#: ``region`` selects which region's gate to run. Every other field, including
#: the 10,000 bootstrap resamples, is a registered pin, checked by
#: :meth:`PreregConfig.assert_production_pins`.
_VARIABLE_FIELDS = frozenset({"delta", "region"})


@dataclass(frozen=True)
class PreregConfig:
    """Everything the gate is allowed to know, frozen before unblinding.

    ``delta`` is ``None`` until :func:`calibrate_delta` has been run on the
    F1-val block. :func:`evaluate_gate` refuses to run while it is ``None``,
    which is the mechanism that enforces "calibrate, commit, then unblind"
    rather than leaving it to discipline.
    """

    delta: float | None = None
    region: str = "CX"
    confirm_split: str = "f2_confirm"
    dev_split: str = "f2_dev"
    calibration_split: str = "f1_val"
    confirm_dataset: str = "f2"
    dev_dataset: str = "f2"
    calibration_dataset: str = "f1"
    n_instances_required: int = 20
    n_seeds_required: int = 3
    degree_setting: str = "degree_given"
    metric: str = "mrr"
    gnn_model: str = "gnn_a4"
    baseline_models: tuple[str, ...] = ("configuration", "logreg", "xgboost")
    n_bootstrap: int = 10_000
    confidence_level: float = 0.95
    sign_test_alpha: float = 0.05
    bootstrap_seed: int = 20260818

    def __post_init__(self) -> None:
        object.__setattr__(self, "region", self.region.upper())
        if self.delta is not None and self.delta < 0:
            raise ValueError(f"delta must be >= 0, got {self.delta}")
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if self.gnn_model in self.baseline_models:
            raise ValueError(
                f"{self.gnn_model!r} is listed as both the GNN and a baseline"
            )
        if not self.baseline_models:
            raise ValueError("at least one baseline model must be pre-registered")

    def assert_production_pins(self) -> None:
        """Refuse a config whose pinned fields deviate from the registered values.

        The pinned parameters "are not knobs". Only the fields
        in :data:`_VARIABLE_FIELDS` may vary per run; a caller cannot slip a
        one-instance/one-seed or low-resample configuration past the gate by
        constructing a weakened :class:`PreregConfig`.
        """
        canonical = PreregConfig()
        deviating = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in _VARIABLE_FIELDS
            and getattr(self, f.name) != getattr(canonical, f.name)
        }
        if deviating:
            raise ValueError(
                f"PreregConfig deviates from the pre-registered pins "
                f"{deviating}. These are not knobs; only "
                f"delta and region may vary per run."
            )


@dataclass(frozen=True)
class GateVerdict:
    """The decision, with every number that produced it."""

    passed: bool
    r_bar: float
    ci_low: float
    ci_high: float
    delta: float
    p_sign: float
    n_instances: int
    n_instances_positive: int
    champion_baseline: str
    ci_method: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    confidence_pct: str = "95%"

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"GATE: {verdict}",
            f"  champion baseline  : {self.champion_baseline}",
            f"  mean log2 ratio    : {self.r_bar:+.4f}  ({2**self.r_bar:.3f}x)",
            (
                f"  {self.confidence_pct} CI ({self.ci_method}) : "
                f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
            ),
            f"  delta (noise floor): {self.delta:+.4f}",
            (
                f"  sign test          : "
                f"{self.n_instances_positive}/{self.n_instances} "
                f"instances positive, p = {self.p_sign:.4g}"
            ),
        ]
        lines.extend(f"  note: {r}" for r in self.reasons)
        return "\n".join(lines)


def _require_columns(results: pd.DataFrame, extra: Sequence[str]) -> None:
    missing = [c for c in (*REQUIRED_COLUMNS, *extra) if c not in results.columns]
    if missing:
        raise ValueError(f"results table is missing column(s): {missing}")


def _validate_metric(frame: pd.DataFrame, prereg: PreregConfig, where: str) -> None:
    """Reject non-numeric, non-finite, or out-of-range metric values.

    MRR is a mean reciprocal rank, so it lies in ``(0, 1]``. An ``inf`` would
    otherwise sail through the non-positive check and pass on a corrupt value,
    and a ``NaN`` would surface only as a late bootstrap failure. Caught here,
    before any aggregation, either is a clear scoring or logging bug.
    """
    values = pd.to_numeric(frame[prereg.metric], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{where}: {prereg.metric} has non-numeric or non-finite value(s); "
            f"this is a scoring or logging bug, not a result the gate may use"
        )
    if ((values <= 0) | (values > 1)).any():
        raise ValueError(
            f"{where}: {prereg.metric} outside (0, 1]; MRR is a mean reciprocal "
            f"rank and cannot leave that range"
        )


def _slice(results: pd.DataFrame, prereg: PreregConfig, split: str) -> pd.DataFrame:
    """The one pre-registered slice of the results table.

    A split is one A/B/C ladder regime, so a non-empty slice
    must carry a single ``mask_regime``; mixed values mean two ladder blocks
    were concatenated. The R/T/S mask regime lives in the ``mask_instance`` id,
    not this column, which is why it is asserted here but never grouped on.

    Each split also belongs to one dataset (F1 trains, F2 is the held-out test
    individual), so a non-empty slice must carry the dataset that split's label
    claims. Rows tagged F1 but wearing an F2 split label would let training-side
    data stand in for the cross-individual test, which the gate exists to make.
    """
    sliced = results[
        (results["region"].astype(str).str.upper() == prereg.region)
        & (results["degree_setting"] == prereg.degree_setting)
        & (results["split"] == split)
    ]
    if sliced.empty:
        return sliced

    regimes = sliced["mask_regime"].unique()
    if regimes.size > 1:
        raise ValueError(
            f"the {split} slice mixes mask_regime values {sorted(regimes)}; a "
            f"split is one ladder regime and must not blend A/B/C blocks"
        )

    expected_dataset = {
        prereg.confirm_split: prereg.confirm_dataset,
        prereg.dev_split: prereg.dev_dataset,
        prereg.calibration_split: prereg.calibration_dataset,
    }.get(split)
    datasets = sliced["dataset"].unique()
    if datasets.size > 1:
        raise ValueError(
            f"the {split} slice mixes dataset values {sorted(datasets)}; a split "
            f"belongs to one dataset"
        )
    if expected_dataset is not None and datasets[0] != expected_dataset:
        raise ValueError(
            f"the {split} slice is dataset {datasets[0]!r}, not the pre-registered "
            f"{expected_dataset!r}; a row labelled {split} must come from that "
            f"dataset or the cross-individual comparison is not the one claimed"
        )
    return sliced


def select_champion_baseline(results: pd.DataFrame, prereg: PreregConfig) -> str:
    """Pick the strongest baseline **on the dev split**.

    One global champion, by mean metric across all dev mask instances -- not
    a per-instance champion, which would reintroduce selection noise into the
    very quantity the CI is computed over.
    """
    _require_columns(results, [prereg.metric])
    dev = _slice(results, prereg, prereg.dev_split)
    dev = dev[dev["model"].isin(prereg.baseline_models)]

    if dev.empty:
        raise ValueError(
            f"no {prereg.dev_split} rows for baselines {prereg.baseline_models} "
            f"in region {prereg.region!r} / {prereg.degree_setting!r}. The "
            f"champion must never be chosen on {prereg.confirm_split}."
        )

    missing = set(prereg.baseline_models) - set(dev["model"].unique())
    if missing:
        raise ValueError(
            f"pre-registered baseline(s) {sorted(missing)} have no "
            f"{prereg.dev_split} results; the champion cannot be selected from "
            f"a partial field"
        )

    _validate_metric(dev, prereg, f"{prereg.dev_split} baselines")

    seed_counts = dev.groupby(["model", "mask_instance"])["seed"].nunique()
    short = seed_counts[seed_counts != prereg.n_seeds_required]
    if not short.empty:
        raise ValueError(
            f"{len(short)} {prereg.dev_split} baseline instance(s) do not carry "
            f"exactly {prereg.n_seeds_required} seeds, e.g. {short.head(3).to_dict()}. "
            f"Champion selection needs identical, complete coverage, not a "
            f"cherry-picked subset."
        )
    if dev.duplicated(["model", "mask_instance", "seed"]).any():
        raise ValueError(
            f"duplicate (model, mask_instance, seed) rows in the "
            f"{prereg.dev_split} baselines reweight the champion means"
        )

    per_model_instances = dev.groupby("model")["mask_instance"].agg(
        lambda s: frozenset(s.unique())
    )
    if per_model_instances.nunique() > 1:
        counts = dev.groupby("model")["mask_instance"].nunique().to_dict()
        raise ValueError(
            f"the {prereg.dev_split} baselines cover different mask-instance sets "
            f"(per-model instance counts {counts}); the champion must be chosen on "
            f"identical, complete coverage so no baseline wins on an easier subset"
        )

    means = dev.groupby("model")[prereg.metric].mean()
    return str(means.idxmax())


def instance_log_ratios(
    results: pd.DataFrame,
    prereg: PreregConfig,
    champion: str | None = None,
) -> pd.DataFrame:
    """Per-mask-instance log2 ratio of GNN to champion baseline.

    Seeds are averaged within an instance before the ratio is taken. That is
    the only aggregation, so the bootstrap in :func:`evaluate_gate` sees one
    independent observation per mask instance.
    """
    _require_columns(results, [prereg.metric])
    if champion is None:
        champion = select_champion_baseline(results, prereg)

    confirm = _slice(results, prereg, prereg.confirm_split)
    if confirm.empty:
        raise ValueError(
            f"no {prereg.confirm_split} rows in region {prereg.region!r} / "
            f"{prereg.degree_setting!r}"
        )

    gnn = confirm[confirm["model"] == prereg.gnn_model]
    base = confirm[confirm["model"] == champion]
    if gnn.empty:
        raise ValueError(
            f"no {prereg.confirm_split} rows for the GNN model "
            f"{prereg.gnn_model!r}"
        )
    if base.empty:
        raise ValueError(
            f"no {prereg.confirm_split} rows for the champion baseline "
            f"{champion!r}"
        )

    for name, frame in (("GNN", gnn), ("champion", base)):
        _validate_metric(frame, prereg, f"{prereg.confirm_split} {name}")
        counts = frame.groupby("mask_instance")["seed"].nunique()
        wrong = counts[counts != prereg.n_seeds_required]
        if not wrong.empty:
            raise ValueError(
                f"{len(wrong)} mask instance(s) do not carry exactly the "
                f"pre-registered {prereg.n_seeds_required} {name} seeds, e.g. "
                f"{wrong.head(3).to_dict()}. The gated MRR is the mean over "
                f"exactly {prereg.n_seeds_required} seeds; more "
                f"or fewer changes the statistic, so the gate refuses it."
            )
        if frame.duplicated(["mask_instance", "seed"]).any():
            raise ValueError(
                f"duplicate (mask_instance, seed) rows in the {name} "
                f"{prereg.confirm_split} results reweight the "
                f"{prereg.n_seeds_required}-seed mean; each seed must appear "
                f"once per instance"
            )

    gnn_mean = gnn.groupby("mask_instance")[prereg.metric].mean()
    base_mean = base.groupby("mask_instance")[prereg.metric].mean()

    orphaned = set(gnn_mean.index) ^ set(base_mean.index)
    if orphaned:
        raise ValueError(
            f"{len(orphaned)} mask instance(s) lack a matching GNN/baseline "
            f"pair, e.g. {sorted(orphaned)[:3]}. The comparison must be paired."
        )

    joined = pd.concat(
        {"gnn": gnn_mean, "base": base_mean}, axis=1, join="inner"
    ).reset_index()

    joined["r_i"] = np.log2(joined["gnn"] / joined["base"])

    return (
        joined[["mask_instance", "r_i"]]
        .sort_values("mask_instance")
        .reset_index(drop=True)
    )


def calibrate_delta(
    results: pd.DataFrame,
    prereg: PreregConfig,
    multiplier: float = 2.0,
) -> tuple[float, float]:
    """Measure the seed noise floor on the **F1-val** mask instances.

    Returns ``(s, delta)`` where ``s`` is the median across F1-val instances
    of the across-seed sample standard deviation of ``log2(metric)``, and
    ``delta = round(multiplier * s, 2)``.

    Every F1-val instance must carry exactly :data:`CALIBRATION_SEEDS` seeds,
    and the suite must be exactly :data:`CALIBRATION_INSTANCES` instances.
    The registered calibration defines
    ``s`` over five seeds on the five fixed F1-val instances; a floor measured
    from a different seed count, or from a different number of instances, is a
    different ``delta``, so neither ``prereg.n_seeds_required`` (the 3-seed gated
    mean) nor a looser instance count applies here. This mirrors the confirm
    side's exact 20.

    Deliberately conservative: ``s`` is the single-seed spread, while the
    gated statistic uses a 3-seed mean whose noise is smaller by roughly
    sqrt(3). The bar therefore sits comfortably above run-to-run variation.

    Must be run, recorded in the region's ``results/<code>/delta.json`` and
    committed **before** any confirm instance is scored.
    """
    prereg.assert_production_pins()
    if multiplier != 2.0:
        raise ValueError(
            f"multiplier is pinned at 2.0 (delta = 2s), got "
            f"{multiplier}; it is not a knob"
        )
    _require_columns(results, [prereg.metric])

    val = _slice(results, prereg, prereg.calibration_split)
    val = val[val["model"] == prereg.gnn_model]
    if val.empty:
        raise ValueError(
            f"no {prereg.calibration_split} rows for {prereg.gnn_model!r}; "
            f"delta cannot be calibrated without them"
        )
    _validate_metric(val, prereg, f"{prereg.calibration_split} {prereg.gnn_model}")
    if val.duplicated(["mask_instance", "seed"]).any():
        raise ValueError(
            f"duplicate (mask_instance, seed) rows on {prereg.calibration_split} "
            f"reweight the across-seed SD; each seed must appear once per instance"
        )

    per_instance = val.groupby("mask_instance").agg(
        n_seeds=("seed", "nunique"),
        sd=(prereg.metric, lambda s: float(np.std(np.log2(s), ddof=1))),
    )
    wrong = per_instance[per_instance["n_seeds"] != CALIBRATION_SEEDS]
    if not wrong.empty:
        offenders = wrong["n_seeds"].to_dict()
        raise ValueError(
            f"delta is defined over exactly {CALIBRATION_SEEDS} seeds per "
            f"{prereg.calibration_split} instance; these carry a "
            f"different count: {offenders}"
        )
    if len(per_instance) != CALIBRATION_INSTANCES:
        raise ValueError(
            f"delta is calibrated on exactly {CALIBRATION_INSTANCES} "
            f"{prereg.calibration_split} mask instances, got "
            f"{len(per_instance)}"
        )

    s = float(np.median(per_instance["sd"]))
    return s, round(multiplier * s, 2)


#: Failures that send _bootstrap_ci to its percentile fallback. RuntimeWarning
#: is here because pytest.ini promotes it to an error; scipy signals BCa
#: degeneracy by warning and returning a non-finite bound, which the isfinite
#: check below catches, so no other warning class need be listed.
_BOOTSTRAP_FAILURES = (
    ValueError,
    FloatingPointError,
    ZeroDivisionError,
    RuntimeWarning,
)


def _bootstrap_ci(
    values: np.ndarray, prereg: PreregConfig
) -> tuple[float, float, str, tuple[str, ...]]:
    """BCa bootstrap over instances, percentile fallback when degenerate."""
    notes: list[str] = []

    if bool(np.all(values == values[0])):
        # Only exact equality is 0/0 for BCa's acceleration term. A tolerance
        # here (np.allclose) would return values[0] as the whole CI for
        # near-equal-but-distinct ratios, making the verdict depend on which
        # instance sorts first; those go through the real bootstrap instead.
        constant = float(values[0])
        return (
            constant,
            constant,
            "degenerate",
            ("every instance has an identical ratio; the CI is a point",),
        )

    # Imported here: scipy.stats adds ~0.4 s to every training process's startup.
    from scipy import stats

    for method in ("BCa", "percentile"):
        try:
            res = stats.bootstrap(
                (values,),
                np.mean,
                n_resamples=prereg.n_bootstrap,
                confidence_level=prereg.confidence_level,
                method=method,
                random_state=np.random.default_rng(prereg.bootstrap_seed),
            )
        except _BOOTSTRAP_FAILURES as exc:  # pragma: no cover - defensive
            notes.append(f"{method} bootstrap raised {exc.__class__.__name__}")
            continue
        low = float(res.confidence_interval.low)
        high = float(res.confidence_interval.high)
        if math.isfinite(low) and math.isfinite(high):
            return low, high, method, tuple(notes)
        notes.append(f"{method} bootstrap returned a non-finite bound")

    raise RuntimeError("both BCa and percentile bootstrap failed to produce a CI")


def evaluate_gate(results: pd.DataFrame, prereg: PreregConfig) -> GateVerdict:
    """Run the pre-registered gate. Call this once per region.

    Raises if ``prereg.delta`` is still ``None`` -- the gate cannot be
    evaluated before the noise floor has been calibrated on F1-val and
    committed.
    """
    prereg.assert_production_pins()
    if prereg.delta is None:
        raise ValueError(
            "prereg.delta is None. Run calibrate_delta() on the F1-val block, "
            "record s and delta in results/<code>/delta.json, commit and tag, and only then "
            "evaluate the gate on the confirm suite."
        )

    champion = select_champion_baseline(results, prereg)
    per_instance = instance_log_ratios(results, prereg, champion)
    values = per_instance["r_i"].to_numpy(dtype=np.float64)

    if values.size != prereg.n_instances_required:
        raise ValueError(
            f"the gate needs exactly {prereg.n_instances_required} "
            f"{prereg.confirm_split} mask instances, got {values.size}"
        )

    ci_low, ci_high, ci_method, notes = _bootstrap_ci(values, prereg)

    n_positive = int((values > 0).sum())
    n_ties = int((values == 0).sum())
    n_effective = values.size - n_ties
    reasons = list(notes)
    if n_ties:
        reasons.append(
            f"{n_ties} instance(s) tied at exactly zero and were dropped from "
            f"the sign test, per convention"
        )
    if n_effective == 0:
        p_sign = 1.0
        reasons.append("every instance tied; the sign test is uninformative")
    else:
        from scipy import stats

        p_sign = float(
            stats.binomtest(
                n_positive, n_effective, p=0.5, alternative="greater"
            ).pvalue
        )

    effect_ok = ci_low > prereg.delta
    sign_ok = p_sign < prereg.sign_test_alpha
    if not effect_ok:
        reasons.append(
            f"condition 1 failed: CI lower bound {ci_low:+.4f} does not exceed "
            f"delta {prereg.delta:+.4f}"
        )
    if not sign_ok:
        reasons.append(
            f"condition 2 failed: sign test p = {p_sign:.4g} is not below "
            f"{prereg.sign_test_alpha}"
        )

    return GateVerdict(
        passed=bool(effect_ok and sign_ok),
        r_bar=float(values.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        delta=float(prereg.delta),
        p_sign=p_sign,
        n_instances=int(values.size),
        n_instances_positive=n_positive,
        champion_baseline=champion,
        ci_method=ci_method,
        reasons=tuple(reasons),
        confidence_pct=f"{prereg.confidence_level:.0%}",
    )
