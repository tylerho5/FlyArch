"""Phase 3's integrated claim: a conjunction of regional guards and boundary comparisons.

The registered integrated claim of Phase 3, and plan 07. Every component is a
per-instance statistic ``r_i = log2(MRR_system(i) / MRR_reference(i))`` with the
bootstrap over mask instances (``src.gate._bootstrap_ci``, called, not edited):

- **Regional guard** (one per gated region, on its F2-confirm suite): the reference is
  the region's registry cohort, ``MRR_reference(i)`` the mean of its three seeds' MRRs.
  It passes when the CI's lower bound is at least ``-delta_r``, or at least 0 for a
  region with no calibrated ``delta``. No sign test: it is a non-inferiority guard.
- **Boundary comparison** (on ``phase3_boundary_f2_confirm``, one per opponent): it
  passes when the lower bound exceeds ``delta_P3`` and the one-sided sign test on
  ``r_i > 0`` is below alpha, as in the regional gate.

:func:`integrated_claim` passes only if every required component exists and passes;
it refuses an incomplete opponent set or a missing region rather than judging what
is there. :func:`calibrate_delta_p3` is the registered recipe on five full system realizations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.gate import PreregConfig, _bootstrap_ci

__all__ = ["Component", "boundary_comparison", "calibrate_delta_p3", "integrated_claim",
           "regional_guard", "required_opponents"]

CONFIRM_INSTANCES = 20
#: The F2 boundary confirm suite skips its T slots (2026-11-10): 20 slots, 4 of them T.
BOUNDARY_CONFIRM_INSTANCES = 16
CALIBRATION_REALIZATIONS = 5
CALIBRATION_INSTANCES = 5
BASELINES = ("configuration", "logreg", "xgboost")


@dataclass(frozen=True)
class Component:
    kind: str  # "regional" or "boundary"
    name: str  # the region, or the opponent
    passed: bool
    r_bar: float
    ci_low: float
    ci_high: float
    margin: float  # the bound the CI's lower end is compared with
    p_sign: float | None
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _check_mrrs(values: dict[str, float], what: str) -> None:
    """A raw MRR is finite and in (0, 1]; checked before any averaging can hide a bad one."""
    bad = sorted(i for i, v in values.items() if not (math.isfinite(v) and 0.0 < v <= 1.0))
    if bad:
        raise ValueError(f"{what}: MRRs must be finite and in (0, 1]; bad at {bad[:3]}")


def _ratios(system: dict[str, float], reference: dict[str, float], n: int) -> np.ndarray:
    _check_mrrs(system, "system")
    _check_mrrs(reference, "reference")
    if set(system) != set(reference):
        raise ValueError("system and reference must score the same instances")
    if len(system) != n:
        raise ValueError(f"a confirm component needs exactly {n} instances, got {len(system)}")
    ids = sorted(system)
    values = np.array([[system[i], reference[i]] for i in ids], dtype=np.float64)
    return np.log2(values[:, 0] / values[:, 1])


def _seed_mean(seeds: list[dict[str, float]], n_seeds: int = 3) -> dict[str, float]:
    if len(seeds) != n_seeds:
        raise ValueError(f"the reference cohort needs exactly {n_seeds} seeds, got {len(seeds)}")
    for k, seed in enumerate(seeds):
        _check_mrrs(seed, f"reference seed {k}")
    ids = set(seeds[0])
    if any(set(s) != ids for s in seeds):
        raise ValueError("every reference seed must score the same instances")
    return {i: float(np.mean([s[i] for s in seeds])) for i in ids}


def regional_guard(region: str, system: dict[str, float], reference_seeds: list[dict[str, float]],
                   delta: float | None, prereg: PreregConfig = PreregConfig(),
                   n_instances: int = CONFIRM_INSTANCES) -> Component:
    """One region's guard: lower bound of mean r_i at least -delta (0 without a delta)."""
    values = _ratios(system, _seed_mean(reference_seeds), n_instances)
    low, high, method, notes = _bootstrap_ci(values, prereg)
    margin = -float(delta) if delta is not None else 0.0
    passed = low >= margin
    reasons = list(notes) + [f"{method} CI"]
    if not passed:
        reasons.append(f"lower bound {low:+.4f} is below {margin:+.4f}")
    return Component("regional", region, bool(passed), float(values.mean()), low, high, margin, None, tuple(reasons))


def boundary_comparison(opponent: str, system: dict[str, float], reference: dict[str, float], delta_p3: float,
                        prereg: PreregConfig = PreregConfig(),
                        n_instances: int = BOUNDARY_CONFIRM_INSTANCES) -> Component:
    """One boundary comparison: lower bound above delta_P3 and the one-sided sign test."""
    from scipy import stats

    if delta_p3 is None or delta_p3 < 0:
        raise ValueError("delta_P3 must be calibrated (and non-negative) before a boundary comparison")
    values = _ratios(system, reference, n_instances)
    low, high, method, notes = _bootstrap_ci(values, prereg)
    positive, effective = int((values > 0).sum()), int((values != 0).sum())
    p_sign = 1.0 if effective == 0 else float(stats.binomtest(positive, effective, 0.5, alternative="greater").pvalue)
    effect, sign = low > delta_p3, p_sign < prereg.sign_test_alpha
    reasons = list(notes) + [f"{method} CI"]
    if not effect:
        reasons.append(f"lower bound {low:+.4f} does not exceed delta_P3 {delta_p3:+.4f}")
    if not sign:
        reasons.append(f"sign test p = {p_sign:.4g} is not below {prereg.sign_test_alpha}")
    return Component("boundary", opponent, bool(effect and sign), float(values.mean()), low, high,
                     float(delta_p3), p_sign, tuple(reasons))


def required_opponents(experts, *, system_is_b0: bool) -> set[str]:
    """Every registry expert, the three baseline families, B0 unless it is the system, and the seed control."""
    names = {f"expert:{e}" for e in experts} | {f"baseline:{b}" for b in BASELINES} | {"seed_control"}
    return names if system_is_b0 else names | {"B0"}


def integrated_claim(regional: list[Component], boundary: list[Component], *, gated_regions,
                     experts, system_is_b0: bool) -> tuple[bool, list[str]]:
    """The conjunction. Refuses (raises) on a missing region or opponent; returns (passed, failures)."""
    if any(c.kind != "regional" for c in regional) or any(c.kind != "boundary" for c in boundary):
        raise ValueError("a component is in the wrong list")
    have_regions = [c.name for c in regional]
    have_opponents = [c.name for c in boundary]
    if len(set(have_regions)) != len(have_regions) or len(set(have_opponents)) != len(have_opponents):
        raise ValueError("a component appears twice")
    if set(have_regions) != set(gated_regions):
        raise ValueError(f"regional guards must be exactly the gated regions; "
                         f"missing {sorted(set(gated_regions) - set(have_regions))}, "
                         f"extra {sorted(set(have_regions) - set(gated_regions))}")
    needed = required_opponents(experts, system_is_b0=system_is_b0)
    if set(have_opponents) != needed:
        raise ValueError(f"the opponent set must be exactly {sorted(needed)}; "
                         f"missing {sorted(needed - set(have_opponents))}, extra {sorted(set(have_opponents) - needed)}")
    failures = [f"{c.kind} {c.name}: {'; '.join(c.reasons)}" for c in (*regional, *boundary) if not c.passed]
    return not failures, failures


def calibrate_delta_p3(realizations: list[dict[str, float]], multiplier: float = 2.0) -> tuple[float, float]:
    """The registered calibration on the Phase 3 system: median over the five F1-val boundary instances of
    the SD of log2(MRR) across five full realizations; delta_P3 = round(2 s, 2)."""
    if multiplier != 2.0:
        raise ValueError("multiplier is pinned at 2.0")
    if len(realizations) != CALIBRATION_REALIZATIONS:
        raise ValueError(f"delta_P3 needs exactly {CALIBRATION_REALIZATIONS} realizations, got {len(realizations)}")
    for k, realization in enumerate(realizations):
        _check_mrrs(realization, f"realization {k}")
    ids = set(realizations[0])
    if any(set(r) != ids for r in realizations) or len(ids) != CALIBRATION_INSTANCES:
        raise ValueError(f"every realization must score the same {CALIBRATION_INSTANCES} F1-val boundary instances")
    logs = np.log2(np.array([[r[i] for r in realizations] for i in sorted(ids)], dtype=np.float64))
    s = float(np.median(logs.std(axis=1, ddof=1)))
    if not math.isfinite(s):
        raise ValueError("non-finite spread")
    return s, round(multiplier * s, 2)

