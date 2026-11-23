"""Per-type readout behind plan 10's F3 dimorphism report.

``per_type_frame`` splits the stored per-positive ranks by the masked
endpoint's type; ``category_table`` pools those rows into dimorphism categories
with a bootstrap over types. Everything here reads frozen scores: nothing is
selected or tuned on F3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.gate import PreregConfig, _bootstrap_ci

UNTYPED = "untyped"
ISOMORPHIC = "isomorphic"
#: The release's ``dimorphism`` labels, in report order; a null label is isomorphic.
CATEGORIES = (
    "male-specific",
    "potentially male-specific",
    "sexually dimorphic",
    "potentially sexually dimorphic",
    ISOMORPHIC,
)
#: The two comparisons the plan allows. Each holds ``type_trained`` fixed, so
#: an untrained (OOV) type is never set against a trained one.
COMPARISONS = (
    ("sexually dimorphic", True, ISOMORPHIC, True),
    ("male-specific", False, ISOMORPHIC, False),
)
_FLAGS = ("type_trained", "type_matched")


def mask_regimes(mask_instance: pd.Series) -> pd.Series:
    """``R``, ``S``, ``T`` or ``H`` from instance ids such as ``f3-ol-R05-s012``."""
    return mask_instance.str.split("-").str[2].str[0]


def per_type_frame(positives: pd.DataFrame, types: pd.Series) -> pd.DataFrame:
    """One row per (seed, mask_instance, masked endpoint's type): n_positives, mrr.

    Single-masked positives only, as the gate scores. ``types`` maps root id to
    the type the row keys on (F3's own ``f3_type``, F2's crosswalked name);
    endpoints it leaves null fall in ``UNTYPED``. ``type_trained`` and
    ``type_matched`` hold only when they hold for every positive in the row,
    and stay null where scoring did not know them.
    """
    single = positives.loc[~positives["both_masked"].astype(bool)]
    frame = single.assign(
        regime=mask_regimes(single["mask_instance"]).to_numpy(),
        type=types.reindex(single["root_id"].to_numpy()).fillna(UNTYPED).to_numpy(),
        **{flag: pd.to_numeric(single[flag], errors="coerce").astype(float) for flag in _FLAGS},
    )
    keys = ["seed", "mask_instance", "regime", "type"]
    out = frame.groupby(keys, sort=True).agg(
        n_positives=("reciprocal_rank", "size"),
        mrr=("reciprocal_rank", "mean"),
        **{flag: (flag, "min") for flag in _FLAGS},
    ).reset_index()
    for flag in _FLAGS:
        out[flag] = out[flag].map({1.0: True, 0.0: False}).astype(object)
    return out


def per_type_mrr(frame: pd.DataFrame) -> pd.DataFrame:
    """Each type's MRR per regime group: over its positives within a seed, then the mean over seeds.

    R and S pool into ``"R/S"``; T stays apart (male-specific types have no F1
    name, so they are never T-masked). H rows are left out.
    """
    frame = frame.loc[frame["regime"] != "H"].assign(
        group=lambda f: np.where(f["regime"] == "T", "T", "R/S"),
        weighted=lambda f: f["mrr"] * f["n_positives"],
    )
    per_seed = frame.groupby(["group", "type", "seed"]).agg(
        weighted=("weighted", "sum"), n_positives=("n_positives", "sum"),
        **{flag: (flag, "first") for flag in _FLAGS},
    )
    per_seed["mrr"] = per_seed["weighted"] / per_seed["n_positives"]
    return per_seed.groupby(["group", "type"]).agg(
        mrr=("mrr", "mean"), n_positives=("n_positives", "sum"),
        **{flag: (flag, "first") for flag in _FLAGS},
    ).reset_index()


def categories(types: pd.Series, dimorphism: pd.Series) -> pd.Series:
    """Each type's dimorphism category: its release label, ``isomorphic`` when null."""
    labels = pd.Series(dimorphism.to_numpy(), index=types.to_numpy())
    labels = labels[~labels.index.duplicated()].fillna(ISOMORPHIC)
    unknown = sorted(set(labels) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"unknown dimorphism labels {unknown}; known: {list(CATEGORIES)}")
    return labels


def category_table(per_type: pd.DataFrame, labels: pd.Series, prereg: PreregConfig) -> pd.DataFrame:
    """One row per (regime group, category, type_trained): types, positives, mean MRR and its interval.

    The interval resamples types through ``src.gate``'s bootstrap with the
    registered seed and resample count. The gate resamples mask instances;
    resampling types is this report's own choice.
    """
    typed = per_type.loc[per_type["type"] != UNTYPED].assign(
        category=lambda f: f["type"].map(labels).fillna(ISOMORPHIC),
        trained=lambda f: f["type_trained"].astype(str),
    )
    rows = []
    for (group, category, trained), part in typed.groupby(["group", "category", "trained"], sort=True):
        values = part["mrr"].to_numpy(dtype=float)
        low, high = _interval(values, prereg)
        rows.append({
            "group": group, "category": category, "type_trained": trained,
            "n_types": len(part), "n_positives": int(part["n_positives"].sum()),
            "mrr": float(values.mean()), "ci_low": low, "ci_high": high,
        })
    return pd.DataFrame(rows, columns=["group", "category", "type_trained", "n_types",
                                       "n_positives", "mrr", "ci_low", "ci_high"])


def _interval(values: np.ndarray, prereg: PreregConfig) -> tuple[float, float]:
    if values.size < 2:
        return float("nan"), float("nan")
    low, high, _, _ = _bootstrap_ci(values, prereg)
    return low, high
