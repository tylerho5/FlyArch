"""B1 candidates over frozen experts on a union graph (Phase 3 plan 05).

Inputs are expert score caches (``train.union_scoring``), each row's candidate pairs,
and node-local features from the union table: per-region innervation, summed over each
region's labels. Membership, regional degree vectors and boundary flags are never
inputs; :func:`region_profile` refuses a table with a column outside the allowlist.

- :func:`profile_route` is #151's innervation-profile router. It has no fitted
  parameters: each expert's row z-scored scores are weighted, per candidate pair, by
  the pair's innervation in that expert's region (source plus target), normalised over
  experts. ``home_share`` adds the hybrid: when the masked endpoint's profile puts at
  least that share in one region, that region's expert alone scores the row.
- :func:`fit_gate` is the regularised linear gate: a softmax over experts of a linear
  map of label-blind features, fit on F1 training rows only with outer folds grouped
  by masked root.
"""

from __future__ import annotations

import numpy as np

from src.boundary_suites import check_inputs
from src.features import FeatureSchema, FeatureTable
from src.regions import region_labels

__all__ = ["apply_gate", "fit_gate", "profile_route", "region_profile"]


def region_profile(table: FeatureTable, schema: FeatureSchema, regions, *, uniform_rows=None) -> np.ndarray:
    """(N, len(regions)) innervation share per region; ``uniform_rows`` (dense indices) get 1/len."""
    check_inputs(schema, table.columns)
    index = {c: k for k, c in enumerate(table.columns)}
    profile = np.stack([table.values[:, [index[f"innervation_{label}"] for label in region_labels(code)]].sum(axis=1)
                        for code in regions], axis=1)
    if uniform_rows is not None:
        profile[np.asarray(uniform_rows, dtype=np.int64)] = 1.0 / len(regions)
    return profile


def _z(scores: np.ndarray) -> np.ndarray:
    mean, sd = scores.mean(axis=-1, keepdims=True), scores.std(axis=-1, keepdims=True)
    return np.divide(scores - mean, sd, out=np.zeros_like(scores), where=sd > 0)


def profile_route(scores: np.ndarray, pairs: np.ndarray, profile: np.ndarray, *,
                  masked: np.ndarray | None = None, home_share: float | None = None) -> np.ndarray:
    """Route ``scores`` (experts, rows, K+1) for ``pairs`` (rows, K+1, 2) by innervation profile.

    Expert ``e`` must be the region in ``profile``'s column ``e``. ``masked`` (rows,) is
    each row's masked endpoint, needed only with ``home_share``.
    """
    if scores.shape[1:] != pairs.shape[:2] or scores.shape[0] != profile.shape[1]:
        raise ValueError("scores, pairs and profile do not align")
    weights = profile[pairs[..., 0]] + profile[pairs[..., 1]]  # rows, K+1, experts
    total = weights.sum(axis=-1, keepdims=True)
    weights = np.divide(weights, total, out=np.full_like(weights, 1.0 / scores.shape[0]), where=total > 0)
    if home_share is not None:
        if masked is None:
            raise ValueError("home_share needs each row's masked endpoint")
        share = profile[masked]
        home = share.max(axis=1) >= home_share
        weights[home] = np.eye(scores.shape[0])[share[home].argmax(axis=1)][:, None, :]
    return (np.moveaxis(_z(scores), 0, -1) * weights).sum(axis=-1)


def _gate_weights(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    logits = features @ theta
    logits -= logits.max(axis=-1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=-1, keepdims=True)


def apply_gate(scores: np.ndarray, features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Gate ``scores`` (experts, rows, K+1) with per-candidate ``features`` (rows, K+1, F)."""
    return (np.moveaxis(_z(scores), 0, -1) * _gate_weights(features, theta)).sum(axis=-1)


def fit_gate(scores: np.ndarray, features: np.ndarray, *, datasets, groups, folds: int = 5,
             l2: float = 1.0, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """Fit the linear gate on F1 training rows; return its parameters and outer-fold losses.

    The loss is the listwise softmax cross-entropy of each row's positive (column 0).
    ``datasets`` (rows,) must all be ``"f1"``: rows from any F2 suite are refused.
    ``groups`` (rows,) is each row's masked root; folds never split a root.
    """
    from scipy.optimize import minimize

    datasets, groups = np.asarray(datasets), np.asarray(groups)
    n_rows = scores.shape[1]
    if len(datasets) != n_rows or len(groups) != n_rows or features.shape[:2] != scores.shape[1:]:
        raise ValueError("scores, features, datasets and groups must describe the same rows and candidates")
    if (datasets != "f1").any():
        raise ValueError("fit_gate refuses rows from F2: the gate fits on F1 training views only")
    z = _z(scores)
    n_features, n_experts = features.shape[-1], scores.shape[0]

    def loss(theta, rows, penalty):
        theta = theta.reshape(n_features, n_experts)
        fused = (np.moveaxis(z[:, rows], 0, -1) * _gate_weights(features[rows], theta)).sum(axis=-1)
        fused = fused - fused.max(axis=1, keepdims=True)
        return float(-(fused[:, 0] - np.log(np.exp(fused).sum(axis=1))).mean() + penalty * (theta ** 2).sum())

    def fit(rows):
        return minimize(loss, np.zeros(n_features * n_experts), args=(rows, l2), method="L-BFGS-B").x

    order = np.random.default_rng(seed).permutation(np.unique(groups))
    fold_of = {g: k % folds for k, g in enumerate(order)}
    assignment = np.array([fold_of[g] for g in groups])
    held = []
    for k in range(folds):
        train, test = np.flatnonzero(assignment != k), np.flatnonzero(assignment == k)
        if not len(test) or not len(train):
            continue
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("a masked root is in both a fit fold and its evaluation fold")
        held.append(loss(fit(train), test, 0.0))  # held-out cross-entropy, no penalty
    return fit(np.arange(len(groups))).reshape(n_features, n_experts), held
