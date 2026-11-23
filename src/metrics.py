"""Evaluation metrics for held-out-neuropil link prediction.

Every function here is a pure map from arrays to numbers. Nothing in this
module touches the connectome, the filesystem, or a model -- which is what
makes the gated statistic testable before any training run exists.

The gated metric is MRR under the personalised, degree-corrected negatives
built by ``src.negatives``. Average precision is computed for the
reconstruction claim but is *not* gated.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TIE_POLICIES",
    "average_precision",
    "expected_calibration_error",
    "hits_at_k",
    "mrr",
    "prevalence",
    "ranks_against_negatives",
    "spearman_weight",
]

#: How a positive scoring exactly equal to negatives is ranked.
#:
#: ``pessimistic`` charges the positive for every tie, so a model emitting a
#: constant score scores MRR = 1/(K+1) rather than a perfect 1.0. That failure
#: mode is real -- a collapsed decoder is a common training bug -- so this is
#: the default and is the registered policy. ``average`` is the midrank
#: convention used by much of the link-prediction literature and is provided
#: for comparability with published numbers, not for the gate.
TIE_POLICIES = ("pessimistic", "average", "optimistic")


def ranks_against_negatives(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    tie_policy: str = "pessimistic",
) -> np.ndarray:
    """Rank each positive among its own negatives.

    Parameters
    ----------
    pos_scores
        Shape ``(P,)``. Score of each true edge.
    neg_scores
        Shape ``(P, K)``. Scores of the K negatives sampled *for that
        positive*. Row ``i`` must correspond to ``pos_scores[i]`` -- these are
        personalised negatives, not a shared pool.
    tie_policy
        One of :data:`TIE_POLICIES`.

    Returns
    -------
    Shape ``(P,)`` of ranks in ``[1, K+1]``. Rank 1 means the positive
    outscored every one of its negatives.
    """
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)

    if pos_scores.ndim != 1:
        raise ValueError(f"pos_scores must be 1-D, got shape {pos_scores.shape}")
    if neg_scores.ndim != 2:
        raise ValueError(f"neg_scores must be 2-D, got shape {neg_scores.shape}")
    if neg_scores.shape[0] != pos_scores.shape[0]:
        raise ValueError(
            f"neg_scores has {neg_scores.shape[0]} rows but there are "
            f"{pos_scores.shape[0]} positives"
        )
    if tie_policy not in TIE_POLICIES:
        raise ValueError(
            f"tie_policy must be one of {TIE_POLICIES}, got {tie_policy!r}"
        )
    if not np.isfinite(pos_scores).all() or not np.isfinite(neg_scores).all():
        raise ValueError("scores must all be finite; got NaN or inf")

    pos_col = pos_scores[:, None]
    n_greater = (neg_scores > pos_col).sum(axis=1)
    n_equal = (neg_scores == pos_col).sum(axis=1)

    if tie_policy == "pessimistic":
        return 1.0 + n_greater + n_equal
    if tie_policy == "optimistic":
        return 1.0 + n_greater
    return 1.0 + n_greater + 0.5 * n_equal


def mrr(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    tie_policy: str = "pessimistic",
) -> float:
    """Mean reciprocal rank. The gated statistic."""
    ranks = ranks_against_negatives(pos_scores, neg_scores, tie_policy)
    return float(np.mean(1.0 / ranks))


def hits_at_k(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    k: int,
    tie_policy: str = "pessimistic",
) -> float:
    """Fraction of positives ranked in the top ``k``. Reported, not gated."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    ranks = ranks_against_negatives(pos_scores, neg_scores, tie_policy)
    return float(np.mean(ranks <= k))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision over a candidate-pair set.

    Equal scores are collapsed into a single threshold, so the result does not
    depend on the order ties happen to arrive in. A model that emits one
    constant score therefore scores exactly the prevalence, which is the
    correct answer for an uninformative model.

    Matches ``sklearn.metrics.average_precision_score``; sklearn is not a
    dependency, so the test module checks against a hand-computed reference and
    against sklearn only when it happens to be installed.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got shape {y_true.shape}")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score must be finite; got NaN or inf")

    labels = y_true.astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        raise ValueError("average_precision is undefined with no positive labels")

    order = np.argsort(-y_score, kind="stable")
    scores_sorted = y_score[order]
    labels_sorted = labels[order]

    # One cut per *distinct* score, taken after the last element of each tied run.
    distinct = np.r_[np.diff(scores_sorted) != 0, True]
    cuts = np.flatnonzero(distinct)

    tp = np.cumsum(labels_sorted)[cuts]
    n_called = cuts + 1

    precision = tp / n_called
    recall = tp / n_pos
    recall_gain = np.diff(np.r_[0.0, recall])

    return float(np.sum(recall_gain * precision))


def prevalence(y_true: np.ndarray) -> float:
    """Positive rate of a candidate set.

    AP is not comparable across neuropils of differing density, so every
    reported AP must be accompanied by this number.
    """
    y_true = np.asarray(y_true).astype(bool)
    if y_true.size == 0:
        raise ValueError("prevalence is undefined on an empty candidate set")
    return float(y_true.mean())


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected calibration error with equal-width bins (Guo et al. 2017).

    Bins are half-open ``[lo, hi)`` except the last, which is closed, so a
    predicted probability of exactly 1.0 lands in the top bin rather than
    falling out of the histogram.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if y_prob.size == 0:
        raise ValueError("ECE is undefined on an empty array")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob must be finite; got NaN or inf")
    if (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("y_prob must lie in [0, 1]; pass probabilities, not logits")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        in_bin = idx == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        accuracy = float(y_true[in_bin].mean())
        confidence = float(y_prob[in_bin].mean())
        ece += (count / y_prob.size) * abs(accuracy - confidence)

    return float(ece)


def spearman_weight(y_true_counts: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rho between true synapse counts and predicted strength.

    Computed on positives only, per the approved plan's regression head. The
    monotone ``log1p`` the head is trained on does not change rank order, so
    this is reported on raw counts.
    """
    y_true_counts = np.asarray(y_true_counts, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true_counts.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true_counts.shape} vs {y_pred.shape}")
    if y_true_counts.size < 2:
        raise ValueError("Spearman rho needs at least 2 points")

    # Imported here: scipy.stats adds ~0.4 s to every training process's startup.
    from scipy import stats

    rho = stats.spearmanr(y_true_counts, y_pred).statistic
    return float(rho)
