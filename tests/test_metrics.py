"""Metric correctness, checked against hand-computed references.

sklearn is not a dependency of this repo, so the average-precision reference
is worked out by hand below and sklearn is consulted only if it happens to be
installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    average_precision,
    expected_calibration_error,
    hits_at_k,
    mrr,
    prevalence,
    ranks_against_negatives,
    spearman_weight,
)


class TestAveragePrecision:
    def test_matches_hand_computed_reference(self):
        # scores strictly descending, so every element is its own threshold
        y_true = np.array([1, 0, 1, 1, 0, 1])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

        # rank : tp  precision  recall  recall gain  contribution
        #   1  :  1     1/1      1/4       1/4         0.250000
        #   2  :  1     1/2      1/4       0           0
        #   3  :  2     2/3      2/4       1/4         0.166667
        #   4  :  3     3/4      3/4       1/4         0.187500
        #   5  :  3     3/5      3/4       0           0
        #   6  :  4     4/6      4/4       1/4         0.166667
        expected = 0.25 + 2 / 3 / 4 + 0.75 / 4 + 2 / 3 / 4

        assert average_precision(y_true, y_score) == pytest.approx(expected)
        assert average_precision(y_true, y_score) == pytest.approx(0.7708333, abs=1e-6)

    def test_perfect_ranking_scores_one(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        assert average_precision(y_true, y_score) == pytest.approx(1.0)

    def test_constant_score_collapses_to_prevalence(self):
        """A degenerate model must score exactly the base rate, not 1.0.

        This is the tie-handling contract: if equal scores were broken by array
        order, a collapsed decoder could score anywhere from prevalence to 1.0
        depending on how the candidate set happened to be sorted.
        """
        y_true = np.array([1, 0, 0, 0, 1, 0])
        y_score = np.full(6, 0.42)
        assert average_precision(y_true, y_score) == pytest.approx(prevalence(y_true))

    def test_invariant_to_input_order(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        y_true[0] = 1
        y_score = rng.choice([0.1, 0.2, 0.3], size=200)  # heavy ties

        baseline = average_precision(y_true, y_score)
        for _ in range(5):
            perm = rng.permutation(200)
            assert average_precision(y_true[perm], y_score[perm]) == pytest.approx(
                baseline
            )

    def test_matches_sklearn_when_available(self):
        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(7)
        for _ in range(20):
            y_true = rng.integers(0, 2, size=100)
            if y_true.sum() == 0:
                continue
            y_score = rng.random(100).round(2)  # forces ties
            assert average_precision(y_true, y_score) == pytest.approx(
                sklearn_metrics.average_precision_score(y_true, y_score)
            )

    def test_rejects_all_negative_labels(self):
        with pytest.raises(ValueError, match="no positive labels"):
            average_precision(np.zeros(5), np.arange(5.0))

    def test_rejects_non_finite_scores(self):
        with pytest.raises(ValueError, match="finite"):
            average_precision(np.array([1, 0]), np.array([np.nan, 0.5]))


class TestRanking:
    def test_hand_worked_example(self):
        pos = np.array([5.0, 1.0, 3.0])
        neg = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],  # positive beats all -> rank 1
                [2.0, 3.0, 4.0, 5.0],  # positive beaten by all -> rank 5
                [1.0, 2.0, 4.0, 5.0],  # two above -> rank 3
            ]
        )
        assert ranks_against_negatives(pos, neg).tolist() == [1, 5, 3]
        assert mrr(pos, neg) == pytest.approx((1 / 1 + 1 / 5 + 1 / 3) / 3)
        assert hits_at_k(pos, neg, 3) == pytest.approx(2 / 3)
        assert hits_at_k(pos, neg, 1) == pytest.approx(1 / 3)

    def test_constant_scores_are_charged_for_every_tie(self):
        """The default tie policy must punish a collapsed decoder.

        Under optimistic tie-breaking a model emitting one constant score
        scores a perfect MRR of 1.0, which is exactly the silent failure this
        default exists to catch.
        """
        pos = np.zeros(4)
        neg = np.zeros((4, 10))

        assert mrr(pos, neg) == pytest.approx(1 / 11)
        assert mrr(pos, neg, tie_policy="optimistic") == pytest.approx(1.0)
        assert mrr(pos, neg, tie_policy="average") == pytest.approx(1 / 6)

    def test_mrr_bounds(self):
        rng = np.random.default_rng(3)
        pos = rng.random(50)
        neg = rng.random((50, 20))
        value = mrr(pos, neg)
        assert 1 / 21 <= value <= 1.0

    def test_rejects_mismatched_rows(self):
        with pytest.raises(ValueError, match="rows"):
            ranks_against_negatives(np.zeros(3), np.zeros((2, 5)))

    def test_rejects_unknown_tie_policy(self):
        with pytest.raises(ValueError, match="tie_policy"):
            ranks_against_negatives(np.zeros(1), np.zeros((1, 2)), "lenient")


class TestCalibration:
    def test_perfect_calibration_is_zero(self):
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        y_true = np.array([0, 0, 1, 1])
        assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0)

    def test_maximally_overconfident(self):
        """Always says 1.0, right half the time -> ECE = 0.5."""
        y_prob = np.ones(100)
        y_true = np.r_[np.ones(50), np.zeros(50)]
        assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.5)

    def test_probability_one_lands_in_the_top_bin(self):
        """A p=1.0 prediction must not fall out of the histogram."""
        assert expected_calibration_error(
            np.ones(10), np.ones(10), n_bins=5
        ) == pytest.approx(0.0)

    def test_rejects_logits(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            expected_calibration_error(np.array([1, 0]), np.array([3.2, -1.5]))


class TestSpearman:
    def test_monotone_relationship(self):
        counts = np.array([1.0, 5.0, 10.0, 50.0])
        assert spearman_weight(counts, np.log1p(counts)) == pytest.approx(1.0)
        assert spearman_weight(counts, -counts) == pytest.approx(-1.0)

    def test_needs_two_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            spearman_weight(np.array([1.0]), np.array([1.0]))


def test_prevalence_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        prevalence(np.array([]))
