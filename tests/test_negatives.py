"""Negative-sampler contracts.

Two of these matter more than the rest:

* ``test_degree_correction_shifts_the_pool`` is the regression test for the
  ``implicit_degree_bias_linkpred`` finding. If alpha silently stopped being
  applied, every other test here would still pass and the benchmark would
  quietly become one a pure-degree predictor can win.
* ``test_digest_is_stable_across_calls`` is what backs the "every model is
  scored against byte-identical negatives" rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluate import CommittedNegatives
from src.negatives import (
    SIDE_SOURCE,
    SIDE_TARGET,
    InsufficientPoolError,
    NegativeSpec,
    _draw_negatives,
    _weights,
    pack_edges,
    sample_negatives,
)
from src.regions import RegionGraph
from src.suites import Suite, suite_negatives


def build_pool(n_high: int = 100, n_low: int = 100):
    """A pool split into high-degree and low-degree halves.

    Node ids ``[0, n_high)`` have degree 10; ``[n_high, n_high + n_low)`` have
    degree 1. Under alpha=1 the high half should absorb ~10/11 of the draws;
    under alpha=0, ~1/2.
    """
    pool = np.arange(n_high + n_low, dtype=np.int64)
    degree = np.r_[np.full(n_high, 10.0), np.full(n_low, 1.0)]
    return pool, degree, n_high


def make_sample(alpha: float = 1.0, k: int = 20, n_positives: int = 40, **kwargs):
    pool, degree, n_high = build_pool()
    rng = np.random.default_rng(11)
    sources = rng.choice(pool, size=n_positives)
    targets = rng.choice(pool, size=n_positives)
    positives = np.c_[sources, targets]
    positives = positives[positives[:, 0] != positives[:, 1]]

    true_edges = pack_edges(map(tuple, positives))
    spec = NegativeSpec(k=k, alpha=alpha, **kwargs)
    return (
        sample_negatives(
            neuropil="TEST_L",
            positives=positives,
            pool=pool,
            in_degree=degree,
            out_degree=degree,
            true_edges=true_edges,
            spec=spec,
        ),
        n_high,
        true_edges,
    )


class TestShapeAndComposition:
    def test_shapes_and_side_split(self):
        sample, _, _ = make_sample(k=21)
        p = sample.n_positives

        assert sample.negatives.shape == (p, 21, 2)
        assert sample.sides.shape == (p, 21)
        # odd K gives the extra corruption to the target side
        assert (sample.sides[0] == SIDE_TARGET).sum() == 11
        assert (sample.sides[0] == SIDE_SOURCE).sum() == 10

    def test_target_side_holds_the_source_fixed(self):
        sample, _, _ = make_sample(k=20)
        for i, (source, target) in enumerate(sample.positives):
            is_target = sample.sides[i] == SIDE_TARGET
            assert (sample.negatives[i, is_target, 0] == source).all()
            assert (sample.negatives[i, ~is_target, 1] == target).all()

    def test_negatives_are_drawn_only_from_the_pool(self):
        sample, _, _ = make_sample()
        pool, _, _ = build_pool()
        assert np.isin(sample.negatives, pool).all()


class TestEligibility:
    def test_no_negative_is_a_true_edge(self):
        sample, _, true_edges = make_sample()
        keys = (sample.negatives[..., 0] << 32) | sample.negatives[..., 1]
        assert not set(keys.ravel().tolist()) & true_edges

    def test_no_duplicates_within_a_row(self):
        """A repeated negative would occupy two of the K slots and distort MRR."""
        sample, _, _ = make_sample(k=20)
        for row in sample.negatives:
            keys = [(int(a) << 32) | int(b) for a, b in row]
            assert len(set(keys)) == len(keys)

    def test_no_self_loops(self):
        sample, _, _ = make_sample()
        assert (sample.negatives[..., 0] != sample.negatives[..., 1]).all()


class TestDegreeCorrection:
    def test_degree_correction_shifts_the_pool(self):
        """alpha=1 must reproduce the degree bias that positives carry.

        With a 10:1 degree split over equal-sized halves, a degree-weighted
        draw lands in the high half with probability 10/11 ~= 0.91, while a
        uniform draw lands there with probability 0.5.
        """
        corrected, n_high, _ = make_sample(alpha=1.0, k=40, n_positives=60)
        uniform, _, _ = make_sample(alpha=0.0, k=40, n_positives=60)

        def high_fraction(sample):
            is_target = sample.sides == SIDE_TARGET
            drawn = sample.negatives[..., 1][is_target]
            return float((drawn < n_high).mean())

        assert high_fraction(corrected) == pytest.approx(10 / 11, abs=0.05)
        assert high_fraction(uniform) == pytest.approx(0.5, abs=0.05)
        assert high_fraction(corrected) > high_fraction(uniform) + 0.3

    def test_zero_degree_nodes_are_never_drawn_when_corrected(self):
        """Consistent with positives, which never have a zero-degree endpoint."""
        pool = np.arange(150, dtype=np.int64)
        degree = np.r_[np.ones(100), np.zeros(50)]
        positives = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)

        sample = sample_negatives(
            neuropil="TEST_L",
            positives=positives,
            pool=pool,
            in_degree=degree,
            out_degree=degree,
            true_edges=pack_edges(map(tuple, positives)),
            spec=NegativeSpec(k=20, alpha=1.0),
        )
        assert (sample.negatives < 100).all()


class TestDeterminism:
    def test_digest_is_stable_across_calls(self):
        first, _, _ = make_sample()
        second, _, _ = make_sample()
        assert first.digest() == second.digest()
        assert np.array_equal(first.negatives, second.negatives)

    def test_different_neuropils_get_different_draws(self):
        """Otherwise every neuropil would share one negative pattern."""
        pool, degree, _ = build_pool()
        positives = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
        kwargs = {
            "positives": positives,
            "pool": pool,
            "in_degree": degree,
            "out_degree": degree,
            "true_edges": pack_edges(map(tuple, positives)),
            "spec": NegativeSpec(k=20),
        }
        left = sample_negatives(neuropil="ME_L", **kwargs)
        right = sample_negatives(neuropil="ME_R", **kwargs)
        assert left.digest() != right.digest()

    def test_spec_change_changes_the_digest(self):
        base, _, _ = make_sample(alpha=1.0)
        free, _, _ = make_sample(alpha=0.0)
        assert base.digest() != free.digest()


class TestMaskedEndpointPreservation:
    """A decoy that drops the masked endpoint is not a query about it.

    ``VisibleView.is_masked`` is handed to every scorer, so a negative with no
    masked endpoint is trivially separable from the positive it is ranked
    against and must never be generated for a masked positive.
    """

    @staticmethod
    def build(positives, masked_ids, *, k=20):
        positives = np.asarray(positives, dtype=np.int64)
        pool, degree, _ = build_pool()
        flags = np.zeros(pool.size, dtype=bool)
        flags[list(masked_ids)] = True
        return sample_negatives(
            neuropil="MASKED",
            positives=positives,
            pool=pool,
            in_degree=degree,
            out_degree=degree,
            true_edges=pack_edges(map(tuple, positives.tolist())),
            spec=NegativeSpec(k=k),
            masked_nodes=flags,
        )

    def test_a_masked_source_keeps_its_source_and_replaces_targets(self):
        sample = self.build([[0, 1]], [0])
        assert (sample.sides == SIDE_TARGET).all()
        assert (sample.negatives[..., 0] == 0).all()

    def test_a_masked_target_keeps_its_target_and_replaces_sources(self):
        sample = self.build([[0, 1]], [1])
        assert (sample.sides == SIDE_SOURCE).all()
        assert (sample.negatives[..., 1] == 1).all()

    def test_a_both_masked_positive_splits_the_draws(self):
        sample = self.build([[0, 1], [2, 3]], [0, 1, 2, 3], k=21)
        # odd K gives the extra corruption to the target side, as the default does
        assert (sample.sides[0] == SIDE_TARGET).sum() == 11
        assert (sample.sides[0] == SIDE_SOURCE).sum() == 10
        assert (sample.sides[1] == SIDE_TARGET).sum() == 11

    def test_a_positive_touching_no_masked_node_is_rejected(self):
        with pytest.raises(ValueError, match="touch no masked node"):
            self.build([[0, 1]], [7])

    def test_a_flags_array_shorter_than_the_endpoints_is_rejected(self):
        pool, degree, _ = build_pool()
        with pytest.raises(ValueError, match="covering every positive endpoint"):
            sample_negatives(
                neuropil="MASKED",
                positives=np.array([[0, 1]], dtype=np.int64),
                pool=pool,
                in_degree=degree,
                out_degree=degree,
                true_edges=pack_edges([(0, 1)]),
                spec=NegativeSpec(k=20),
                masked_nodes=np.array([True]),
            )

    def test_omitting_the_mask_keeps_the_plain_split(self):
        pool, degree, _ = build_pool()
        plain = sample_negatives(
            neuropil="MASKED",
            positives=np.array([[0, 1]], dtype=np.int64),
            pool=pool,
            in_degree=degree,
            out_degree=degree,
            true_edges=pack_edges([(0, 1)]),
            spec=NegativeSpec(k=20),
        )
        assert (plain.sides[0] == SIDE_TARGET).sum() == 10
        assert (plain.sides[0] == SIDE_SOURCE).sum() == 10


class TestGuards:
    def test_pool_smaller_than_k_is_rejected(self):
        pool = np.arange(10, dtype=np.int64)
        positives = np.array([[0, 1]], dtype=np.int64)
        with pytest.raises(InsufficientPoolError, match="cannot supply"):
            sample_negatives(
                neuropil="TINY",
                positives=positives,
                pool=pool,
                in_degree=np.ones(10),
                out_degree=np.ones(10),
                true_edges=pack_edges([(0, 1)]),
                spec=NegativeSpec(k=100),
            )

    def test_saturated_node_is_reported_not_silently_shortened(self):
        """A hub connected to almost everything cannot supply K negatives."""
        pool = np.arange(60, dtype=np.int64)
        hub = 0
        edges = [(hub, other) for other in range(1, 60)]
        positives = np.array([[hub, 1]], dtype=np.int64)

        with pytest.raises(InsufficientPoolError, match="eligible partners"):
            sample_negatives(
                neuropil="HUBBY",
                positives=positives,
                pool=pool,
                in_degree=np.ones(60),
                out_degree=np.ones(60),
                true_edges=pack_edges(edges),
                spec=NegativeSpec(k=50),
            )

    @staticmethod
    def _stalled_hub(n_rare: int):
        """Hub 0 links to every heavy node; only ``n_rare`` light and 30 zero-degree remain."""
        n_pool, n_dead = 120, 30
        heavy = range(1, n_pool - n_dead - n_rare)
        in_degree = np.zeros(n_pool)
        in_degree[list(heavy)] = 1000.0
        in_degree[n_pool - n_dead - n_rare : n_pool - n_dead] = 1.0
        masked = np.zeros(n_pool, dtype=bool)
        masked[0] = True
        return {
            "neuropil": "STALL",
            "positives": np.array([[0, 1]], dtype=np.int64),
            "pool": np.arange(n_pool, dtype=np.int64),
            "in_degree": in_degree,
            "out_degree": np.ones(n_pool),
            "true_edges": pack_edges([(0, w) for w in heavy]),
            "spec": NegativeSpec(k=5),
            "masked_nodes": masked,
        }

    def test_stalled_row_draws_no_zero_weight_decoy(self):
        case = self._stalled_hub(n_rare=6)
        sample = sample_negatives(**case)
        assert sample.negatives.shape == (1, 5, 2)
        decoys = sample.negatives[0, :, 1]
        assert len(set(decoys.tolist())) == 5
        assert (case["in_degree"][decoys] > 0).all()

    def test_stalled_row_with_too_few_weighted_partners_raises(self):
        with pytest.raises(InsufficientPoolError, match="only 4 eligible partners with nonzero"):
            sample_negatives(**self._stalled_hub(n_rare=4))

    def test_positive_endpoint_outside_pool_is_rejected(self):
        pool = np.arange(200, dtype=np.int64)
        positives = np.array([[0, 999]], dtype=np.int64)
        with pytest.raises(ValueError, match="absent from the candidate pool"):
            sample_negatives(
                neuropil="TEST_L",
                positives=positives,
                pool=pool,
                in_degree=np.ones(200),
                out_degree=np.ones(200),
                true_edges=pack_edges([(0, 999)]),
                spec=NegativeSpec(k=20),
            )

    def test_pack_edges_rejects_wide_ids(self):
        with pytest.raises(ValueError, match="densely reindexed"):
            pack_edges([(2**33, 1)])

    @pytest.mark.parametrize("bad_id", [-1, 2**32])
    def test_pool_ids_pack_edges_cannot_pack_are_rejected(self, bad_id):
        pool = np.r_[np.arange(30), bad_id].astype(np.int64)
        with pytest.raises(ValueError, match="densely reindexed"):
            sample_negatives(
                neuropil="WIDE",
                positives=np.array([[0, 1]], dtype=np.int64),
                pool=pool,
                in_degree=np.ones(31),
                out_degree=np.ones(31),
                true_edges=pack_edges([(0, 1)]),
                spec=NegativeSpec(k=20),
            )

    @pytest.mark.parametrize("bad_key", [-1, 2**64])
    def test_true_edges_outside_packed_keys_are_rejected(self, bad_key):
        with pytest.raises(ValueError, match="build it with pack_edges"):
            sample_negatives(
                neuropil="BADKEY",
                positives=np.array([[0, 1]], dtype=np.int64),
                pool=np.arange(30, dtype=np.int64),
                in_degree=np.ones(30),
                out_degree=np.ones(30),
                true_edges=pack_edges([(0, 1)]) | {bad_key},
                spec=NegativeSpec(k=20),
            )

    def test_spec_validates_its_arguments(self):
        with pytest.raises(ValueError, match="k must be"):
            NegativeSpec(k=0)
        with pytest.raises(ValueError, match="alpha must be"):
            NegativeSpec(alpha=-1.0)

    def test_degree_free_helper_only_changes_alpha(self):
        spec = NegativeSpec(k=100, alpha=1.0, master_seed=42)
        free = spec.for_degree_free()
        assert free.alpha == 0.0
        assert (free.k, free.master_seed) == (spec.k, spec.master_seed)


def _reference_sample_side(
    rng, pool, weights, n_wanted, fixed_node, is_target_side, true_edges, neuropil
):
    """The original one-draw-at-a-time sampler, kept as the bitwise oracle."""
    cumulative = _cumulative(weights)
    chosen: list[int] = []
    seen: set[int] = set()
    max_draws = 50 * max(n_wanted, 1) + 100

    for _ in range(max_draws):
        if len(chosen) == n_wanted:
            return np.asarray(chosen, dtype=np.int64)
        candidate = int(pool[np.searchsorted(cumulative, rng.random())])
        if candidate in seen or candidate == fixed_node:
            continue
        key = (
            (fixed_node << 32) | candidate
            if is_target_side
            else (candidate << 32) | fixed_node
        )
        if key in true_edges:
            continue
        seen.add(candidate)
        chosen.append(candidate)

    if len(chosen) == n_wanted:
        return np.asarray(chosen, dtype=np.int64)

    keep = np.array([
        int(w) != fixed_node
        and (
            ((fixed_node << 32) | int(w))
            if is_target_side
            else ((int(w) << 32) | fixed_node)
        )
        not in true_edges
        for w in pool
    ], dtype=bool)
    eligible_weights = weights[keep]
    n_positive = int((eligible_weights > 0).sum())
    if n_positive < n_wanted:
        side = "target" if is_target_side else "source"
        raise InsufficientPoolError(
            f"neuropil {neuropil!r}: node {fixed_node} has only {n_positive} "
            f"eligible partners with nonzero {side}-side weight but {n_wanted} "
            f"negatives are required, so this positive cannot be ranked against "
            f"K degree-weighted negatives."
        )
    return rng.choice(
        pool[keep], size=n_wanted, replace=False,
        p=eligible_weights / eligible_weights.sum(),
    )


def _reference_draw(
    rng, positives, pool, w_in, w_out, n_target, k, true_edges, neuropil
):
    negatives = np.empty((positives.shape[0], k, 2), dtype=np.int64)
    sides = np.empty((positives.shape[0], k), dtype=np.uint8)
    for i, (source, target) in enumerate(positives):
        source, target = int(source), int(target)
        n_tgt = int(n_target[i])
        replacements = _reference_sample_side(
            rng, pool, w_in, n_tgt, source, True, true_edges, neuropil
        )
        negatives[i, :n_tgt, 0] = source
        negatives[i, :n_tgt, 1] = replacements
        sides[i, :n_tgt] = SIDE_TARGET
        replacements = _reference_sample_side(
            rng, pool, w_out, k - n_tgt, target, False, true_edges, neuropil
        )
        negatives[i, n_tgt:, 0] = replacements
        negatives[i, n_tgt:, 1] = target
        sides[i, n_tgt:] = SIDE_SOURCE
    return negatives, sides


def _cumulative(weights):
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return cumulative


def _random_case(
    seed, *, k, n_pool, density, n_positives=12, ids="dense", degrees="int"
):
    """Random pool, degrees, edges and side counts for one equivalence check.

    ``ids`` is ``"dense"`` or ``"wide"`` (up to ``2**32``). ``degrees`` is
    ``"int"`` (small counts with zeros) or ``"float"`` (heavy-tailed, with
    zeros, last node zero).
    """
    case_rng = np.random.default_rng(seed)
    high = 2**32 if ids == "wide" else 4 * n_pool
    pool = case_rng.choice(high, size=n_pool, replace=False).astype(np.int64)
    if degrees == "int":
        in_degree, out_degree = case_rng.integers(0, 6, size=(2, n_pool)).astype(np.float64)
        in_degree[case_rng.integers(n_pool)] += 1  # never all zero
        out_degree[case_rng.integers(n_pool)] += 1
    else:
        heavy = case_rng.pareto(1.5, (2, n_pool))
        in_degree, out_degree = heavy * (case_rng.random((2, n_pool)) < 0.7)
        in_degree[0] = out_degree[0] = 1.0
        in_degree[-1] = out_degree[-1] = 0.0
    n_edges = max(1, int(density * n_pool * n_pool))
    ends = case_rng.integers(n_pool, size=(n_edges, 2))
    ends = ends[ends[:, 0] != ends[:, 1]]
    edges = pool[ends] if len(ends) else pool[None, :2]
    positives = edges[case_rng.integers(len(edges), size=n_positives)]
    choices = np.array([k - k // 2, k, 0])
    alpha = float(case_rng.choice([0.0, 0.5, 1.0]))
    return {
        "positives": positives,
        "pool": pool,
        "w_in": _weights(in_degree, alpha),
        "w_out": _weights(out_degree, alpha),
        "n_target": choices[case_rng.integers(3, size=len(positives))],
        "k": k,
        "true_edges": pack_edges(edges.tolist()),
        "neuropil": f"CASE_{seed}",
    }


def _stalling_case(n_rare: int):
    """Hub 0 links both ways to every node except ``n_rare`` near-weightless ones.

    Rejection sampling cannot find five distinct partners for it within its
    draw budget, so it reaches the enumerating fallback, which then draws
    among the rare nodes by weight or raises when there are fewer than five. Ordinary
    positives before and after make the stream continue past the fallback.
    """
    n_pool = 120
    pool = np.arange(n_pool, dtype=np.int64)
    degree = np.full(n_pool, 1000.0)
    degree[n_pool - n_rare :] = 1e-6
    linked = range(1, n_pool - n_rare)
    edges = [(0, w) for w in linked] + [(w, 0) for w in linked] + [(5, 6), (7, 8)]
    return {
        "positives": np.array([[5, 6], [0, 1], [7, 8], [1, 0], [5, 6]], dtype=np.int64),
        "pool": pool,
        "w_in": _weights(degree, 1.0),
        "w_out": _weights(degree, 1.0),
        "n_target": np.array([3, 5, 2, 0, 5]),
        "k": 5,
        "true_edges": pack_edges(edges),
        "neuropil": "STALL",
    }


def _outcome(draw, case, seed, **options):
    """Arrays or error message, plus the generator state left behind."""
    rng = np.random.default_rng(seed)
    try:
        result = draw(rng, **case, **options)
    except InsufficientPoolError as exc:
        result = str(exc)
    return result, rng.bit_generator.state


#: Read-ahead sizes down to one draw, so refills land inside windows and inside
#: the fallback's draw budget, plus the default.
_BLOCKS = (1, 2, 7, 64, 1 << 16)


def _assert_same_outcome(case, seed):
    expected, expected_state = _outcome(_reference_draw, case, seed)
    for block in _BLOCKS:
        actual, actual_state = _outcome(_draw_negatives, case, seed, block=block)
        assert type(actual) is type(expected)
        if isinstance(expected, str):
            assert actual == expected
        else:
            for want, got in zip(expected, actual):
                assert got.dtype == want.dtype
                assert np.array_equal(got, want)
        assert actual_state == expected_state, f"block={block}"


class TestMatchesReferenceSampler:
    """The block-drawn sampler must reproduce the scalar one draw for draw.

    The committed suite digests and every training run's negatives depend on
    it, so equality covers the output arrays and the generator state after.
    """

    @pytest.mark.parametrize(
        ("k", "n_pool", "density"),
        [(1, 3, 0.3), (2, 6, 0.5), (3, 8, 0.6), (5, 12, 0.2), (7, 40, 0.1),
         (20, 60, 0.3), (20, 200, 0.02), (100, 300, 0.05)],
    )
    @pytest.mark.parametrize("degrees", ["int", "float"])
    def test_random_graphs(self, k, n_pool, density, degrees):
        for seed in range(25):
            case = _random_case(seed, k=k, n_pool=n_pool, density=density, degrees=degrees)
            _assert_same_outcome(case, seed)

    def test_node_ids_beyond_31_bits(self):
        for seed in range(25):
            case = _random_case(seed, k=10, n_pool=50, density=0.2, ids="wide")
            _assert_same_outcome(case, seed)

    @pytest.mark.parametrize("n_pool", [5_000, 100_000])
    def test_pools_that_widen_the_lookup_table(self, n_pool):
        density = 3 / n_pool
        for seed in range(2):
            case = _random_case(
                seed, k=100, n_pool=n_pool, density=density, n_positives=30, degrees="float"
            )
            _assert_same_outcome(case, seed)

    def test_degenerate_cumulative_weights(self):
        """Cumulative sums that pass 1.0 before the end, or are NaN throughout."""
        case = _random_case(0, k=20, n_pool=300, density=0.05, degrees="float")
        assert _cumulative(case["w_in"])[-2] > 1.0
        _assert_same_outcome(case, 0)
        nan = _weights(np.full(300, np.nan), 1.0)
        _assert_same_outcome({**case, "w_in": nan, "w_out": nan}, 0)

    def test_long_stream_crosses_read_ahead_blocks(self):
        case = _random_case(0, k=100, n_pool=400, density=0.05, n_positives=1500)
        _assert_same_outcome(case, 0)

    @pytest.mark.parametrize("n_rare", [5, 6, 30, 4, 0])
    def test_stalled_rejection_falls_back_in_step(self, n_rare):
        for seed in range(10):
            _assert_same_outcome(_stalling_case(n_rare), seed)

    def test_zero_requests_draw_nothing(self):
        case = _random_case(3, k=6, n_pool=30, density=0.1)
        for count in (0, 6):
            n_target = np.full(len(case["positives"]), count)
            _assert_same_outcome({**case, "n_target": n_target}, 3)

    def test_no_positives_leave_the_generator_untouched(self):
        case = _random_case(3, k=1, n_pool=30, density=0.1)
        empty = {"positives": case["positives"][:0], "n_target": case["n_target"][:0]}
        _assert_same_outcome({**case, **empty}, 3)


_DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(
    not (_DATA / "processed" / "f1" / "cx").is_dir(),
    reason="needs the F1/CX region extract in data/processed",
)
def test_committed_f1_val_digests_reproduce():
    graph = RegionGraph.read(_DATA / "processed" / "f1" / "cx")
    suite_dir = _DATA / "suites" / "f1" / "cx"
    committed = CommittedNegatives.read(suite_dir / "f1_val.negatives.json")
    samples = suite_negatives(graph, Suite.read(suite_dir / "f1_val.json"))
    assert {iid: sample.digest() for iid, sample in samples.items()} == dict(
        committed.digests
    )
