"""Tests for the evaluation harness: task construction, per-instance scoring,
and the results/strata reporting frames."""

import json
from dataclasses import replace

import numpy as np
import pytest

from src.evaluate import (
    STRATA_AXES,
    CommittedNegatives,
    InstanceResult,
    ScoringTask,
    build_scoring_task,
    candidate_pairs,
    constant_scorer,
    gated_mrr,
    random_scorer,
    results_frame,
    score_instance,
    strata_frame,
)
from src.gate import REQUIRED_COLUMNS
from src.masks import apply_mask, random_mask
from src.negatives import DEFAULT_SPEC, NegativeSpec
from src.suites import Suite, SuiteInstance

_FRAME_KWARGS = dict(model="gnn_a4", seed=0, regime="c")

#: The registered suite name for each dataset the tests build graphs from.
_SUITE_FOR_DATASET = {"f2": "f2_dev", "f1": "f1_val"}


def _suite_of(graph, *masks, name):
    """A Suite wrapping the given masks, for exercising the frozen contract.

    The pinned suite sizes and tags are plan 04's concern; these tests need
    specific mask shapes, so they wrap single instances in a real ``Suite``
    (same dataset/region/graph pinning) rather than using ``random_mask``
    directly.
    """
    return Suite(
        name=name,
        dataset=graph.dataset,
        region=graph.region,
        graph_digest=graph.digest(),
        threshold=graph.threshold,
        master_seed=DEFAULT_SPEC.master_seed,
        instances=tuple(
            SuiteInstance(
                instance_id=mask.instance_id,
                regime=mask.regime,
                seed=mask.seed,
                params=dict(mask.params),
                masked_root_ids=graph.node_ids[mask.masked],
            )
            for mask in masks
        ),
    )


def _committed(name, spec, **digests):
    return CommittedNegatives(name=name, spec=spec, digests=digests)


def _build(graph, sample_for, *, fraction=0.2, seed=1, types=None, spec=DEFAULT_SPEC):
    """(view, sample, task) for one freshly masked instance of ``graph``."""
    mask = random_mask(graph, fraction=fraction, seed=seed)
    view = apply_mask(graph, mask)
    sample = sample_for(view, spec=spec)
    name = _SUITE_FOR_DATASET[graph.dataset]
    return view, sample, build_scoring_task(
        view,
        sample,
        suite=_suite_of(graph, mask, name=name),
        negatives=_committed(name, spec, **{mask.instance_id: sample.digest()}),
        types=types,
    )


def test_build_scoring_task_copies_negative_sample_arrays_unchanged(cx_graph, sample_for):
    view, sample, task = _build(cx_graph, sample_for)

    assert isinstance(task, ScoringTask)
    assert task.instance_id == view.instance.instance_id
    assert np.array_equal(task.positives, sample.positives)
    assert np.array_equal(task.negatives, sample.negatives)
    assert np.array_equal(task.sides, sample.sides)
    assert task.sides.dtype == np.uint8
    assert task.positives.shape == sample.positives.shape
    assert task.negatives.shape == sample.negatives.shape


def test_task_carries_provenance_from_the_suite(cx_graph, sample_for):
    view, sample, task = _build(cx_graph, sample_for)

    assert task.suite == "f2_dev"
    assert task.dataset == cx_graph.dataset
    assert task.region == cx_graph.region
    assert task.degree_setting == "degree_given"
    assert task.negatives_digest == sample.digest()


def test_strata_shape_and_both_masked_and_direction_untyped(cx_graph, sample_for):
    view, sample, task = _build(cx_graph, sample_for)
    strata = task.strata

    assert len(strata) == sample.n_positives
    assert list(strata.columns) == [
        "both_masked", "direction", "source_type", "target_type", "type_matched",
        "type_trained", "siblings",
    ]
    assert np.array_equal(strata["both_masked"].to_numpy(), view.both_masked())

    source = sample.positives[:, 0]
    expect_direction = np.where(view.is_masked[source], "out", "in")
    assert np.array_equal(strata["direction"].to_numpy(), expect_direction)

    # Sanity: a 20% random mask over 300 nodes produces both both_masked
    # states and both directions, so the equality assertions above are not
    # vacuously true.
    assert strata["both_masked"].any() and not strata["both_masked"].all()
    assert (strata["direction"] == "out").any() and (strata["direction"] == "in").any()

    assert strata["source_type"].isna().all()
    assert strata["target_type"].isna().all()
    assert strata["type_matched"].isna().all()


def test_strata_types_filled_by_node_id_when_types_given(cx_graph, sample_for, cx_types):
    view, sample, task = _build(cx_graph, sample_for, types=cx_types)
    strata = task.strata

    values = cx_types.reindex(view.graph.node_ids).to_numpy()
    source, target = sample.positives[:, 0], sample.positives[:, 1]
    assert np.array_equal(strata["source_type"].to_numpy(), values[source])
    assert np.array_equal(strata["target_type"].to_numpy(), values[target])
    # cx_types has no missing values, so every masked endpoint is typed.
    assert strata["type_matched"].all()


def test_type_matched_is_false_when_a_masked_endpoints_type_is_missing(
    cx_graph, sample_for, cx_types
):
    view, sample, task = _build(cx_graph, sample_for, types=cx_types)

    both_masked = view.both_masked()
    idx = int(np.flatnonzero(~both_masked)[0])
    source, target = sample.positives[idx]
    masked_node = source if view.is_masked[source] else target
    root_id = view.graph.node_ids[masked_node]

    assert task.strata["type_matched"].iloc[idx] == True  # noqa: E712

    types = cx_types.copy()
    types.loc[root_id] = np.nan
    mask = random_mask(cx_graph, fraction=0.2, seed=1)
    strata = build_scoring_task(
        view,
        sample,
        suite=_suite_of(cx_graph, mask, name="f2_dev"),
        negatives=_committed("f2_dev", DEFAULT_SPEC, **{mask.instance_id: sample.digest()}),
        types=types,
    ).strata
    assert strata["type_matched"].iloc[idx] == False  # noqa: E712


def test_mismatched_positives_raises_value_error(cx_graph, sample_for):
    view, sample, _ = _build(cx_graph, sample_for)
    bad_positives = sample.positives.copy()
    bad_positives[0, 0] = (bad_positives[0, 0] + 1) % view.graph.n_nodes
    bad_sample = replace(sample, positives=bad_positives)
    mask = random_mask(cx_graph, fraction=0.2, seed=1)

    with pytest.raises(ValueError, match="hidden_edges"):
        build_scoring_task(
            view,
            bad_sample,
            suite=_suite_of(cx_graph, mask, name="f2_dev"),
            negatives=_committed("f2_dev", DEFAULT_SPEC, **{mask.instance_id: sample.digest()}),
        )


def test_wrong_k_spec_is_rejected(cx_graph, sample_for):
    """A K=2 sample cannot reach a scorer: MRR is a function of K."""
    mask = random_mask(cx_graph, fraction=0.2, seed=1)
    view = apply_mask(cx_graph, mask)
    tiny = sample_for(view, spec=NegativeSpec(k=2))
    assert tiny.negatives.shape[1] == 2

    # Committed spec is the pinned K=100, so the sample's spec does not match.
    with pytest.raises(ValueError, match="spec"):
        build_scoring_task(
            view,
            tiny,
            suite=_suite_of(cx_graph, mask, name="f2_dev"),
            negatives=_committed("f2_dev", DEFAULT_SPEC, **{mask.instance_id: tiny.digest()}),
        )


def test_committed_spec_cannot_leave_the_pinned_k(cx_graph, sample_for):
    """A K=2 sample cannot reach a scorer: MRR is a function of K."""
    mask = random_mask(cx_graph, fraction=0.2, seed=1)
    view = apply_mask(cx_graph, mask)
    tiny = sample_for(view, spec=NegativeSpec(k=2))
    assert tiny.negatives.shape[1] == 2

    # Even a self-consistent K=2 digest is refused: the committed spec must be
    # the pinned one, so K cannot be redefined by the artefact that carries it.
    with pytest.raises(ValueError, match="pinned"):
        _committed("f2_dev", NegativeSpec(k=2), **{mask.instance_id: tiny.digest()})

    with pytest.raises(ValueError, match="spec"):
        build_scoring_task(
            view,
            tiny,
            suite=_suite_of(cx_graph, mask, name="f2_dev"),
            negatives=_committed("f2_dev", DEFAULT_SPEC, **{mask.instance_id: tiny.digest()}),
        )


def test_foreign_neuropil_identity_is_rejected(cx_graph, sample_for):
    view, sample, _ = _build(cx_graph, sample_for)
    mask = random_mask(cx_graph, fraction=0.2, seed=1)
    foreign = replace(sample, neuropil=f"f1-cx-R01-s999")

    with pytest.raises(ValueError, match="neuropil"):
        build_scoring_task(
            view,
            foreign,
            suite=_suite_of(cx_graph, mask, name="f2_dev"),
            negatives=_committed("f2_dev", DEFAULT_SPEC, **{mask.instance_id: sample.digest()}),
        )


def test_instance_outside_the_frozen_suite_is_rejected(cx_graph, sample_for):
    view, sample, _ = _build(cx_graph, sample_for)
    other = random_mask(cx_graph, fraction=0.2, seed=2)

    with pytest.raises(ValueError, match="frozen instances"):
        build_scoring_task(
            view,
            sample,
            suite=_suite_of(cx_graph, other, name="f2_dev"),
            negatives=_committed("f2_dev", DEFAULT_SPEC, **{view.instance.instance_id: sample.digest()}),
        )


def test_reused_instance_id_with_changed_mask_is_rejected(cx_graph, sample_for):
    view, sample, _ = _build(cx_graph, sample_for)
    mask = random_mask(cx_graph, fraction=0.2, seed=1)
    replacement = next(i for i in range(cx_graph.n_nodes) if i not in mask.masked)
    changed_nodes = mask.masked.copy()
    changed_nodes[0] = replacement
    changed = replace(mask, masked=np.sort(changed_nodes))
    changed_view = apply_mask(cx_graph, changed)
    changed_sample = sample_for(changed_view)
    with pytest.raises(ValueError, match="frozen mask"):
        build_scoring_task(
            changed_view, changed_sample,
            suite=_suite_of(cx_graph, mask, name="f2_dev"),
            negatives=_committed("f2_dev", DEFAULT_SPEC, **{mask.instance_id: changed_sample.digest()}),
        )


def test_suite_and_negatives_name_mismatch_is_rejected(cx_graph, sample_for):
    view, sample, _ = _build(cx_graph, sample_for)
    mask = random_mask(cx_graph, fraction=0.2, seed=1)

    with pytest.raises(ValueError, match="disagree"):
        build_scoring_task(
            view,
            sample,
            suite=_suite_of(cx_graph, mask, name="f2_dev"),
            negatives=_committed("f1_val", DEFAULT_SPEC, **{mask.instance_id: sample.digest()}),
        )


def test_tampered_negatives_are_rejected_by_the_committed_digest(cx_graph, sample_for):
    view, sample, _ = _build(cx_graph, sample_for)
    mask = random_mask(cx_graph, fraction=0.2, seed=1)

    tampered = sample.negatives.copy()
    tampered[:, 0, 1] = (tampered[:, 0, 1] + 1) % view.graph.n_nodes
    bad_sample = replace(sample, negatives=tampered)
    assert bad_sample.digest() != sample.digest()

    with pytest.raises(ValueError, match="digest"):
        build_scoring_task(
            view,
            bad_sample,
            suite=_suite_of(cx_graph, mask, name="f2_dev"),
            negatives=_committed("f2_dev", DEFAULT_SPEC, **{mask.instance_id: sample.digest()}),
        )


def test_out_of_range_negative_endpoint_is_rejected(cx_graph, sample_for):
    view, sample, _ = _build(cx_graph, sample_for)
    mask = random_mask(cx_graph, fraction=0.2, seed=1)

    tampered = sample.negatives.copy()
    tampered[:, 0, 1] = view.graph.n_nodes + 3

    with pytest.raises(ValueError, match="outside"):
        build_scoring_task(
            view,
            replace(sample, negatives=tampered),
            suite=_suite_of(cx_graph, mask, name="f2_dev"),
            negatives=_committed("f2_dev", DEFAULT_SPEC, **{mask.instance_id: sample.digest()}),
        )


def test_malformed_sides_are_rejected(cx_graph, sample_for):
    view, sample, _ = _build(cx_graph, sample_for)
    mask = random_mask(cx_graph, fraction=0.2, seed=1)

    with pytest.raises(ValueError, match="SIDE_TARGET"):
        build_scoring_task(
            view,
            replace(sample, sides=np.full_like(sample.sides, 7)),
            suite=_suite_of(cx_graph, mask, name="f2_dev"),
            negatives=_committed("f2_dev", DEFAULT_SPEC, **{mask.instance_id: sample.digest()}),
        )

    with pytest.raises(ValueError, match="sides"):
        build_scoring_task(
            view,
            replace(sample, sides=sample.sides[:, :-1]),
            suite=_suite_of(cx_graph, mask, name="f2_dev"),
            negatives=_committed("f2_dev", DEFAULT_SPEC, **{mask.instance_id: sample.digest()}),
        )


def test_unregistered_alpha_cannot_be_committed(cx_graph, sample_for):
    mask = random_mask(cx_graph, fraction=0.2, seed=1)
    view = apply_mask(cx_graph, mask)
    sample = sample_for(view)

    with pytest.raises(ValueError, match="degree correction"):
        _committed("f2_dev", NegativeSpec(alpha=0.5), **{mask.instance_id: sample.digest()})


def test_degree_free_spec_maps_to_the_degree_free_setting(cx_graph, sample_for):
    free = DEFAULT_SPEC.for_degree_free()
    _, _, task = _build(cx_graph, sample_for, spec=free)

    assert task.degree_setting == "degree_free"


def test_committed_negatives_round_trip(tmp_path, cx_graph):
    mask = random_mask(cx_graph, fraction=0.2, seed=1)
    path = tmp_path / "f2_dev.negatives.json"
    path.write_text(
        json.dumps({
            "name": "f2_dev",
            "spec": {"k": 100, "alpha": 1.0, "master_seed": 20260818},
            "digests": {mask.instance_id: "a" * 64},
        }),
        encoding="utf-8",
    )

    committed = CommittedNegatives.read(path)

    assert committed.name == "f2_dev"
    assert committed.spec == DEFAULT_SPEC
    assert committed.digest_for(mask.instance_id) == "a" * 64

    with pytest.raises(ValueError, match="no committed digest"):
        committed.digest_for("f2-cx-R01-s999")


def test_committed_negatives_rejects_a_non_digest(tmp_path):
    path = tmp_path / "f2_dev.negatives.json"
    path.write_text(
        json.dumps({
            "name": "f2_dev",
            "spec": {"k": 100, "alpha": 1.0, "master_seed": 20260818},
            "digests": {"f2-cx-R01-s1000": "not-a-digest"},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sha256"):
        CommittedNegatives.read(path)


def test_committed_negatives_rejects_a_missing_spec(tmp_path):
    path = tmp_path / "f2_dev.negatives.json"
    path.write_text(json.dumps({"name": "f2_dev", "digests": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="not a committed negatives file"):
        CommittedNegatives.read(path)


def _perfect_scorer(task):
    """Test-local helper: scores 1.0 for a task's true positive pairs, else 0.0."""
    true_pairs = {tuple(p) for p in task.positives.tolist()}

    def _score(pairs):
        return np.array(
            [1.0 if tuple(p) in true_pairs else 0.0 for p in pairs.tolist()],
            dtype=np.float64,
        )

    return _score


def test_random_scorer_mean_mrr_matches_harmonic_expectation(cx_graph, cx_types, sample_for):
    _, _, task = _build(cx_graph, sample_for, types=cx_types)

    k = DEFAULT_SPEC.k
    expected = float(np.sum(1.0 / np.arange(1, k + 2)) / (k + 1))
    assert expected == pytest.approx(0.05146, abs=1e-4)

    mrrs = [score_instance(task, random_scorer(seed)).mrr for seed in range(30)]
    assert np.mean(mrrs) == pytest.approx(expected, abs=0.01)


def test_constant_scorer_gives_pessimistic_worst_case(cx_graph, sample_for):
    _, _, task = _build(cx_graph, sample_for)

    result = score_instance(task, constant_scorer(0.5))

    assert isinstance(result, InstanceResult)
    assert result.mrr == pytest.approx(1.0 / (DEFAULT_SPEC.k + 1))
    assert result.recall_at_10 == 0.0
    assert result.n_positives == int((~task.strata["both_masked"]).sum())
    assert result.suite == task.suite
    assert result.dataset == task.dataset
    assert result.region == task.region
    assert result.degree_setting == task.degree_setting
    assert result.negatives_digest == task.negatives_digest


def test_perfect_scorer_gives_mrr_and_recall_of_one(cx_graph, sample_for):
    _, _, task = _build(cx_graph, sample_for)

    result = score_instance(task, _perfect_scorer(task))

    assert result.mrr == pytest.approx(1.0)
    assert result.recall_at_10 == pytest.approx(1.0)


def test_score_instance_calls_score_fn_once_over_the_full_block(cx_graph, sample_for):
    # The sanity scorers are all per-pair, so they cannot catch a split into
    # separate positive/negative calls; a per-batch-normalising model could.
    # Pin the single-call contract directly instead.
    _, _, task = _build(cx_graph, sample_for)

    block_sizes = []

    def counting_scorer(pairs):
        block_sizes.append(pairs.shape[0])
        return np.full(pairs.shape[0], 0.5)

    score_instance(task, counting_scorer)

    assert block_sizes == [task.positives.shape[0] * (DEFAULT_SPEC.k + 1)]


def test_gated_mrr_over_candidate_pairs_is_score_instances_mrr(cx_graph, sample_for):
    _, _, task = _build(cx_graph, sample_for, fraction=0.30, seed=5)
    pairs = candidate_pairs(task)
    scores = np.random.default_rng(0).normal(size=len(pairs)).astype(np.float32)

    assert pairs.shape == (task.positives.shape[0] * (DEFAULT_SPEC.k + 1), 2)
    assert np.array_equal(pairs[:: DEFAULT_SPEC.k + 1], task.positives)
    assert task.strata["both_masked"].any()
    assert gated_mrr(task, scores) == score_instance(task, lambda _pairs: scores).mrr


def test_strata_rollup_covers_all_axes_and_partitions_positives(cx_graph, cx_types, sample_for):
    _, _, task = _build(cx_graph, sample_for, fraction=0.30, seed=5, types=cx_types)

    result = score_instance(task, constant_scorer(0.1))
    strata = result.strata

    assert set(strata["axis"]) == set(STRATA_AXES)
    for axis in STRATA_AXES:
        axis_rows = strata[strata["axis"] == axis]
        expected = len(task.positives) if axis == "masked" else result.n_positives
        assert axis_rows["n_positives"].sum() == expected

    masked_strata = set(strata.loc[strata["axis"] == "masked", "stratum"])
    assert masked_strata == {"both_masked", "single_masked"}

    direction_strata = set(strata.loc[strata["axis"] == "direction", "stratum"])
    assert direction_strata == {"in", "out"}

    type_match_strata = set(strata.loc[strata["axis"] == "type_match", "stratum"])
    assert type_match_strata <= {"matched", "unmatched"}


@pytest.mark.parametrize("spec", [DEFAULT_SPEC, DEFAULT_SPEC.for_degree_free()])
def test_mask_count_scorer_cannot_improve_gated_mrr(cx_graph, sample_for, spec):
    view, sample, task = _build(cx_graph, sample_for, fraction=0.3, seed=5, spec=spec)
    digest = sample.digest()

    def mask_count(pairs):
        return view.is_masked[pairs].sum(axis=1).astype(float)

    result = score_instance(task, mask_count)
    assert (~task.strata["both_masked"]).sum() == result.n_positives
    assert result.n_positives < len(sample.positives)
    assert result.mrr == pytest.approx(1 / (spec.k + 1))
    assert result.recall_at_10 == 0
    both = result.strata.query("axis == 'masked' and stratum == 'both_masked'")
    assert both.iloc[0]["mrr"] > 1 / (spec.k + 1)
    if spec.alpha == 1.0:
        assert both.iloc[0]["mrr"] == 1.0
    assert sample.digest() == digest
    assert task.negatives_digest == digest


def test_scoring_rejects_instance_without_single_masked_positive(cx_graph, sample_for):
    _, _, task = _build(cx_graph, sample_for, fraction=0.3, seed=5)
    only_both = task.strata.copy()
    only_both["both_masked"] = True
    with pytest.raises(ValueError, match="single-masked"):
        score_instance(replace(task, strata=only_both), constant_scorer(0.5))


def test_strata_rollup_omits_type_match_axis_when_types_is_none(cx_graph, sample_for):
    _, _, task = _build(cx_graph, sample_for, fraction=0.30, seed=5)

    result = score_instance(task, constant_scorer(0.1))

    assert "type_match" not in set(result.strata["axis"])


def _results(cx_graph, cx_types, sample_for, seeds=(1, 2)):
    results = []
    for seed in seeds:
        _, _, task = _build(cx_graph, sample_for, fraction=0.25, seed=seed, types=cx_types)
        results.append(score_instance(task, constant_scorer(0.1)))
    return results


def test_results_frame_has_one_row_per_result_with_required_columns(cx_graph, cx_types, sample_for):
    results = _results(cx_graph, cx_types, sample_for)

    frame = results_frame(results, **_FRAME_KWARGS)

    assert len(frame) == len(results)
    assert set(REQUIRED_COLUMNS) <= set(frame.columns)
    assert {"mrr", "recall_at_10", "n_positives"} <= set(frame.columns)
    assert list(frame["mask_instance"]) == [r.instance_id for r in results]
    assert (frame["mask_regime"] == "c").all()
    assert (frame["model"] == "gnn_a4").all()
    assert (frame["seed"] == 0).all()
    assert list(frame["mrr"]) == [r.mrr for r in results]
    assert list(frame["recall_at_10"]) == [r.recall_at_10 for r in results]
    assert list(frame["n_positives"]) == [r.n_positives for r in results]


def test_results_frame_derives_its_labels_from_result_provenance(cx_graph, cx_types, sample_for):
    """Dataset/region/split/degree_setting come from the results, not the caller."""
    results = _results(cx_graph, cx_types, sample_for)

    frame = results_frame(results, **_FRAME_KWARGS)

    assert (frame["dataset"] == cx_graph.dataset).all()
    assert (frame["region"] == cx_graph.region).all()
    assert (frame["split"] == "f2_dev").all()
    assert (frame["degree_setting"] == "degree_given").all()
    assert list(frame["negatives_digest"]) == [r.negatives_digest for r in results]


def test_results_frame_rejects_mixed_provenance(cx_graph, cx_types, sample_for):
    results = _results(cx_graph, cx_types, sample_for)
    mixed = [*results, replace(results[0], dataset="f1")]

    with pytest.raises(ValueError, match="different suites"):
        results_frame(mixed, **_FRAME_KWARGS)


def test_results_frame_rejects_a_regime_that_does_not_match_the_dataset(
    cx_graph, cx_types, sample_for
):
    results = _results(cx_graph, cx_types, sample_for)

    with pytest.raises(ValueError, match="rung"):
        results_frame(results, model="gnn_a4", seed=0, regime="b")


def test_results_frame_of_empty_results_has_expected_columns_and_no_rows():
    frame = results_frame([], **_FRAME_KWARGS)

    assert len(frame) == 0
    assert set(REQUIRED_COLUMNS) <= set(frame.columns)
    assert {"mrr", "recall_at_10", "n_positives"} <= set(frame.columns)


def test_strata_frame_has_one_row_per_instance_axis_stratum(cx_graph, cx_types, sample_for):
    results = _results(cx_graph, cx_types, sample_for)

    frame = strata_frame(results, **_FRAME_KWARGS)

    assert len(frame) == sum(len(r.strata) for r in results)
    assert set(REQUIRED_COLUMNS) <= set(frame.columns)
    assert {"axis", "stratum", "n_positives", "mrr", "recall_at_10"} <= set(frame.columns)
    assert (frame["mask_regime"] == "c").all()
    assert (frame["split"] == "f2_dev").all()

    expected_instances = []
    for r in results:
        expected_instances.extend([r.instance_id] * len(r.strata))
    assert list(frame["mask_instance"]) == expected_instances


def test_strata_frame_of_empty_results_has_the_same_columns(cx_graph, cx_types, sample_for):
    results = _results(cx_graph, cx_types, sample_for)

    populated = strata_frame(results, **_FRAME_KWARGS)
    empty = strata_frame([], **_FRAME_KWARGS)

    assert len(empty) == 0
    assert list(empty.columns) == list(populated.columns)


def test_f1_suite_results_carry_their_own_dataset_and_split(cx_graph_f1, sample_for):
    """A result scored against F1 cannot be presented as an F2 confirm row."""
    _, _, task = _build(cx_graph_f1, sample_for)
    result = score_instance(task, constant_scorer(0.1))

    frame = results_frame([result], model="gnn_a4", seed=0, regime="b")

    assert frame["dataset"].iloc[0] == "f1"
    assert frame["split"].iloc[0] == "f1_val"

    with pytest.raises(ValueError, match="rung"):
        # "c" is the cross-individual F2 rung; F1 rows cannot wear it.
        results_frame([result], model="gnn_a4", seed=0, regime="c")
