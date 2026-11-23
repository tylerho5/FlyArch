"""The pair-feature builder, the type-pair encoding, and the three baselines."""

import dataclasses
import importlib
import weakref

import numpy as np
import pandas as pd
import pytest

from src.boundary_suites import BoundarySuite
from src.evaluate import CommittedNegatives, build_scoring_task, score_instance
from src.masks import MaskInstance, apply_mask, random_mask, typed_mask
from src.negatives import DEFAULT_SPEC, SIDE_SOURCE, SIDE_TARGET, pack_edges
from src.suites import Suite, SuiteInstance, build_suite, suite_negatives
from train import baselines
from train.baselines import (
    BASELINES,
    GLOBAL_MEAN_KEY,
    LOGREG_GRID,
    XGBOOST_GRID,
    fit_type_encoding,
    pair_features,
)

#: The suite name F2-labelled fixture graphs are scored under (plan 05).
_SUITE_NAME = "f2_dev"


def _require_sklearn():
    pytest.importorskip("sklearn", exc_type=ImportError)


def _import_xgboost_or_skip():
    try:
        return importlib.import_module("xgboost")
    except (ImportError, OSError) as exc:
        pytest.skip(f"xgboost cannot be imported: {exc}")
    except Exception as exc:
        error_type = type(exc)
        if (
            error_type.__name__ == "XGBoostError"
            and error_type.__module__.startswith("xgboost")
        ):
            pytest.skip(f"xgboost is unusable: {exc}")
        raise


def _fixture_view(graph, *, fraction=0.05, seed=13):
    return apply_mask(graph, random_mask(graph, fraction=fraction, seed=seed))


def _frozen_task(graph, view, sample):
    """Wrap one fixture instance in a real Suite and its committed digest."""
    mask = view.instance
    suite = Suite(
        name=_SUITE_NAME,
        dataset=graph.dataset,
        region=graph.region,
        graph_digest=graph.digest(),
        threshold=graph.threshold,
        master_seed=DEFAULT_SPEC.master_seed,
        instances=(
            SuiteInstance(
                instance_id=mask.instance_id,
                regime=mask.regime,
                seed=mask.seed,
                params=dict(mask.params),
                masked_root_ids=graph.node_ids[mask.masked],
            ),
        ),
    )
    committed = CommittedNegatives(
        name=_SUITE_NAME, spec=DEFAULT_SPEC, digests={mask.instance_id: sample.digest()}
    )
    return build_scoring_task(view, sample, suite=suite, negatives=committed)


def _score_task(model, view, table, task):
    return score_instance(task, lambda pairs: model.score(view.model_view(), table, pairs))


def _context(graph, *, excluded_seed=44):
    exclusion = _fixture_view(graph, seed=excluded_seed)
    return baselines.F1TrainingContext(graph, _frozen_suite(graph, exclusion, name="f1_val"))


def _fit_view(context, *, fraction=0.05, seed=13):
    """A fit view on the pruned F1-train graph, drawn from the complement pool."""
    return context.draw_fit_view(np.random.default_rng(seed), fraction=fraction)


def _excluded_neurons(context):
    """Every dense index the exclusion suite reserves."""
    return np.unique(
        np.concatenate(
            [
                np.asarray(v.masked)
                for v in context.exclusion_suite.resolve(context.graph)
            ]
        )
    )


def _frozen_suite(graph, view, *, name="f2_dev"):
    mask = view.instance
    return Suite(
        name=name, dataset=graph.dataset, region=graph.region,
        graph_digest=graph.digest(), threshold=graph.threshold,
        master_seed=DEFAULT_SPEC.master_seed,
        instances=(SuiteInstance(
            instance_id=mask.instance_id, regime=mask.regime, seed=mask.seed,
            params=dict(mask.params), masked_root_ids=graph.node_ids[mask.masked],
        ),),
    )


def _dev_artifacts(graph, view, *, spec=DEFAULT_SPEC):
    sample = baselines._candidate_negatives(view, spec=spec)
    return _frozen_suite(graph, view), CommittedNegatives(
        "f2_dev", spec, {view.instance.instance_id: sample.digest()}
    )


def _f1_train(graph, *, excluded_seed=44, fraction=0.05, seed=13):
    """The F1 training context plus one fit view on its pruned graph."""
    context = _context(graph, excluded_seed=excluded_seed)
    return context, _fit_view(context, fraction=fraction, seed=seed)


def _context_avoiding(graph, masked):
    """A context whose F1-val reservation misses every neuron in ``masked``."""
    for excluded_seed in range(1, 500):
        context = _context(graph, excluded_seed=excluded_seed)
        if not np.intersect1d(context.excluded, np.asarray(masked)).size:
            return context
    raise AssertionError("no F1-val exclusion seed avoids the mask")


def _fit_learned(model, context, train, validation, table):
    suite, committed = _dev_artifacts(validation.graph, validation, spec=model.negative_spec)
    return model.fit(
        [train], [table], context=context,
        val_views=[validation], val_tables=[table],
        val_suite=suite, val_negatives=committed,
    )


def test_context_rejects_relabelled_f1_val_and_wrong_graph(cx_graph_f1, cx_graph):
    context = _context(cx_graph_f1)
    excluded = apply_mask(
        context.train_graph, random_mask(cx_graph_f1, fraction=0.05, seed=44)
    )
    relabelled = dataclasses.replace(
        excluded, instance=dataclasses.replace(excluded.instance, instance_id="f1-cx-R05-s999")
    )
    with pytest.raises(ValueError, match="F1-val"):
        context.validate(relabelled)
    with pytest.raises(ValueError, match="F1 graph"):
        context.validate(_fixture_view(cx_graph))


def test_context_prunes_every_f1_val_edge_from_the_train_graph(cx_graph_f1):
    """F1-train is the F1 graph minus the F1-val masks (phase-1 split table)."""
    context = _context(cx_graph_f1)
    excluded = _excluded_neurons(context)

    assert context.train_graph.n_edges < cx_graph_f1.n_edges
    assert not np.isin(context.train_graph.edges, excluded).any()
    # Node ids, dense indexing and row order are unchanged, so a mask drawn
    # against the original graph still addresses the same neurons.
    assert context.train_graph.n_nodes == cx_graph_f1.n_nodes
    assert np.array_equal(context.train_graph.node_ids, cx_graph_f1.node_ids)
    # A fit view may hide only the neurons the exclusion suite did not reserve.
    assert np.array_equal(
        context.fit_pool, np.setdiff1d(np.arange(cx_graph_f1.n_nodes), excluded)
    )
    # The sampler keeps the original edge set as its true-edge reference.
    assert context.true_edges == pack_edges(cx_graph_f1.edges)


def test_pruned_graph_matches_the_m1_training_graph(cx_graph_f1):
    """The baseline copy prunes and draws exactly what M1's training loop does.

    The two helpers are duplicated because the training modules must not
    import each other; these assertions pin them to the same result so they
    cannot drift.
    """
    from train.train import _training_graph, simulate_gaps

    context = _context(cx_graph_f1)
    # The union comes from the suite, not from the context, so the pin covers
    # M1's pruner and the exclusion resolution independently.
    m1 = _training_graph(cx_graph_f1, _excluded_neurons(context))

    assert np.array_equal(context.train_graph.edges, m1.edges)
    assert np.array_equal(context.train_graph.weights, m1.weights)
    # The same rng and pool give the same gap from either entry point.
    mine = context.draw_fit_view(np.random.default_rng(7)).instance
    m1_gap = simulate_gaps(
        context.train_graph, np.random.default_rng(7), pool=context.fit_pool
    )
    assert mine.instance_id == m1_gap.instance_id
    assert np.array_equal(mine.masked, m1_gap.masked)


def test_fit_view_reports_the_unpruned_degree_given_total(cx_graph_f1):
    """A fit row and a scored row report the same granted degree."""
    context = _context(cx_graph_f1)
    view = _fit_view(context)
    complete = cx_graph_f1.in_strength() + cx_graph_f1.out_strength()

    granted = view.degree_given_total()
    assert np.array_equal(granted[view.is_masked], complete[view.is_masked])
    assert not granted[~view.is_masked].any()
    # The pruned copy is what the fit trains on, so the view-relative value
    # would be smaller wherever a masked neuron's F1-val edges went.
    pruned = context.train_graph.in_strength() + context.train_graph.out_strength()
    assert not np.array_equal(granted[view.is_masked], pruned[view.is_masked])
    # A scored F1-val view grants the same quantity.
    scored = apply_mask(cx_graph_f1, context.exclusion_suite.resolve(cx_graph_f1)[0])
    assert np.array_equal(
        scored.degree_given_total()[scored.is_masked], complete[scored.is_masked]
    )


def test_context_reserves_the_real_f1_val_suites_union(cx_graph_f1, cx_types, cx_centroids):
    """The five R/T/S instances' union is what F1-train prunes.

    The hand-made single-instance suites elsewhere cover the plumbing; this
    pins the shape the phase-1 split actually uses.
    """
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    context = baselines.F1TrainingContext(cx_graph_f1, suite)
    excluded = np.unique(
        np.concatenate([np.asarray(v.masked) for v in suite.resolve(cx_graph_f1)])
    )

    assert np.array_equal(context.excluded, excluded)
    assert np.array_equal(
        context.fit_pool, np.setdiff1d(np.arange(cx_graph_f1.n_nodes), excluded)
    )
    assert not np.isin(context.train_graph.edges, excluded).any()
    assert context.train_graph.n_edges < cx_graph_f1.n_edges


def test_context_rejects_a_suite_that_reserves_nothing(cx_graph_f1):
    suite = _frozen_suite(
        cx_graph_f1, _fixture_view(cx_graph_f1), name="f1_val"
    )
    empty = dataclasses.replace(suite, instances=())

    with pytest.raises(ValueError, match="reserves no neurons"):
        baselines.F1TrainingContext(cx_graph_f1, empty)


def test_context_rejects_a_fit_view_on_the_unpruned_f1_graph(cx_graph_f1, cx_features):
    context = _context(cx_graph_f1)
    stale = _fixture_view(cx_graph_f1)

    with pytest.raises(ValueError, match="pruned F1-train graph"):
        context.validate(stale)
    with pytest.raises(ValueError, match="pruned F1-train graph"):
        fit_type_encoding([stale], [cx_features], positives_per_view=200, context=context)
    with pytest.raises(ValueError, match="pruned F1-train graph"):
        BASELINES["logreg"]().fit(
            [stale], [cx_features], context=context,
            val_views=[], val_tables=[], val_suite=None, val_negatives=None,
        )


def test_context_rejects_a_fit_mask_that_touches_a_reserved_neuron(cx_graph_f1):
    context = _context(cx_graph_f1)
    reserved = np.asarray(
        context.exclusion_suite.resolve(cx_graph_f1)[0].masked, dtype=np.int64
    )
    kept = np.setdiff1d(context.fit_pool, reserved)[:1]
    masked = np.sort(np.concatenate([reserved, kept])).astype(np.int32)
    instance = MaskInstance(
        instance_id="f1-cx-gap-s900", dataset=cx_graph_f1.dataset,
        region=cx_graph_f1.region, regime="R", seed=900,
        params={"fraction": 0.05}, masked=masked,
    )

    with pytest.raises(ValueError, match="F1-val"):
        context.validate(apply_mask(context.train_graph, instance))


def test_fit_trains_only_on_the_pruned_f1_train_graph(
    monkeypatch, cx_graph_f1, cx_features
):
    """No F1-val neuron supervises a training row or appears in a positive."""
    context = _context(cx_graph_f1)
    view = _fit_view(context)
    excluded = _excluded_neurons(context)
    train_edges = pack_edges(context.train_graph.edges)
    captured = []
    original = baselines._candidate_negatives

    def spy(view, *, spec, pool=None, true_edges=None):
        sample = original(view, spec=spec, pool=pool, true_edges=true_edges)
        captured.append((sample, true_edges))
        return sample

    monkeypatch.setattr(baselines, "_candidate_negatives", spy)
    encoding = fit_type_encoding(
        [view], [cx_features], positives_per_view=200, context=context
    )
    baselines._training_block(
        [view], [cx_features], encoding, DEFAULT_SPEC, context=context
    )

    assert captured
    for sample, true_edges in captured:
        positives = np.asarray(sample.positives, dtype=np.int64).reshape(-1, 2)
        assert positives.size
        assert not np.isin(positives, excluded).any()
        assert all(((int(u) << 32) | int(v)) in train_edges for u, v in positives)
        # The pruned copy lost the validation edges, so the sampler is handed
        # the original edge set to keep refusing those pairs as negatives.
        assert true_edges == pack_edges(cx_graph_f1.edges)


def test_pair_features_rejects_masked_view(cx_graph, cx_features):
    view = _fixture_view(cx_graph)
    with pytest.raises(TypeError, match="VisibleView"):
        pair_features(view, view.hidden_edges[:1], cx_features, type_encoding={})
    assert pair_features(view.model_view(), view.hidden_edges[:1], cx_features, type_encoding={}).shape[0] == 1
    with pytest.raises(TypeError, match="VisibleView"):
        BASELINES["configuration"]().score(view, cx_features, view.hidden_edges[:1])


def _renamed(suite, name):
    """``suite`` under ``name``, read as a boundary suite when the name is one."""
    cls = BoundarySuite if name.startswith("phase3_boundary_") else Suite
    return cls(**{f.name: getattr(suite, f.name) for f in dataclasses.fields(suite)} | {"name": name})


@pytest.mark.parametrize("name, accepted", [
    ("f1_val", True), ("phase3_boundary_f1_val", True), ("f2_dev", False), ("phase3_boundary_f2_confirm", False),
])
def test_context_excludes_a_region_or_boundary_f1_val_suite_only(cx_graph_f1, name, accepted):
    suite = _renamed(_frozen_suite(cx_graph_f1, _fixture_view(cx_graph_f1, seed=44)), name)
    if accepted:
        assert baselines.F1TrainingContext(cx_graph_f1, suite).excluded.size
    else:
        with pytest.raises(ValueError, match="F1-val exclusion suite"):
            baselines.F1TrainingContext(cx_graph_f1, suite)


@pytest.mark.parametrize("name, accepted", [
    ("f2_dev", True), ("phase3_boundary_f2_dev", True), ("f2_confirm", False), ("phase3_boundary_f2_confirm", False),
])
def test_fit_tunes_on_a_region_or_boundary_dev_suite_only(
    monkeypatch, cx_graph, cx_graph_f1, cx_features, name, accepted
):
    context, train = _f1_train(cx_graph_f1)
    dev = _fixture_view(cx_graph)
    suite, committed = _dev_artifacts(cx_graph, dev)
    model = BASELINES["logreg"]()
    monkeypatch.setattr(model, "_estimator", lambda *a: (_ for _ in ()).throw(AssertionError("fitted")))
    with pytest.raises(AssertionError if accepted else ValueError, match="fitted" if accepted else "F2-dev"):
        model.fit([train], [cx_features], context=context,
                  val_views=[dev], val_tables=[cx_features], val_suite=_renamed(suite, name),
                  val_negatives=dataclasses.replace(committed, name=name))


def test_fit_rejects_a_changed_dev_digest_before_estimator(
    monkeypatch, cx_graph, cx_graph_f1, cx_features
):
    context, train = _f1_train(cx_graph_f1)
    dev = _fixture_view(cx_graph)
    suite, committed = _dev_artifacts(cx_graph, dev)
    model = BASELINES["logreg"]()
    monkeypatch.setattr(model, "_estimator", lambda *a: (_ for _ in ()).throw(AssertionError("fitted")))
    changed = dataclasses.replace(committed, digests={dev.instance.instance_id: "a" * 64})
    with pytest.raises(ValueError, match="digest"):
        model.fit([train], [cx_features], context=context,
                  val_views=[dev], val_tables=[cx_features], val_suite=suite,
                  val_negatives=changed)
    other = _fixture_view(cx_graph, seed=27)
    with pytest.raises(ValueError, match="frozen instances"):
        model.fit([train], [cx_features], context=context,
                  val_views=[other], val_tables=[cx_features], val_suite=suite,
                  val_negatives=committed)


def test_pair_features_reads_only_post_mask_quantities(cx_graph, cx_features):
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=7))
    features = pair_features(view.model_view(), view.hidden_edges, cx_features, type_encoding={})
    rebuilt = apply_mask(cx_graph, view.instance)
    assert features.shape[0] == view.hidden_edges.shape[0]
    assert np.allclose(
        features,
        pair_features(rebuilt.model_view(), rebuilt.hidden_edges, cx_features, type_encoding={}),
    )
    assert np.isfinite(features).all()

    # A builder that secretly indexed hidden_edges would pass the check above,
    # since the rebuilt view holds the same arrays. Scramble them instead: any
    # read of a hidden-edge array has to show up in the result.
    order = np.random.default_rng(11).permutation(len(view.hidden_edges))
    scrambled = dataclasses.replace(
        view,
        hidden_edges=view.hidden_edges[order],
        hidden_weights=view.hidden_weights[order],
    )
    assert not np.array_equal(scrambled.hidden_edges, view.hidden_edges)
    assert np.allclose(
        pair_features(scrambled.model_view(), view.hidden_edges, cx_features, type_encoding={}),
        features,
    )


def test_fit_type_encoding_keys_leave_the_feature_count_unchanged(cx_graph_f1, cx_features):
    context, view = _f1_train(cx_graph_f1, seed=11)
    encoding = fit_type_encoding(
        [view], [cx_features], positives_per_view=200, context=context
    )
    type_pairs = [key for key in encoding if key != GLOBAL_MEAN_KEY]
    assert type_pairs
    assert all(isinstance(source, int) and isinstance(target, int) for source, target in type_pairs)
    assert all(0.0 <= value <= 1.0 for value in encoding.values())
    empty = pair_features(view.model_view(), view.hidden_edges, cx_features, type_encoding={})
    fitted = pair_features(view.model_view(), view.hidden_edges, cx_features, type_encoding=encoding)
    assert fitted.shape == empty.shape
    assert np.isfinite(fitted).all()


def test_baselines_registry_is_the_preregistered_three():
    assert set(BASELINES) == {"configuration", "logreg", "xgboost"}
    assert all(isinstance(cls, type) for cls in BASELINES.values())


def test_grids_span_the_registered_hyperparameters():
    assert [point["C"] for point in LOGREG_GRID] == list(10.0 ** np.arange(-3, 4))

    assert len(XGBOOST_GRID) == 3 * 2 * 2 * 2
    assert {point["max_depth"] for point in XGBOOST_GRID} == {3, 6, 9}
    assert {point["n_estimators"] for point in XGBOOST_GRID} == {200, 600}
    assert {point["learning_rate"] for point in XGBOOST_GRID} == {0.03, 0.1}
    assert {point["subsample"] for point in XGBOOST_GRID} == {0.8, 1.0}


def test_fit_type_encoding_refuses_a_non_f1_view(cx_graph, cx_graph_f1, cx_features):
    view = _fixture_view(cx_graph)
    context = _context(cx_graph_f1)

    with pytest.raises(ValueError, match="F1 graph"):
        fit_type_encoding([view], [cx_features], positives_per_view=200, context=context)
    with pytest.raises(ValueError, match="F1 graph"):
        BASELINES["logreg"]().fit(
            [view], [cx_features], context=context,
            val_views=[], val_tables=[], val_suite=None, val_negatives=None,
        )


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baseline_scores_are_finite_deterministic_and_beat_random(
    name, cx_graph, cx_graph_f1, cx_features, sample_for
):
    if name != "configuration":
        _require_sklearn()
    if name == "xgboost":
        _import_xgboost_or_skip()
    view = _fixture_view(cx_graph)
    task = _frozen_task(cx_graph, view, sample_for(view))
    context, train = _f1_train(cx_graph_f1)
    model = BASELINES[name]()
    if name == "configuration":
        model.fit([train], [cx_features])
    else:
        _fit_learned(model, context, train, view, cx_features)

    first = model.score(view.model_view(), cx_features, task.positives)
    second = model.score(view.model_view(), cx_features, task.positives)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)

    result = _score_task(model, view, cx_features, task)
    assert result.mrr > 1.0 / (DEFAULT_SPEC.k + 1)
    assert _score_task(model, view, cx_features, task).mrr == result.mrr


def test_configuration_baseline_is_the_degree_product_of_both_endpoints(
    cx_graph, cx_graph_f1, cx_features
):
    view = _fixture_view(cx_graph)
    model = BASELINES["configuration"]().fit([_fixture_view(cx_graph_f1)], [cx_features])
    masked = np.asarray(view.is_masked, dtype=bool)
    hidden = np.asarray(view.hidden_edges, dtype=np.int64)
    unmasked = np.flatnonzero(~masked)

    u, v = (int(x) for x in hidden[masked[hidden[:, 0]] & ~masked[hidden[:, 1]]][0])
    w = int(unmasked[unmasked != v][0])
    x, y = (int(z) for z in hidden[~masked[hidden[:, 0]] & masked[hidden[:, 1]]][0])

    total = view.degree_given_total()
    in_strength = view.visible_in_strength()
    out_strength = view.visible_out_strength()
    scores = model.score(view.model_view(), cx_features, np.array([[u, v], [u, w], [w, v], [x, y]]))

    assert scores[0] == float(total[u]) * float(in_strength[v])
    assert scores[1] == float(total[u]) * float(in_strength[w])
    assert scores[2] == float(out_strength[w]) * float(in_strength[v])
    assert scores[3] == float(out_strength[x]) * float(total[y])


def test_fit_ranks_grid_points_on_the_supplied_validation_views(
    cx_graph, cx_graph_f1, cx_features
):
    _require_sklearn()
    context, train = _f1_train(cx_graph_f1, seed=13)
    validation = _fixture_view(cx_graph, seed=21)
    model = _fit_learned(BASELINES["logreg"](), context, train, validation, cx_features)

    assert len(model.grid_mrr) == len(LOGREG_GRID)
    assert np.isfinite(model.grid_mrr).all()
    assert model.champion_params == LOGREG_GRID[int(np.argmax(model.grid_mrr))]
    assert model.training_view_ids == (train.instance.instance_id,)
    assert model.ranking_view_ids == (validation.instance.instance_id,)
    assert np.isfinite(model.score(validation.model_view(), cx_features, validation.hidden_edges)).all()
    with pytest.raises(TypeError, match="VisibleView"):
        model.score(validation, cx_features, validation.hidden_edges)


def test_xgboost_200_tree_prefix_matches_separate_fit():
    # xgboost's sklearn wrapper needs sklearn; the grid baseline uses it too.
    _require_sklearn()
    _import_xgboost_or_skip()
    from xgboost import XGBClassifier
    rng = np.random.default_rng(73)
    features = rng.normal(size=(180, 4)).astype(np.float32)
    labels = ((features[:, 0] + features[:, 1] * 0.3) > 0).astype(int)
    params = dict(max_depth=3, learning_rate=0.1, subsample=0.8,
                  tree_method="hist", n_jobs=1, random_state=19,
                  eval_metric="logloss")
    short = XGBClassifier(n_estimators=200, **params).fit(features, labels)
    long = XGBClassifier(n_estimators=600, **params).fit(features, labels)
    assert np.allclose(
        short.predict_proba(features),
        long.predict_proba(features, iteration_range=(0, 200)), atol=1e-7, rtol=1e-7,
    )


def test_xgboost_guard_skips_a_broken_native_import(monkeypatch):
    class XGBoostError(ValueError):
        pass

    XGBoostError.__module__ = "xgboost._c_api"

    def broken_import(name):
        assert name == "xgboost"
        raise XGBoostError("native library is unavailable")

    monkeypatch.setattr(importlib, "import_module", broken_import)
    with pytest.raises(pytest.skip.Exception, match="xgboost is unusable"):
        _import_xgboost_or_skip()


def test_fit_type_encoding_requires_f1_train_provenance(cx_graph_f1, cx_features):
    context = _context(cx_graph_f1)
    reserved = apply_mask(
        context.train_graph, random_mask(cx_graph_f1, fraction=0.05, seed=44)
    )
    with pytest.raises(ValueError, match="F1-val"):
        fit_type_encoding(
            [reserved], [cx_features], positives_per_view=200, context=context
        )


def test_learned_fit_requires_f2_dev_validation(cx_graph, cx_graph_f1, cx_features):
    context, train = _f1_train(cx_graph_f1)
    f1_val = _fixture_view(cx_graph_f1, seed=21)
    f2_view = _fixture_view(cx_graph, seed=34)
    model = BASELINES["logreg"]()
    with pytest.raises(ValueError, match="F2-dev"):
        model.fit([train], [cx_features], context=context,
                  val_views=[], val_tables=[], val_suite=None, val_negatives=None)
    suite, committed = _dev_artifacts(cx_graph, f2_view)
    with pytest.raises(ValueError, match="dataset"):
        model.fit([train], [cx_features], context=context,
                  val_views=[f1_val], val_tables=[cx_features],
                  val_suite=suite, val_negatives=committed)
    with pytest.raises(ValueError, match="F2-dev"):
        model.fit([train], [cx_features], context=context,
                  val_views=[f2_view], val_tables=[cx_features],
                  val_suite=dataclasses.replace(suite, name="f2_confirm"),
                  val_negatives=committed)


def test_fit_subset_keeps_ten_and_balances_both_masked_sides(cx_graph_f1):
    view = _fixture_view(cx_graph_f1, fraction=0.3, seed=5)
    sample = baselines._candidate_negatives(view, spec=DEFAULT_SPEC)
    first = baselines._fit_subset(sample, view, seed=101)
    again = baselines._fit_subset(sample, view, seed=101)
    other = baselines._fit_subset(sample, view, seed=202)
    assert first.negatives.shape == (len(sample.positives), 10, 2)
    assert np.array_equal(first.negatives, again.negatives)
    assert not np.array_equal(first.negatives, other.negatives)
    both = view.both_masked()
    assert np.all(np.sum(first.sides[both] == SIDE_TARGET, axis=1) == 5)
    assert np.all(np.sum(first.sides[both] == SIDE_SOURCE, axis=1) == 5)
    for i in np.flatnonzero(~both)[:10]:
        assert np.unique(first.sides[i]).size == 1
    assert sample.negatives.shape[1] == 100
    assert sample.digest() == baselines._candidate_negatives(view, spec=DEFAULT_SPEC).digest()


def test_training_feature_rows_use_ten_negatives(cx_graph_f1, cx_features):
    context, view = _f1_train(cx_graph_f1)
    full = fit_type_encoding([view], [cx_features], positives_per_view=200,
                             context=context)
    features, labels = baselines._training_block(
        [view], [cx_features], full, DEFAULT_SPEC, context=context
    )
    assert features.shape[0] == len(view.hidden_edges) * 11
    assert features.dtype == np.float32
    assert int(labels.sum()) == len(view.hidden_edges)


def test_training_encoding_column_matches_the_scoring_encoding_scale(
    cx_graph, cx_graph_f1, cx_features
):
    """A training row's encoding and a ranked row's encoding are the same quantity.

    Both must estimate the positive rate of the candidate distribution the
    models are ranked against (``positives + K negatives``), so the feature has
    one scale at fit time and at scoring time. A 10-negative training draw
    would put the column at 1/(10+1) while ranking sees 1/(K+1): the estimator
    then learns on a scale ~9x the one it is applied to, which is the undertuned
    baseline that flatters the GNN (phase-1-poc.md).
    """
    context, train = _f1_train(cx_graph_f1)
    dev = _fixture_view(cx_graph)
    scoring = fit_type_encoding([train], [cx_features], positives_per_view=200,
                                context=context)
    training_features, _ = baselines._training_block(
        [train], [cx_features], scoring, DEFAULT_SPEC, context=context
    )
    candidates = baselines._candidate_block(
        baselines._candidate_negatives(dev, spec=DEFAULT_SPEC)
    )
    ranking_features = pair_features(
        dev.model_view(), candidates.reshape(-1, 2), cx_features, type_encoding=scoring
    )

    base_rate = 1.0 / (DEFAULT_SPEC.k + 1)
    training_encoding = training_features[:, 1].mean()
    assert training_encoding == pytest.approx(base_rate, rel=0.3)
    assert ranking_features[:, 1].mean() == pytest.approx(base_rate, rel=0.3)
    assert training_encoding == pytest.approx(ranking_features[:, 1].mean(), rel=0.5)


def test_crossfit_fold_counts_use_the_full_candidate_draw(
    monkeypatch, cx_graph_f1, cx_features
):
    context, view = _f1_train(cx_graph_f1)
    scoring = fit_type_encoding([view], [cx_features], positives_per_view=200,
                                context=context)
    rates = []
    original = baselines.pair_features

    def capture(*args, **kwargs):
        rates.append(kwargs["type_encoding"][GLOBAL_MEAN_KEY])
        return original(*args, **kwargs)

    monkeypatch.setattr(baselines, "pair_features", capture)
    baselines._training_block([view], [cx_features], scoring, DEFAULT_SPEC, context=context)
    assert rates
    assert all(rate == pytest.approx(scoring[GLOBAL_MEAN_KEY], rel=0.3) for rate in rates)


def test_candidate_negatives_match_suite_negatives(cx_graph):
    view = _fixture_view(cx_graph)
    mask = view.instance
    suite = Suite(
        name="f2_dev",
        dataset=cx_graph.dataset,
        region=cx_graph.region,
        graph_digest=cx_graph.digest(),
        threshold=cx_graph.threshold,
        master_seed=DEFAULT_SPEC.master_seed,
        instances=(
            SuiteInstance(
                instance_id=mask.instance_id,
                regime=mask.regime,
                seed=mask.seed,
                params=dict(mask.params),
                masked_root_ids=cx_graph.node_ids[mask.masked],
            ),
        ),
    )

    expected = suite_negatives(cx_graph, suite)[mask.instance_id]
    actual = baselines._candidate_negatives(view, spec=DEFAULT_SPEC)

    assert actual.digest() == expected.digest()
    assert np.array_equal(actual.positives, expected.positives)
    assert np.array_equal(actual.negatives, expected.negatives)
    assert np.array_equal(actual.sides, expected.sides)


def test_type_encoding_uses_the_masked_candidate_distribution(
    cx_graph_f1, cx_features, cx_types
):
    instance = typed_mask(cx_graph_f1, cx_types, seed=7)
    context = _context_avoiding(cx_graph_f1, instance.masked)
    view = apply_mask(context.train_graph, instance)
    # The fit draw is what fit_type_encoding will use: the pruned graph's
    # visible degrees, the original edge set as the true-edge reference.
    sample = baselines._candidate_negatives(
        view, spec=DEFAULT_SPEC, true_edges=context.true_edges
    )
    candidates = np.concatenate(
        [sample.positives[:, None, :], sample.negatives], axis=1
    )
    truth = np.zeros(candidates.shape[:2], dtype=float)
    truth[:, 0] = 1.0
    type_ids = baselines._type_ids(cx_features, cx_graph_f1.n_nodes).astype(np.int64)
    candidate_types = type_ids[candidates.reshape(-1, 2)]
    expected = {}
    for pair in np.unique(candidate_types, axis=0):
        rows = np.all(candidate_types == pair, axis=1)
        expected[(int(pair[0]), int(pair[1]))] = float(truth.reshape(-1)[rows].mean())

    encoding = fit_type_encoding(
        [view],
        [cx_features],
        positives_per_view=len(view.hidden_edges),
        context=context,
    )

    assert encoding[GLOBAL_MEAN_KEY] == pytest.approx(1.0 / (DEFAULT_SPEC.k + 1))
    for pair, rate in expected.items():
        assert encoding[pair] == pytest.approx(rate)


def test_crossfit_excludes_heldout_group_labels_and_uses_unseen_fallback():
    pair_types = np.array([
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
    ])
    folds = np.array([0, 1])
    labels = np.array([[1., 0.], [1., 0.]])
    original = baselines._crossfit_encodings(pair_types, labels, folds)[0]
    changed = labels.copy()
    changed[0] = [0., 1.]
    after = baselines._crossfit_encodings(pair_types, changed, folds)[0]
    assert original == after
    assert original[(1, 2)] == 1.0
    assert original[GLOBAL_MEAN_KEY] == 0.5
    assert baselines._encoding_column(np.array([9]), np.array([9]), original)[0] == 0.5


def test_one_view_crossfit_training_features_do_not_use_full_encoding(
    cx_graph_f1, cx_features
):
    context, view = _f1_train(cx_graph_f1)
    full = fit_type_encoding([view], [cx_features], positives_per_view=200,
                             context=context)
    features, labels = baselines._training_block(
        [view], [cx_features], full, DEFAULT_SPEC, context=context
    )
    assert len(features) == len(labels)
    assert np.isfinite(features).all()
    assert not np.array_equal(features[:, 1], np.full(len(features), full[GLOBAL_MEAN_KEY]))


def test_two_missing_sides_do_not_match(cx_graph, cx_features):
    view = _fixture_view(cx_graph)
    side_columns = [i for i, name in enumerate(cx_features.columns) if name.startswith("side_")]
    values = cx_features.values.copy()
    values[:2, side_columns] = 0
    table = dataclasses.replace(cx_features, values=values)
    row = pair_features(view.model_view(), np.array([[0, 1]]), table, type_encoding={})
    assert row[0, -2] == 0.0


def test_degree_free_removes_degree_inputs_and_uses_uniform_negatives(
    monkeypatch, cx_graph, cx_features
):
    view = _fixture_view(cx_graph)
    pairs = view.hidden_edges[:10]
    degree_given = pair_features(
        view.model_view(),
        pairs,
        cx_features,
        type_encoding={},
        negative_spec=DEFAULT_SPEC,
    )
    degree_free_spec = DEFAULT_SPEC.for_degree_free()

    def forbidden_degree_total(self):
        raise AssertionError("degree-free scoring read degree_given_total")

    monkeypatch.setattr(type(view.model_view()), "degree_given_total", forbidden_degree_total)
    degree_free = pair_features(
        view.model_view(),
        pairs,
        cx_features,
        type_encoding={},
        negative_spec=degree_free_spec,
    )

    assert degree_free.shape[1] == degree_given.shape[1] - 1
    assert np.array_equal(degree_free, degree_given[:, :-1])
    assert baselines._candidate_negatives(view, spec=degree_free_spec).spec == degree_free_spec
    scores = baselines.ConfigurationBaseline(negative_spec=degree_free_spec).score(
        view.model_view(), cx_features, pairs
    )
    assert np.array_equal(scores, np.zeros(len(pairs)))


def test_degree_free_learned_baseline_fits_and_scores(
    cx_graph, cx_graph_f1, cx_features
):
    _require_sklearn()
    context, train = _f1_train(cx_graph_f1)
    validation = _fixture_view(cx_graph, seed=21)
    model = _fit_learned(
        BASELINES["logreg"](negative_spec=DEFAULT_SPEC.for_degree_free()),
        context, train, validation, cx_features,
    )

    scores = model.score(validation.model_view(), cx_features, validation.hidden_edges[:10])
    assert np.isfinite(scores).all()


def test_grid_search_prepares_each_view_once(
    monkeypatch, cx_graph, cx_graph_f1, cx_features
):
    _require_sklearn()
    context, train = _f1_train(cx_graph_f1)
    validation = _fixture_view(cx_graph, seed=21)
    suite, committed = _dev_artifacts(cx_graph, validation)
    calls = {train.instance.instance_id: 0, validation.instance.instance_id: 0}
    original = baselines._candidate_negatives

    def counted(view, *, spec, **kwargs):
        calls[view.instance.instance_id] += 1
        return original(view, spec=spec, **kwargs)

    monkeypatch.setattr(baselines, "_candidate_negatives", counted)
    BASELINES["logreg"]().fit(
        [train], [cx_features], context=context,
        val_views=[validation], val_tables=[cx_features],
        val_suite=suite, val_negatives=committed,
    )

    assert calls == {
        train.instance.instance_id: 1,
        validation.instance.instance_id: 1,
    }


def test_score_requires_a_completed_fit(cx_graph, cx_features):
    model = BASELINES["logreg"]()
    view = _fixture_view(cx_graph)

    with pytest.raises(RuntimeError, match="fit must be called"):
        model.score(view.model_view(), cx_features, view.hidden_edges[:1])
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.score(view.model_view(), cx_features, np.empty((0, 2), dtype=np.int64))


def test_three_seeds_are_recorded_and_predict_from_visible_view(cx_graph, cx_graph_f1, cx_features):
    _require_sklearn()
    context, train = _f1_train(cx_graph_f1)
    validation = _fixture_view(cx_graph)
    model = _fit_learned(BASELINES["logreg"](), context, train, validation, cx_features)
    assert len(model.seeds) == 3
    assert len(set(model.seeds)) == 3
    assert len(model.seed_models) == 3
    assert len(model.grid_seed_mrr) == len(LOGREG_GRID)
    assert all(len(values) == 3 for values in model.grid_seed_mrr)
    assert model.grid_mrr[model.champion_index] == pytest.approx(
        np.mean(model.grid_seed_mrr[model.champion_index])
    )
    for seed in model.seeds:
        scores = model.score_seed(validation.model_view(), cx_features, validation.hidden_edges[:5], seed=seed)
        assert scores.shape == (5,)
        assert np.isfinite(scores).all()
    with pytest.raises(TypeError, match="VisibleView"):
        model.score_seed(validation, cx_features, validation.hidden_edges[:5], seed=model.seeds[0])


def test_grid_rank_uses_gated_rows_and_seed_mrr_not_averaged_probabilities(tmp_path):
    # First row: A wins under 2/3 seeds, B is always rank two. Second row is
    # exploratory: B wins it, so including it would choose B instead.
    from types import SimpleNamespace
    path = tmp_path / "features.npy"
    np.save(path, np.arange(6, dtype=np.float32).reshape(-1, 1))
    block = SimpleNamespace(path=path, shape=(2, 3), gated=np.array([True, False]))

    class FixedEstimator:
        def __init__(self, scores):
            self.scores = np.asarray(scores)
        def predict_proba(self, features):
            p = self.scores[features[:, 0].astype(int)]
            return np.column_stack([1 - p, p])

    a = [
        FixedEstimator([.51, .49, .1, .1, .8, .9]),
        FixedEstimator([.51, .49, .1, .1, .8, .9]),
        FixedEstimator([0., .99, .98, .1, .8, .9]),
    ]
    b = [FixedEstimator([.6, .7, .1, .9, .8, .1]) for _ in range(3)]
    a_mrr = np.mean([baselines._mean_mrr(model, [block]) for model in a])
    b_mrr = np.mean([baselines._mean_mrr(model, [block]) for model in b])
    assert a_mrr > b_mrr
    averaged_a = np.mean([model.scores for model in a], axis=0)
    assert averaged_a[0] < averaged_a[1]
    assert np.mean([baselines._mean_mrr(model, [block]) for model in b]) == 0.5


def test_grid_selects_mean_seed_mrr_on_gated_rows(
    monkeypatch, cx_graph, cx_graph_f1, cx_features
):
    from types import SimpleNamespace
    seeds = tuple(int(s) for s in np.random.SeedSequence(DEFAULT_SPEC.master_seed).generate_state(3))

    class FixedEstimator:
        def __init__(self, choice, seed):
            self.choice, self.seed = choice, seed
        def fit(self, features, labels):
            return self
        def predict_proba(self, features):
            if self.choice == "A":
                first = [0., .99, .98] if self.seed == seeds[2] else [.51, .49, .1]
                scores = first + [.1, .8, .9]
            else:
                scores = [.6, .7, .1, .9, .8, .1]
            p = np.asarray(scores)[features[:, 0].astype(int)]
            return np.column_stack([1 - p, p])

    class TinyGrid(baselines._PairFeatureBaseline):
        grid = ({"choice": "A"}, {"choice": "B"})
        def _estimator(self, params, seed):
            return FixedEstimator(params["choice"], seed)

    def fake_blocks(views, tables, tasks, encoding, spec, directory):
        path = __import__("pathlib").Path(directory) / "features.npy"
        np.save(path, np.arange(6, dtype=np.float32).reshape(-1, 1))
        gated = baselines.evaluated_rows(
            dataclasses.replace(tasks[0], strata=pd.DataFrame({"both_masked": [False, True]}))
        )
        return (SimpleNamespace(path=path, shape=(2, 3), gated=gated),)

    monkeypatch.setattr(baselines, "_ranking_blocks", fake_blocks)
    context, train = _f1_train(cx_graph_f1)
    dev = _fixture_view(cx_graph)
    model = _fit_learned(TinyGrid(), context, train, dev, cx_features)
    assert model.champion_index == 0
    assert model.grid_mrr[0] == pytest.approx((1 + 1 + 1 / 3) / 3)
    assert model.grid_mrr[1] == pytest.approx(0.5)


def test_validation_feature_arrays_do_not_accumulate_across_views(
    monkeypatch, tmp_path, cx_graph, cx_features
):
    views = [_fixture_view(cx_graph, seed=seed) for seed in (13, 21, 33)]
    records = tuple(_frozen_suite(cx_graph, view).instances[0] for view in views)
    suite = dataclasses.replace(_frozen_suite(cx_graph, views[0]), instances=records)
    samples = [baselines._candidate_negatives(view, spec=DEFAULT_SPEC) for view in views]
    committed = CommittedNegatives(
        "f2_dev", DEFAULT_SPEC,
        {view.instance.instance_id: sample.digest() for view, sample in zip(views, samples)},
    )
    tasks = [build_scoring_task(view, sample, suite=suite, negatives=committed)
             for view, sample in zip(views, samples)]
    live = []
    peak = [0]
    original = baselines.pair_features

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        live.append(weakref.ref(result))
        peak[0] = max(peak[0], sum(ref() is not None for ref in live))
        return result

    monkeypatch.setattr(baselines, "pair_features", counted)
    blocks = baselines._ranking_blocks(
        views, [cx_features] * len(views), tasks, {}, DEFAULT_SPEC, str(tmp_path)
    )
    assert len(blocks) == 3
    assert peak[0] == 1
    assert all(ref() is None for ref in live)
    assert all(block.path.exists() and block.gated.any() for block in blocks)


def test_grid_temporary_features_are_removed_on_success_and_error(
    monkeypatch, tmp_path, cx_graph, cx_graph_f1, cx_features
):
    _require_sklearn()
    monkeypatch.setattr(baselines.tempfile, "tempdir", str(tmp_path))
    context, train = _f1_train(cx_graph_f1)
    validation = _fixture_view(cx_graph)
    model = BASELINES["logreg"]()
    _fit_learned(model, context, train, validation, cx_features)
    assert list(tmp_path.iterdir()) == []
    class BrokenEstimator:
        def fit(self, features, labels):
            raise RuntimeError("estimator failed")
    monkeypatch.setattr(model, "_estimator", lambda params, seed: BrokenEstimator())
    with pytest.raises(RuntimeError, match="estimator failed"):
        _fit_learned(model, context, train, validation, cx_features)
    assert list(tmp_path.iterdir()) == []


def test_failed_refit_preserves_the_previous_fitted_state(
    monkeypatch, cx_graph, cx_graph_f1, cx_features
):
    _require_sklearn()
    context, train = _f1_train(cx_graph_f1)
    validation = _fixture_view(cx_graph, seed=21)
    model = _fit_learned(BASELINES["logreg"](), context, train, validation, cx_features)
    before_encoding = dict(model.type_encoding)
    before_scores = model.score(validation.model_view(), cx_features, validation.hidden_edges[:10])

    class BrokenEstimator:
        def fit(self, features, labels):
            raise RuntimeError("fit failed")

    monkeypatch.setattr(model, "_estimator", lambda params, seed: BrokenEstimator())
    with pytest.raises(RuntimeError, match="fit failed"):
        _fit_learned(model, context, _fit_view(context, seed=55), validation, cx_features)

    assert model.type_encoding == before_encoding
    assert np.array_equal(
        model.score(validation.model_view(), cx_features, validation.hidden_edges[:10]), before_scores
    )
