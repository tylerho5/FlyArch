"""The opt-in MRR knobs: each is off by default and does what its docstring says."""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import _write_committed_negatives
from src.masks import apply_mask, random_mask
from src.negatives import pack_edges
from src.suites import build_suite, suite_negatives
from train import train as training
from train.model import M1, MODEL_KNOBS, _group_pairs, _type_pair_columns
from train.train import TrainConfig, fit


def _located(table, seed=0):
    values = table.values.copy()
    geometry = table.blocks["geometry"]
    values[:, geometry.start:geometry.start + 6] = np.random.default_rng(seed).normal(
        size=(len(values), 6)
    )
    return dataclasses.replace(table, values=values, geometry_present=np.ones(len(values), bool))


def test_defaults_add_no_parameters_or_columns(cx_schema):
    torch.manual_seed(0)
    default = M1(cx_schema, hidden=8)
    torch.manual_seed(0)
    explicit = M1(cx_schema, hidden=8, **MODEL_KNOBS)
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert not any(k.startswith("pool") or "nt_gates" in k or "attention" in k
                   for k in default.state_dict())
    for name, off in MODEL_KNOBS.items():
        assert getattr(TrainConfig(epochs=1), name) == off


def test_type_pair_columns_count_type_to_type_edges():
    types = torch.tensor([1, 1, 2, 2, 0])
    masked = torch.tensor([True, False, False, False, False])
    edges = torch.tensor([[1, 2], [1, 3]])
    weights = torch.tensor([0.5, 0.25])
    table = _group_pairs(edges, weights, types, masked)
    classes = _group_pairs(edges, weights, torch.zeros(5, dtype=torch.long), masked)
    queries = torch.tensor([[0, 2], [2, 0], [0, 4], [1, 2]])
    columns = _type_pair_columns(queries, table, classes, masked)
    # type 1 -> type 2 has 2 edges over 1 x 2 possible pairs; the reverse has none.
    assert columns[0].tolist() == pytest.approx([1.0, 0.375, np.log1p(2), 0.0])
    assert columns[1].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert columns[2].tolist() == [0.0, 0.0, 0.0, 0.0]  # untyped candidate
    assert columns[3].tolist() == [0.0, 0.0, 0.0, 0.0]  # neither end masked
    dropped = torch.tensor([True, False, False, False, False])
    assert _type_pair_columns(queries, table, classes, masked, dropped)[0].abs().sum() == 0


def test_pair_geometry_reads_centroid_differences(cx_schema, cx_features):
    table = _located(cx_features)
    model = M1(cx_schema, hidden=8, pair_geometry=True)
    x = model._node_features(table)
    columns = model._geometry_columns_for(x, torch.tensor([[0, 1]])).detach()
    names = list(table.columns)
    centroid = [names.index(f"centroid_{a}") for a in "xyz"]
    expected = np.abs(table.values[0, centroid] - table.values[1, centroid])
    assert columns[0, :3].numpy() == pytest.approx(expected, rel=1e-5)
    assert columns[0, 3].item() == pytest.approx(np.linalg.norm(expected), rel=1e-5)
    assert columns[0, 4].item() == 1.0


@pytest.mark.parametrize("knobs", [
    {"type_pair_features": True},
    {"pair_geometry": True},
    {"sibling_pooling": True},
    {"aggregation": "degree_norm"},
    {"aggregation": "attention"},
    {"edge_weight": "count"},
    {"nt_edges": True},
])
def test_each_model_knob_scores_the_same_on_training_and_inference_paths(
    cx_graph, cx_schema, cx_features, knobs
):
    table = _located(cx_features)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.1, seed=1)).model_view()
    hidden = np.asarray(apply_mask(cx_graph, random_mask(cx_graph, fraction=0.1, seed=1)).hidden_edges)
    pairs = np.concatenate([hidden[:5], hidden[:5, ::-1]])
    torch.manual_seed(0)
    model = M1(cx_schema, hidden=8, **knobs)
    if knobs.get("sibling_pooling"):
        torch.nn.init.normal_(model.pool.weight)
    model.eval()
    with torch.no_grad():
        direct = model.score(view, table, pairs)[0]
    inferred = model.score_inference(view, table, pairs)[0]
    assert torch.isfinite(direct).all()
    assert torch.allclose(direct, inferred, atol=1e-5)


@pytest.mark.parametrize("knob", [{"sibling_pooling": True}, {"nt_edges": True}])
def test_pooling_and_nt_gates_start_as_the_unchanged_model(cx_graph, cx_schema, cx_features, knob):
    table = _located(cx_features)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.1, seed=1)).model_view()
    pairs = np.asarray(cx_graph.edges[:6], dtype=np.int64)
    base = M1(cx_schema, hidden=8).eval()
    changed = M1(cx_schema, hidden=8, **knob).eval()
    missing, unexpected = changed.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing
    assert torch.allclose(
        base.score_inference(view, table, pairs)[0], changed.score_inference(view, table, pairs)[0]
    )


def test_feature_noise_perturbs_only_training_scores(cx_graph, cx_schema, cx_features):
    table = _located(cx_features)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.1, seed=1)).model_view()
    pairs = np.asarray(cx_graph.edges[:6], dtype=np.int64)
    model = M1(cx_schema, hidden=8, pair_geometry=True).eval()
    with torch.no_grad():
        clean = model.score(view, table, pairs)[0]
        assert torch.equal(clean, model.score(view, table, pairs)[0])
        assert not torch.allclose(clean, model.score(view, table, pairs, feature_noise=0.5)[0])


def test_type_negatives_are_untrue_type_mates_of_the_partner(cx_graph, cx_features):
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.1, seed=1))
    types = training._type_ids(cx_features, cx_graph.n_nodes).astype(np.int64)
    true = pack_edges(cx_graph.edges)
    pairs, p, k = training._ranking_pairs(view, None, type_negatives=4, types=types)
    base_pairs, _, base_k = training._ranking_pairs(view, None)
    assert k == base_k + 4
    positives = pairs[:p]
    rows = pairs[p:].reshape(p, k, 2)
    masked = np.asarray(view.is_masked)
    for (source, target), extra, first in zip(positives, rows[:, base_k:], rows[:, 0]):
        partner = target if masked[source] else source
        for decoy in extra:
            if (decoy == first).all():
                continue  # padding when the partner type has too few members
            assert (int(decoy[0]) << 32 | int(decoy[1])) not in true
            end = decoy[1] if masked[source] else decoy[0]
            assert types[end] == types[partner] and end != partner


def test_train_negatives_widens_training_rows_only(cx_graph):
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.1, seed=1))
    _, _, k = training._ranking_pairs(view, None, k=120)
    assert k == 120


@pytest.fixture
def _val_suite(cx_graph_f1, cx_types, cx_centroids, tmp_path):
    suite = build_suite(cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids)
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    return suite, suites_dir


def test_holdout_probe_keeps_probe_neurons_out_of_training(
    cx_graph_f1, cx_features_f1, cx_schema, _val_suite, tmp_path, monkeypatch
):
    suite, suites_dir = _val_suite
    seen = []
    real = training._epoch_batch
    monkeypatch.setattr(training, "_epoch_batch",
                        lambda graph, *a, **k: seen.append(graph) or real(graph, *a, **k))
    config = TrainConfig(epochs=2, hidden=8, eval_every=1, holdout_probe=True,
                         run_dir=tmp_path, suites_dir=suites_dir)
    result = fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)
    validation = training.validate_m1_inputs(cx_graph_f1, suite, suites_dir)
    pool = np.setdiff1d(np.arange(cx_graph_f1.n_nodes),
                        training._masked_union(validation.instances))
    held = training.simulate_gaps(
        training._training_graph(cx_graph_f1, training._masked_union(validation.instances)),
        np.random.default_rng([0, 1]), config.gap_fraction, pool,
    ).masked
    trained_edges = seen[0].edges
    assert not np.isin(trained_edges, held).any()
    assert np.isfinite(result.history["train_mrr"]).all()


def test_ema_validates_and_keeps_the_averaged_weights(
    cx_graph_f1, cx_features_f1, cx_schema, _val_suite, tmp_path
):
    suite, suites_dir = _val_suite
    config = TrainConfig(epochs=3, hidden=8, eval_every=1, ema_decay=0.5,
                         run_dir=tmp_path, suites_dir=suites_dir)
    result = fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)
    last = torch.load(tmp_path / result.manifest["run_id"] / "last.pt", weights_only=True)
    assert "ema" in last
    # The live weights moved on from the average they started at.
    assert any(not torch.equal(last["ema"][k], last["model"][k])
               for k in last["model"] if last["model"][k].is_floating_point())
    if result.best_epoch == 2:
        assert all(torch.equal(result.state_dict[k], last["ema"][k].cpu()) for k in last["ema"])


def _with_extras(table, seed=0):
    rng = np.random.default_rng(seed)
    n = len(table.values)
    hist = rng.random((n, 64)).astype(np.float32)
    extras = {
        "out_xyz": rng.normal(size=(n, 3)).astype(np.float32),
        "in_xyz": rng.normal(size=(n, 3)).astype(np.float32),
        "out_ext": rng.random((n, 3)).astype(np.float32),
        "in_ext": rng.random((n, 3)).astype(np.float32),
        "out_hist": hist / hist.sum(axis=1, keepdims=True),
        "in_hist": hist[::-1] / hist[::-1].sum(axis=1, keepdims=True),
        "out_present": np.ones(n, bool),
        "in_present": rng.random(n) > 0.1,
        "supertype": rng.integers(0, 6, n),
        "hemilineage": rng.integers(0, 4, n),
    }
    return dataclasses.replace(_located(table, seed), extras=extras)


def test_group_shares_divide_by_each_groups_total_output_and_input():
    from train.model import _group_pair_shares
    types = torch.tensor([1, 1, 2, 2, 3])
    masked = torch.tensor([True, False, False, False, False])
    edges = torch.tensor([[1, 2], [1, 3], [1, 4], [2, 1]])
    table = _group_pairs(edges, torch.ones(4), types, masked)
    shares = _group_pair_shares(torch.tensor([[0, 2], [0, 4]]), table, torch.tensor([True, True]))
    # type 1 sends 3 edges, 2 of them to type 2; type 2 receives 2 edges, both from type 1.
    assert shares[0].tolist() == pytest.approx([2 / 3, 1.0, 2 / 3, 1.0])
    assert shares[1].tolist() == pytest.approx([1 / 3, 1.0, 1 / 3, 1.0])


def test_low_rank_is_deterministic_and_recovers_a_rank_one_matrix():
    from train.model import _low_rank
    matrix = torch.outer(torch.arange(1.0, 7.0), torch.arange(2.0, 8.0))
    first, second = _low_rank(matrix, 2), _low_rank(matrix, 2)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    left, values, right = first
    assert torch.allclose(left @ torch.diag(values) @ right.T, matrix, atol=1e-3)


def test_nt_pair_columns_read_the_receivers_input_transmitters():
    from train.model import _nt_context, _nt_pair_columns
    # nodes 0..3; NT one-hot over two classes; node 0 masked (type 1), node 3 shares its type.
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    types = torch.tensor([1, 2, 2, 1])
    masked = torch.tensor([True, False, False, False])
    edges = torch.tensor([[1, 2], [2, 3], [1, 3]])  # node 3 receives one class-0 and one class-1 edge
    context = _nt_context(x, [0, 1], edges, types, masked, 3)
    columns = _nt_pair_columns(torch.tensor([[0, 3], [1, 0]]), context, types, masked,
                               torch.tensor([True, True]))
    assert columns[0, 0].item() == pytest.approx(0.5)  # masked source (class 0) into node 3
    assert columns[1, 0].item() == pytest.approx(0.5)  # into masked node 0: its type-mate 3's inputs


@pytest.mark.parametrize("knobs", [
    {"type_pair_shares": True},
    {"annotation_groups": True},
    {"type_profile_rank": 4},
    {"nt_pair_features": True},
    {"type_pair_features": True, "pair_geometry": True, "type_pair_shares": True,
     "annotation_groups": True, "type_profile_rank": 4, "nt_pair_features": True},
])
def test_new_pair_blocks_score_the_same_on_both_paths(cx_graph, cx_schema, cx_features, knobs):
    table = _with_extras(cx_features)
    masked_view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.1, seed=1))
    hidden = np.asarray(masked_view.hidden_edges)
    pairs = np.concatenate([hidden[:5], hidden[:5, ::-1]])
    torch.manual_seed(0)
    model = M1(cx_schema, hidden=8, **knobs).eval()
    with torch.no_grad():
        direct = model.score(masked_view.model_view(), table, pairs)[0]
    inferred = model.score_inference(masked_view.model_view(), table, pairs)[0]
    assert torch.isfinite(direct).all()
    assert torch.allclose(direct, inferred, atol=1e-5)


def test_extras_switches_refuse_a_table_without_extras(cx_graph, cx_schema, cx_features):
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.1, seed=1)).model_view()
    model = M1(cx_schema, hidden=8, annotation_groups=True)
    with pytest.raises(ValueError, match="extras"):
        model.score(view, _located(cx_features), np.asarray(cx_graph.edges[:4]))


def test_extras_are_read_when_present_and_must_match_the_graph(tmp_path):
    from src.feature_io import _read_extras
    assert _read_extras(tmp_path / "extras.npz", 3) is None
    np.savez(tmp_path / "extras.npz", out_xyz=np.zeros((3, 3)), supertype=np.zeros(3))
    assert set(_read_extras(tmp_path / "extras.npz", 3)) == {"out_xyz", "supertype"}
    with pytest.raises(ValueError, match="one row per graph node"):
        _read_extras(tmp_path / "extras.npz", 4)
