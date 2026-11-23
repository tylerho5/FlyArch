"""The M1 encoder/decoder, the gap-simulation training loop, and run manifests."""

import copy
import dataclasses
import gc
import json
import os
import re
import subprocess
import sys
import weakref
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import _write_committed_negatives
from src.features import build_features
from src.masks import MASK_REGIMES, VisibleView, apply_mask, random_mask
from src.negatives import DEFAULT_SPEC, pack_edges
from src.run_control import RunStopped
from src.run_labels import RunLabels
from src.suites import build_suite, f1_train_types, suite_negatives, view_negatives
from train import train as training
from train.model import M1, PairDecoder, _DirectedSAGELayer
from train.train import TrainConfig, TrainResult, fit, simulate_gaps, validation_mrr

#: The pre-ablation regime, for the seeds that pin the resident and streamed paths.
_REAL = TrainConfig(epochs=1, masked_innervation="real")


def test_directed_sage_layer_matches_the_gather_first_form():
    """Evaluating the linear on node rows and gathering is algebraically exact.

    ``nn.Linear`` and ReLU are row-wise, so ``relu(W x[source])`` equals
    ``relu(W x)[source]``; the per-edge weight multiplies after the gather
    either way. Hoisting the two linears off the edge rows is therefore a
    pure performance change, which this pins against a hand-computed
    gather-first reference.
    """
    torch.manual_seed(0)
    n, in_dim, hidden = 12, 5, 7
    layer = _DirectedSAGELayer(in_dim, hidden)
    x = torch.randn(n, in_dim)
    edges = torch.tensor([[0, 1], [1, 2], [3, 0], [2, 5], [5, 5]], dtype=torch.long)
    weights = torch.rand(len(edges))

    got = layer(x, edges, weights)

    source, target = edges[:, 0], edges[:, 1]
    in_message = torch.relu(layer.in_linear(x[source])) * weights[:, None]
    out_message = torch.relu(layer.out_linear(x[target])) * weights[:, None]
    in_aggregate = torch.zeros(n, hidden)
    in_aggregate.index_add_(0, target, in_message)
    out_aggregate = torch.zeros(n, hidden)
    out_aggregate.index_add_(0, source, out_message)
    self_features = torch.relu(layer.self_linear(x))
    expected = torch.cat(
        [in_aggregate + self_features, out_aggregate + self_features], dim=1
    )

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_epoch_loss_applies_type_feature_dropout(cx_graph, cx_features, monkeypatch):
    """Training must exercise the OOV type value the F2 neurons will carry.

    The spec names type-feature dropout as the mitigation for F2's untyped
    (and so OOV) neurons (about 40% of OL, under 9% of each other region on v888):
    without it the encoder never sees the sentinel during F1 training and is
    free to be type-dependent.
    """
    config = TrainConfig(epochs=1, hidden=8)
    model = M1(training._table_schema(cx_features), hidden=8)
    captured = {}
    def spy(_module, inputs, _output):
        captured["ids"] = inputs[0].clone()

    model.type_embedding.register_forward_hook(spy)
    original = cx_features.values.copy()

    training._epoch_loss(
        model, cx_features,
        training._epoch_batch(cx_graph, cx_features, np.random.default_rng(0), config),
        config,
    )

    column = cx_features.columns.index("cell_type")
    ids = captured["ids"].numpy()
    newly_oov = (ids == 0) & (original[:, column] != 0)
    assert newly_oov.sum() > 0
    assert np.array_equal(cx_features.values, original)


def test_cell_type_reaches_the_encoder_only_through_the_embedding(
    cx_graph, cx_schema, cx_features
):
    """Type ids are labels: read as a magnitude, id 8,000 would swamp every other feature."""
    model = M1(cx_schema, hidden=8)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    x = model._node_features(cx_features)

    column = cx_features.columns.index("cell_type")
    ids = torch.as_tensor(cx_features.values[:, column], dtype=torch.long)
    rest = np.delete(cx_features.values, column, axis=1)
    assert x.shape == (cx_graph.n_nodes, rest.shape[1] + model.type_embedding.embedding_dim)
    assert torch.allclose(x[:, : rest.shape[1]], torch.as_tensor(rest, dtype=torch.float32))
    assert torch.equal(x[:, rest.shape[1] :], model.type_embedding(ids))
    assert model.encode(view.model_view(), cx_features).shape[0] == cx_graph.n_nodes


def test_dropout_acts_only_in_training_mode(cx_graph, cx_schema, cx_features):
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1)).model_view()
    pairs = np.array([[0, 1], [2, 3], [4, 5]])
    torch.manual_seed(0)
    plain = M1(cx_schema, hidden=8)
    torch.manual_seed(0)
    dropped = M1(cx_schema, hidden=8, encoder_dropout=0.5, decoder_dropout=0.5)

    dropped.eval()
    for left, right in zip(plain.score(view, cx_features, pairs), dropped.score(view, cx_features, pairs)):
        assert torch.allclose(left, right)
    dropped.train()
    assert not torch.allclose(
        plain.score(view, cx_features, pairs)[0], dropped.score(view, cx_features, pairs)[0]
    )


def test_trained_types_outside_the_vocabulary_name_the_schema(cx_schema):
    model = M1(cx_schema, hidden=8)

    with pytest.raises(ValueError, match="FeatureSchema"):
        model.set_trained_types(np.array([1, model.type_embedding.num_embeddings]))


def test_an_untrained_type_reads_as_oov(cx_schema, cx_features):
    model = M1(cx_schema, hidden=8)
    column = cx_features.columns.index("cell_type")
    ids = cx_features.values[:, column].astype(int)
    kept = np.unique(ids[ids > 0])[:2]

    model.set_trained_types(kept)
    x = model._node_features(cx_features)[:, -model.type_embedding.embedding_dim :]

    oov = model.type_embedding(torch.tensor([0]))[0]
    reads_oov = ~np.isin(ids, kept)
    assert reads_oov.any() and (~reads_oov).any()
    assert torch.equal(x[torch.as_tensor(reads_oov)], oov.expand(int(reads_oov.sum()), -1))
    assert torch.equal(
        x[torch.as_tensor(~reads_oov)],
        model.type_embedding(torch.as_tensor(ids[~reads_oov])),
    )


def test_fit_trains_only_the_types_with_an_f1_train_edge(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path
):
    """A type whose every neuron an F1-val instance hides gets no gradient."""
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    config = TrainConfig(
        epochs=1, hidden=8, patience=1, run_dir=tmp_path, suites_dir=suites_dir
    )

    result = fit(
        cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema
    )

    excluded = training._masked_union(suite.resolve(cx_graph_f1))
    train_graph = training._training_graph(cx_graph_f1, excluded)
    column = cx_features_f1.columns.index("cell_type")
    ids = cx_features_f1.values[:, column].astype(int)
    with_edges = np.unique(train_graph.edges.reshape(-1))
    expected = np.zeros(len(cx_schema.type_vocab) + 1, dtype=bool)
    expected[ids[with_edges]] = True
    expected[0] = True
    trained = result.state_dict["trained_types"].numpy()
    assert np.array_equal(trained, expected)
    val_only = np.setdiff1d(ids, ids[with_edges])
    val_only = val_only[val_only > 0]
    assert val_only.size, "the typed F1-val instance hides every neuron of its type"
    assert not trained[val_only].any()
    # F2 suites draw T types from this same set; the two must not drift.
    pool = f1_train_types(cx_graph_f1, cx_types, suite)
    assert np.flatnonzero(trained[1:]).tolist() == sorted(cx_schema.type_id(t) - 1 for t in pool)

    torch.manual_seed(0)
    fresh = M1(cx_schema, hidden=8, sibling_features=True)
    initial = fresh.type_embedding.weight.detach()[val_only]
    final = result.state_dict["type_embedding.weight"][val_only]
    decay = (final * initial).sum() / initial.square().sum()
    assert torch.allclose(final, decay * initial), "AdamW decay is the only change"
    fresh.load_state_dict(result.state_dict)


def test_fit_rejects_a_table_shorter_than_the_graph(
    cx_graph_f1, cx_features_f1, cx_schema, tmp_path
):
    table = dataclasses.replace(cx_features_f1, values=cx_features_f1.values[:-1])
    config = TrainConfig(epochs=1, hidden=8, patience=1, run_dir=tmp_path)

    with pytest.raises(ValueError, match="not in this graph's node order"):
        fit(cx_graph_f1, table, None, seed=0, config=config, schema=cx_schema)


def test_forward_shapes(cx_graph, cx_schema, cx_features):
    model = M1(cx_schema, hidden=16)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))

    encoded = model.encode(view.model_view(), cx_features)
    assert encoded.shape == (cx_graph.n_nodes, 32)

    logits, counts = model.score(view.model_view(), cx_features, view.hidden_edges)
    assert logits.shape == (len(view.hidden_edges),)
    assert counts.shape == (len(view.hidden_edges),)


#: ``M1(cx_schema, hidden=16)``'s state dict before the knobs existed, in order.
_PRE_KNOB_LAYOUT = [
    ("trained_types", (11,)), ("type_embedding.weight", (11, 32)),
    ("encoder.layers.0.self_linear.weight", (16, 69)), ("encoder.layers.0.self_linear.bias", (16,)),
    ("encoder.layers.0.in_linear.weight", (16, 69)), ("encoder.layers.0.in_linear.bias", (16,)),
    ("encoder.layers.0.out_linear.weight", (16, 69)), ("encoder.layers.0.out_linear.bias", (16,)),
    ("encoder.layers.1.self_linear.weight", (16, 32)), ("encoder.layers.1.self_linear.bias", (16,)),
    ("encoder.layers.1.in_linear.weight", (16, 32)), ("encoder.layers.1.in_linear.bias", (16,)),
    ("encoder.layers.1.out_linear.weight", (16, 32)), ("encoder.layers.1.out_linear.bias", (16,)),
    ("decoder.trunk.0.weight", (32, 65)), ("decoder.trunk.0.bias", (32,)),
    ("decoder.trunk.2.weight", (32, 32)), ("decoder.trunk.2.bias", (32,)),
    ("decoder.edge_head.weight", (1, 32)), ("decoder.edge_head.bias", (1,)),
    ("decoder.count_head.weight", (1, 32)), ("decoder.count_head.bias", (1,)),
]


def test_default_knobs_build_the_pre_knob_model(cx_schema):
    """Default knobs build the parameters, in the order, the model had before they existed."""
    torch.manual_seed(0)
    default = M1(cx_schema, hidden=16)
    torch.manual_seed(0)
    explicit = M1(
        cx_schema, hidden=16, encoder_dropout=0.0, decoder_dropout=0.0,
        sibling_features=False,
    )
    assert [(key, tuple(value.shape)) for key, value in default.state_dict().items()] == (
        _PRE_KNOB_LAYOUT
    )
    for left, right in zip(default.state_dict().values(), explicit.state_dict().values()):
        assert torch.equal(left, right)


@pytest.mark.parametrize("knobs", [
    {"encoder_dropout": 0.2, "decoder_dropout": 0.2}, {"sibling_features": True},
])
def test_architecture_knobs_score(cx_graph, cx_schema, cx_features, knobs):
    model = M1(cx_schema, hidden=16, **knobs)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    logits, counts = model.score(view.model_view(), cx_features, view.hidden_edges)
    assert logits.shape == counts.shape == (len(view.hidden_edges),)
    assert torch.isfinite(logits).all()


def _sibling_columns(count, group):
    count, group = torch.tensor(count), torch.tensor(group)
    return torch.stack([count / group.clamp_min(1), torch.log1p(count), torch.log1p(group)], dim=1)


def test_sibling_features_count_same_type_partners():
    from train.model import PreparedPairs, _sibling_features

    # Masked: 0 and 8 (type 1), 6 (type 4, its only member), 7 (OOV). 1 and 2
    # are 0's visible siblings; masked neurons have no visible edges.
    edges = torch.tensor([[1, 4], [2, 4], [4, 2], [4, 1], [3, 5]], dtype=torch.long)
    type_ids = torch.tensor([1, 1, 1, 2, 3, 3, 4, 0, 1])
    masked = torch.zeros(9, dtype=torch.bool)
    masked[[0, 6, 7, 8]] = True
    pairs = torch.tensor(
        [[0, 4], [4, 0], [0, 5], [3, 4], [6, 4], [7, 4], [0, 8], [8, 0]], dtype=torch.long
    )
    prepared = PreparedPairs(
        n_nodes=9, edges=edges, weights=torch.ones(5), index=pairs,
        pair_features=torch.zeros(8, 1), row_width=1, is_masked=masked,
    )

    got = _sibling_features(prepared, type_ids)
    # Source-masked, target-masked (two siblings), no partner, neither masked,
    # sole member masked, OOV, and both masked (the source's type) each way.
    expected = _sibling_columns(
        [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0]
    )
    assert torch.allclose(got, expected)

    # Type dropout on a sibling leaves its anchor's counts; on the anchor it zeros the row.
    sibling_dropped = torch.zeros(9, dtype=torch.bool)
    sibling_dropped[2] = True
    assert torch.allclose(_sibling_features(prepared, type_ids, sibling_dropped), expected)
    anchor_dropped = torch.zeros(9, dtype=torch.bool)
    anchor_dropped[0] = True
    got = _sibling_features(prepared, type_ids, anchor_dropped)
    assert not got[[0, 1, 2, 6]].any() and torch.allclose(got[7], expected[7])


def test_sibling_features_match_a_per_neuron_count_over_visible_edges(
    cx_graph, cx_schema, cx_features
):
    """Only the visible graph enters: every hidden edge touches a masked neuron."""
    from train.model import _sibling_features

    model = M1(cx_schema, hidden=8, sibling_features=True)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    rng = np.random.default_rng(0)
    pairs = np.concatenate(
        [view.hidden_edges, rng.integers(0, cx_graph.n_nodes, size=(200, 2))]
    )
    type_ids = model.type_ids(cx_features)
    got = _sibling_features(model.prepare(view.model_view(), pairs), type_ids).numpy()

    types, masked = type_ids.numpy(), np.asarray(view.is_masked)
    visible = {tuple(edge) for edge in np.asarray(view.visible_edges).tolist()}
    assert not any(masked[s] or masked[t] for s, t in visible)
    connected = np.zeros(cx_graph.n_nodes, dtype=bool)
    connected[[node for edge in visible for node in edge]] = True
    expected = []
    for s, t in pairs.tolist():
        anchor = s if masked[s] else t
        if not (masked[s] or masked[t]) or types[anchor] == 0:
            expected.append((0.0, 0.0))
            continue
        k = types[anchor]
        siblings = np.flatnonzero((types == k) & connected & ~masked)
        linked = (
            (sibling, t) in visible if masked[s] else (s, sibling) in visible
            for sibling in siblings
        )
        expected.append((float(sum(linked)), float(len(siblings))))
    count, group = map(list, zip(*expected))
    assert np.allclose(got, _sibling_columns(count, group).numpy())


def _decoder_batches(model, monkeypatch, *, chunk):
    """Set the decode chunk and record each decoder batch's pair count."""
    batches = []
    decode = model.decoder.decode

    def spy(source, target, ends, pair_features):
        batches.append(len(ends))
        return decode(source, target, ends, pair_features)

    monkeypatch.setattr(model.decoder, "decode", spy)
    monkeypatch.setattr("train.model._DECODE_CHUNK", chunk)
    return batches


@pytest.mark.parametrize("grad", [False, True])
def test_score_decodes_pairs_in_chunks_with_the_same_scores(
    cx_graph, cx_schema, cx_features, monkeypatch, grad
):
    """Validation decodes ~1.4M F1/CX pairs; one block overflows an 8 GB card."""
    model = M1(cx_schema, hidden=16)
    model.eval()
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    pairs = view.hidden_edges
    with torch.no_grad():
        whole = model.score(view.model_view(), cx_features, pairs)

    batches = _decoder_batches(model, monkeypatch, chunk=7)
    with torch.set_grad_enabled(grad):
        chunked = model.score(view.model_view(), cx_features, pairs)

    assert max(batches) == 7 and sum(batches) == len(pairs)
    assert chunked[0].requires_grad == grad
    for got, expected in zip(chunked, whole):
        assert torch.allclose(got.detach(), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("sibling_features", [False, True])
def test_score_inference_matches_score_without_grad_and_preserves_training_gradients(
    cx_graph, cx_schema, cx_features, sibling_features
):
    model = M1(cx_schema, hidden=16, sibling_features=sibling_features)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    pairs = view.hidden_edges
    model.eval()

    expected = model.score(view.model_view(), cx_features, pairs)
    actual = model.score_inference(view.model_view(), cx_features, pairs)

    for got, want in zip(actual, expected):
        assert torch.allclose(got, want)
        assert not got.requires_grad

    model.train()
    model.score(view.model_view(), cx_features, pairs)[0].sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_score_inference_keeps_ranking_rows_together_with_a_partial_final_chunk(
    cx_graph, cx_schema, cx_features, monkeypatch
):
    model = M1(cx_schema, hidden=16)
    model.eval()
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    pairs = np.tile(cx_graph.edges[:3], (5, 1))
    batches = _decoder_batches(model, monkeypatch, chunk=8)

    model.score_inference(view.model_view(), cx_features, pairs, row_width=3)

    assert batches == [6, 6, 3]


def test_score_inference_bounds_pair_preparation_and_encodes_once(
    cx_graph, cx_schema, cx_features, monkeypatch
):
    model = M1(cx_schema, hidden=16)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    pairs = np.tile(cx_graph.edges[:3], (5, 1))
    monkeypatch.setattr("train.model._DECODE_CHUNK", 8)
    prepared_sizes = []
    feature_sizes = []
    encode_calls = []
    prepare = model.prepare
    pair_features = model._pair_features_from_context
    encode = model.encoder.forward

    def track_prepare(view, pairs, *, row_width=1):
        prepared_sizes.append(len(pairs))
        return prepare(view, pairs, row_width=row_width)

    def track_features(pairs, degree, masked):
        feature_sizes.append(len(pairs))
        return pair_features(pairs, degree, masked)

    def track_encode(*args, **kwargs):
        encode_calls.append(1)
        return encode(*args, **kwargs)

    monkeypatch.setattr(model, "prepare", track_prepare)
    monkeypatch.setattr(model, "_pair_features_from_context", track_features)
    monkeypatch.setattr(model.encoder, "forward", track_encode)

    model.score_inference(view.model_view(), cx_features, pairs, row_width=3)

    assert prepared_sizes == []
    assert feature_sizes == [6, 6, 3]
    assert encode_calls == [1]


@pytest.mark.parametrize(
    ("pairs", "row_width", "message"),
    [
        (np.zeros((2, 2), dtype=np.int64), 0, "positive integer"),
        (np.zeros((2, 2), dtype=np.int64), 3, "not a multiple"),
    ],
)
def test_score_inference_rejects_invalid_ranking_row_width(
    cx_graph, cx_schema, cx_features, pairs, row_width, message
):
    model = M1(cx_schema, hidden=16)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))

    with pytest.raises(ValueError, match=message):
        model.score_inference(view.model_view(), cx_features, pairs, row_width=row_width)


@pytest.mark.parametrize("sibling_features", [False, True])
def test_score_inference_matches_score_across_decoder_chunks(
    cx_graph, cx_schema, cx_features, sibling_features, monkeypatch
):
    """Chunking the pair block must not move a logit, sibling features included."""
    model = M1(cx_schema, hidden=16, sibling_features=sibling_features)
    model.eval()
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    pairs = np.tile(cx_graph.edges[:3], (5, 1))

    expected = model.score(view.model_view(), cx_features, pairs)
    batches = _decoder_batches(model, monkeypatch, chunk=8)
    actual = model.score_inference(view.model_view(), cx_features, pairs, row_width=3)

    assert batches == [6, 6, 3]
    for got, want in zip(actual, expected):
        assert torch.allclose(got, want, atol=1e-6, rtol=1e-5)


def test_score_inference_returns_empty_outputs_for_no_pairs(
    cx_graph, cx_schema, cx_features
):
    model = M1(cx_schema, hidden=16)
    model.eval()
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))

    logits, counts = model.score_inference(
        view.model_view(), cx_features, np.zeros((0, 2), dtype=np.int64), row_width=4
    )

    assert logits.shape == (0,)
    assert counts.shape == (0,)


def test_score_inference_refuses_a_pair_count_below_one_ranking_row(
    cx_graph, cx_schema, cx_features
):
    model = M1(cx_schema, hidden=16)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))

    with pytest.raises(ValueError, match="not a multiple"):
        model.score_inference(
            view.model_view(), cx_features, view.hidden_edges[:1], row_width=4
        )


def test_score_inference_leaves_the_models_mode_alone(cx_graph, cx_schema, cx_features):
    model = M1(cx_schema, hidden=16)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    pairs = view.hidden_edges

    model.train()
    model.score_inference(view.model_view(), cx_features, pairs)
    assert model.training

    model.eval()
    model.score_inference(view.model_view(), cx_features, pairs)
    assert not model.training


def test_score_inference_scores_in_eval_mode_from_a_training_model(
    cx_graph, cx_schema, cx_features
):
    """Dropout is off during inference even when the caller left the model training."""
    model = M1(cx_schema, hidden=16, decoder_dropout=0.5, encoder_dropout=0.5)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    pairs = view.hidden_edges

    model.train()
    training_mode = model.score_inference(view.model_view(), cx_features, pairs)[0]
    model.eval()
    eval_mode = model.score_inference(view.model_view(), cx_features, pairs)[0]

    assert torch.equal(training_mode, eval_mode)


def _concatenated_heads(decoder, pair_repr):
    """The decoder over the concatenated ``[h_u, h_v, pair_features]`` rows."""
    features = decoder.trunk(pair_repr)
    return decoder.edge_head(features).squeeze(-1), decoder.count_head(features).squeeze(-1)


def _concatenated_score(model, view, table, pairs):
    """``M1.score`` with the decoder's concatenated ``[h_u, h_v, pair_features]`` form."""
    hidden = model.encode(view, table)
    pair_features = model._pair_features(view, pairs)
    index = torch.as_tensor(pairs, dtype=torch.long)
    return _concatenated_heads(
        model.decoder,
        torch.cat([hidden[index[:, 0]], hidden[index[:, 1]], pair_features], dim=1),
    )


@pytest.mark.parametrize("pair_dim", [1, 3])
def test_factored_decoder_matches_the_concatenated_first_layer(pair_dim):
    """Gathering per-node first-layer terms is the concatenated layer, reordered."""
    torch.manual_seed(0)
    decoder = PairDecoder(32, pair_dim=pair_dim).double()
    nodes = torch.randn(200, 32, dtype=torch.float64)
    ends = torch.randint(0, 200, (1000, 2))
    pair_features = torch.randn(1000, pair_dim, dtype=torch.float64)

    with torch.no_grad():
        factored = decoder.decode(*decoder.project(nodes), ends, pair_features)
        concatenated = _concatenated_heads(
            decoder, torch.cat([nodes[ends[:, 0]], nodes[ends[:, 1]], pair_features], dim=1)
        )

    for got, expected in zip(factored, concatenated):
        assert got.dtype == torch.float64
        assert torch.allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_factored_decoder_matches_the_concatenated_form_in_float32(
    cx_graph, cx_schema, cx_features
):
    torch.manual_seed(0)
    model = M1(cx_schema, hidden=16)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1)).model_view()
    pairs = np.random.default_rng(0).integers(0, cx_graph.n_nodes, size=(500, 2))

    factored = model.score(view, cx_features, pairs)
    concatenated = _concatenated_score(model, view, cx_features, pairs)
    for got, expected in zip(factored, concatenated):
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)

    loss = sum(head.sum() for head in factored)
    loss.backward()
    factored_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad()
    sum(head.sum() for head in concatenated).backward()
    for name, parameter in model.named_parameters():
        assert torch.allclose(factored_grads[name], parameter.grad, atol=1e-4, rtol=1e-4), name


def test_a_committed_checkpoint_loads_and_decodes_like_the_concatenated_form():
    """The factored decoder reads the same ``trunk.0`` weights old checkpoints carry."""
    path = Path(__file__).resolve().parent.parent / "results/checkpoints/f1-cx-s000-d835ecda/best.pt"
    if not path.exists():
        pytest.skip("committed checkpoint not present")
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    prefix = "decoder."
    decoder_state = {
        key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
    }
    width = decoder_state["trunk.0.weight"].shape[0]
    decoder = PairDecoder(width, pair_dim=1)
    decoder.load_state_dict(decoder_state)
    assert set(decoder.state_dict()) == set(decoder_state)

    decoder = decoder.double()
    generator = torch.Generator().manual_seed(0)
    nodes = torch.randn(300, decoder.hidden, generator=generator, dtype=torch.float64)
    ends = torch.randint(0, 300, (2000, 2), generator=generator)
    pair_features = torch.rand(2000, 1, generator=generator, dtype=torch.float64) * 10
    with torch.no_grad():
        factored = decoder.decode(*decoder.project(nodes), ends, pair_features)
        concatenated = _concatenated_heads(
            decoder, torch.cat([nodes[ends[:, 0]], nodes[ends[:, 1]], pair_features], dim=1)
        )
    for got, expected in zip(factored, concatenated):
        assert torch.allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_score_of_no_pairs_is_empty(cx_graph, cx_schema, cx_features):
    model = M1(cx_schema, hidden=16)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    logits, counts = model.score(view.model_view(), cx_features, view.hidden_edges[:0])
    assert logits.shape == counts.shape == (0,)


@pytest.mark.parametrize("row_width, batch", [(3, 6), (10, 10)])
def test_decoder_batches_never_split_a_ranking_row(
    cx_graph, cx_schema, cx_features, monkeypatch, row_width, batch
):
    model = M1(cx_schema, hidden=16)
    model.eval()
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    n_rows = len(view.hidden_edges) // row_width
    pairs = view.hidden_edges[: n_rows * row_width]
    with torch.no_grad():
        whole = model.score(view.model_view(), cx_features, pairs)[0]
        batches = _decoder_batches(model, monkeypatch, chunk=7)
        chunked = model.score(
            view.model_view(), cx_features, pairs, row_width=row_width
        )[0]

    assert all(size % row_width == 0 for size in batches)
    assert max(batches) == batch and sum(batches) == len(pairs)

    def pessimistic_ranks(scores):
        rows = scores.view(n_rows, row_width)
        return 1 + (rows[:, 1:] >= rows[:, :1]).sum(dim=1)

    assert torch.equal(pessimistic_ranks(chunked), pessimistic_ranks(whole))


def test_masked_nodes_contribute_features_but_no_edges(cx_graph, cx_schema, cx_features):
    model = M1(cx_schema, hidden=16)
    model.eval()
    instance = random_mask(cx_graph, fraction=0.05, seed=1)
    view = apply_mask(cx_graph, instance)
    target = int(instance.masked[0])
    assert bool(view.is_masked[target])

    before = model.encode(view.model_view(), cx_features).detach()
    after = model.encode(view.model_view(), cx_features.perturb(target)).detach()
    delta = (after - before).abs().amax(dim=1)

    # The masked node's own features must reach its embedding ...
    assert float(delta[target]) > 1e-6
    # ... while no unmasked node may see them, since every edge touching a
    # masked endpoint is hidden.
    unmasked = torch.as_tensor(~view.is_masked)
    assert float(delta[unmasked].max()) <= 1e-6


def test_model_refuses_the_evaluator_view(cx_graph, cx_schema, cx_features):
    """M1 may only be handed the leak-free projection of a masked view.

    A MaskedView carries hidden_edges, hidden_weights and both_masked -- the
    answer key -- so accepting one would leave the section 7 experiment open to
    a silent leak in a later iteration.
    """
    model = M1(cx_schema, hidden=16)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))

    with pytest.raises(TypeError, match="model_view"):
        model.encode(view, cx_features)
    with pytest.raises(TypeError, match="model_view"):
        model.score(view, cx_features, view.hidden_edges)


def test_gap_simulation_draws_a_fresh_mask_every_epoch(cx_graph):
    rng = np.random.default_rng(0)
    masks = [simulate_gaps(cx_graph, rng) for _ in range(5)]

    draws = {tuple(int(index) for index in mask.masked) for mask in masks}
    assert len(draws) == 5
    for mask in masks:
        assert mask.regime == "R"
        assert mask.dataset == cx_graph.dataset
        assert mask.region == cx_graph.region
        assert mask.instance_id.startswith(f"{cx_graph.dataset}-cx-gap-s")
        assert mask.n_masked == round(0.05 * cx_graph.n_nodes)


def test_fit_rejects_a_validation_suite_that_is_not_f1_val(
    cx_graph, cx_features, cx_types, cx_centroids, tmp_path
):
    """Only f1_val may early-stop: the registered rules reserve the F2 splits for evaluation."""
    suite = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    runs_dir = tmp_path / "runs"
    config = TrainConfig(epochs=2, hidden=16, patience=2, run_dir=runs_dir)

    with pytest.raises(ValueError, match="f1_val"):
        fit(cx_graph, cx_features, suite, seed=0, config=config)
    assert not runs_dir.exists()


def test_fit_rejects_a_graph_that_is_not_f1(cx_graph, cx_features, tmp_path):
    """F2 is the generalization test and F3 exploratory, so neither trains."""
    runs_dir = tmp_path / "runs"
    config = TrainConfig(epochs=2, hidden=16, patience=2, run_dir=runs_dir)

    with pytest.raises(ValueError, match="fit trains only on F1"):
        fit(cx_graph, cx_features, None, seed=0, config=config)
    assert not runs_dir.exists()


def test_fit_rejects_an_f1_val_suite_for_a_different_region(
    cx_graph_f1, cx_features_f1, cx_types, cx_centroids, tmp_path
):
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    wrong_region = dataclasses.replace(suite, region="OL")
    runs_dir = tmp_path / "runs"

    with pytest.raises(ValueError):
        fit(
            cx_graph_f1,
            cx_features_f1,
            wrong_region,
            seed=0,
            config=TrainConfig(epochs=1, run_dir=runs_dir),
        )
    assert not runs_dir.exists()


def test_fit_excludes_every_f1_val_masked_neuron_from_training(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    """F1-train is the CX graph minus the F1-val masks (phase-1 split table).

    An F1-val neuron may not be a training gap, supervise an epoch, or take
    part in a pair the model is trained on: F1-val MRR is the early-stopping
    and delta-calibration signal, so it must rank held-out edges.
    """
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    union = np.unique(
        np.concatenate(
            [np.asarray(instance.masked) for instance in suite.resolve(cx_graph_f1)]
        )
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    config = TrainConfig(
        epochs=3, hidden=16, patience=3, run_dir=tmp_path, suites_dir=suites_dir,
        steps_per_epoch=1,
    )

    epochs = []
    real_negatives = training._training_negatives

    def spy_negatives(view, true_edges=None, positives=None):
        negatives = real_negatives(view, true_edges, positives)
        epochs.append((view, negatives))
        return negatives

    task_graphs = []
    real_tasks = training._build_validation_tasks

    def spy_tasks(graph, val_suite, instances, samples, committed):
        task_graphs.append(graph)
        return real_tasks(graph, val_suite, instances, samples, committed)

    monkeypatch.setattr(training, "_training_negatives", spy_negatives)
    monkeypatch.setattr(training, "_build_validation_tasks", spy_tasks)

    fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)

    # One view per epoch, plus the fixed train-side MRR probe drawn first.
    assert len(epochs) == config.epochs + 1
    original_edges = pack_edges(cx_graph_f1.edges)
    complete = cx_graph_f1.in_strength() + cx_graph_f1.out_strength()
    for view, negatives in epochs:
        gap = np.asarray(view.instance.masked)
        positives = np.asarray(view.hidden_edges, dtype=np.int64).reshape(-1, 2)
        # No F1-val neuron is ever hidden by a training gap ...
        assert not np.isin(gap, union).any()
        # ... supervises an epoch through a hidden edge ...
        assert not np.isin(positives, union).any()
        # ... or appears in the edge set training runs on.
        assert not np.isin(view.graph.edges, union).any()
        assert view.graph.n_nodes == cx_graph_f1.n_nodes
        assert np.array_equal(view.graph.node_ids, cx_graph_f1.node_ids)
        # The granted degree-given total is the unpruned F1 value, so a training
        # row reports the same quantity a scored one does.
        granted = view.degree_given_total()
        assert np.array_equal(granted[view.is_masked], complete[view.is_masked])
        # A removed F1-val edge must not come back as a negative either.
        assert not any(
            ((int(u) << 32) | int(v)) in original_edges
            for u, v in negatives.reshape(-1, 2)
        )
    # Validation keeps resolving against the original graph, whose digest the
    # suite (and the manifest) pins.
    assert len(task_graphs) == 1
    assert task_graphs[0] is cx_graph_f1


def test_one_training_run_reduces_the_loss(cx_graph_f1, cx_features_f1, tmp_path):
    """Each epoch draws a fresh mask, so compare the first and last five epochs."""
    config = TrainConfig(epochs=20, hidden=16, patience=20, run_dir=tmp_path)
    result = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    assert isinstance(result, TrainResult)
    loss = result.history["loss"]
    assert len(loss) == config.epochs
    assert np.mean(loss[-5:]) < np.mean(loss[:5])


@pytest.mark.parametrize("steps", [1, 3])
def test_each_epoch_takes_steps_per_epoch_steps_over_fresh_gaps(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch, steps
):
    calls = {"loss": 0, "step": 0}
    gaps = set()
    epoch_loss, make_optimizer = training._epoch_loss, training._optimizer

    def counting_loss(model, table, batch, config):
        calls["loss"] += 1
        gaps.add(tuple(np.flatnonzero(batch.view.is_masked)))
        return epoch_loss(model, table, batch, config)

    def counting_optimizer(*args, **kwargs):
        optimizer = make_optimizer(*args, **kwargs)
        step = optimizer.step

        def counted(*a, **k):
            calls["step"] += 1
            return step(*a, **k)

        optimizer.step = counted
        return optimizer

    monkeypatch.setattr(training, "_epoch_loss", counting_loss)
    monkeypatch.setattr(training, "_optimizer", counting_optimizer)
    config = TrainConfig(epochs=2, hidden=8, run_dir=tmp_path, steps_per_epoch=steps)
    result = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    assert calls == {"loss": 2 * steps, "step": 2 * steps}
    assert len(gaps) == 2 * steps
    assert len(result.history["loss"]) == 2


def test_count_head_starts_at_the_mean_log_count(cx_graph_f1, cx_schema):
    model = M1(cx_schema, hidden=8)

    training._init_count_head(model, cx_graph_f1)

    assert float(model.decoder.count_head.bias.detach()) == pytest.approx(
        float(np.log1p(cx_graph_f1.weights).mean())
    )


def test_validation_path_tracks_best_mrr(
    cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids, tmp_path
):
    table = build_features(cx_annotations, cx_graph_f1, cx_schema, dataset="f1")
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    config = TrainConfig(
        epochs=6,
        hidden=16,
        patience=6,
        eval_every=2,
        run_dir=tmp_path,
        suites_dir=suites_dir,
    )

    result = fit(cx_graph_f1, table, suite, seed=0, config=config, schema=cx_schema)

    history = result.history
    assert "val_mrr" not in history
    scored, means = validation_mrr(history)
    assert scored == history["val_mrr_epoch"] == [0, 2, 4, 5]
    assert result.best_epoch == scored[int(np.argmax(means))]
    by_instance = history["val_mrr_by_instance"]
    assert set(by_instance) == {instance.instance_id for instance in suite.instances}
    assert all(len(values) == len(scored) for values in by_instance.values())
    selected = [i.instance_id for i in suite.instances if i.regime != "T"]
    assert len(selected) < len(by_instance)
    assert history["val_selection_instances"] == selected
    assert means == [
        float(np.mean([by_instance[i][k] for i in selected])) for k in range(len(scored))
    ]


def test_validation_mrr_selects_on_the_listed_instances_and_reports_the_rest():
    history = {
        "val_mrr_epoch": [0, 5],
        "val_mrr_by_instance": {"r": [0.5, 0.75], "t": [0.75, 0.25], "s": [0.25, 0.5]},
        "val_selection_instances": ["r", "s"],
    }

    assert validation_mrr(history) == ([0, 5], [0.375, 0.625])
    del history["val_selection_instances"]
    assert validation_mrr(history) == ([0, 5], [0.5, 0.5])


def test_validation_path_uses_the_committed_negatives(
    cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids, tmp_path
):
    """A wrong committed digest must stop the run, not be quietly reconciled.

    The val path may only score against the suite's committed negatives
    (the registered negatives rule); deriving the digest map from the samples just drawn
    would make build_scoring_task's check unfalsifiable.
    """
    table = build_features(cx_annotations, cx_graph_f1, cx_schema, dataset="f1")
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    path = suites_dir / suite.dataset / suite.region.lower() / "f1_val.negatives.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "f1_val",
                "spec": {
                    "k": DEFAULT_SPEC.k,
                    "alpha": DEFAULT_SPEC.alpha,
                    "master_seed": DEFAULT_SPEC.master_seed,
                },
                "digests": {
                    instance.instance_id: "0" * 64 for instance in suite.instances
                },
            }
        )
    )
    runs_dir = tmp_path / "runs"
    config = TrainConfig(
        epochs=2, hidden=16, patience=2, run_dir=runs_dir, suites_dir=suites_dir
    )

    with pytest.raises(ValueError, match="committed digest"):
        fit(cx_graph_f1, table, suite, seed=0, config=config, schema=cx_schema)
    assert not runs_dir.exists()


def test_validation_path_raises_when_the_committed_negatives_are_missing(
    cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids, tmp_path
):
    """A missing committed-negatives artefact must fail, not fabricate digests."""
    table = build_features(cx_annotations, cx_graph_f1, cx_schema, dataset="f1")
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    expected = (
        suites_dir / suite.dataset / suite.region.lower() / "f1_val.negatives.json"
    )
    runs_dir = tmp_path / "runs"
    config = TrainConfig(
        epochs=2, hidden=16, patience=2, run_dir=runs_dir, suites_dir=suites_dir
    )

    with pytest.raises(FileNotFoundError, match=re.escape(str(expected))):
        fit(cx_graph_f1, table, suite, seed=0, config=config)
    assert not runs_dir.exists()


def test_suites_dir_is_part_of_the_run_identity(
    cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids, tmp_path
):
    """Two runs differing only in suites_dir score different committed files."""
    table = build_features(cx_annotations, cx_graph_f1, cx_schema, dataset="f1")
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    negatives = suite_negatives(cx_graph_f1, suite)
    first_dir = tmp_path / "suites_a"
    second_dir = tmp_path / "suites_b"
    _write_committed_negatives(first_dir, suite, negatives)
    _write_committed_negatives(second_dir, suite, negatives)

    first = fit(
        cx_graph_f1, table, suite, seed=0,
        config=TrainConfig(
            epochs=2, hidden=16, patience=2, run_dir=tmp_path, suites_dir=first_dir
        ),
    )
    second = fit(
        cx_graph_f1, table, suite, seed=0,
        config=TrainConfig(
            epochs=2, hidden=16, patience=2, run_dir=tmp_path, suites_dir=second_dir
        ),
    )

    assert first.manifest["run_id"] != second.manifest["run_id"]
    assert first.manifest["config"]["suites_dir"] == first_dir.as_posix()
    assert second.manifest["config"]["suites_dir"] == second_dir.as_posix()


def test_fit_returns_the_weights_of_the_checkpoint_it_selected(
    cx_graph_f1, cx_annotations, cx_schema, cx_types, cx_centroids, tmp_path
):
    """The returned weights are the selected checkpoint's, not the live model's.

    Early stopping breaks the loop up to ``patience`` epochs after the best-val
    epoch, and the live model keeps stepping through those epochs, so returning
    it hands back weights worse than the run's own selection (``best.pt``).
    """
    table = build_features(cx_annotations, cx_graph_f1, cx_schema, dataset="f1")
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(
        suites_dir, suite, suite_negatives(cx_graph_f1, suite, spec=DEFAULT_SPEC)
    )
    config = TrainConfig(
        epochs=20,
        hidden=16,
        patience=1,
        lr_schedule="constant",
        lr=0.003,
        run_dir=tmp_path / "runs",
        suites_dir=suites_dir,
    )

    result = fit(cx_graph_f1, table, suite, seed=0, config=config, schema=cx_schema)

    run_dir = tmp_path / "runs" / result.manifest["run_id"]
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)

    def same(left, right):
        return all(torch.equal(left[key], right[key]) for key in left)

    scored, means = validation_mrr(result.history)
    assert result.best_epoch == scored[int(np.argmax(means))]
    assert result.best_epoch < len(result.history["loss"]) - 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["best_val_mrr"] == pytest.approx(max(means))
    assert same(result.state_dict, best["model"])
    assert not same(result.state_dict, last["model"])


def test_fit_is_idempotent_after_early_stop(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path
):
    """A second identical call must not train more epochs or change the result.

    Early stopping is the run's terminal state. Without persisting it, resume
    re-enters the loop, extends the history, and can overwrite ``best.pt`` and
    the returned weights with epochs the caller never asked for.
    """
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    config = TrainConfig(
        epochs=20,
        hidden=16,
        patience=1,
        lr_schedule="constant",
        lr=0.003,
        run_dir=tmp_path / "runs",
        suites_dir=suites_dir,
    )

    first = fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)
    assert len(first.history["loss"]) < config.epochs
    assert first.resumed is False

    second = fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)

    run_dir = tmp_path / "runs" / first.manifest["run_id"]
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert last["stopped_early"] is True
    assert second.manifest["run_id"] == first.manifest["run_id"]
    assert second.history == first.history
    assert second.best_epoch == first.best_epoch
    assert second.resumed is True
    assert len(second.manifest["invocations"]) == 2
    assert len(second.history["loss"]) == len(first.history["loss"])
    for key, value in first.state_dict.items():
        assert torch.equal(second.state_dict[key], value)


def test_a_paused_resume_records_the_restored_best_val_mrr(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    """A run paused again after a resume reports the score its checkpoint selected so far."""
    suite = build_suite(cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids)
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    config = TrainConfig(
        epochs=6, hidden=8, patience=50, eval_every=1, steps_per_epoch=1, run_dir=tmp_path / "runs", suites_dir=suites_dir
    )
    real_loss = training._epoch_loss

    def preempt_on(call):
        calls = {"n": 0}

        def loss(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == call:
                raise KeyboardInterrupt
            return real_loss(*args, **kwargs)

        return loss

    for call in (3, 1):
        with monkeypatch.context() as patch:
            patch.setattr(training, "_epoch_loss", preempt_on(call))
            with pytest.raises(RunStopped):
                fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)
        (run_dir,) = (tmp_path / "runs").iterdir()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if call == 3:
            assert manifest["best_val_mrr"] is None

    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    _, means = validation_mrr(last["history"])
    assert len(means) == 2
    assert manifest["best_val_mrr"] == pytest.approx(max(means))


def test_resume_appends_the_second_invocation_and_keeps_the_first_provenance(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """A resume must not relabel the first epochs with the second host's stack.

    Git, image, and runtime are per invocation: the run's top level stays the
    first process's, and each later process appends its own record instead of
    overwriting the provenance the run was created with.
    """
    first_commit, second_commit = "a" * 40, "b" * 40
    first_image = "sha256:" + "1" * 64
    second_image = "sha256:" + "2" * 64
    config = TrainConfig(epochs=4, hidden=16, patience=4, run_dir=tmp_path)

    monkeypatch.setenv(
        "FLYARCH_GIT_STATE", json.dumps({"commit": first_commit, "dirty": False})
    )
    monkeypatch.setenv("FLYARCH_IMAGE_IDENTIFIER", first_image)
    monkeypatch.setattr(training.torch, "__version__", "2.7.1")
    first = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    assert first.resumed is False
    assert len(first.manifest["invocations"]) == 1

    monkeypatch.setenv(
        "FLYARCH_GIT_STATE", json.dumps({"commit": second_commit, "dirty": True})
    )
    monkeypatch.setenv("FLYARCH_IMAGE_IDENTIFIER", second_image)
    monkeypatch.setattr(training.torch, "__version__", "2.8.0")
    second = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    manifest = json.loads(
        (tmp_path / first.manifest["run_id"] / "manifest.json").read_text()
    )
    assert second.resumed is True
    assert second.history["loss"] == first.history["loss"]
    # The top level keeps the first invocation's stack, Git, and image.
    assert manifest["git_commit"] == first_commit
    assert manifest["image_identifier"] == first_image
    assert manifest["torch_version"] == "2.7.1"
    assert manifest["device"] == "cpu"
    assert manifest["precision"] == "fp32"
    assert manifest["metadata_unavailable"] == []
    first_record, second_record = manifest["invocations"]
    assert first_record["git"]["commit"] == first_commit
    assert first_record["git"]["dirty"] is False
    assert first_record["image_identifier"] == first_image
    assert first_record["runtime"]["torch"] == "2.7.1"
    assert first_record["resumed"] is False
    assert second_record["git"]["commit"] == second_commit
    assert second_record["git"]["dirty"] is True
    assert second_record["image_identifier"] == second_image
    assert second_record["runtime"]["torch"] == "2.8.0"
    assert second_record["resumed"] is True
    assert first_record["timestamp_utc"] < second_record["timestamp_utc"]
    assert second_record["timestamp_utc"].endswith("Z")


def test_each_invocation_records_its_own_title_and_branch(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    config = TrainConfig(epochs=4, hidden=16, patience=4, run_dir=tmp_path)
    monkeypatch.setattr(training, "git_subject", lambda root, commit, environ=None: "s")

    monkeypatch.setattr(training, "git_branch", lambda root, environ=None: "one")
    first = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config,
                labels=RunLabels(title="first"))
    monkeypatch.setattr(training, "git_branch", lambda root, environ=None: "two")
    fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config,
        labels=RunLabels(title="second"))

    manifest = json.loads(
        (tmp_path / first.manifest["run_id"] / "manifest.json").read_text()
    )
    first_record, second_record = manifest["invocations"]
    assert (first_record["title"], first_record["git"]["branch"]) == ("first", "one")
    assert (second_record["title"], second_record["git"]["branch"]) == ("second", "two")
    assert first_record["git"]["subject"] == "s"


def test_validation_tasks_are_built_once_per_fit(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    """The per-instance scoring task is verification work, not per-epoch work.

    ``build_scoring_task`` re-checks the suite digest and re-hashes every
    committed sample. Suite, graph and committed negatives are fixed for the
    whole fit, so that work belongs before the epoch loop while the model is
    what changes.
    """
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    config = TrainConfig(
        epochs=4, hidden=16, patience=4, run_dir=tmp_path, suites_dir=suites_dir
    )

    calls = {"n": 0}
    real = training.build_scoring_task

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "build_scoring_task", counted)

    fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)

    assert calls["n"] == len(suite.instances)


def test_validation_scores_match_score_instance_and_score_one_instance_at_a_time(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    """Selection reads the gated MRR the evaluator reports, from inputs built once."""
    from src.evaluate import score_instance

    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    inputs = training.validate_m1_inputs(cx_graph_f1, suite, suites_dir)
    torch.manual_seed(0)
    model = M1(cx_schema, hidden=16)
    validation = training._Validation(inputs.tasks)
    monkeypatch.setattr(training, "_RESIDENT_PAIRS", 0)

    def reference():
        model.eval()
        with torch.no_grad():
            return {
                view.instance.instance_id: score_instance(
                    task,
                    lambda pairs, view=view, task=task: model.score(
                        view.model_view(), cx_features_f1, pairs,
                        row_width=task.negatives.shape[1] + 1,
                    )[0].numpy(),
                ).mrr
                for view, task in inputs.tasks
            }

    expected = reference()
    cached = validation.prepared()
    assert cached is not None
    assert all(isinstance(visible, VisibleView) for visible, _ in cached)
    assert [width for _, width in cached] == [
        task.negatives.shape[1] + 1 for _, task in inputs.tasks
    ]
    built = []
    real_candidate_pairs = training.candidate_pairs

    def track_candidate_pairs(task):
        pairs = real_candidate_pairs(task)
        built.append(weakref.ref(pairs))
        return pairs

    monkeypatch.setattr(training, "candidate_pairs", track_candidate_pairs)
    output_refs = []
    real_inference = model.score_inference

    def track_inference(*args, **kwargs):
        gc.collect()
        assert not any(ref() is not None for ref in output_refs)
        output = real_inference(*args, **kwargs)
        output_refs.append(weakref.ref(output[0]))
        return output

    with monkeypatch.context() as patch:
        patch.setattr(model, "score_inference", track_inference)
        first = training._validation_scores(model, cx_features_f1, validation, _REAL)

    assert first == expected
    assert len(output_refs) == len(inputs.tasks)
    assert len(built) == len(inputs.tasks)
    gc.collect()
    assert not any(ref() is not None for ref in output_refs)
    assert not any(ref() is not None for ref in built)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    second = training._validation_scores(model, cx_features_f1, validation, _REAL)
    assert second == reference()
    assert second != first
    assert validation.views is cached
    assert len(cached) == len(inputs.tasks)
    assert validation.batches is None


def test_small_validation_suites_keep_their_pairs_resident_with_bitwise_scores(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    from src.evaluate import candidate_pairs, gated_mrr

    inputs = training.validate_m1_inputs(cx_graph_f1, suite, suites_dir)
    torch.manual_seed(0)
    model = M1(cx_schema, hidden=16)
    validation = training._Validation(inputs.tasks)
    assert validation.n_pairs <= training._RESIDENT_PAIRS

    def reference():
        model.eval()
        with torch.no_grad():
            return {
                view.instance.instance_id: gated_mrr(
                    task,
                    model.score(
                        view.model_view(), cx_features_f1, candidate_pairs(task),
                        row_width=task.negatives.shape[1] + 1,
                    )[0].numpy(),
                )
                for view, task in inputs.tasks
            }

    built = {"n": 0}
    real_prepare = model.prepare

    def counted(*args, **kwargs):
        built["n"] += 1
        return real_prepare(*args, **kwargs)

    def scores():
        with monkeypatch.context() as patch:
            patch.setattr(model, "prepare", counted)
            return training._validation_scores(model, cx_features_f1, validation, _REAL)

    first = scores()
    assert first == reference()
    assert built["n"] == len(inputs.tasks)
    resident = validation.batches
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    second = scores()
    assert second == reference() and second != first
    assert built["n"] == len(inputs.tasks) and validation.batches is resident
    monkeypatch.setattr(training, "_RESIDENT_PAIRS", validation.n_pairs - 1)
    assert scores() == second
    assert built["n"] == len(inputs.tasks)


def test_the_ablated_validation_keeps_its_pairs_resident(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path
):
    """The ablation rewrites the masked roots' rows only, so the pairs never leave the device."""
    suite = build_suite(cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids)
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    inputs = training.validate_m1_inputs(cx_graph_f1, suite, suites_dir)
    torch.manual_seed(0)
    model = M1(cx_schema, hidden=16)
    validation = training._Validation(inputs.tasks)
    training._validation_scores(model, cx_features_f1, validation, TrainConfig(epochs=1))
    assert validation.batches is not None  # a streamed validation is what the union cannot afford


def test_validation_inputs_are_prepared_once_per_fit(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    config = TrainConfig(
        epochs=3, hidden=8, patience=3, eval_every=1, run_dir=tmp_path,
        suites_dir=suites_dir,
    )
    builds = {"n": 0}
    real = training._Validation.prepared

    def counted(self):
        before = self.views
        result = real(self)
        if self.views is not before:
            builds["n"] += 1
        return result

    monkeypatch.setattr(training._Validation, "prepared", counted)
    result = fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)

    assert result.history["val_mrr_epoch"] == [0, 1, 2]
    assert builds["n"] == 1


def test_gpu_backends_step_a_fused_adamw(cx_schema, monkeypatch):
    assert training._BACKENDS["cuda"].fused_adamw
    assert training._BACKENDS["mps"].fused_adamw
    assert not training._BACKENDS["cpu"].fused_adamw
    config = TrainConfig(epochs=1, hidden=8)
    model = M1(cx_schema, hidden=8)

    assert training._optimizer(model, config, "cpu").defaults["fused"] is None
    monkeypatch.setitem(
        training._BACKENDS, "cpu",
        dataclasses.replace(training._BACKENDS["cpu"], fused_adamw=True),
    )
    optimizer = training._optimizer(model, config, "cpu")
    assert optimizer.defaults["fused"] is True
    assert optimizer.defaults["lr"] == config.lr


@pytest.fixture
def restore_deterministic_algorithms():
    enabled = torch.are_deterministic_algorithms_enabled()
    workspace = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    yield
    torch.use_deterministic_algorithms(enabled)
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if workspace is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def test_cuda_runs_use_deterministic_algorithms(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch, restore_deterministic_algorithms
):
    assert training._BACKENDS["cuda"].deterministic
    assert not training._BACKENDS["mps"].deterministic
    assert not training._BACKENDS["cpu"].deterministic
    config = TrainConfig(epochs=1, hidden=8, run_dir=tmp_path)

    first = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)
    assert first.manifest["invocations"][0]["runtime"]["deterministic"] is False

    monkeypatch.setitem(
        training._BACKENDS, "cpu",
        dataclasses.replace(training._BACKENDS["cpu"], deterministic=True),
    )
    second = fit(
        cx_graph_f1, cx_features_f1, None, seed=0,
        config=dataclasses.replace(config, run_dir=tmp_path / "deterministic"),
    )

    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert second.manifest["invocations"][0]["runtime"]["deterministic"] is True


def test_fit_fails_when_the_model_scores_validation_pairs_non_finite(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    """A non-finite val score cannot select a checkpoint, so it must stop the run.

    ``nan > -inf`` is false, so an all-NaN epoch would never update
    ``best_epoch`` and patience would never fire: the run would train to budget
    and return the last weights labelled as the selected checkpoint.
    """
    from src.run_control import read_state

    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    runs = tmp_path / "runs"
    config = TrainConfig(
        epochs=2, hidden=16, patience=2, run_dir=runs, suites_dir=suites_dir
    )
    real_init = training._init_count_head

    def nan_edge_head(model, graph):
        real_init(model, graph)
        with torch.no_grad():
            model.decoder.edge_head.bias.fill_(float("nan"))

    monkeypatch.setattr(training, "_init_count_head", nan_edge_head)
    # The train-side probe ranks the same logits; it only logs, so keep it out
    # of the way of the validation path under test.
    monkeypatch.setattr(training, "_train_mrr", lambda *args: float("nan"))

    with pytest.raises(ValueError, match="scores must all be finite") as exc:
        fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)
    assert any(entry.name == "_validation_scores" for entry in exc.traceback)
    (run_dir,) = runs.iterdir()
    assert read_state(run_dir)["status"] == "crashed"
    assert not (run_dir / "best.pt").exists()


def test_fit_writes_a_manifest(cx_graph_f1, cx_features_f1, tmp_path):
    config = TrainConfig(epochs=3, hidden=16, patience=3, run_dir=tmp_path)
    result = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    run_dir = tmp_path / result.manifest["run_id"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["seed"] == 0
    assert manifest["config"]["epochs"] == 3
    assert manifest["config"]["hidden"] == 16
    assert manifest["config"]["lr"] == config.lr
    assert manifest["config"]["patience"] == 3
    assert manifest["config"]["gap_fraction"] == config.gap_fraction
    assert manifest["config"]["aux_weight"] == config.aux_weight
    assert manifest["graph_digest"] == cx_graph_f1.digest()
    assert manifest["feature_digest"]
    assert manifest["schema_digest"] == "none"
    assert manifest["suite_digest"] == "none"
    assert "git_commit" in manifest
    assert manifest["torch_version"] == torch.__version__
    assert "cuda_version" in manifest
    assert "gpu_name" in manifest
    assert "tf32" in manifest
    assert "cudnn_tf32" in manifest["invocations"][0]["runtime"]
    assert manifest["invocations"][0]["runtime"]["fused_adamw"] is False
    assert manifest["device"] == "cpu"
    assert manifest["precision"] == "fp32"
    assert manifest["run_id"] == result.manifest["run_id"]
    assert manifest["best_epoch"] == result.best_epoch
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "last.pt").exists()


def _tensorboard_available() -> bool:
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        return False
    return True


#: Event-file assertions need TensorBoard, which the CPU checkout omits and the
#: training image pins. The writer contract below is tested without it.
requires_tensorboard = pytest.mark.skipif(
    not _tensorboard_available(), reason="TensorBoard is not installed"
)


def _events(run_dir):
    """The scalars TensorBoard itself would display for ``run_dir``."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    return accumulator


@pytest.fixture
def recording_writers(monkeypatch):
    """Capture writer construction and closure without TensorBoard installed."""
    writers = []

    class RecordingWriter:
        def __init__(self, log_dir, purge_step=None):
            self.log_dir = log_dir
            self.purge_step = purge_step
            self.closed = False
            writers.append(self)

        def add_scalar(self, tag, value, step):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(training, "_summary_writer_class", lambda device: RecordingWriter)
    return writers


def _suite_fit(tmp_path, graph, table, schema, types, centroids, *, epochs, patience):
    """One real F1-val-selected fit over the synthetic CX graph."""
    suite = build_suite(graph, name="f1_val", types=types, centroids=centroids)
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(graph, suite))
    config = TrainConfig(
        epochs=epochs,
        hidden=8,
        patience=patience,
        run_dir=tmp_path,
        suites_dir=suites_dir,
    )
    result = fit(graph, table, suite, seed=0, config=config, schema=schema)
    return tmp_path / result.manifest["run_id"], suite, config, result


def test_manifest_records_the_validated_suite_digest(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path
):
    """The milestone row's suite digest must name the suite that selected the run."""
    run_dir, suite, _, _ = _suite_fit(
        tmp_path, cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids,
        epochs=2, patience=2,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["suite_digest"] == suite.digest()


@requires_tensorboard
def test_fit_writes_loss_and_val_mrr_events_at_their_epoch_indices(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path
):
    from src.evaluate import evaluated_rows

    run_dir, suite, config, result = _suite_fit(
        tmp_path, cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids,
        epochs=3, patience=3,
    )

    accumulator = _events(run_dir)
    epochs = list(range(config.epochs))
    by_instance = result.history["val_mrr_by_instance"]
    assert sorted(accumulator.Tags()["scalars"]) == sorted(
        ["loss", "lr", "train_mrr", "val_mrr", "val_mrr/selection", "val_mrr/pooled"]
        + [f"val_mrr/{instance_id}" for instance_id in by_instance]
    )
    positives = {
        view.instance.instance_id: int(evaluated_rows(task).sum())
        for view, task in training.validate_m1_inputs(
            cx_graph_f1, suite, config.suites_dir).tasks
    }
    assert len(set(positives.values())) > 1
    pooled = [
        sum(by_instance[i][k] * n for i, n in positives.items()) / sum(positives.values())
        for k in range(len(result.history["val_mrr_epoch"]))
    ]
    scored, means = validation_mrr(result.history)
    for tag, steps, values in (
        ("loss", epochs, result.history["loss"]),
        ("train_mrr", result.history["train_mrr_epoch"], result.history["train_mrr"]),
        ("val_mrr", scored, list(np.mean(list(by_instance.values()), axis=0))),
        ("val_mrr/selection", scored, means),
        ("val_mrr/pooled", scored, pooled),
        *(
            (f"val_mrr/{instance_id}", scored, instance_values)
            for instance_id, instance_values in by_instance.items()
        ),
    ):
        events = accumulator.Scalars(tag)
        assert [event.step for event in events] == steps
        assert [event.value for event in events] == pytest.approx(
            values, rel=1e-6, abs=1e-6
        )


@requires_tensorboard
def test_resumed_fit_replaces_the_preempted_epochs_event(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    """The curve stays single-valued after a resume, not doubled at the preempted epoch.

    The preemption lands inside validation, after epoch 1's loss event was
    written but before its checkpoint, so that stale event has to disappear
    when the resumed invocation rewrites the epoch.
    """
    from tensorboard.summary.writer import event_file_writer

    # Both writers start in one process and often one second; a fresh counter keeps their files in name order.
    monkeypatch.setattr(event_file_writer, "_global_uid", event_file_writer.AtomicCounter(0))
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    config = TrainConfig(
        epochs=4,
        hidden=8,
        patience=4,
        eval_every=1,
        run_dir=tmp_path,
        suites_dir=suites_dir,
    )
    real_validation = training._validation_scores
    calls = {"n": 0}

    def preempted(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_validation(*args, **kwargs)

    monkeypatch.setattr(training, "_validation_scores", preempted)
    with pytest.raises(RunStopped):
        fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)
    monkeypatch.setattr(training, "_validation_scores", real_validation)

    result = fit(
        cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema
    )

    accumulator = _events(tmp_path / result.manifest["run_id"])
    epochs = list(range(config.epochs))
    scored, means = validation_mrr(result.history)
    for tag, steps, values in (
        ("loss", epochs, result.history["loss"]),
        ("train_mrr", result.history["train_mrr_epoch"], result.history["train_mrr"]),
        ("val_mrr/selection", scored, means),
    ):
        events = accumulator.Scalars(tag)
        assert [event.step for event in events] == steps
        assert [event.value for event in events] == pytest.approx(
            values, rel=1e-6, abs=1e-6
        )


def test_resumed_fit_purges_events_from_the_first_epoch_it_retrains(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path,
    monkeypatch, recording_writers,
):
    """``purge_step`` is the resumed checkpoint's next epoch, not the run's first.

    Events at or after it belong to an epoch whose checkpoint never landed, so
    TensorBoard must drop them; every earlier epoch stays valid.
    """
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    config = TrainConfig(
        epochs=4,
        hidden=8,
        patience=4,
        eval_every=1,
        run_dir=tmp_path,
        suites_dir=suites_dir,
    )
    real_validation = training._validation_scores
    calls = {"n": 0}

    def preempted(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_validation(*args, **kwargs)

    monkeypatch.setattr(training, "_validation_scores", preempted)
    with pytest.raises(RunStopped):
        fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)
    monkeypatch.setattr(training, "_validation_scores", real_validation)

    fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)

    assert [writer.purge_step for writer in recording_writers] == [None, 1]


def test_fit_closes_the_event_writer_when_an_epoch_raises(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch, recording_writers
):
    """A failed epoch must not leak a writer holding unflushed events."""
    config = TrainConfig(epochs=2, hidden=8, patience=2, run_dir=tmp_path)

    def failing(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(training, "_epoch_loss", failing)
    with pytest.raises(RunStopped):
        fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    (writer,) = recording_writers
    assert Path(writer.log_dir).parent == tmp_path
    assert Path(writer.log_dir).is_dir()
    assert writer.closed is True


def test_fit_rejects_cuda_without_tensorboard_before_creating_run_directory(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """A CUDA run's curves are part of its result, so its writer must exist."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    runs_dir = tmp_path / "runs"
    config = TrainConfig(epochs=1, hidden=8, run_dir=runs_dir)

    with pytest.raises(RuntimeError, match="TensorBoard"):
        fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config, device="cuda")

    assert not runs_dir.exists()


def test_fit_rejects_mps_without_tensorboard_before_creating_run_directory(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    runs_dir = tmp_path / "runs"
    config = TrainConfig(epochs=1, hidden=8, run_dir=runs_dir)

    with pytest.raises(RuntimeError, match="TensorBoard") as exc:
        fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config, device="mps")

    assert "active Python environment" in str(exc.value)
    assert "requirements/training.lock" not in str(exc.value)
    assert not runs_dir.exists()


def test_fit_runs_without_tensorboard_and_writes_no_event_file(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """A CPU checkout without TensorBoard trains normally and records no events."""
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    config = TrainConfig(epochs=2, hidden=8, patience=2, run_dir=tmp_path)

    result = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    run_dir = tmp_path / result.manifest["run_id"]
    assert len(result.history["loss"]) == config.epochs
    assert not list(run_dir.glob("events.out.tfevents*"))


def test_fit_records_no_gpu_name_for_a_cpu_run(
    cx_graph_f1, cx_features_f1, tmp_path
):
    """A CPU invocation must not be labelled with a GPU it never used.

    A visible GPU says nothing about where the epochs ran; naming it in the
    manifest lets a reader mistake a CPU run for a CUDA one.
    """
    config = TrainConfig(epochs=1, hidden=8, patience=1, run_dir=tmp_path)

    result = fit(
        cx_graph_f1, cx_features_f1, None, seed=0, config=config, device="cpu"
    )

    assert result.manifest["device"] == "cpu"
    assert result.manifest["precision"] == "fp32"
    assert result.manifest["gpu_name"] is None
    assert result.manifest["invocations"][0]["runtime"]["gpu"] is None


def test_cpu_provenance_records_no_gpu_name_with_a_visible_gpu(monkeypatch):
    """A visible GPU must not name a CPU invocation's hardware.

    The simulated visibility is scoped to the provenance check: a step with
    ``torch.cuda.is_available`` patched True enters torch's CUDA graph health
    check, which needs a driver the CPU-only training image does not have.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "RTX 5090")

    assert training._gpu_name("cuda") == "RTX 5090"
    assert training._gpu_name("cpu") is None
    record = training._invocation("cpu", "fp32", resumed=False)
    assert record["device"] == "cpu"
    assert record["runtime"]["gpu"] is None


def test_direct_run_marks_unavailable_source_metadata(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """Direct runs have no launcher image and may have no checkout either.

    The record names what was unavailable instead of omitting the fields or
    substituting an unverified value.
    """
    monkeypatch.delenv("FLYARCH_GIT_STATE", raising=False)
    monkeypatch.delenv("FLYARCH_IMAGE_IDENTIFIER", raising=False)
    monkeypatch.setattr(training, "_git_commit", lambda: None)
    monkeypatch.setattr(training, "git_branch", lambda root, environ=None: None)
    monkeypatch.setattr(training, "git_subject", lambda root, commit, environ=None: None)
    config = TrainConfig(epochs=1, hidden=8, patience=1, run_dir=tmp_path)

    result = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    record = result.manifest["invocations"][0]
    assert record["git"] == {
        "commit": None, "dirty": None, "branch": None, "subject": None
    }
    assert record["title"] is None
    assert record["image_identifier"] is None
    assert record["metadata_unavailable"] == ["git", "image_identifier"]
    assert result.manifest["git_commit"] is None
    assert result.manifest["image_identifier"] is None
    assert result.manifest["metadata_unavailable"] == ["git", "image_identifier"]


def test_fit_builds_every_tensor_on_the_model_device(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """A tensor built for another device would break the forward pass.

    ``encode`` builds the feature, edge-index and weight tensors from numpy on
    every call, and the loss builds its labels and aux targets. The default is
    CPU; an explicit ``device="cpu"`` must place all of them where the
    parameters are.
    """
    config = TrainConfig(epochs=2, hidden=8, patience=2, run_dir=tmp_path)
    seen = []
    real_forward = _DirectedSAGELayer.forward

    def spy(self, x, edges, weights):
        seen.append(
            (x.device, edges.device, weights.device, self.in_linear.weight.device)
        )
        return real_forward(self, x, edges, weights)

    monkeypatch.setattr(_DirectedSAGELayer, "forward", spy)
    result = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config, device="cpu")

    assert seen
    for x, edges, weights, parameters in seen:
        assert x == edges == weights == parameters
        assert x.type == "cpu"
    assert result.manifest["device"] == "cpu"
    assert result.manifest["precision"] == "fp32"


def test_fit_rejects_cuda_with_tf32_before_creating_run_directory(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """A TF32 matmul would train CUDA below the fp32 its manifest records."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    runs_dir = tmp_path / "runs"
    config = TrainConfig(epochs=1, hidden=8, run_dir=runs_dir)

    with pytest.raises(ValueError, match="TF32"):
        fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config, device="cuda")

    assert not runs_dir.exists()


@pytest.mark.skipif(
    not hasattr(torch.backends.cuda.matmul, "fp32_precision"),
    reason="this PyTorch has only the legacy TF32 flag",
)
def test_fit_rejects_tf32_set_through_the_newer_flag(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """Newer PyTorch raises on the legacy read once ``fp32_precision`` is set."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "tf32")
    runs_dir = tmp_path / "runs"
    config = TrainConfig(epochs=1, hidden=8, run_dir=runs_dir)

    with pytest.raises(ValueError, match="TF32"):
        fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config, device="cuda")

    assert not runs_dir.exists()


def test_fit_rejects_mps_when_unavailable_before_creating_run_directory(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    runs_dir = tmp_path / "runs"
    config = TrainConfig(epochs=1, hidden=8, run_dir=runs_dir)

    with pytest.raises(ValueError, match="torch.backends.mps.is_available"):
        fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config, device="mps")

    assert not runs_dir.exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="a CUDA device is visible")
def test_fit_rejects_cuda_when_the_build_has_no_gpu(
    cx_graph_f1, cx_features_f1, tmp_path
):
    """Asking for a GPU that is not there must fail before any run exists."""
    config = TrainConfig(epochs=1, hidden=8, run_dir=tmp_path / "runs")

    with pytest.raises(ValueError, match="cuda"):
        fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config, device="cuda")

    assert not (tmp_path / "runs").exists()


def _record_autocast(monkeypatch, device_type):
    """Spy on each encoder layer call; returns whether autocast was on for each."""
    enabled = []
    real_forward = _DirectedSAGELayer.forward

    def record(self, x, edges, weights):
        enabled.append(torch.is_autocast_enabled(device_type))
        return real_forward(self, x, edges, weights)

    monkeypatch.setattr(_DirectedSAGELayer, "forward", record)
    return enabled


@requires_tensorboard
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_fit_trains_a_tiny_run_on_mps(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    config = TrainConfig(epochs=2, hidden=8, patience=2, run_dir=tmp_path)
    autocast_enabled = _record_autocast(monkeypatch, "mps")
    result = fit(
        cx_graph_f1, cx_features_f1, None, seed=0, config=config, device="mps"
    )

    assert autocast_enabled and not any(autocast_enabled)
    assert result.manifest["device"] == "mps"
    assert result.manifest["precision"] == "fp32"
    assert result.manifest["gpu_name"] is None
    assert result.manifest["invocations"][0]["runtime"]["gpu"] is None
    assert len(result.history["loss"]) == config.epochs
    assert np.isfinite(result.history["loss"]).all()
    events = _events(tmp_path / result.manifest["run_id"]).Scalars("loss")
    assert [event.step for event in events] == list(range(config.epochs))


def test_fit_empties_the_backend_cache_once_per_epoch(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    calls = []
    backend = dataclasses.replace(
        training._BACKENDS["cpu"], empty_cache=lambda: calls.append(1)
    )
    monkeypatch.setitem(training._BACKENDS, "cpu", backend)
    config = TrainConfig(epochs=3, hidden=8, patience=3, run_dir=tmp_path)

    fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config, device="cpu")

    assert len(calls) == config.epochs


def test_gpu_backends_empty_their_cache():
    assert training._BACKENDS["mps"].empty_cache is torch.mps.empty_cache
    assert training._BACKENDS["cuda"].empty_cache is torch.cuda.empty_cache
    assert training._BACKENDS["cpu"].empty_cache is None


def test_recorded_device_keeps_the_cuda_ordinal_and_drops_the_mps_one():
    assert training._recorded_device(torch.device("cuda", 1)) == "cuda:1"
    assert training._recorded_device(torch.device("mps", 0)) == "mps"
    assert training._recorded_device(torch.device("cpu")) == "cpu"


def test_fit_records_the_model_device_not_the_requested_string(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    seen = []
    real = training._recorded_device

    def record(device):
        seen.append(device)
        return real(device)

    monkeypatch.setattr(training, "_recorded_device", record)
    config = TrainConfig(epochs=1, hidden=8, patience=1, run_dir=tmp_path)
    result = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    assert seen == [torch.device("cpu")]
    assert result.manifest["device"] == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fit_trains_a_tiny_run_on_cuda(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """The documented 3090 environment must run on the GPU, in fp32."""
    config = TrainConfig(epochs=2, hidden=8, patience=2, run_dir=tmp_path)
    autocast_enabled = _record_autocast(monkeypatch, "cuda")
    result = fit(
        cx_graph_f1, cx_features_f1, None, seed=0, config=config, device="cuda"
    )

    assert autocast_enabled and not any(autocast_enabled)
    assert result.manifest["device"] == f"cuda:{torch.cuda.current_device()}"
    assert result.manifest["precision"] == "fp32"
    assert len(result.history["loss"]) == config.epochs


def test_run_id_tracks_every_input(cx_graph_f1, cx_features_f1, tmp_path):
    config = TrainConfig(epochs=4, hidden=16, patience=4, run_dir=tmp_path)
    first = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    # Changed features must open a separate run directory, not resume the first.
    changed = fit(cx_graph_f1, cx_features_f1.perturb(0), None, seed=0, config=config)
    assert changed.manifest["run_id"] != first.manifest["run_id"]
    assert (tmp_path / changed.manifest["run_id"]).is_dir()

    # Identical inputs reuse the run id and continue from the stored last.pt.
    resumed = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)
    assert resumed.manifest["run_id"] == first.manifest["run_id"]
    assert resumed.history["loss"] == first.history["loss"]


def test_the_model_and_schedule_knobs_are_always_recorded(
    cx_graph_f1, cx_features_f1, tmp_path
):
    """Each knob is in the hashed config at any value; moved off its default, it opens a new run."""
    knobs = ("sibling_features", "encoder_dropout", "decoder_dropout", "steps_per_epoch")
    default = TrainConfig(epochs=4, run_dir=tmp_path)
    assert [training._scientific_config(default)[knob] for knob in knobs] == [True, 0.1, 0.1, 8]
    digest = training._feature_digest(cx_features_f1, extras=())
    run_ids = {training._run_id(cx_graph_f1, None, default, 0, digest)}
    for knob, value in zip(knobs, (False, 0.0, 0.0, 1)):
        config = TrainConfig(epochs=4, run_dir=tmp_path, **{knob: value})
        assert training._scientific_config(config)[knob] == value
        run_ids.add(training._run_id(cx_graph_f1, None, config, 0, digest))
    assert len(run_ids) == len(knobs) + 1


def test_run_id_ignores_the_runs_root(cx_graph_f1, cx_features_f1, tmp_path):
    """A storage root is not an input that shapes the run.

    Hashing it would give the same experiment a different id under a different
    runs root, so a resumed run would restart from zero instead of continuing.
    """
    config = dict(epochs=2, hidden=16, patience=2)
    first = fit(
        cx_graph_f1, cx_features_f1, None, seed=0,
        config=TrainConfig(**config, run_dir=tmp_path / "a"),
    )
    second = fit(
        cx_graph_f1, cx_features_f1, None, seed=0,
        config=TrainConfig(**config, run_dir=tmp_path / "b"),
    )

    assert first.manifest["run_id"] == second.manifest["run_id"]
    assert "run_dir" not in first.manifest["config"]


def test_run_id_is_the_same_for_a_windows_and_a_posix_suites_dir(cx_graph_f1, cx_features_f1):
    """A suites path hashes the same on every host, so a cohort can span Windows and Linux."""
    digest = training._feature_digest(cx_features_f1, extras=())
    ids = {
        training._run_id(
            cx_graph_f1, None, TrainConfig(epochs=1, suites_dir=suites_dir), 0, digest
        )
        for suites_dir in (PureWindowsPath("data/suites"), PurePosixPath("data/suites"))
    }
    assert len(ids) == 1


def test_run_id_separates_every_backend(cx_graph_f1, cx_features_f1):
    """A preempted run must not resume under other arithmetic, fp32 or not."""
    digest = training._feature_digest(cx_features_f1, extras=())
    config = TrainConfig(epochs=1)

    ids = {
        training._run_id(cx_graph_f1, None, config, 0, digest, None, device)
        for device in training.DEVICES
    }

    assert len(ids) == len(training.DEVICES)


def test_feature_digest_hashes_only_the_extras_the_config_reads(cx_features_f1):
    """An extras.npz no switch reads leaves the run id's feature digest unchanged."""
    n = cx_features_f1.values.shape[0]
    with_file = dataclasses.replace(cx_features_f1, extras={
        "supertype": np.arange(n), "hemilineage": np.zeros(n, dtype=np.int64),
    })
    plain = training.extras_read(TrainConfig(epochs=1))
    reads = training.extras_read(TrainConfig(epochs=1, annotation_groups=True))

    assert reads == training.extras_read({"annotation_groups": True})
    assert (training._feature_digest(with_file, extras=plain)
            == training._feature_digest(cx_features_f1, extras=plain))
    assert (training._feature_digest(with_file, extras=reads)
            != training._feature_digest(with_file, extras=plain))


def test_run_id_separates_cpu_architectures(cx_graph_f1, cx_features_f1, monkeypatch):
    """An x86 CPU run and an arm64 CPU run accumulate differently."""
    digest = training._feature_digest(cx_features_f1, extras=())
    config = TrainConfig(epochs=1)
    ids = []
    for machine in ("x86_64", "arm64"):
        monkeypatch.setattr(training.platform, "machine", lambda m=machine: m)
        ids.append(
            training._run_id(cx_graph_f1, None, config, 0, digest, None, "cpu")
        )

    assert ids[0] != ids[1]


def test_run_id_tracks_the_training_code(cx_graph_f1, cx_features_f1, tmp_path, monkeypatch):
    """A code change must open a new run, not continue the old checkpoint.

    Hashing the inputs alone would let a fixed training bug resume a run whose
    stored history was produced by the buggy revision, returning one curve
    spanning two revisions of the code.
    """
    config = TrainConfig(epochs=2, hidden=16, patience=2, run_dir=tmp_path)

    monkeypatch.setattr(training, "_code_digest", lambda: "aaaa")
    first = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)
    monkeypatch.setattr(training, "_code_digest", lambda: "bbbb")
    second = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    assert first.manifest["run_id"] != second.manifest["run_id"]
    assert first.manifest["code_digest"] == "aaaa"
    assert second.manifest["code_digest"] == "bbbb"
    assert (tmp_path / first.manifest["run_id"]).is_dir()
    assert (tmp_path / second.manifest["run_id"]).is_dir()


@pytest.mark.parametrize(
    "relative",
    [
        "train/train.py",
        "train/model.py",
        "train/__init__.py",
        "src/masks.py",
        "src/negatives.py",
        "src/suites.py",
        "src/evaluate.py",
        "src/metrics.py",
        "src/features.py",
        "src/regions.py",
    ],
)
def test_code_digest_covers_the_training_path(relative, monkeypatch):
    """Every module that shapes a run must enter the code digest.

    The digest exists so a code fix opens a new run instead of appending
    new-code epochs to a checkpoint trained by the old revision; leaving the
    sampling or scoring modules out defeats it. This changes the bytes
    ``_code_digest`` reads for each file, so a module missing from its list
    leaves the digest unchanged.
    """
    root = Path(training.__file__).resolve().parents[1]
    target = root / relative
    assert target.is_file()
    baseline = training._code_digest()
    real = Path.read_bytes

    def edited(self):
        data = real(self)
        return data + b"\n# edited\n" if self == target else data

    monkeypatch.setattr(Path, "read_bytes", edited)

    assert training._code_digest() != baseline


def test_run_id_tracks_the_schema(cx_graph_f1, cx_features_f1, cx_schema, tmp_path):
    """A schema change the feature table cannot show must still open a new run.

    ``type_vocab`` sizes the type embedding but not the table's columns, so
    without the schema in the identity a widened vocab reuses the old run and
    the resume raises a shape mismatch instead of training a new run.
    """
    config = TrainConfig(epochs=2, hidden=16, patience=2, run_dir=tmp_path)
    widened = dataclasses.replace(
        cx_schema,
        type_vocab={**cx_schema.type_vocab, "NEW_TYPE": len(cx_schema.type_vocab) + 1},
    )

    first = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config, schema=cx_schema)
    second = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config, schema=widened)

    assert first.manifest["run_id"] != second.manifest["run_id"]
    assert first.manifest["schema_digest"] == cx_schema.digest()
    assert second.manifest["schema_digest"] == widened.digest()


def test_checkpoints_load_with_weights_only(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """Checkpoint loads must not execute arbitrary pickle objects.

    The payload is tensors, optimizer state, primitives and generator states,
    so the unrestricted unpickler buys nothing and widens the blast radius of
    a checkpoint written by another process on the shared host.
    """
    config = TrainConfig(epochs=2, hidden=16, patience=2, run_dir=tmp_path)
    real = torch.load
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights_only"))
        return real(*args, **kwargs)

    monkeypatch.setattr(training.torch, "load", spy)

    fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)
    fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    assert seen
    assert all(flag is True for flag in seen)


def test_git_commit_records_this_repo_from_any_working_directory(tmp_path, monkeypatch):
    """The manifest must name this checkout's commit, not the process cwd's repo.

    The pinned training image installs no ``git`` executable, so the probe's
    subprocess result is injected rather than shelling out.
    """
    root = Path(training.__file__).resolve().parents[1]
    commit = "a" * 40
    calls = {}

    def fake_run(args, **kwargs):
        calls["args"] = args
        calls["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(args, 0, stdout=commit + "\n", stderr="")

    monkeypatch.setattr(training.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    assert training._git_commit() == commit
    assert calls["args"] == ["git", "rev-parse", "HEAD"]
    assert Path(calls["cwd"]).resolve() == root


def test_git_commit_records_nothing_when_git_is_absent(monkeypatch):
    """A checkout with no ``git`` on PATH records an unavailable commit."""

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(training.subprocess, "run", missing)

    assert training._git_commit() is None


def test_manifest_write_replaces_atomically(tmp_path, monkeypatch):
    """A preemption mid-write must leave the previous manifest intact.

    The manifest is rewritten more often than a run is created, and a
    truncated provenance record is unrecoverable; the new bytes have to land
    on the final path only once they are complete.
    """
    path = tmp_path / "manifest.json"
    training._write_manifest(path, {"best_epoch": 1})
    assert not path.with_name(path.name + ".tmp").exists()

    real_write = Path.write_text

    def truncated(self, data, *args, **kwargs):
        real_write(self, data, *args, **kwargs)
        raise OSError("preempted mid-write")

    monkeypatch.setattr(Path, "write_text", truncated)
    with pytest.raises(OSError, match="preempted"):
        training._write_manifest(path, {"best_epoch": 2})
    monkeypatch.undo()

    assert not path.with_name(path.name + ".tmp").exists()
    assert json.loads(path.read_text()) == {"best_epoch": 1}


def test_save_checkpoint_replaces_atomically(cx_features, tmp_path, monkeypatch):
    """A preemption mid-write must leave the previous checkpoint intact.

    Writing the pickle straight to ``last.pt`` leaves a truncated file that
    every later resumed run trips over; the new bytes have to land on the
    final path only once they are complete.
    """
    path = tmp_path / "last.pt"
    model = M1(training._table_schema(cx_features), hidden=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    training._save_checkpoint(path, model, optimizer, 0, {"loss": [1.0]}, 0, rng.bit_generator.state)
    assert not path.with_name(path.name + ".tmp").exists()

    real_save = torch.save

    def truncated(obj, target, *args, **kwargs):
        real_save(obj, target, *args, **kwargs)
        raise OSError("preempted mid-write")

    monkeypatch.setattr(training.torch, "save", truncated)
    with pytest.raises(OSError, match="preempted"):
        training._save_checkpoint(path, model, optimizer, 1, {"loss": [2.0]}, 1, rng.bit_generator.state)
    monkeypatch.undo()

    assert not path.with_name(path.name + ".tmp").exists()
    restored = torch.load(path, weights_only=False)
    assert restored["epoch"] == 0
    assert restored["history"] == {"loss": [1.0]}


def test_resume_continues_the_same_mask_sequence(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """A preempted run must resume the sequence it was drawing, not a new one."""
    uninterrupted = fit(
        cx_graph_f1, cx_features_f1, None, seed=0,
        config=TrainConfig(epochs=6, hidden=16, patience=6, run_dir=tmp_path / "whole"),
    ).history["loss"]

    config = TrainConfig(epochs=6, hidden=16, patience=6, run_dir=tmp_path / "preempted")
    real = training._epoch_loss
    calls = {"n": 0}

    def preempted(*args, **kwargs):
        if calls["n"] == 2 * config.steps_per_epoch:
            raise KeyboardInterrupt
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "_epoch_loss", preempted)
    with pytest.raises(RunStopped):
        fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)
    monkeypatch.undo()

    restored = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config).history["loss"]
    assert restored == uninterrupted


def test_first_invocation_is_recorded_before_any_checkpoint_exists(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """An invocation preempted before its first checkpoint is still provenance.

    Waiting for the checkpoint write would lose the environment of a run that
    died in epoch 0, which is what a later reader needs to interpret the run.
    """
    config = TrainConfig(epochs=2, hidden=8, patience=2, run_dir=tmp_path)

    def preempted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(training, "_epoch_loss", preempted)
    with pytest.raises(RunStopped):
        fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)
    monkeypatch.undo()

    (run_dir,) = [path for path in tmp_path.iterdir() if path.is_dir()]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    record = manifest["invocations"][0]
    assert not (run_dir / "last.pt").exists()
    assert manifest["best_epoch"] is None
    assert len(manifest["invocations"]) == 1
    assert record["resumed"] is False
    assert record["timestamp_utc"].endswith("Z")


def test_training_negatives_come_from_the_pinned_sampler(cx_graph):
    """Training must draw negatives the way the gate scores them."""
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=3))
    negatives = training._training_negatives(view)

    assert np.array_equal(negatives, view_negatives(view, spec=DEFAULT_SPEC).negatives)
    assert negatives.shape[1] == DEFAULT_SPEC.k
    flat = negatives.reshape(-1, 2)
    assert view.is_masked[flat].any(axis=1).all()
    true_edges = pack_edges(cx_graph.edges)
    assert not any(((int(u) << 32) | int(v)) in true_edges for u, v in flat)


def test_epoch_loss_ranks_each_positive_against_its_own_negatives(
    cx_graph, cx_features, monkeypatch
):
    """Each row is one positive and all K of its negatives, the row the gate ranks.

    Pointwise BCE at 1:K is minimized by scoring every pair negative, so the
    loss is a softmax over the row with the positive as the target. Intact
    rows have no row in that question: the gate never scores an edge without
    a masked endpoint.
    """
    config = TrainConfig(epochs=1, hidden=8)
    model = M1(training._table_schema(cx_features), hidden=8)
    seen = {}
    real_negatives = training._training_negatives

    def spy_negatives(view, true_edges=None, positives=None):
        negatives = real_negatives(view, true_edges, positives)
        seen["positives"] = np.asarray(view.hidden_edges, dtype=np.int64).reshape(-1, 2)
        seen["negatives"] = negatives
        return negatives

    real_score = M1.score

    def spy_score(self, view, table, pairs, **kwargs):
        logits, counts = real_score(self, view, table, pairs, **kwargs)
        seen["logits"] = logits.detach().clone()
        return logits, counts

    captured = {}
    real_ce = torch.nn.functional.cross_entropy

    def capture(rows, targets, *args, **kwargs):
        captured["rows"] = rows.detach().clone()
        captured["targets"] = targets.detach().clone()
        return real_ce(rows, targets, *args, **kwargs)

    monkeypatch.setattr(training, "_training_negatives", spy_negatives)
    monkeypatch.setattr(M1, "score", spy_score)
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", capture)

    training._epoch_loss(
        model, cx_features,
        training._epoch_batch(cx_graph, cx_features, np.random.default_rng(0), config),
        config,
    )

    n_positives = len(seen["positives"])
    assert n_positives > 0
    assert captured["rows"].shape == (n_positives, 1 + DEFAULT_SPEC.k)
    assert bool(captured["targets"].eq(0).all())
    # Row i holds positive i then its own negatives, in sampler order.
    logits = seen["logits"]
    assert torch.equal(captured["rows"][:, 0], logits[:n_positives])
    assert torch.equal(
        captured["rows"][:, 1:], logits[n_positives:].view(n_positives, DEFAULT_SPEC.k)
    )


def test_epoch_loss_scores_exactly_positives_and_their_negatives(
    cx_graph, cx_features, monkeypatch
):
    """The epoch's scored rows are the hidden positives and their K negatives.

    An intact row is a pair the encoder already message-passes over, and the
    pair-feature column separates it from every negative for free; training on
    it spends capacity on a rule the gated evaluation never asks about.
    """
    config = TrainConfig(epochs=1, hidden=8)
    model = M1(training._table_schema(cx_features), hidden=8)
    captured = {}

    real_negatives = training._training_negatives

    def spy_negatives(view, true_edges=None, positives=None):
        captured["positives"] = np.asarray(view.hidden_edges, dtype=np.int64).reshape(-1, 2)
        negatives = real_negatives(view, true_edges, positives)
        captured["negatives"] = negatives
        return negatives

    real_score = M1.score

    def spy_score(self, view, table, pairs, **kwargs):
        captured["pairs"] = np.asarray(pairs, dtype=np.int64)
        return real_score(self, view, table, pairs, **kwargs)

    monkeypatch.setattr(training, "_training_negatives", spy_negatives)
    monkeypatch.setattr(M1, "score", spy_score)

    training._epoch_loss(
        model, cx_features,
        training._epoch_batch(cx_graph, cx_features, np.random.default_rng(0), config),
        config,
    )

    positives = captured["positives"]
    negatives = captured["negatives"]
    assert negatives.shape == (len(positives), DEFAULT_SPEC.k, 2)
    assert np.array_equal(
        captured["pairs"], np.concatenate([positives, negatives.reshape(-1, 2)])
    )


def test_table_schema_sizes_the_type_embedding_like_the_real_schema(
    cx_schema, cx_features
):
    """A checkpoint built from the table shim must load into a real-schema model.

    The shim is sized from the schema, so it matches even when the table omits
    the highest-numbered type; the inferred shim, which reads ``cell_type``
    from the table, comes out narrower and would fail ``load_state_dict``.
    """
    cell_type_col = cx_features.columns.index("cell_type")
    cell_type = cx_features.values[:, cell_type_col]
    max_id = int(cell_type.max())
    keep = cell_type != max_id
    table = dataclasses.replace(
        cx_features,
        values=cx_features.values[keep],
        coverage=cx_features.coverage,  # per-feature rows, not per-node
    )

    assert (
        M1(training._table_schema(table, cx_schema), hidden=16).type_embedding.num_embeddings
        == M1(cx_schema, hidden=16).type_embedding.num_embeddings
    )
    assert (
        M1(training._table_schema(table), hidden=16).type_embedding.num_embeddings
        < M1(cx_schema, hidden=16).type_embedding.num_embeddings
    )


def test_fit_sizes_the_type_embedding_from_the_schema(
    cx_graph_f1, cx_schema, cx_features_f1, tmp_path
):
    """``fit(..., schema=...)`` writes a checkpoint that loads into ``M1(schema)``.

    The table keeps its rows so training aligns with the graph, but the
    highest-numbered type is mapped to the OOV sentinel, leaving the observed
    max type id below the schema's count. Without the schema the inferred
    embedding would be one row short; with it the checkpoint matches.
    """
    cell_type_col = cx_features_f1.columns.index("cell_type")
    cell_type = cx_features_f1.values[:, cell_type_col]
    max_id = int(cell_type.max())
    values = cx_features_f1.values.copy()
    values[cell_type == max_id, cell_type_col] = 0
    table = dataclasses.replace(cx_features_f1, values=values, coverage=cx_features_f1.coverage)

    config = TrainConfig(epochs=2, hidden=16, patience=2, run_dir=tmp_path)
    result = fit(cx_graph_f1, table, None, seed=0, config=config, schema=cx_schema)

    assert (
        result.state_dict["type_embedding.weight"].shape[0]
        == M1(cx_schema, hidden=16).type_embedding.num_embeddings
    )


def test_pair_features_log_scale_the_degree_column(cx_graph, cx_schema, cx_features):
    """The masked endpoint's synapse count enters the decoder at log1p scale.

    The raw total (up to ~23k on real CX) dominates the decoder's first layer
    by ~200x over the encoder half, so it is compressed to the scale the count
    head already regresses on. Values are unchanged: ``log1p`` is monotone.
    """
    model = M1(cx_schema, hidden=8)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1))
    pairs = np.asarray(view.hidden_edges, dtype=np.int64)

    degree = np.asarray(view.degree_given_total(), dtype=np.float64)
    masked = np.asarray(view.is_masked, dtype=bool)
    source, target = pairs[:, 0], pairs[:, 1]
    raw = np.where(
        masked[source], degree[source], np.where(masked[target], degree[target], 0.0)
    )

    got = model._pair_features(view.model_view(), pairs).detach().numpy()[:, 0]

    assert raw.max() > 1.0
    assert np.allclose(got, np.log1p(raw), rtol=1e-6, atol=1e-6)
    assert got.max() < raw.max()


def test_encode_rejects_a_table_with_permuted_columns(cx_graph, cx_schema, cx_features):
    """A same-width table in another column order must fail loudly.

    Column order is the model's only key to which value is which; a permuted
    table has the right shape and produces plausible but meaningless rankings.
    """
    model = M1(cx_schema, hidden=16)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1)).model_view()
    columns = list(cx_features.columns)
    order = [1, 0] + list(range(2, len(columns)))
    permuted = dataclasses.replace(
        cx_features,
        values=cx_features.values[:, order],
        columns=tuple(columns[index] for index in order),
    )

    with pytest.raises(ValueError, match="column"):
        model.encode(view, permuted)


def test_encode_rejects_a_misaligned_table(cx_graph, cx_schema, cx_features):
    """A table in another node order must fail loudly, not train on nonsense."""
    model = M1(cx_schema, hidden=16)
    view = apply_mask(cx_graph, random_mask(cx_graph, fraction=0.05, seed=1)).model_view()
    short = dataclasses.replace(cx_features, values=cx_features.values[:-1])

    with pytest.raises(ValueError, match="rows"):
        model.encode(view, short)


def test_train_mrr_is_scored_every_tenth_epoch(cx_graph_f1, cx_features_f1, tmp_path):
    config = TrainConfig(epochs=11, hidden=8, patience=11, run_dir=tmp_path)

    result = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    assert result.history["train_mrr_epoch"] == [0, 10]
    assert all(np.isfinite(result.history["train_mrr"]))


def test_the_probe_keeps_its_stride_under_a_shorter_eval_every(
    cx_graph_f1, cx_features_f1, tmp_path
):
    config = TrainConfig(epochs=12, hidden=8, patience=12, eval_every=3, run_dir=tmp_path)

    result = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    assert result.history["train_mrr_epoch"] == [0, 10, 11]


def test_the_probe_follows_a_longer_eval_every(cx_graph_f1, cx_features_f1, tmp_path):
    config = TrainConfig(epochs=14, hidden=8, patience=14, eval_every=12, run_dir=tmp_path)

    result = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    assert result.history["train_mrr_epoch"] == [0, 12, 13]


def test_patience_stops_only_on_a_scored_epoch(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    """Epoch 2 is past patience, but the run waits for the next score at epoch 3."""
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    config = TrainConfig(
        epochs=12,
        hidden=8,
        patience=2,
        eval_every=3,
        lr_schedule="constant",
        run_dir=tmp_path,
        suites_dir=suites_dir,
    )
    scores = iter([0.5, 0.1, 0.1, 0.1])

    def fake_scores(*args, **kwargs):
        value = next(scores)
        return {instance.instance_id: value for instance in suite.instances}

    monkeypatch.setattr(training, "_validation_scores", fake_scores)

    result = fit(
        cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema
    )

    assert result.best_epoch == 0
    assert result.history["val_mrr_epoch"] == [0, 3]
    assert len(result.history["loss"]) == 4


def test_lr_warms_up_then_follows_the_schedule():
    cosine = TrainConfig(epochs=12, lr=1e-2, warmup_epochs=2, lr_schedule="cosine")
    constant = TrainConfig(epochs=12, lr=1e-2, lr_schedule="constant")

    assert [training._lr_at(epoch, cosine) for epoch in (0, 1)] == pytest.approx(
        [5e-3, 1e-2]
    )
    assert training._lr_at(2, cosine) == pytest.approx(1e-2)
    assert training._lr_at(7, cosine) == pytest.approx(5e-3)
    assert training._lr_at(11, cosine) < 3e-4
    assert {training._lr_at(epoch, constant) for epoch in range(12)} == {1e-2}


def _falling_val_fit(tmp_path, graph, table, schema, types, centroids, monkeypatch, **fields):
    """A fit whose F1-val score only falls after epoch 0."""
    suite = build_suite(graph, name="f1_val", types=types, centroids=centroids)
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(graph, suite))
    config = TrainConfig(hidden=8, eval_every=1, run_dir=tmp_path, suites_dir=suites_dir, **fields)
    scores = iter(np.linspace(0.5, 0.1, config.epochs))

    def fake_scores(*args, **kwargs):
        value = float(next(scores))
        return {instance.instance_id: value for instance in suite.instances}

    monkeypatch.setattr(training, "_validation_scores", fake_scores)
    return fit(graph, table, suite, seed=0, config=config, schema=schema)


def test_a_signal_during_teardown_leaves_a_patience_stop_finished(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path,
    monkeypatch, recording_writers,
):
    """After the loop ends, last.pt already records the stop; a signal cannot relabel it."""
    import os
    import signal
    import time
    from concurrent.futures import ThreadPoolExecutor

    from src.run_control import read_state

    real_shutdown = ThreadPoolExecutor.shutdown
    signalled = []

    def signalling_shutdown(self, *args, **kwargs):
        signalled.append(1)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.01)
        return real_shutdown(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "shutdown", signalling_shutdown)
    monkeypatch.setitem(
        training._BACKENDS, "cpu",
        dataclasses.replace(training._BACKENDS["cpu"], prefetch=True),
    )
    result = _falling_val_fit(
        tmp_path, cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids,
        monkeypatch, epochs=10, patience=2, lr_schedule="constant",
    )

    assert signalled and result.best_epoch == 0
    (run_dir,) = [path for path in tmp_path.iterdir() if path.name.startswith("f1-")]
    assert read_state(run_dir)["status"] == "stopped_early"
    assert torch.load(run_dir / "last.pt", weights_only=False)["stopped_early"] is True
    (writer,) = recording_writers
    assert writer.closed


def test_patience_counts_from_the_end_of_warmup(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    result = _falling_val_fit(
        tmp_path, cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids,
        monkeypatch, epochs=12, patience=2, warmup_epochs=6, lr_schedule="constant",
    )

    assert result.best_epoch == 0
    assert len(result.history["loss"]) == 9


def test_a_decaying_schedule_trains_every_epoch(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    result = _falling_val_fit(
        tmp_path, cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids,
        monkeypatch, epochs=6, patience=1, lr_schedule="cosine",
    )

    assert result.best_epoch == 0
    assert len(result.history["loss"]) == 6


def test_the_default_schedule_trains_the_full_budget(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path, monkeypatch
):
    """The default schedule ignores patience; a constant default stops this run early."""
    assert TrainConfig.lr_schedule == "cosine"
    result = _falling_val_fit(
        tmp_path, cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids,
        monkeypatch, epochs=6, patience=1,
    )

    assert result.best_epoch == 0
    assert len(result.history["loss"]) == 6


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("epochs", 0, "epochs"),
        ("eval_every", 0, "eval_every"),
        ("warmup_epochs", -1, "warmup_epochs"),
        ("warmup_epochs", 10, "warmup_epochs"),
        ("lr_schedule", "linear", "lr_schedule"),
        ("gap_regimes", (), "gap_regimes"),
        ("gap_regimes", ("R", "X"), "gap_regimes"),
    ],
)
def test_train_config_rejects_out_of_range_values(field, value, message):
    fields = {"epochs": 10, field: value}

    with pytest.raises(ValueError, match=message):
        TrainConfig(**fields)


def test_a_probe_mask_with_no_edges_logs_nan_instead_of_aborting(
    cx_graph, cx_schema, cx_features
):
    edgeless = dataclasses.replace(
        cx_graph, edges=cx_graph.edges[:0], weights=cx_graph.weights[:0]
    )
    config = TrainConfig(epochs=1)

    probe = training._probe(edgeless, 0, config, None, None, edgeless)

    assert probe is None
    assert np.isnan(training._train_mrr(M1(cx_schema, hidden=8), probe))


def test_the_prepared_probe_scores_bitwise_like_a_fresh_score(
    cx_graph_f1, cx_schema, cx_features_f1, monkeypatch
):
    from src.metrics import mrr

    torch.manual_seed(0)
    model = M1(cx_schema, hidden=8)
    probe = training._probe(cx_graph_f1, 0, TrainConfig(epochs=1), None, None, cx_graph_f1)
    view, pairs, n_positives, k = probe
    prepared = training._prepare_probe(model, probe, cx_features_f1, _REAL)
    assert isinstance(prepared[0], type(view.model_view()))
    assert prepared[1] is pairs

    model.eval()
    with torch.no_grad():
        logits = model.score(view.model_view(), cx_features_f1, pairs)[0]
    rows = training._ranking_rows(logits, n_positives, k).double().numpy()
    expected = mrr(rows[:, 0], rows[:, 1:])
    calls = []
    real_inference = model.score_inference

    def tracked_inference(view, table, pairs, *, row_width):
        calls.append(row_width)
        return real_inference(view, table, pairs, row_width=row_width)

    monkeypatch.setattr(model, "score_inference", tracked_inference)
    assert training._train_mrr(model, prepared) == expected
    assert training._train_mrr(model, prepared) == expected  # a prepared probe re-scores bitwise
    assert calls == [1, 1]


def test_the_probe_ranks_only_single_masked_rows_like_val_mrr(cx_graph_f1):
    """val_mrr gates out both-masked rows, so a probe that kept them would score another stratum."""
    config = TrainConfig(epochs=1, gap_fraction=0.3)

    view, pairs, n_positives, k = training._probe(
        cx_graph_f1, 0, config, None, None, cx_graph_f1
    )

    both = view.both_masked()
    assert both.any()
    assert n_positives == int((~both).sum())
    masked = view.is_masked
    positives = pairs[:n_positives]
    assert not (masked[positives[:, 0]] & masked[positives[:, 1]]).any()
    assert len(pairs) == n_positives * (1 + k)


def test_a_finished_run_does_not_rebuild_the_probe(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    config = TrainConfig(epochs=1, hidden=8, run_dir=tmp_path)
    fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    def refuse(*args, **kwargs):
        raise AssertionError("the probe was built for a run with no epochs left")

    monkeypatch.setattr(training, "_probe", refuse)
    again = fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)

    assert again.resumed is True


def test_encode_converts_each_feature_table_once(cx_graph, cx_schema, cx_features):
    model = M1(cx_schema, hidden=8)

    first = model._static_features(cx_features)
    again = model._static_features(cx_features)
    other = model._static_features(cx_features.perturb(0))

    assert all(left is right for left, right in zip(first, again))
    assert all(left is not right for left, right in zip(first, other))
    assert not torch.equal(first[0], other[0])


def test_the_feature_cache_sees_an_in_place_edit(cx_schema, cx_features):
    model = M1(cx_schema, hidden=8)
    table = dataclasses.replace(cx_features, values=cx_features.values.copy())
    before = model._static_features(table)[0].clone()

    table.values[0, 0] += 1.0

    assert not torch.equal(model._static_features(table)[0], before)


def test_the_feature_cache_holds_no_table_and_is_not_copied(cx_schema, cx_features):
    model = M1(cx_schema, hidden=8)
    table = dataclasses.replace(cx_features, values=cx_features.values.copy())
    model._static_features(table)
    reference = weakref.ref(table)

    assert copy.deepcopy(model)._feature_cache is None
    del table
    gc.collect()
    assert reference() is None


def test_a_type_id_outside_the_embedding_fails_by_name(cx_schema, cx_features):
    model = M1(cx_schema, hidden=8)
    values = cx_features.values.copy()
    values[0, cx_features.columns.index("cell_type")] = model.type_embedding.num_embeddings
    table = dataclasses.replace(cx_features, values=values)

    with pytest.raises(ValueError, match="cell_type ids"):
        model._static_features(table)


def test_table_schema_refuses_a_table_without_cell_type(cx_features):
    column = cx_features.columns.index("cell_type")
    table = dataclasses.replace(
        cx_features,
        values=np.delete(cx_features.values, column, axis=1),
        columns=tuple(name for name in cx_features.columns if name != "cell_type"),
    )

    with pytest.raises(ValueError, match="cell_type"):
        training._table_schema(table)


#: Configs a preempted run must resume bit-identically under.
_RESUME_CONFIGS = [
    {"eval_every": 2, "warmup_epochs": 2, "lr_schedule": "cosine"},
    {"eval_every": 3, "gap_regimes": ("R", "T", "S")},
    {"steps_per_epoch": 3, "sibling_features": True, "encoder_dropout": 0.1, "decoder_dropout": 0.1},
]

#: Every knob that changes a step: torch-side dropout draws and several draws per epoch.
_STEP_KNOBS = {"steps_per_epoch": 3, "sibling_features": True, "encoder_dropout": 0.1, "decoder_dropout": 0.1}


def _with_geometry(table, seed=0, located=None):
    """``table`` with random synapse geometry, which the fixtures build without."""
    values = table.values.copy()
    geometry = table.blocks["geometry"]
    values[:, geometry.start:geometry.start + 6] = np.random.default_rng(seed).normal(
        size=(len(values), 6)
    )
    if located is None:
        located = np.ones(len(values), dtype=bool)
    return dataclasses.replace(table, values=values, geometry_present=located)


@pytest.mark.parametrize("extra", _RESUME_CONFIGS)
def test_a_resumed_run_matches_the_uninterrupted_one(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path,
    monkeypatch, extra,
):
    """Preempted between scored epochs, the resume must redraw nothing and rescore nothing."""
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))

    table = _with_geometry(cx_features_f1)

    def run(name):
        config = TrainConfig(
            epochs=7, hidden=8, patience=7, run_dir=tmp_path / name,
            suites_dir=suites_dir, **extra,
        )
        return fit(
            cx_graph_f1, table, suite, seed=0, config=config, schema=cx_schema
        )

    whole = run("whole")
    real = training._epoch_loss
    calls = {"n": 0}

    def preempted(*args, **kwargs):
        if calls["n"] == 3 * TrainConfig(epochs=7, **extra).steps_per_epoch:
            raise KeyboardInterrupt
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "_epoch_loss", preempted)
    with pytest.raises(RunStopped):
        run("preempted")
    monkeypatch.undo()
    resumed = run("preempted")

    assert resumed.resumed
    assert resumed.history == whole.history
    assert resumed.best_epoch == whole.best_epoch
    for key, value in whole.state_dict.items():
        assert torch.equal(resumed.state_dict[key], value), key


def _prefetch_on_cpu(monkeypatch, prefetch=True):
    monkeypatch.setitem(
        training._BACKENDS, "cpu",
        dataclasses.replace(training._BACKENDS["cpu"], prefetch=prefetch),
    )


def _prefetch_fit(
    tmp_path, name, graph, table, suite, schema, suites_dir, *, epochs=6, **knobs
):
    config = TrainConfig(
        epochs=epochs, hidden=8, patience=epochs, eval_every=2,
        run_dir=tmp_path / name, suites_dir=suites_dir,
        **{"steps_per_epoch": 1, **knobs},
    )
    return fit(graph, table, suite, seed=0, config=config, schema=schema)


def _last_checkpoint(tmp_path, name):
    (run_dir,) = [path for path in (tmp_path / name).iterdir() if path.is_dir()]
    return run_dir, torch.load(run_dir / "last.pt", weights_only=False)


@pytest.fixture
def prefetch_inputs(cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path):
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    return cx_graph_f1, cx_features_f1, suite, cx_schema, suites_dir


@pytest.mark.parametrize("knobs", [{}, _STEP_KNOBS])
def test_prefetching_the_next_epoch_trains_bitwise_like_the_serial_loop(
    prefetch_inputs, tmp_path, monkeypatch, knobs
):
    """Drawing epoch e+1 on a worker during epoch e's step changes no draw and no weight."""
    from concurrent.futures import ThreadPoolExecutor

    assert training._BACKENDS["cuda"].prefetch and training._BACKENDS["mps"].prefetch
    assert not training._BACKENDS["cpu"].prefetch
    submitted = []
    real_submit = ThreadPoolExecutor.submit

    def counted(self, *args, **kwargs):
        submitted.append(1)
        return real_submit(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "submit", counted)
    serial = _prefetch_fit(tmp_path, "serial", *prefetch_inputs, **knobs)
    assert not submitted
    _prefetch_on_cpu(monkeypatch)
    prefetched = _prefetch_fit(tmp_path, "prefetched", *prefetch_inputs, **knobs)

    # One per draw after the first, none past the last.
    assert len(submitted) == 6 * knobs.get("steps_per_epoch", 1) - 1
    assert prefetched.history == serial.history
    for key, value in serial.state_dict.items():
        assert torch.equal(prefetched.state_dict[key], value), key
    _, a = _last_checkpoint(tmp_path, "prefetched")
    _, b = _last_checkpoint(tmp_path, "serial")
    assert a["rng"] == b["rng"]
    assert torch.equal(a["torch_rng"], b["torch_rng"])


@pytest.mark.parametrize("knobs", [{}, _STEP_KNOBS])
def test_a_paused_prefetching_run_checkpoints_the_serial_rng_and_resumes_exactly(
    prefetch_inputs, tmp_path, monkeypatch, knobs
):
    """The pause lands while epoch 3's batch is in flight; it is discarded, not saved."""
    whole = _prefetch_fit(tmp_path, "whole", *prefetch_inputs, **knobs)

    def pause_after(epoch):
        calls = {"n": 0}

        def request(run_dir):
            calls["n"] += 1
            return {"status": "paused"} if calls["n"] == epoch + 1 else None

        return request

    _prefetch_on_cpu(monkeypatch)
    monkeypatch.setattr(training, "read_request", pause_after(2))
    with pytest.raises(RunStopped) as stop:
        _prefetch_fit(tmp_path, "paused", *prefetch_inputs, **knobs)
    assert stop.value.resume_epoch == 3
    _prefetch_on_cpu(monkeypatch, False)
    monkeypatch.setattr(training, "read_request", pause_after(2))
    with pytest.raises(RunStopped):
        _prefetch_fit(tmp_path, "paused-serial", *prefetch_inputs, **knobs)
    _, paused = _last_checkpoint(tmp_path, "paused")
    _, paused_serial = _last_checkpoint(tmp_path, "paused-serial")
    assert paused["epoch"] == paused_serial["epoch"] == 2
    assert paused["rng"] == paused_serial["rng"]
    monkeypatch.undo()
    _prefetch_on_cpu(monkeypatch)

    resumed = _prefetch_fit(tmp_path, "paused", *prefetch_inputs, **knobs)

    assert resumed.resumed
    assert resumed.history == whole.history
    for key, value in whole.state_dict.items():
        assert torch.equal(resumed.state_dict[key], value), key


def test_a_prefetching_run_preempted_inside_an_epoch_resumes_exactly(
    prefetch_inputs, tmp_path, monkeypatch
):
    """Killed after epoch 1's first step, the resume retrains epoch 1 from its first draw."""
    whole = _prefetch_fit(tmp_path, "whole", *prefetch_inputs, **_STEP_KNOBS)
    _prefetch_on_cpu(monkeypatch)
    real = training._epoch_loss
    calls = {"n": 0}

    def preempted(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == _STEP_KNOBS["steps_per_epoch"] + 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "_epoch_loss", preempted)
    with pytest.raises(RunStopped):
        _prefetch_fit(tmp_path, "preempted", *prefetch_inputs, **_STEP_KNOBS)
    monkeypatch.setattr(training, "_epoch_loss", real)
    resumed = _prefetch_fit(tmp_path, "preempted", *prefetch_inputs, **_STEP_KNOBS)

    assert resumed.resumed
    assert resumed.history == whole.history
    for key, value in whole.state_dict.items():
        assert torch.equal(resumed.state_dict[key], value), key


def test_a_checkpoint_carries_and_restores_the_device_generator(
    cx_graph_f1, cx_features_f1, tmp_path, monkeypatch
):
    """Dropout draws from the device's generator, so a resume must restore it too."""
    saved = torch.tensor([7], dtype=torch.uint8)
    restored = []
    monkeypatch.setitem(
        training._BACKENDS, "cpu",
        dataclasses.replace(training._BACKENDS["cpu"], rng=(saved.clone, restored.append)),
    )
    config = TrainConfig(epochs=4, hidden=8, patience=4, run_dir=tmp_path / "run",
        steps_per_epoch=1,
    )
    real = training._epoch_loss
    calls = {"n": 0}

    def preempted(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "_epoch_loss", preempted)
    with pytest.raises(RunStopped):
        fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config)
    _, checkpoint = _last_checkpoint(tmp_path, "run")
    assert torch.equal(checkpoint["device_rng"], saved)
    monkeypatch.setattr(training, "_epoch_loss", real)
    assert fit(cx_graph_f1, cx_features_f1, None, seed=0, config=config).resumed
    assert len(restored) == 1 and torch.equal(restored[0], saved)


@pytest.mark.parametrize("device", [
    pytest.param("mps", marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS is not available"
    )),
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is not available"
    )),
])
def test_the_device_generator_resumes_the_dropout_mask_sequence(tmp_path, device):
    path = tmp_path / "last.pt"
    model = torch.nn.Linear(2, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    ones = torch.ones(4096, device=device)
    dropout = torch.nn.functional.dropout
    torch.manual_seed(0)
    first = dropout(ones, 0.5)
    rng_state = np.random.default_rng(0).bit_generator.state
    training._save_checkpoint(path, model, optimizer, 0, {}, 0, rng_state, device=device)
    expected = dropout(ones, 0.5)

    # Reseeding alone, as a resumed fit does, would replay the first mask.
    torch.manual_seed(0)
    assert torch.equal(dropout(ones, 0.5), first) and not torch.equal(first, expected)
    torch.manual_seed(0)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training._BACKENDS[device].rng[1](checkpoint["device_rng"])
    assert torch.equal(dropout(ones, 0.5), expected)


@pytest.mark.parametrize("prefetch", [True, False])
def test_a_failed_epoch_draw_crashes_the_run_at_its_own_epoch(
    prefetch_inputs, tmp_path, monkeypatch, prefetch
):
    """A worker error is the serial error, raised before that epoch's step or checkpoint."""
    from src.run_control import read_state

    real = training._ranking_pairs
    calls = {"n": 0}

    def failing(view, true_edges, *, single_masked=False, rows=None):
        if not single_masked:
            calls["n"] += 1
            if calls["n"] == 3:
                raise ValueError("sampler failed for epoch 2")
        return real(view, true_edges, single_masked=single_masked, rows=rows)

    monkeypatch.setattr(training, "_ranking_pairs", failing)
    _prefetch_on_cpu(monkeypatch, prefetch)
    with pytest.raises(ValueError, match="sampler failed for epoch 2"):
        _prefetch_fit(tmp_path, "failed", *prefetch_inputs)

    run_dir, last = _last_checkpoint(tmp_path, "failed")
    assert last["epoch"] == 1
    assert len(last["history"]["loss"]) == 2
    assert read_state(run_dir)["status"] == "crashed"


def test_a_failed_draw_for_an_epoch_that_never_runs_is_discarded(
    prefetch_inputs, tmp_path, monkeypatch
):
    real = training._ranking_pairs
    calls = {"n": 0}

    def failing(view, true_edges, *, single_masked=False, rows=None):
        if not single_masked:
            calls["n"] += 1
            if calls["n"] == 3:
                raise ValueError("sampler failed for epoch 2")
        return real(view, true_edges, single_masked=single_masked, rows=rows)

    requests = {"n": 0}

    def request(run_dir):
        requests["n"] += 1
        return {"status": "paused"} if requests["n"] == 2 else None

    monkeypatch.setattr(training, "_ranking_pairs", failing)
    monkeypatch.setattr(training, "read_request", request)
    _prefetch_on_cpu(monkeypatch)
    with pytest.raises(RunStopped) as stop:
        _prefetch_fit(tmp_path, "paused", *prefetch_inputs)

    assert stop.value.status == "paused" and stop.value.resume_epoch == 2
    assert calls["n"] == 3


def test_fit_starts_the_count_head_from_the_pruned_training_graph(
    cx_graph_f1, cx_features_f1, cx_schema, cx_types, cx_centroids, tmp_path,
    monkeypatch,
):
    """The F1-val edges are not training data, so their counts may not set the bias."""
    suite = build_suite(
        cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids
    )
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(cx_graph_f1, suite))
    seen = []
    real = training._init_count_head

    def spy(model, graph):
        seen.append(len(graph.edges))
        return real(model, graph)

    monkeypatch.setattr(training, "_init_count_head", spy)
    config = TrainConfig(
        epochs=1, hidden=8, patience=1, run_dir=tmp_path, suites_dir=suites_dir
    )
    fit(cx_graph_f1, cx_features_f1, suite, seed=0, config=config, schema=cx_schema)

    excluded = training._masked_union(suite.resolve(cx_graph_f1))
    pruned = training._training_graph(cx_graph_f1, excluded)
    assert seen == [len(pruned.edges)]
    assert seen[0] < len(cx_graph_f1.edges)


def test_every_mask_regime_has_a_gap_draw():
    assert tuple(training._GAP_REGIMES) == MASK_REGIMES


def test_an_r_only_config_draws_the_mask_it_drew_before_regimes(cx_graph, cx_features):
    """One regime makes no extra draw, so the R-only gap stream is unchanged."""
    config = TrainConfig(epochs=1)

    instance = training._gap_sampler(cx_graph, cx_features, config, None)(
        np.random.default_rng(4)
    )
    before = training.simulate_gaps(
        cx_graph, np.random.default_rng(4), config.gap_fraction
    )

    assert instance.regime == "R"
    assert np.array_equal(instance.masked, before.masked)


def test_mixed_gap_regimes_draw_each_regime_at_the_same_size(cx_graph, cx_features):
    config = TrainConfig(epochs=1, gap_regimes=("R", "T", "S"))
    rng = np.random.default_rng(0)
    draw = training._gap_sampler(cx_graph, _with_geometry(cx_features), config, None)
    target = round(config.gap_fraction * cx_graph.n_nodes)

    instances = [draw(rng) for _ in range(30)]

    assert {instance.regime for instance in instances} == {"R", "T", "S"}
    for instance in instances:
        if instance.regime != "T":
            assert instance.n_masked == target


def test_a_spatial_gap_is_sized_to_the_pool_and_skips_unlocated_neurons(
    cx_graph, cx_features
):
    config = TrainConfig(epochs=1, gap_regimes=("S",))
    located = np.arange(cx_graph.n_nodes) % 2 == 1
    table = _with_geometry(cx_features, located=located)

    instance = training._gap_sampler(cx_graph, table, config, None)(
        np.random.default_rng(0)
    )

    assert instance.n_masked == round(config.gap_fraction * cx_graph.n_nodes)
    assert located[instance.masked].all()


@pytest.mark.parametrize(
    "regimes, message",
    [
        (("S",), "rebuild the features"),
        (("R", "S"), "only 0 have synapse geometry"),
        (("T",), "cell types"),
    ],
)
def test_a_regime_its_table_cannot_support_fails_before_the_run_starts(
    cx_graph_f1, cx_features_f1, tmp_path, regimes, message
):
    table = cx_features_f1
    assert not table.geometry_present.any(), "the fixture has no synapse geometry"
    if regimes == ("S",):
        table = dataclasses.replace(table, geometry_present=None)
    if regimes == ("T",):
        values = table.values.copy()
        values[:, table.columns.index("cell_type")] = 0
        table = dataclasses.replace(table, values=values)
    config = TrainConfig(epochs=2, hidden=8, gap_regimes=regimes, run_dir=tmp_path)

    with pytest.raises(ValueError, match=message):
        fit(cx_graph_f1, table, None, seed=0, config=config)

    assert not any(tmp_path.iterdir())


def test_fit_trains_on_mixed_gap_regimes(cx_graph_f1, cx_features_f1, tmp_path):
    config = TrainConfig(
        epochs=6, hidden=8, patience=6, gap_regimes=("R", "T", "S"), run_dir=tmp_path
    )

    result = fit(cx_graph_f1, _with_geometry(cx_features_f1), None, seed=0, config=config)

    assert len(result.history["loss"]) == config.epochs
    assert np.isfinite(result.history["loss"]).all()
    assert result.manifest["config"]["gap_regimes"] == ["R", "T", "S"]


def _dense_sibling_features(prepared, type_ids):
    """The dense ``n_types * n_nodes`` block the sparse count replaced."""
    n_nodes = prepared.n_nodes
    n_types = int(type_ids.max().item()) + 1
    masked, edges = prepared.is_masked, prepared.edges
    counts_out = torch.zeros(n_types * n_nodes)
    counts_in = torch.zeros(n_types * n_nodes)
    ones = torch.ones(edges.shape[0])
    counts_out.index_add_(0, type_ids[edges[:, 0]] * n_nodes + edges[:, 1], ones)
    counts_in.index_add_(0, type_ids[edges[:, 1]] * n_nodes + edges[:, 0], ones)
    connected_nodes = torch.zeros(n_nodes, dtype=torch.bool)
    connected_nodes[edges.reshape(-1)] = True
    members = type_ids[connected_nodes & ~masked]
    siblings = torch.zeros(n_types).index_add_(0, members, torch.ones(members.shape[0]))
    source, target = prepared.index[:, 0], prepared.index[:, 1]
    source_masked = masked[source]
    anchor = torch.where(source_masked, type_ids[source], type_ids[target])
    flat = torch.where(source_masked, anchor * n_nodes + target, anchor * n_nodes + source)
    connected = torch.where(source_masked, counts_out[flat], counts_in[flat])
    group = siblings[anchor]
    live = (source_masked | masked[target]) & (anchor > 0)
    connected = torch.where(live, connected, torch.zeros_like(connected))
    group = torch.where(live, group, torch.zeros_like(group))
    return torch.stack([connected / group.clamp_min(1), torch.log1p(connected), torch.log1p(group)], 1)


def test_sparse_sibling_counts_equal_the_dense_columns(paired_graph_f1, paired_annotations):
    """Masked and unmasked ends, the untyped neuron (OOV) and the single-side type, every pair."""
    from src.features import FeatureSchema
    from train.model import _sibling_features

    schema = FeatureSchema.from_f1(paired_annotations, "AL")
    table = build_features(paired_annotations, paired_graph_f1, schema, dataset="f1")
    model = M1(schema, hidden=8, sibling_features=True)
    type_ids = model.type_ids(table)
    assert (type_ids == 0).any()
    n = paired_graph_f1.n_nodes
    pairs = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    for seed in range(3):
        view = apply_mask(paired_graph_f1, random_mask(paired_graph_f1, fraction=0.3, seed=seed))
        prepared = model.prepare(view.model_view(), pairs)
        assert torch.equal(
            _sibling_features(prepared, type_ids), _dense_sibling_features(prepared, type_ids)
        )


def _batch(graph, table, config, rng):
    return training._epoch_batch(graph, table, rng, config)


def test_pair_batch_unset_or_covering_the_gap_keeps_every_row(cx_graph_f1, cx_features_f1):
    full = TrainConfig(epochs=1)
    wide = TrainConfig(epochs=1, pair_batch=10**9)
    rng_full, rng_wide = np.random.default_rng(4), np.random.default_rng(4)
    a, b = _batch(cx_graph_f1, cx_features_f1, full, rng_full), _batch(
        cx_graph_f1, cx_features_f1, wide, rng_wide
    )
    assert a.rows is None and b.rows is None
    assert np.array_equal(a.pairs, b.pairs)
    assert rng_full.bit_generator.state == rng_wide.bit_generator.state


def test_pair_batch_draws_the_same_rows_after_a_resume(cx_graph_f1, cx_features_f1):
    config = TrainConfig(epochs=1, pair_batch=3)
    straight = np.random.default_rng(5)
    expected = [_batch(cx_graph_f1, cx_features_f1, config, straight) for _ in range(2)]
    first = np.random.default_rng(5)
    _batch(cx_graph_f1, cx_features_f1, config, first)
    resumed = np.random.default_rng()
    resumed.bit_generator.state = first.bit_generator.state
    again = _batch(cx_graph_f1, cx_features_f1, config, resumed)
    assert len(again.rows) == 3
    assert np.array_equal(again.rows, expected[1].rows)
    assert np.array_equal(again.pairs, expected[1].pairs)


def test_pair_batch_count_loss_reads_only_the_selected_positives(
    cx_graph_f1, cx_features_f1, cx_schema, monkeypatch
):
    config = TrainConfig(epochs=1, pair_batch=3)
    batch = _batch(cx_graph_f1, cx_features_f1, config, np.random.default_rng(6))
    assert batch.n_positives == 3 < len(batch.view.hidden_edges)
    hidden = np.asarray(batch.view.hidden_edges)
    assert np.array_equal(batch.pairs[:3], hidden[batch.rows])

    captured = {}
    real = torch.nn.functional.smooth_l1_loss

    def spy(counts, targets, *args, **kwargs):
        captured["counts"], captured["targets"] = counts, targets
        return real(counts, targets, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "smooth_l1_loss", spy)
    model = M1(training._table_schema(cx_features_f1), hidden=8)
    training._epoch_loss(model, cx_features_f1, batch, config)
    weights = np.asarray(batch.view.hidden_weights)[batch.rows]
    assert captured["counts"].shape == (3,)
    assert torch.equal(captured["targets"], torch.log1p(torch.as_tensor(weights, dtype=torch.float32)))


def test_decoder_product_rank_adds_a_pairwise_term_and_defaults_to_none():
    torch.manual_seed(0)
    plain = PairDecoder(8, pair_dim=1)
    assert not hasattr(plain, "prod_source")
    dec = PairDecoder(8, pair_dim=1, product_rank=4)
    nodes = torch.randn(5, 8)
    src, tgt = dec.project(nodes)
    assert src.shape == (5, 12)
    ends = torch.tensor([[0, 1], [2, 3]])
    feats = torch.randn(2, 1)
    base = dec.decode(src, tgt, ends, feats)[0]
    # swapping the product factors of one endpoint must change the logits
    tgt2 = tgt.clone()
    tgt2[:, 8:] = 0
    assert not torch.allclose(base, dec.decode(src, tgt2, ends, feats)[0])
    assert base.shape == (2,)
