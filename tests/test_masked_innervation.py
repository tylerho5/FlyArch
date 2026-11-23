"""Seams of the masked-innervation ablation: the table a training step and a scorer may read."""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.features import FeatureTable  # noqa: E402
from train import train as training  # noqa: E402
from train.train import MASKED_INNERVATION, TrainConfig  # noqa: E402


def _table(columns):
    values = np.arange(len(columns) * 3, dtype=float).reshape(3, len(columns))
    return FeatureTable(values, tuple(columns), {}, None)


def test_the_ablation_touches_only_the_rows_and_columns_it_names():
    columns = ["centroid_x", "innervation_AL_L", "innervation_AL_R", "innervation_LH_L"]
    table = _table(columns)
    out = table.with_uniform_innervation([1])
    inn = [1, 2, 3]
    np.testing.assert_allclose(out.values[1, inn], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_array_equal(out.values[0], table.values[0])
    np.testing.assert_array_equal(out.values[2], table.values[2])
    np.testing.assert_array_equal(out.values[:, [0]], table.values[:, [0]])
    np.testing.assert_array_equal(table.values[1], _table(columns).values[1])  # the input is untouched
    assert table.with_uniform_innervation([]) is table
    with pytest.raises(ValueError, match="no innervation column"):
        _table(["centroid_x"]).with_uniform_innervation([0])


def test_the_regime_is_validated():
    assert TrainConfig(epochs=1).masked_innervation == "uniform"
    assert set(MASKED_INNERVATION) == {"uniform", "real"}
    with pytest.raises(ValueError, match="masked_innervation must be one of"):
        TrainConfig(epochs=1, masked_innervation="hidden")


class _Spy:
    """Stands in for the model: records the table each step hands it."""

    _device = torch.device("cpu")

    def __init__(self, n_positives):
        self.n_positives = n_positives
        self.tables = []

    def score(self, view, table, pairs, **_kwargs):
        self.tables.append(table)
        return torch.zeros(len(pairs)), torch.zeros(self.n_positives)


@pytest.mark.parametrize("regime", MASKED_INNERVATION)
def test_a_training_step_reads_masked_roots_as_uniform_or_as_scored(cx_graph_f1, cx_features_f1, regime):
    graph = cx_graph_f1
    columns = [k for k, name in enumerate(cx_features_f1.columns) if name.startswith("innervation_")]
    values = cx_features_f1.values.copy()
    profile = np.arange(len(columns)) + 1.0
    values[:, columns] = profile / profile.sum()  # non-uniform, so the ablation has work to do
    table = dataclasses.replace(cx_features_f1, values=values)
    config = TrainConfig(epochs=1, hidden=8, steps_per_epoch=1, gap_fraction=0.2,
                         masked_innervation=regime)
    batch = training._epoch_batch(graph, table, np.random.default_rng(0), config)
    masked = np.flatnonzero(batch.view.is_masked)
    assert masked.size  # a 20% gap always hides something
    spy = _Spy(batch.n_positives)
    training._epoch_loss(spy, table, batch, config)
    frame = spy.tables[0]
    if regime == "real":
        assert frame is table
        return
    assert not np.allclose(table.values[masked][:, columns], 1.0 / len(columns))
    np.testing.assert_allclose(frame.values[masked][:, columns], 1.0 / len(columns))
    keep = np.ones(len(table.values), dtype=bool)
    keep[masked] = False
    np.testing.assert_array_equal(frame.values[keep], table.values[keep])
