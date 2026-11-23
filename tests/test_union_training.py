"""Seams of union training (Phase 3 plan 04): holdout, preflight, the probe and queue admission."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.boundary_suites import BoundarySuite, build_boundary_suite  # noqa: E402
from src.features import FeatureSchema, build_features  # noqa: E402
from src.negatives import DEFAULT_SPEC  # noqa: E402
from src.suites import suite_negatives  # noqa: E402
from src.union import boundary_flags, pair_counts, union_graph, union_labels  # noqa: E402
from train import train as training  # noqa: E402
from train.region_config import RegionConfig, region_paths  # noqa: E402
from train.train import TrainConfig, fit, validate_m1_inputs  # noqa: E402

B = 10**17


@pytest.fixture(scope="module")
def union(tmp_path_factory):
    rng = np.random.default_rng(0)
    rows = []
    for pre, post in sorted({(int(a), int(b)) for a, b in rng.integers(0, 300, size=(3000, 2)) if a != b}):
        rows += ([(pre, post, "FB", 3), (pre, post, "LH_L", 2)] if rng.random() < 0.5 else [(pre, post, "FB", 6)])
    pre, post, neuropil, count = zip(*rows)
    connections = pd.DataFrame({"pre_pt_root_id": np.add(pre, B), "post_pt_root_id": np.add(post, B),
                                "neuropil": neuropil, "syn_count": count})
    pc = pair_counts(connections)
    graph = union_graph(pc, "CX+LH")
    boundary = boundary_flags(graph, pc)
    ids = graph.node_ids
    annotations = pd.DataFrame({"root_id": ids, "super_class": ["central"] * len(ids), "class": ["a"] * len(ids),
                                "cell_type": [f"t{k % 6}" for k in range(len(ids))], "side": ["left"] * len(ids),
                                "flow": ["intrinsic"] * len(ids), "nt_type": ["ACH"] * len(ids)})
    labels = union_labels("CX+LH")
    schema = dataclasses.replace(FeatureSchema.from_f1(annotations, "CX"), region="CX+LH", labels=labels)
    innervation = pd.DataFrame(rng.random((len(ids), len(labels))), index=pd.Index(ids, name="root_id"),
                               columns=list(labels))
    table = build_features(annotations, graph, schema, dataset="f1", innervation=innervation)
    types = annotations.set_index("root_id")["cell_type"]
    suite = build_boundary_suite(graph, boundary, name="phase3_boundary_f1_val", types=types,
                                 centroids=rng.normal(size=(len(ids), 3)))
    suites = tmp_path_factory.mktemp("suites")
    path = suites / "f1" / "cx+lh" / "phase3_boundary_f1_val.negatives.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"name": suite.name, "spec": {"k": DEFAULT_SPEC.k, "alpha": DEFAULT_SPEC.alpha,
                                                            "master_seed": DEFAULT_SPEC.master_seed},
                                "digests": {k: v.digest() for k, v in suite_negatives(graph, suite).items()}}))
    return graph, table, schema, suite, suites


def test_union_training_never_sees_an_edge_of_a_boundary_suite_root(union, tmp_path, monkeypatch):
    graph, table, schema, suite, suites = union
    held = np.unique(np.concatenate([inst.masked for inst in suite.resolve(graph)]))
    seen = []
    real = training._epoch_batch

    def spy(train_graph, *args, **kwargs):
        seen.append(train_graph)
        return real(train_graph, *args, **kwargs)

    monkeypatch.setattr(training, "_epoch_batch", spy)
    config = TrainConfig(epochs=2, hidden=8, steps_per_epoch=2, eval_every=1, run_dir=tmp_path, suites_dir=suites)
    fit(graph, table, suite, seed=0, config=config, schema=schema)
    assert seen and np.isin(graph.edges, held).any()
    for train_graph in seen:
        assert not np.isin(train_graph.edges, held).any()


def test_preflight_pairs_each_graph_kind_with_its_own_early_stopping_suite(union):
    graph, _, _, suite, suites = union
    validate_m1_inputs(graph, suite, suites)
    wrong = dataclasses.replace(suite, name="f1_val")
    with pytest.raises(ValueError, match="phase3_boundary_f1_val"):
        validate_m1_inputs(graph, wrong, suites)
    assert isinstance(suite, BoundarySuite)


def test_configs_and_paths_accept_unions_but_not_strangers():
    assert RegionConfig("UNION12", train={"epochs": 1}).region == "UNION12"
    assert region_paths("UNION12", "f1", "data").processed.as_posix() == "data/processed/f1/union12"
    for bad in ("CX+CX", "NOPE", "LH+CX"):
        with pytest.raises(ValueError, match="unknown region"):
            RegionConfig(bad, train={"epochs": 1})


class _NoMemory:
    """Stands in for the probe's memory reader; the CPU reader is POSIX-only and this test compares inputs."""

    def __init__(self, device):
        pass

    def reset(self):
        pass

    def peak(self):
        return 0

    def peak_allocated(self):
        return None

    def total(self):
        return None


def test_the_probe_builds_the_same_training_inputs_as_fit_on_a_union(union, tmp_path, monkeypatch):
    # Plan 04's seam: the probe measures the step training would take, on the same held-out graph.
    from scripts import probe_memory

    graph, table, schema, suite, suites = union
    seen = {"probe": [], "fit": []}
    stage, real = ["probe"], training._epoch_batch

    def spy(train_graph, *args, **kwargs):
        seen[stage[0]].append((train_graph, kwargs["complete_graph"]))
        return real(train_graph, *args, **kwargs)

    monkeypatch.setattr(training, "_epoch_batch", spy)
    monkeypatch.setattr(probe_memory, "_Memory", _NoMemory)
    config = TrainConfig(epochs=1, hidden=8, steps_per_epoch=1, eval_every=1, run_dir=tmp_path, suites_dir=suites)
    probe_memory.probe(graph, table, schema, suite, config, device="cpu", draws=1)
    stage[0] = "fit"
    fit(graph, table, suite, seed=0, config=config, schema=schema)
    (probed, probed_full), (fitted, fitted_full) = seen["probe"][0], seen["fit"][0]
    assert np.array_equal(probed.edges, fitted.edges) and np.array_equal(probed.weights, fitted.weights)
    assert probed_full is graph and fitted_full is graph
