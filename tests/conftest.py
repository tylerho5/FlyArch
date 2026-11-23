"""Shared synthetic Central Complex fixtures.

Built in-fixture (no ``data/`` dependency) so any suite can score against a
small, fast, reproducible CX graph. Edges are drawn type-structured -- each
node gets a cell type first, then a per-type-pair affinity decides whether an
edge is drawn -- so a type-pair predictor has real signal on this graph,
unlike a uniform random graph.
"""

import json
import os

# torch and xgboost each link their own libomp on macOS, and initialising both
# thread pools in one process segfaults the first xgboost fit. This is asserted
# rather than defaulted: a pre-set OMP_NUM_THREADS in the shell would otherwise
# defeat the pin and crash the suite. Conftest loads before any test module
# imports torch, so the pin is in place for every ordering. Test-process only:
# the co-import hazard for training scripts is a separate, documented concern.
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pytest

import dataclasses

from src.features import FeatureSchema, build_features
from src.negatives import DEFAULT_SPEC, pack_edges, sample_negatives
from src.regions import build_region_graph
from src.suites import build_suite, suite_negatives

# Ten cell types (>= the 7 distinct a disjoint f2_dev + f2_confirm needs): dev's
# three T instances plus confirm's four, which draw types disjoint from dev's.
_N = 300  # enough visible-degree partners for K=100 negatives after masking
_TYPES = ("ET1", "ET2", "ET3", "ET4", "ET5", "ET6", "ET7", "ET8", "ET9", "ET10")
_SEED = 20260912


def _synthetic_connections() -> tuple[pd.DataFrame, np.ndarray]:
    """A CX-labeled connections frame plus each root id's assigned type index."""
    rng = np.random.default_rng(_SEED)
    root_ids = np.arange(1, _N + 1, dtype=np.int64)
    type_idx = np.arange(_N) % len(_TYPES)

    # Same-type pairs are far likelier to connect, so a type-pair predictor
    # beats chance on this synthetic graph (plan 06's beats-random smoke test).
    affinity = np.full((len(_TYPES), len(_TYPES)), 0.03)
    np.fill_diagonal(affinity, 0.15)

    draws = rng.random((_N, _N))
    connect = draws < affinity[type_idx[:, None], type_idx[None, :]]
    np.fill_diagonal(connect, False)
    source, target = np.nonzero(connect)

    connections = pd.DataFrame({
        "pre_pt_root_id": root_ids[source],
        "post_pt_root_id": root_ids[target],
        "neuropil": "FB",
        "syn_count": 9,  # >= DEFAULT_THRESHOLD so every drawn pair is an edge
    })
    return connections, type_idx


def _write_committed_negatives(suites_dir, suite, negatives):
    """Write the committed-negatives artefact the way build_suites.py does."""
    path = (
        suites_dir / suite.dataset / suite.region.lower() / f"{suite.name}.negatives.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": suite.name,
                "spec": {
                    "k": DEFAULT_SPEC.k,
                    "alpha": DEFAULT_SPEC.alpha,
                    "master_seed": DEFAULT_SPEC.master_seed,
                },
                "digests": {iid: sample.digest() for iid, sample in negatives.items()},
            }
        )
    )
    return path


def _bypass_result_verification(monkeypatch, raw):
    """Stub the scoring CLIs' input and provenance checks around ``raw`` result rows."""
    from scripts import scoring_common

    monkeypatch.setattr(scoring_common, "load_result_tables", lambda path, repo_root=None: (raw, raw.copy()))
    monkeypatch.setattr(scoring_common, "validate_f1_cohort_inputs",
                        lambda *a, **k: {"inputs": (None,) * 7})
    for name in ("load_scoring_inputs", "validate_scored_suite", "validate_expert_provenance",
                 "validate_baseline_provenance", "dev_fit_identity"):
        monkeypatch.setattr(scoring_common, name, lambda *a, **k: None)


def write_release_evidence(repo, calibrations=(), scored_files=()):
    """Write ``results/release-evidence.json`` (schema_version 1) under ``repo``; commit it yourself."""
    path = repo / "results" / "release-evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1, "source_snapshot": "0" * 40,
        "calibrations": list(calibrations), "scored_files": list(scored_files),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def evidence_entry(delta_path, repo, *, source_tag_commit="1" * 40):
    """A calibration entry pinning ``delta_path``'s current bytes, as the release generator writes it."""
    import hashlib

    document = json.loads(delta_path.read_bytes())
    return {"region": document["region"], "path": delta_path.relative_to(repo).as_posix(),
            "cohort": document["cohort"], "tag": document["tag"], "source_tag_commit": source_tag_commit,
            "delta_sha256": hashlib.sha256(delta_path.read_bytes()).hexdigest()}


def write_training_inputs(tmp_path, graph, annotations, schema, types, centroids):
    """What ``train_m1.py`` reads: a processed directory and a one-instance ``f1_val`` suite.

    Returns ``(processed, suite_path, suites_dir)``; the committed negatives sit
    under ``<suites_dir>/f1/<region>/``.
    """
    processed = tmp_path / "processed"
    processed.mkdir()
    graph.write(processed)
    table = build_features(annotations, graph, schema, dataset="f1")
    frame = pd.DataFrame(table.values, columns=table.columns)
    frame.insert(0, "root_id", graph.node_ids)
    frame.to_parquet(processed / "features.parquet", index=False)
    (processed / "schema.json").write_text(schema.to_json())
    table.coverage.to_csv(processed / "coverage.csv", index=False)

    suite = build_suite(graph, name="f1_val", types=types, centroids=centroids)
    suite = dataclasses.replace(suite, instances=suite.instances[:1])
    suite_path = tmp_path / "f1_val.json"
    suite.write(suite_path)
    suites_dir = tmp_path / "suites"
    _write_committed_negatives(suites_dir, suite, suite_negatives(graph, suite))
    return processed, suite_path, suites_dir


@pytest.fixture(scope="session")
def _synthetic_cx():
    # Drawn once and shared, so cx_graph and cx_annotations cannot drift apart.
    return _synthetic_connections()


@pytest.fixture(scope="session")
def cx_graph(_synthetic_cx):
    connections, _ = _synthetic_cx
    return build_region_graph(connections, region="CX", dataset="f2")


@pytest.fixture(scope="session")
def cx_graph_f1(_synthetic_cx):
    """The same synthetic region labelled F1, for the ``f1_val`` suite.

    Suite names are coupled to a dataset (``f1_val`` calibrates on F1), so tests
    that build one need an F1-labelled graph. Node ids and ordering are
    identical to ``cx_graph``'s, so ``cx_types``/``cx_centroids`` fit both.
    """
    connections, _ = _synthetic_cx
    return build_region_graph(connections, region="CX", dataset="f1")


def _synthetic_annotations(graph, type_idx: np.ndarray) -> pd.DataFrame:
    """Per-node annotations for a graph built from ``_synthetic_connections``."""
    # Root ids are 1..N contiguous, so root_id - 1 recovers the same
    # per-node type assignment used to draw the graph's edges.
    pos = graph.node_ids - 1
    node_type_idx = type_idx[pos]
    n = graph.n_nodes
    types = np.asarray(_TYPES, dtype=object)
    return pd.DataFrame({
        "root_id": graph.node_ids,
        "super_class": np.array(["cx_a", "cx_b"], dtype=object)[node_type_idx % 2],
        "class": types[node_type_idx],
        "cell_type": types[node_type_idx],
        "side": np.array(["L", "R", "M"], dtype=object)[np.arange(n) % 3],
        "flow": np.array(["afferent", "efferent", "intrinsic"], dtype=object)[np.arange(n) % 3],
        "nt_type": np.array(
            ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine"],
            dtype=object,
        )[np.arange(n) % 6],
    })


def _synthetic_centroids(n_nodes: int) -> np.ndarray:
    rng = np.random.default_rng(_SEED + 1)
    return rng.normal(size=(n_nodes, 3))


@pytest.fixture(scope="session")
def cx_annotations(cx_graph, _synthetic_cx):
    _, type_idx = _synthetic_cx
    return _synthetic_annotations(cx_graph, type_idx)


@pytest.fixture(scope="session")
def cx_types(cx_graph, cx_annotations):
    return cx_annotations.set_index("root_id")["cell_type"].reindex(cx_graph.node_ids)


@pytest.fixture(scope="session")
def cx_centroids(cx_graph):
    return _synthetic_centroids(cx_graph.n_nodes)


@pytest.fixture(scope="session")
def cx_schema(cx_annotations):
    return FeatureSchema.from_f1(cx_annotations, "CX")


@pytest.fixture(scope="session")
def cx_features(cx_annotations, cx_graph, cx_schema):
    return build_features(cx_annotations, cx_graph, cx_schema, dataset="f2")


@pytest.fixture(scope="session")
def cx_features_f1(cx_annotations, cx_graph_f1, cx_schema):
    """The F1-labelled counterpart of ``cx_features``, for ``fit`` runs."""
    return build_features(cx_annotations, cx_graph_f1, cx_schema, dataset="f1")


# A toy paired region (AL): types present on both sides with several copies per
# side, one single-side type, and one untyped neuron, in F1's side vocabulary.
_PAIRED_TYPES = ("PA1", "PA2", "PA3", "PA4")
_PAIRED_COPIES = 40


def _paired_neurons() -> pd.DataFrame:
    rows = [
        (cell_type, side)
        for cell_type in _PAIRED_TYPES
        for side in ("left", "right")
        for _ in range(_PAIRED_COPIES)
    ]
    rows += [("PS1", "left")] * 3 + [(None, "center")]
    frame = pd.DataFrame(rows, columns=["cell_type", "side"])
    frame.insert(0, "root_id", np.arange(1, len(frame) + 1, dtype=np.int64))
    return frame


def _paired_connections(neurons: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Same-type pairs connect most, same-side pairs next; each edge lands in its source's AL side."""
    rng = np.random.default_rng(seed)
    types = neurons["cell_type"].fillna("").to_numpy()
    sides = neurons["side"].to_numpy()
    same_type = (types[:, None] == types[None, :]) & (types[:, None] != "")
    same_side = sides[:, None] == sides[None, :]
    p = np.where(same_type, 0.25, np.where(same_side, 0.06, 0.02))
    connect = rng.random(p.shape) < p
    np.fill_diagonal(connect, False)
    source, target = np.nonzero(connect)
    root_ids = neurons["root_id"].to_numpy()
    return pd.DataFrame({
        "pre_pt_root_id": root_ids[source],
        "post_pt_root_id": root_ids[target],
        "neuropil": np.where(sides[source] == "right", "AL_R", "AL_L"),
        "syn_count": 9,
    })


@pytest.fixture(scope="session")
def _paired():
    neurons = _paired_neurons()
    return neurons, _paired_connections(neurons, _SEED + 2), _paired_connections(neurons, _SEED + 3)


@pytest.fixture(scope="session")
def paired_graph_f1(_paired):
    _, f1, _ = _paired
    return build_region_graph(f1, region="AL", dataset="f1")


@pytest.fixture(scope="session")
def paired_graph_f2(_paired):
    """A second individual: the same neurons, wired by a different draw."""
    _, _, f2 = _paired
    return build_region_graph(f2, region="AL", dataset="f2")


@pytest.fixture(scope="session")
def paired_annotations(_paired):
    neurons, _, _ = _paired
    n = len(neurons)
    return pd.DataFrame({
        "root_id": neurons["root_id"],
        "super_class": "central",
        "class": "ALPN",
        "cell_type": neurons["cell_type"],
        "side": neurons["side"],
        "flow": np.array(["afferent", "efferent", "intrinsic"], dtype=object)[np.arange(n) % 3],
        "nt_type": np.array(["acetylcholine", "gaba", "glutamate"], dtype=object)[np.arange(n) % 3],
    })


@pytest.fixture
def paired_inputs(tmp_path, paired_graph_f1, paired_annotations):
    """``(processed, suite_path, suites_dir)`` for an AL ``train_m1.py`` run."""
    graph = paired_graph_f1
    types = paired_annotations.set_index("root_id")["cell_type"].reindex(graph.node_ids)
    centroids = _synthetic_centroids(graph.n_nodes)
    schema = FeatureSchema.from_f1(paired_annotations, "AL")
    return write_training_inputs(
        tmp_path, graph, paired_annotations, schema, types, centroids
    )


@pytest.fixture(scope="session")
def sample_for():
    def _sample_for(view, *, spec=DEFAULT_SPEC):
        # Degrees for alpha come from the visible post-mask graph
        # only, never the full graph, so a masked neuron's true degree is
        # never handed to the model as a negative-sampling weight. Passing
        # is_masked keeps a masked endpoint in every decoy, exactly as
        # suite_negatives does.
        return sample_negatives(
            neuropil=view.instance.instance_id,
            positives=view.hidden_edges,
            pool=np.arange(view.graph.n_nodes, dtype=np.int64),
            in_degree=view.visible_in_strength(),
            out_degree=view.visible_out_strength(),
            true_edges=pack_edges(view.graph.edges),
            spec=spec,
            masked_nodes=view.is_masked,
        )

    return _sample_for
