"""Side holdouts and sibling visibility at the suite/evaluation boundary."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src import masks
from src.evaluate import (
    STRATA_AXES,
    CommittedNegatives,
    build_scoring_task,
    constant_scorer,
    results_frame,
    score_instance,
    strata_frame,
)
from src.regions import RegionGraph
from src.suites import Suite, SuiteInstance, build_suite, suite_negatives


def test_side_suite_round_trip_and_scoring(paired_graph_f2, paired_annotations, tmp_path):
    from scripts.prepare_suite_inputs import build_sides

    graph = paired_graph_f2
    sides = build_sides(paired_annotations, graph).set_index("root_id")["side"]
    suite = build_suite(graph, name="f2_side", soma_sides=sides)
    assert [i.instance_id for i in suite.instances] == ["f2-al-HL-s4000", "f2-al-HR-s4001"]
    path = tmp_path / "f2_side.json"
    suite.write(path)
    restored = Suite.read(path)
    assert restored.to_json() == suite.to_json()
    samples = suite_negatives(graph, restored)
    committed = CommittedNegatives(
        name="f2_side", spec=next(iter(samples.values())).spec,
        digests={iid: sample.digest() for iid, sample in samples.items()},
    )
    results = []
    types = paired_annotations.set_index("root_id")["cell_type"]
    for side, mask in zip(("L", "R"), restored.resolve(graph)):
        assert mask.regime == "H"
        np.testing.assert_array_equal(graph.node_ids[mask.masked], sides.index[sides == side])
        assert not np.isin(sides.index[sides.isna()], graph.node_ids[mask.masked]).any()
        view = masks.apply_mask(graph, mask)
        assert not view.is_masked[view.visible_edges].any()
        task = build_scoring_task(
            view, samples[mask.instance_id], suite=restored, negatives=committed,
            types=types, soma_sides=sides, trained_types={"PA1", "PA2", "PA3", "PS1"},
        )
        results.append(score_instance(task, constant_scorer(0.5)))
    report = strata_frame(results, model="gnn_a4", seed=0, regime="c")
    assert {"contra", "untrained"} <= set(report.loc[report.axis == "siblings", "stratum"])
    assert set(report.axis) == set(STRATA_AXES)
    assert set(results_frame(results, model="gnn_a4", seed=0, regime="c").split) == {"f2_side"}


def test_categorical_sides_survive_producer_mask_and_scoring(paired_graph_f2, paired_annotations):
    from scripts.prepare_suite_inputs import build_sides

    annotations = paired_annotations.copy()
    annotations["side"] = pd.Categorical(annotations.side.where(annotations.side.isin(("left", "right"))))
    sides = build_sides(annotations, paired_graph_f2).set_index("root_id")["side"]
    assert isinstance(sides.dtype, pd.CategoricalDtype)
    suite = build_suite(paired_graph_f2, name="f2_side", soma_sides=sides)
    samples = suite_negatives(paired_graph_f2, suite)
    committed = CommittedNegatives("f2_side", next(iter(samples.values())).spec,
        {iid: sample.digest() for iid, sample in samples.items()})
    mask = suite.resolve(paired_graph_f2)[0]
    task = build_scoring_task(
        masks.apply_mask(paired_graph_f2, mask), samples[mask.instance_id], suite=suite,
        negatives=committed, types=annotations.set_index("root_id")["cell_type"], soma_sides=sides,
    )
    result = score_instance(task, constant_scorer(0.5))
    assert "contra" in set(result.strata.loc[result.strata.axis == "siblings", "stratum"])


def test_side_mask_validates_inputs_and_preserves_training_regimes(paired_graph_f2):
    graph = paired_graph_f2
    sides = pd.Series("L", index=graph.node_ids)
    assert masks.MASK_REGIMES == ("R", "T", "S")
    assert masks.SUITE_REGIMES == ("R", "T", "S", "H")
    with pytest.raises(ValueError, match="side"):
        masks.side_mask(graph, sides, side="left", seed=4000)
    with pytest.raises(ValueError, match="no.*R"):
        masks.side_mask(graph, sides, side="R", seed=4001)
    with pytest.raises(ValueError, match="duplicate"):
        masks.side_mask(graph, pd.concat([sides, sides.iloc[:1]]), side="L", seed=4000)
    with pytest.raises(ValueError, match="missing"):
        masks.side_mask(graph, sides.iloc[1:], side="L", seed=4000)
    with pytest.raises(ValueError, match="L.*R"):
        masks.side_mask(graph, sides.replace("L", "left"), side="L", seed=4000)
    with pytest.raises(ValueError, match="soma_sides"):
        build_suite(graph, name="f2_side")
    with pytest.raises(ValueError, match="registered to dataset"):
        build_suite(replace(graph, dataset="f1"), name="f2_side", soma_sides=sides)
    mask = masks.side_mask(graph, sides, side="L", seed=4000)
    assert masks.MaskInstance.from_dict(mask.to_dict()).digest() == mask.digest()


@pytest.mark.parametrize("name,dataset", [("f1_val", "f1"), ("f2_dev", "f2"), ("f2_confirm", "f2")])
def test_h_artifacts_cannot_be_relabelled_as_selection_or_gate_suites(name, dataset, paired_graph_f2, tmp_path):
    graph = replace(paired_graph_f2, dataset=dataset)
    sides = pd.Series(np.where(np.arange(graph.n_nodes) % 2, "R", "L"), index=graph.node_ids)
    mask = masks.side_mask(graph, sides, side="L", seed=4000)
    suite = Suite(
        name, dataset, graph.region, graph.digest(), graph.threshold, 20260818,
        (SuiteInstance(mask.instance_id, "H", 4000, dict(mask.params), graph.node_ids[mask.masked]),),
    )
    path = tmp_path / f"{name}.json"
    suite.write(path)
    with pytest.raises(ValueError, match="H.*f2_side"):
        Suite.read(path).resolve(graph)


@pytest.mark.parametrize("keep", [slice(0, 1), slice(0, 0), [0, 0], [1, 0]])
def test_side_suite_rejects_incomplete_or_reordered_artifacts(keep, paired_graph_f2, paired_annotations):
    from scripts.prepare_suite_inputs import build_sides

    sides = build_sides(paired_annotations, paired_graph_f2).set_index("root_id")["side"]
    built = build_suite(paired_graph_f2, name="f2_side", soma_sides=sides)
    instances = built.instances[keep] if isinstance(keep, slice) else tuple(built.instances[i] for i in keep)
    with pytest.raises(ValueError, match="only H"):
        replace(built, instances=instances).resolve(paired_graph_f2)


@pytest.mark.parametrize("regime", ["R", "T", "S"])
def test_side_suite_rejects_non_h_artifacts(regime, paired_graph_f2):
    graph = paired_graph_f2
    suite = Suite(
        "f2_side", "f2", graph.region, graph.digest(), graph.threshold, 20260818,
        (SuiteInstance("f2-al-HL-s4000", regime, 4000, {}, graph.node_ids[:1]),),
    )
    with pytest.raises(ValueError, match="only H"):
        suite.resolve(graph)


def _sibling_task(sample_for, *, soma_sides=True, trained_types=None, extra_hidden=False):
    edges = [(i, 140) for i in range(8)] + [(141, i) for i in range(8)]
    edges += [(i, 140) for i in range(10, 16)] + [(0, 1)]
    edges += [(i, i + 1) for i in range(20, 139)] + [(139, 20)]
    if extra_hidden:
        edges += [(2, 17), (17, 2)]
    graph = RegionGraph(
        dataset="f2", region="AL", threshold=5,
        node_ids=np.arange(1000, 1150, dtype=np.int64),
        edges=np.array(edges, dtype=np.int32), weights=np.full(len(edges), 9, dtype=np.int64),
    )
    mask = masks.MaskInstance(
        "f2-al-R06-s001", "f2", "AL", "R", 1, {}, np.arange(9, dtype=np.int32),
    )
    types = pd.Series(None, index=graph.node_ids, dtype=object)
    sides = pd.Series(None, index=graph.node_ids, dtype=object)
    names = ["contra_type", "ipsi_type", "hidden_type", "untrained_type", "singleton", "sideless", None, "unknown"]
    for i, name in enumerate(names):
        types.iloc[i] = name
        sides.iloc[i] = "L" if i != 5 else None
    for i, name, side in [
        (8, "hidden_type", "R"), (10, "contra_type", "R"), (11, "contra_type", "L"),
        (12, "ipsi_type", "L"), (13, "untrained_type", "R"),
        (14, "sideless", "R"), (15, "unknown", None), (17, "hidden_type", "R"),
    ]:
        types.iloc[i], sides.iloc[i] = name, side
    view = masks.apply_mask(graph, mask)
    suite = Suite(
        name="f2_dev", dataset="f2", region="AL", graph_digest=graph.digest(),
        threshold=5, master_seed=20260818,
        instances=(SuiteInstance(mask.instance_id, "R", 1, {}, graph.node_ids[mask.masked]),),
    )
    sample = sample_for(view)
    committed = CommittedNegatives("f2_dev", sample.spec, {mask.instance_id: sample.digest()})
    return build_scoring_task(
        view, sample, suite=suite, negatives=committed, types=types,
        soma_sides=sides if soma_sides else None, trained_types=trained_types,
    )


def test_sibling_strata_cover_visibility_precedence_and_trained_types(sample_for):
    trained = {"contra_type", "ipsi_type", "hidden_type", "singleton", "sideless", "unknown"}
    task = _sibling_task(sample_for, trained_types=trained)
    anchors = np.where(task.strata.direction == "out", task.positives[:, 0], task.positives[:, 1])
    expected = {0: "contra", 1: "ipsi", 2: "hidden", 3: "untrained", 4: "none", 5: "unsided", 6: "none", 7: "unsided"}
    single = ~task.strata.both_masked
    assert task.strata.loc[single, "siblings"].tolist() == [expected[i] for i in anchors[single]]
    assert task.strata.loc[single & (anchors == 3), "type_matched"].all()
    assert not task.strata.loc[single & (anchors == 3), "type_trained"].any()
    result = score_instance(task, constant_scorer(0.5))
    siblings = result.strata.loc[result.strata.axis == "siblings"]
    assert set(siblings.stratum) == {"contra", "ipsi", "hidden", "unsided", "untrained", "none"}
    assert siblings.n_positives.sum() == single.sum()


def test_sibling_visibility_ignores_edges_incident_to_masked_nodes(sample_for):
    plain = _sibling_task(sample_for)
    altered = _sibling_task(sample_for, extra_hidden=True)
    for task in (plain, altered):
        anchors = np.where(task.strata.direction == "out", task.positives[:, 0], task.positives[:, 1])
        assert set(task.strata.loc[(anchors == 2) & ~task.strata.both_masked, "siblings"]) == {"hidden"}


def test_old_callers_report_none_and_unknown_training_status(sample_for):
    task = _sibling_task(sample_for, soma_sides=False)
    assert set(task.strata.siblings) == {"none"}
    assert task.strata.type_trained.isna().all()
    assert "untrained" not in set(_sibling_task(sample_for).strata.siblings)


def test_untrained_needs_no_soma_sides(sample_for):
    trained = {"contra_type", "ipsi_type", "hidden_type", "singleton", "sideless", "unknown"}
    task = _sibling_task(sample_for, soma_sides=False, trained_types=trained)
    anchors = np.where(task.strata.direction == "out", task.positives[:, 0], task.positives[:, 1])
    single = ~task.strata.both_masked
    assert set(task.strata.loc[single & (anchors == 3), "siblings"]) == {"untrained"}
    assert set(task.strata.loc[single & (anchors != 3), "siblings"]) == {"none"}


def test_soma_sides_are_not_read_without_types(paired_graph_f2, paired_annotations):
    from scripts.prepare_suite_inputs import build_sides

    graph = paired_graph_f2
    sides = build_sides(paired_annotations, graph).set_index("root_id")["side"]
    suite = build_suite(graph, name="f2_side", soma_sides=sides)
    samples = suite_negatives(graph, suite)
    committed = CommittedNegatives("f2_side", next(iter(samples.values())).spec,
        {iid: sample.digest() for iid, sample in samples.items()})
    mask = suite.resolve(graph)[0]
    task = build_scoring_task(
        masks.apply_mask(graph, mask), samples[mask.instance_id], suite=suite,
        negatives=committed, soma_sides=sides.iloc[1:],
    )
    assert set(task.strata.siblings) == {"none"}


def test_side_suite_cli_requires_only_sides(paired_graph_f2, paired_annotations, tmp_path, capsys):
    from scripts.build_suites import main
    from scripts.prepare_suite_inputs import build_sides

    graph = paired_graph_f2
    graph.write(tmp_path / "graph")
    sides_path = tmp_path / "sides.parquet"
    build_sides(paired_annotations, graph).iloc[::-1].to_parquet(sides_path, index=False)
    args = ["--graph", str(tmp_path / "graph"), "--name", "f2_side", "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        main(args)
    assert "--sides" in capsys.readouterr().err
    assert main([*args, "--sides", str(sides_path)]) == 0
    suite = Suite.read(tmp_path / "out" / "f2_side.json")
    committed = CommittedNegatives.read(tmp_path / "out" / "f2_side.negatives.json")
    assert committed.digests == {iid: s.digest() for iid, s in suite_negatives(graph, suite).items()}
    assert "Suite digests" not in capsys.readouterr().out


def test_h_results_cannot_change_gate_or_baseline_selection():
    from test_gate import build_results

    from src.gate import PreregConfig, evaluate_gate, select_champion_baseline

    results = build_results({f"f2-al-R05-s{i:04d}": 0.8 + i / 100 for i in range(20)}, region="al", mask_regime="c")
    prereg = PreregConfig(region="al", delta=0.1)
    side = results.loc[results.dataset == "f2"].copy()
    side["split"] = "f2_side"
    side["mask_instance"] = "f2-al-HL-s4000"
    side["mrr"] = np.where(side.model == "configuration", 1.0, 0.001)
    combined = pd.concat([results, side], ignore_index=True)
    assert select_champion_baseline(combined, prereg) == select_champion_baseline(results, prereg)
    assert evaluate_gate(combined, prereg) == evaluate_gate(results, prereg)


@pytest.mark.parametrize("name,digest", [
    ("f1_val", "29b730f95ca7f0641b84e9013dd9f3528fbdaaf1323e5dc74b1dee46b7865836"),
    ("f2_dev", "6ec905f7065290b2e8dd14ca0be980fd968dd8c8cd430ea98e335a2e223be62a"),
    ("f2_confirm", "0c0c07bc86bada0e2887a2231b41933141105c5dd611329ddb680efb4625857c"),
])
def test_existing_cx_suites_keep_their_pre_side_strata_bytes(name, digest, cx_graph_f1, cx_types, cx_centroids):
    graph = replace(cx_graph_f1, dataset="f1" if name == "f1_val" else "f2")
    # Suite code from e40acdd, before H was registered, on the 300-node fixture.
    assert build_suite(graph, name=name, types=cx_types, centroids=cx_centroids).digest() == digest


def test_committed_cx_suite_digest_remains_valid():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data/suites/f1/cx/f1_val.json"
    assert Suite.read(path).digest() == "84d9c330afe25ad2bd4a76b8816f3aee5e551155188de608b6dd1fa5b5236d61"
