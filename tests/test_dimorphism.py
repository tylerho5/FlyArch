import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.dimorphism import UNTYPED, categories, category_table, per_type_frame, per_type_mrr
from src.gate import PreregConfig


def test_per_type_rows_reproduce_the_gated_instance_mrr(cx_graph, cx_features, cx_schema, cx_types):
    from src.evaluate import CommittedNegatives
    from src.masks import apply_mask, random_mask
    from src.negatives import DEFAULT_SPEC
    from src.suites import Suite, SuiteInstance, view_negatives
    from train.model import M1
    from train.scoring import score_expert

    instance = random_mask(cx_graph, fraction=0.10, seed=71)
    suite = Suite(
        name="f2_dev", dataset="f2", region=cx_graph.region, graph_digest=cx_graph.digest(),
        threshold=cx_graph.threshold, master_seed=DEFAULT_SPEC.master_seed,
        instances=(SuiteInstance(instance.instance_id, instance.regime, instance.seed,
                                 dict(instance.params), cx_graph.node_ids[instance.masked]),),
    )
    sample = view_negatives(apply_mask(cx_graph, instance))
    negatives = CommittedNegatives("f2_dev", DEFAULT_SPEC, {instance.instance_id: sample.digest()})
    model = M1(cx_schema, hidden=8)
    model.set_trained_types(np.arange(1, len(cx_schema.type_vocab) // 2))
    rows, positives = score_expert(
        model, cx_graph, cx_features, suite, types=cx_types, negatives=negatives, rung="c"
    )
    assert positives["both_masked"].any() and positives["type_trained"].notna().all()

    # One masked endpoint loses its type, so its positives fall in the untyped bucket.
    types = cx_types.copy()
    types[cx_graph.node_ids[instance.masked[0]]] = np.nan
    frame = per_type_frame(positives, types)

    assert UNTYPED in set(frame["type"])
    assert frame["n_positives"].sum() == (~positives["both_masked"]).sum()
    weighted = (frame["mrr"] * frame["n_positives"]).sum() / frame["n_positives"].sum()
    assert weighted == pytest.approx(rows.set_index("mask_instance").at[instance.instance_id, "mrr"])


def test_category_rows_hold_type_trained_fixed():
    # Two seeds, R and T instances. "Lx" is matched but untrained, so it must
    # land in the untrained rows beside the male-specific type, never the trained ones.
    rows = []
    for seed in (0, 1):
        for type_, trained, rr in (("Iso1", True, .5), ("Iso2", True, .3), ("Dim1", True, .2),
                                   ("Dim2", True, .1), ("Male1", False, .05), ("Lx", False, .4)):
            regimes = ("R05",) if type_ == "Male1" else ("R05", "T")
            for tag in regimes:
                rows.append({"seed": seed, "mask_instance": f"f3-snp-{tag}-s{seed:03d}",
                             "root_id": type_, "reciprocal_rank": rr, "both_masked": False,
                             "type_trained": trained, "type_matched": type_ != "Male1"})
    positives = pd.DataFrame(rows)
    types = pd.Series({t: t for t in ("Iso1", "Iso2", "Dim1", "Dim2", "Male1", "Lx")})
    per_type = per_type_mrr(per_type_frame(positives, types))
    labels = categories(pd.Series(["Iso1", "Iso2", "Dim1", "Dim2", "Male1", "Lx"]),
                        pd.Series([None, None, "sexually dimorphic", "sexually dimorphic",
                                   "male-specific", None]))

    table = category_table(per_type, labels, PreregConfig()).set_index(
        ["group", "category", "type_trained"]
    )

    assert set(table.index) == {
        ("R/S", "isomorphic", "True"), ("R/S", "isomorphic", "False"),
        ("R/S", "sexually dimorphic", "True"), ("R/S", "male-specific", "False"),
        ("T", "isomorphic", "True"), ("T", "isomorphic", "False"), ("T", "sexually dimorphic", "True"),
    }
    assert table.at[("R/S", "isomorphic", "True"), "n_types"] == 2
    assert table.at[("R/S", "isomorphic", "False"), "n_types"] == 1  # Lx, matched but untrained
    assert table.at[("R/S", "male-specific", "False"), "n_positives"] == 2
    assert table.at[("R/S", "sexually dimorphic", "True"), "mrr"] == pytest.approx(.15)


def test_the_untyped_bucket_never_counts_as_an_isomorphic_type():
    from scripts.dimorphism_report import _category_section

    per_type = pd.DataFrame(
        [["SNP", "R/S", UNTYPED, False, 100, .9], ["SNP", "R/S", "Iso", False, 1, .1]],
        columns=["region", "group", "type", "type_trained", "n_positives", "mrr"],
    )
    rows = _category_section(per_type, pd.Series({"Iso": "isomorphic"}), PreregConfig())

    assert "| R/S | isomorphic | False | 1 | 1 | 0.1000 | n/a (one type) |" in rows
    assert _category_section(per_type.iloc[:1], pd.Series(dtype=object), PreregConfig()) == [
        "No typed masked neuron was scored, so there is no category table."
    ]


def test_report_reads_ranks_through_the_verified_loader(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import dimorphism_report
    from train.region_config import config_path, load_region_config

    seen = {}

    def fake(config, **kwargs):
        seen.update(kwargs, region=config.region)
        ranks = pd.DataFrame({"split": ["f3_eval", "f2_dev"], "reciprocal_rank": [.5, .4]})
        return kwargs["cohort"], [], pd.DataFrame(), ranks

    monkeypatch.setattr(dimorphism_report, "load_verified_results", fake)
    registry = pd.DataFrame({"region": ["SNP", "SNP", "SNP", "AL"], "gate_seed": ["true", "false", "true", "true"],
                             "cohort": ["c1", "c1", "c1", "c9"]})
    args = SimpleNamespace(results_root=tmp_path, runs_dir=tmp_path / "runs", data_root=tmp_path)

    ranks = dimorphism_report.expert_positives(registry, "SNP", args)

    assert (seen["region"], seen["cohort"], seen["models"]) == ("SNP", "c1", ("gnn_a4",))
    assert seen["splits"] == ("f3_eval", "f2_dev")
    assert seen["seeds"] == load_region_config(config_path("SNP")).seeds
    assert ranks["f3"]["reciprocal_rank"].tolist() == [.5] and ranks["f2"]["reciprocal_rank"].tolist() == [.4]
    registry.loc[2, "cohort"] = "c2"
    with pytest.raises(ValueError, match="one cohort"):
        dimorphism_report.expert_positives(registry, "SNP", args)
