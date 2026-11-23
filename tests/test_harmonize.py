"""The F2 threshold is calibrated, never assumed."""

import pandas as pd
import pytest

from src.harmonize import (
    calibrate_threshold,
    lamina_dependent_types,
    lamina_fractions,
    type_crosswalk,
    type_pair_density,
)
from src.regions import build_region_graph


def _graph(rows, dataset, threshold=5):
    return build_region_graph(
        pd.DataFrame(rows), region="CX", dataset=dataset, threshold=threshold
    )


def test_crosswalk_maps_f2_neurons_through_their_fafb_match_id_then_their_own_name():
    annotations = pd.DataFrame(
        {
            "root_id": [10, 11, 12, 13, 14, 15],
            "fafb_match_id": [1, 2, pd.NA, pd.NA, pd.NA, 3],
            "cell_type": ["PEN_a", "local", "EPG", "auto:EPG", "BANC_only", "PEN_a"],
        }
    )
    crosswalk = type_crosswalk(
        annotations, dataset="f2", f1_types=pd.Series({1: "EPG", 2: "PEN_a", 3: None})
    )
    # A match wins over the neuron's own differing name.
    assert crosswalk[10] == "EPG"
    assert crosswalk[11] == "PEN_a"
    # No match, or a match with no F1 type: the exact own name, if F1 has it.
    assert crosswalk[12] == "EPG"
    assert crosswalk[15] == "PEN_a"
    assert pd.isna(crosswalk[13]) and pd.isna(crosswalk[14])


def test_crosswalk_maps_nullable_match_ids_without_float_coercion():
    ids = [72057594037927937, 72057594037927938]
    f1_types = pd.Series(["EPG", "PEN_a"], index=pd.Index(ids, name="root_id"))
    annotations = pd.DataFrame({
        "root_id": [10, 11, 12],
        "fafb_match_id": pd.array([ids[0], ids[1], pd.NA], dtype="Int64"),
        "cell_type": [pd.NA] * 3,
    })
    crosswalk = type_crosswalk(annotations, dataset="f2", f1_types=f1_types)
    # Adjacent 18-digit ids collapse to one float64; the join must not pass
    # through one.
    assert crosswalk.tolist()[:2] == ["EPG", "PEN_a"]
    assert pd.isna(crosswalk.iloc[2])


def test_pair_density_is_edges_over_the_possible_ordered_pairs():
    graph = _graph(
        {
            "pre_pt_root_id": [1, 1, 2],
            "post_pt_root_id": [3, 4, 3],
            "neuropil": ["FB", "FB", "FB"],
            "syn_count": [9, 9, 9],
        },
        "toy",
    )
    types = pd.Series({1: "A", 2: "A", 3: "B", 4: "B"})
    density = type_pair_density(graph, types).set_index(["source_type", "target_type"])
    # A->B: 3 of the 2 x 2 = 4 possible ordered pairs carry an edge.
    assert density.loc[("A", "B"), "n_edges"] == 3
    assert density.loc[("A", "B"), "density"] == pytest.approx(0.75)


def test_calibration_picks_the_threshold_that_matches_f1_density():
    # F1: two A neurons, two B neurons, all four A->B pairs connected.
    f1_rows = {
        "pre_pt_root_id": [1, 1, 2, 2],
        "post_pt_root_id": [3, 4, 3, 4],
        "neuropil": ["FB"] * 4,
        "syn_count": [20, 20, 20, 20],
    }
    # F2: the same four pairs, but its detector reports far fewer synapses, so
    # only a low threshold recovers the same density.
    f2_rows = pd.DataFrame(
        {
            "pre_pt_root_id": [10, 10, 11, 11],
            "post_pt_root_id": [12, 13, 12, 13],
            "neuropil": ["FB"] * 4,
            "syn_count": [3, 3, 3, 3],
        }
    )
    result = calibrate_threshold(
        _graph(f1_rows, "f1"),
        pd.Series({1: "A", 2: "A", 3: "B", 4: "B"}),
        f2_rows,
        pd.Series({10: "A", 11: "A", 12: "B", 13: "B"}),
        region="CX",
        candidates=range(1, 6),
        min_neurons=2,
    )
    assert result.chosen == 3  # at 4 and above, every F2 edge vanishes
    assert set(result.curve.columns) >= {
        "threshold", "n_edges", "n_pairs", "median_log2_ratio", "spearman"
    }
    assert (result.curve["threshold"] == result.chosen).any()


def test_calibration_leaves_excluded_types_out_of_the_density_comparison():
    f1_rows = {
        "pre_pt_root_id": [1, 1, 2, 2, 5],
        "post_pt_root_id": [3, 4, 3, 4, 3],
        "neuropil": ["FB"] * 5,
        "syn_count": [20] * 5,
    }
    f2_rows = pd.DataFrame(
        {
            "pre_pt_root_id": [10, 10, 11, 11, 14],
            "post_pt_root_id": [12, 13, 12, 13, 12],
            "neuropil": ["FB"] * 5,
            "syn_count": [3, 3, 3, 3, 3],
        }
    )
    f1_types = pd.Series({1: "A", 2: "A", 3: "B", 4: "B", 5: "X"})
    f2_types = pd.Series({10: "A", 11: "A", 12: "B", 13: "B", 14: "X"})
    curves = [
        calibrate_threshold(
            _graph(f1_rows, "f1"), f1_types, f2_rows, f2_types,
            region="CX", candidates=range(1, 4), min_neurons=1, exclude_types=exclude,
        ).curve
        for exclude in (("X",), ())
    ]
    assert curves[0]["n_pairs"].max() < curves[1]["n_pairs"].max()


def test_calibration_refuses_when_no_type_pair_survives():
    rows = {
        "pre_pt_root_id": [1],
        "post_pt_root_id": [2],
        "neuropil": ["FB"],
        "syn_count": [9],
    }
    with pytest.raises(ValueError, match="no cell-type pair"):
        calibrate_threshold(
            _graph(rows, "f1"),
            pd.Series({1: "A", 2: "B"}),
            pd.DataFrame(rows),
            pd.Series({1: "A", 2: "B"}),
            region="CX",
            candidates=range(1, 3),
            min_neurons=99,
        )


def test_banc_neuropil_labels_take_v783_side_suffixes():
    from src.harmonize import map_neuropil_labels

    neuropil = pd.Series(["FB", "AL", "AL", "AL", "ANm"])
    side = pd.Series(["left", "left", "right", None, "right"])
    labels = map_neuropil_labels(neuropil, side)
    assert labels.isna().tolist() == [False, False, False, True, False]
    assert labels.dropna().tolist() == ["FB", "AL_L", "AL_R", "ANm_R"]


def _innervation(rows):
    return pd.DataFrame(rows, columns=["root_id", "neuropil", "count"])


def test_lamina_share_counts_pre_and_post_over_ol_labels_only():
    pre = _innervation([(1, "LA_L", 3), (1, "ME_R", 1), (1, "FB", 100), (2, "LO_L", 4)])
    post = _innervation([(1, "LA_R", 4), (1, "AME_L", 2), (2, "LA_L", 1), (2, "PB", 50)])
    types = pd.Series({1: "L1", 2: "Tm1"})
    share = lamina_fractions(pre, post, types)
    assert share.to_dict() == pytest.approx({"L1": 0.7, "Tm1": 0.2})


def test_lamina_dependent_types_include_the_boundary_and_come_back_sorted():
    pre = _innervation([(1, "LA_L", 1), (1, "ME_L", 3), (2, "LA_L", 1), (2, "ME_L", 1),
                        (3, "LA_R", 1), (3, "LO_R", 9)])
    post = _innervation([(1, "ME_R", 0)])
    types = pd.Series({1: "Z", 2: "A", 3: "M"})
    assert lamina_dependent_types(pre, post, types, min_fraction=0.25) == ("A", "Z")
    assert lamina_dependent_types(pre, post, types, min_fraction=0.5) == ("A",)


def test_lamina_share_ignores_untyped_neurons_and_types_without_ol_synapses():
    pre = _innervation([(1, "LA_L", 1), (2, "LA_L", 50), (3, "FB", 9)])
    post = _innervation([(1, "ME_L", 1)])
    types = pd.Series({1: "L1", 2: None, 3: "EPG"})
    assert lamina_fractions(pre, post, types).to_dict() == {"L1": 0.5}


def test_lamina_share_keeps_adjacent_18_digit_root_ids_apart():
    ids = [72057594037927937, 72057594037927938]
    pre = _innervation([(ids[0], "LA_L", 1), (ids[1], "ME_L", 1)])
    post = _innervation([(ids[1], "ME_L", 0)])
    types = pd.Series(["L1", "Tm1"], index=pd.Index(ids, name="root_id"))
    assert lamina_fractions(pre, post, types).to_dict() == {"L1": 1.0, "Tm1": 0.0}


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_lamina_dependent_types_reject_a_min_fraction_outside_zero_one(bad):
    empty = _innervation([])
    with pytest.raises(ValueError, match="min_fraction"):
        lamina_dependent_types(empty, empty, pd.Series(dtype=object), min_fraction=bad)
