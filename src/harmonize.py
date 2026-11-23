"""Harmonize F1/F2/F3 cell types and calibrate the F2 edge threshold.

Two rules govern this module:

* **Synapse counts are not comparable across datasets** (rule 4). BANC's
  detector has different recall from FAFB's -- its Kenyon-cell mean
  out-connection count is 139 against FAFB's 213 (the paper's own example) --
  so the ``>=5`` threshold that reproduces F1's published connection count is
  not reused for F2. Instead :func:`calibrate_threshold` sweeps candidate F2
  thresholds and picks the one whose matched cell-type pair densities are
  closest in level to F1's, because a threshold is a level knob: it can push
  the overall density up or down, but it cannot fix a *ranking* that has
  degraded. Spearman rho is carried alongside as the diagnostic that the
  ordering survived, not as the primary target.
* **The type vocabulary comes from F1** (rule 2). F2 neurons join it through
  their FAFB match id, or else their own name where F1 has it; the rest (about
  40% of OL, under 9% of each other region on v888, per rule 2) get ``NaN``
  rather than a bucket, because
  :func:`type_pair_density` drops
  untyped neurons rather than lumping them into one enormous out-of-vocabulary
  pseudo-type that would swamp the fit.

The F2 tables below put BANC's spellings into F1's terms (harmonization.md
rules 3 and 6): its column names, annotation values, neuropil labels and axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from src.regions import UNPAIRED_NEUROPILS, RegionGraph, build_region_graph, region_labels

__all__ = [
    "F2_AXIS_MAP",
    "F2_COLUMNS",
    "F2_VALUE_MAPS",
    "F3_AXIS_MAP",
    "F3_COLUMNS",
    "F3_MIDLINE_X_NM",
    "F3_VALUE_MAPS",
    "ThresholdCalibration",
    "banc_neuron_ids",
    "calibrate_threshold",
    "lamina_dependent_types",
    "lamina_fractions",
    "f3_hemisphere",
    "f3_neuron_ids",
    "map_f2_values",
    "map_neuropil_labels",
    "normalize_f3_annotations",
    "type_crosswalk",
    "type_pair_density",
]

#: BANC meta columns no ``src.features`` alias resolves, under their canonical names.
F2_COLUMNS = {"fafb_match": "fafb_match_id", "neurotransmitter_predicted": "nt_type"}

#: BANC value -> F1 value per canonical annotation field. Each BANC value takes the
#: F1 value most common among its proofread-tier neurons whose FAFB match has an
#: F1 row, given at least five of them; shares under 80% are noted. A value left
#: out has no F1 counterpart (its matches' F1 value is mostly null) and encodes
#: as OOV, or as zeros for side and flow, which have no OOV column.
F2_VALUE_MAPS = {
    "super_class": {
        "central_brain_intrinsic": "central",
        "optic_lobe_intrinsic": "optic",
        "visual_projection": "visual_projection",
        "sensory": "sensory",
        "ascending": "ascending",
        "descending": "descending",
        "visual_centrifugal": "visual_centrifugal",
        "sensory_ascending": "sensory_ascending",
        "motor": "motor",
        "visceral_circulatory": "endocrine",
        "sensory_descending": "sensory",
    },
    "class": {
        "transverse_neuron": "LO>LOP",  # 48%, ME>LOP 42%
        "kenyon_cell": "Kenyon_Cell",
        "olfactory_receptor_neuron": "olfactory",
        "medulla_intrinsic": "ME",
        "ascending_neuron": "AN",
        "bristle_neuron": "mechanosensory",
        "central_complex_intrinsic_neuron": "CX",
        "chordotonal_organ_neuron": "mechanosensory",
        "distal_medulla": "ME",
        "lobula_intrinsic": "LO",
        "transmedullary": "ME>LO",
        "central_complex_input_neuron": "CX",
        "central_complex_output_neuron": "CX",
        "antennal_lobe_projection_neuron": "ALPN",
        "transmedullary_y": "ME>LO.LOP",  # 50%, null 42%
        "lateral_horn_local_neuron": "LHLN",
        "y_neuron": "LOP>ME.LO",  # 78%
        "serpentine_medulla": "ME",
        "antennal_lobe_local_neuron": "ALLN",
        "proximal_medulla": "ME",
        "lamina_monopolar": "LA>ME",
        "campaniform_sensillum_neuron": "AN",
        "taste_bristle_gustatory_neuron": "gustatory",
        "optic_lobe_intrinsic_centrifugal": "ME>LA",
        "mushroom_body_dopaminergic_neuron": "DAN",
        "lobula_plate_intrinsic": "LOP",
        "medulla_lobula_tangential": "ME>LO",
        "translobula_plate": "LOP>LO",
        "tubercular_bulbar_neuron": "TuBu",
        "mushroom_body_output_neuron": "MBON",
        "lobula_medulla_tangential": "LO>ME",
        "hygrosensory_receptor_neuron": "hygrosensory",
        "subesophageal_zone_projection_neuron": "mAL",
        "internal_taste_sensillum_gustatory_neuron": "gustatory",  # 53%
        "taste_peg_tactile_neuron": "mechanosensory",
        "medulla_medulla": "bilateral",
        "lobula_medulla_amacrine": "LO>ME",  # 33%
        "internal_sensillum_tactile_neuron": "mechanosensory",
        "photoreceptor_neuron": "visual",
        "taste_peg_gustatory_neuron": "gustatory",
        "lateral_horn_centrifugal_neuron": "LHCENT",
        "lobula_lobula_plate_tangential": "LO>ME",
        "multidendritic_neuron": "unknown_sensory",  # 76%
        "pars_intercerebralis_neurosecretory_cell": "pars_intercerebralis",
        "proboscis_motor_neuron": "brain_motor_neuron",
        "orphan_neuron": "mechanosensory",  # 52%
        "thermosensory_receptor_neuron": "thermosensory",
        "pars_lateralis_neurosecretory_cell": "pars_lateralis",
        "neck_motor_neuron": "brain_motor_neuron",
        "antennal_lobe_centrifugal_neuron": "ALIN",
        "ocellar_centrifugal_neuron": "ocellar",
        "ocellar_projection_neuron": "ocellar",
        "pharynx_motor_neuron": "brain_motor_neuron",
        "ocellar_intrinsic_neuron": "ocellar",
        "antennal_lobe_output_neuron": "ALON",
        "oxygenation_neuron": "AN",
        "antenna_motor_neuron": "brain_motor_neuron",
        "lamina_wide_field": "ME>LA",
        "distal_medulla_dorsal_rim_area": "ME",
        "crop_motor_neuron": "brain_motor_neuron",
    },
    "side": {"left": "left", "right": "right"},
    "flow": {"intrinsic": "intrinsic", "afferent": "afferent", "efferent": "efferent"},
}

#: F1 axis <- (BANC axis, sign), in F1's x, y, z order: F1 (x, y, z) = (-X, -Z, +Y).
F2_AXIS_MAP = (("x", -1), ("z", -1), ("y", 1))

#: Male CNS release field -> canonical column.
F3_COLUMNS = {
    "bodyId": "root_id",
    "flywireType": "cell_type",
    "type": "f3_type",
    "superclass": "super_class",
    "class": "class",
    "somaSide": "side",
    "dimorphism": "dimorphism",
    "predicted_nt": "nt_type",
}

#: Release value -> F1 value per canonical field; a value left out has no F1 counterpart.
F3_VALUE_MAPS = {
    "super_class": {
        "ol_intrinsic": "optic",
        "cb_intrinsic": "central",
        "visual_projection": "visual_projection",
        "cb_sensory": "sensory",
        "ol_sensory": "sensory",
        "descending_neuron": "descending",
        "ascending_neuron": "ascending",
        "visual_centrifugal": "visual_centrifugal",
        "cb_motor": "motor",
        "cb_endocrine": "endocrine",
        "sensory_ascending": "sensory_ascending",
        "sensory_descending": "sensory",
    },
    "class": {
        "Kenyon_Cell": "Kenyon_Cell",
        "visual": "visual",
        "olfactory": "olfactory",
        "CX": "CX",
        "mechanosensory": "mechanosensory",
        "ALPN": "ALPN",
        "ALLN": "ALLN",
        "DAN": "DAN",
        "gustatory": "gustatory",
        "ol_bilateral": "bilateral",
        "MBON": "MBON",
        "unknown_sensory": "unknown_sensory",  # 99%
        "hygrosensory": "hygrosensory",
        "thermosensory": "thermosensory",
        "ALIN": "ALIN",
        "ALON": "ALON",
    },
    "side": {"L": "left", "R": "right", "M": "center"},
    "flow": {
        "ol_intrinsic": "intrinsic",
        "cb_intrinsic": "intrinsic",
        "visual_projection": "intrinsic",
        "cb_sensory": "afferent",
        "ol_sensory": "afferent",
        "descending_neuron": "efferent",
        "ascending_neuron": "afferent",
        "visual_centrifugal": "intrinsic",
        "cb_motor": "efferent",
        "cb_endocrine": "efferent",
        "sensory_ascending": "afferent",
        "sensory_descending": "afferent",
    },
}

#: F1 axis <- (male CNS axis, sign), in F1's x, y, z order: F1 (x, y, z) = (-X, +Y, +Z).
F3_AXIS_MAP = (("x", -1), ("y", 1), ("z", 1))

#: Male CNS native x of the midline, nm; the fly's left is above it.
F3_MIDLINE_X_NM = 386416.0

_F3_POPULATION_STATUS = ("Traced", "Anchor")

_BANC_NON_NEURONS = frozenset({"glia", "trachea", "not_a_neuron"})
_SIDE_SUFFIXES = {"left": "_L", "right": "_R"}


def banc_neuron_ids(meta: pd.DataFrame) -> np.ndarray:
    """The proofread-tier BANC neurons F1's proofread table corresponds to.

    Rows with ``proofread`` or ``roughly_proofread`` equal to ``"TRUE"``, minus
    glia, trachea and ``not_a_neuron`` by ``super_class`` or a ``status`` tag
    (155,848 on v888). ``meta`` is keyed on
    ``banc_888_id`` as ``root_id``, as ``build_features.read_table`` returns it.
    """
    tier = (meta["proofread"] == "TRUE") | (meta["roughly_proofread"] == "TRUE")
    tags = meta["status"].fillna("").astype(str).str.lower().str.split(",")
    tagged = tags.map(lambda values: bool(set(values) & _BANC_NON_NEURONS))
    neurons = tier & ~meta["super_class"].isin(_BANC_NON_NEURONS) & ~tagged
    return np.unique(meta.loc[neurons, "root_id"].to_numpy(dtype=np.int64))


def map_neuropil_labels(neuropil: pd.Series, side: pd.Series) -> pd.Series:
    """BANC base neuropil name and synapse side -> the v783 label ``src.regions`` filters on.

    Unpaired bases keep their name and drop the side; paired ones take ``_L``
    for ``left`` and ``_R`` for ``right``. A paired base with any other side
    maps to null, so no region claims the row.
    """
    pairs = pd.DataFrame({"neuropil": neuropil, "side": side}, index=neuropil.index)
    combos = pairs.drop_duplicates().copy()
    combos["label"] = [
        base if base in UNPAIRED_NEUROPILS
        else f"{base}{_SIDE_SUFFIXES[where]}" if where in _SIDE_SUFFIXES
        else None
        for base, where in zip(combos["neuropil"], combos["side"])
    ]
    labels = pairs.merge(combos, how="left", on=["neuropil", "side"])["label"]
    return labels.set_axis(neuropil.index)


def map_f2_values(annotations: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Canonical BANC annotation columns rewritten in F1's values, and what had none.

    Returns the mapped frame and, per field, the non-null BANC values with no
    F1 value and their neuron counts. Those become null, the encoding F1's own
    unannotated neurons get.
    """
    mapped = annotations.copy()
    unmapped = {}
    for field, table in F2_VALUE_MAPS.items():
        values = annotations[field]
        out = values.map(table)
        missed = values[values.notna() & out.isna()]
        if len(missed):
            unmapped[field] = {str(k): int(v) for k, v in missed.astype(str).value_counts().items()}
        mapped[field] = out
    return mapped, unmapped


def f3_hemisphere(x_nm: np.ndarray) -> np.ndarray:
    """Male CNS native x (nm) -> ``"L"``/``"R"``, the fly's side as the release's ROI names give it."""
    return np.where(np.asarray(x_nm, dtype=float) > F3_MIDLINE_X_NM, "L", "R")


def _f3_population(annotations: pd.DataFrame) -> pd.DataFrame:
    keep = annotations["status"].isin(_F3_POPULATION_STATUS) & annotations["superclass"].notna()
    return annotations.loc[keep]


def f3_neuron_ids(annotations: pd.DataFrame) -> np.ndarray:
    """Sorted ``bodyId`` of the male CNS neurons kept: status Traced or Anchor, with a superclass."""
    return np.unique(_f3_population(annotations)["bodyId"].to_numpy(dtype=np.int64))


def normalize_f3_annotations(
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Raw F3 table (release annotations plus ``predicted_nt``) -> one row per population neuron in F1's values, and what had none.

    Columns: root_id, cell_type (``flywireType``, else ``type``; not crosswalked),
    f3_type, side, super_class, class, flow, nt_type, dimorphism. Returns the frame
    and, per field, the non-null release values with no F1 value and their neuron
    counts. Those become null.
    """
    pop = _f3_population(annotations).sort_values("bodyId")
    side = pop["somaSide"].where(pop["somaSide"].notna(), pop["rootSide"])
    nt = pop["predicted_nt"].where(pop["predicted_nt"] != "unclear")
    keys = {"super_class": pop["superclass"], "class": pop["class"], "flow": pop["superclass"], "side": side}
    out = pd.DataFrame({
        "root_id": pop["bodyId"].to_numpy(dtype=np.int64),
        "cell_type": pop["flywireType"].where(pop["flywireType"].notna(), pop["type"]).to_numpy(),
        "f3_type": pop["type"].to_numpy(),
        **{field: values.map(F3_VALUE_MAPS[field]).to_numpy() for field, values in keys.items()},
        "nt_type": nt.to_numpy(),
        "dimorphism": pop["dimorphism"].to_numpy(),
    })[["root_id", "cell_type", "f3_type", "side", "super_class", "class", "flow", "nt_type", "dimorphism"]]
    unmapped = {}
    for field, values in keys.items():
        missed = values[values.notna() & ~values.isin(F3_VALUE_MAPS[field])]
        if len(missed):
            unmapped[field] = {str(k): int(v) for k, v in missed.astype(str).value_counts().items()}
    return out, unmapped


@dataclass(frozen=True)
class ThresholdCalibration:
    """The chosen F2 threshold, the curve behind it, and what it was fit on."""

    chosen: int
    curve: pd.DataFrame
    concordance: float  # spearman at the chosen threshold
    n_pairs: int  # cell-type pairs the fit used
    frozen_at: str  # ISO date (YYYY-MM-DD); the freeze is a fact, not a vibe


def _root_id_keys(values: pd.Series | pd.Index) -> pd.Index:
    """Exact decimal string keys for root-id-typed values.

    Float columns go through Int64 so ``1.0`` and ``1`` produce the same
    key; every other route must not pass through float64, where adjacent
    18-digit ids collide.
    """
    if pd.api.types.is_float_dtype(values):
        values = values.astype("Int64")
    return pd.Index(values).astype("string")


_ROOT_ID_COLUMNS = ("root_id", "pt_root_id", "pre_pt_root_id", "post_pt_root_id")
_LAMINA_LABELS = ("LA_L", "LA_R")


def _ol_counts_by_type(counts: pd.DataFrame, typed: pd.Series) -> pd.DataFrame:
    lookup = {str(column).lower(): column for column in counts.columns}
    root = next((lookup[name] for name in _ROOT_ID_COLUMNS if name in lookup), None)
    if root is None or "neuropil" not in lookup or "count" not in lookup:
        raise ValueError(
            "innervation table needs a root id, neuropil and count column; "
            f"available: {sorted(map(str, counts.columns))}"
        )
    rows = counts[counts[lookup["neuropil"]].isin(region_labels("OL"))]
    cell_type = _root_id_keys(rows[root]).to_series(index=rows.index).map(typed)
    return pd.DataFrame({
        "cell_type": cell_type,
        "lamina": rows[lookup["neuropil"]].isin(_LAMINA_LABELS),
        "count": rows[lookup["count"]],
    }).dropna(subset=["cell_type"])


def lamina_fractions(
    pre_counts: pd.DataFrame, post_counts: pd.DataFrame, f1_types: pd.Series
) -> pd.Series:
    """F1 cell type -> share of its OL synapses that sit in the lamina.

    Reads F1 only. Each type's pre- and postsynaptic counts (long-form
    ``per_neuron_neuropil_count_{pre,post}_783`` tables) are summed over the
    OL labels; the share is the ``LA_L`` + ``LA_R`` part of that sum. Untyped
    neurons are ignored and a type with no OL synapses has no share. Root ids
    join as exact decimal keys, never through float64.
    """
    typed = f1_types.dropna()
    typed = typed.set_axis(_root_id_keys(typed.index))
    rows = pd.concat([_ol_counts_by_type(t, typed) for t in (pre_counts, post_counts)])
    total = rows.groupby("cell_type")["count"].sum()
    lamina = rows[rows["lamina"]].groupby("cell_type")["count"].sum()
    share = (lamina.reindex(total.index, fill_value=0) / total.where(total > 0)).dropna()
    return share.sort_index().rename("lamina_fraction")


def lamina_dependent_types(
    pre_counts: pd.DataFrame,
    post_counts: pd.DataFrame,
    f1_types: pd.Series,
    *,
    min_fraction: float,
) -> tuple[str, ...]:
    """Sorted F1 cell types whose lamina share of OL synapses is at least ``min_fraction``.

    The rule reads F1 only (:func:`lamina_fractions`); a share equal to
    ``min_fraction`` counts.
    """
    if not 0 < min_fraction <= 1:
        raise ValueError(f"min_fraction must be in (0, 1]; got {min_fraction!r}")
    share = lamina_fractions(pre_counts, post_counts, f1_types)
    return tuple(share.index[share >= min_fraction])


def _in_vocabulary(annotations: pd.DataFrame, f1_types: pd.Series) -> pd.Series:
    cell_type = annotations.set_index("root_id")["cell_type"]
    return cell_type.where(cell_type.isin(set(f1_types.dropna().unique())))


def type_crosswalk(
    annotations: pd.DataFrame, *, dataset: str, f1_types: pd.Series
) -> pd.Series:
    """Root id -> F1 cell type, ``NaN`` where the neuron has none.

    F1 neurons already carry their own type (identity map from the
    ``cell_type`` column). F2 neurons take the F1 type of their FAFB match id
    first, then their own ``cell_type`` where that exact name is an F1 type
    (rule 2); a neuron with neither takes
    the ``NaN`` out-of-vocabulary path rather than being silently dropped.
    F3 neurons take their FAFB-aligned name (``cell_type``, the release's
    ``flywireType``) where it is an F1 type, then their own ``f3_type`` where
    that exact name is one, the same order as F2; a neuron with neither is
    out-of-vocabulary, not kept as a stray F3-only label.
    """
    if dataset == "f1":
        return annotations.set_index("root_id")["cell_type"]
    if dataset == "f2":
        match_id = annotations.set_index("root_id")["fafb_match_id"]
        # Join on exact decimal keys: Series.map/reindex coerce nullable
        # integer ids with missings to float64, where adjacent 18-digit root
        # ids collide and silently map to the wrong type.
        table = f1_types.set_axis(_root_id_keys(f1_types.index))
        matched = _root_id_keys(match_id).to_series(index=match_id.index).map(table)
        return matched.where(matched.notna(), _in_vocabulary(annotations, f1_types).to_numpy())
    if dataset == "f3":
        named = _in_vocabulary(annotations, f1_types)
        if "f3_type" not in annotations.columns:
            return named
        own = _in_vocabulary(annotations.assign(cell_type=annotations["f3_type"]), f1_types)
        return named.where(named.notna(), own.to_numpy())
    raise ValueError(f"unknown dataset {dataset!r}; expected 'f1', 'f2' or 'f3'")


def type_pair_density(graph: RegionGraph, types: pd.Series) -> pd.DataFrame:
    """Ordered cell-type pair densities within one region graph.

    ``density = edges(A->B) / (n_A * n_B)``. Untyped neurons are dropped, not
    bucketed: one enormous OOV pseudo-type would swamp the fit. Only type
    pairs carrying at least one edge appear as rows -- a pair with no
    connection has nothing to divide, and (by construction) always yields a
    positive, comparable density on the datasets it does appear for.
    """
    node_type = pd.Series(graph.node_ids).map(types)
    typed = node_type.to_numpy()
    valid = node_type.notna().to_numpy()

    sources, targets = graph.edges[:, 0], graph.edges[:, 1]
    edge_valid = valid[sources] & valid[targets]

    pairs = pd.DataFrame(
        {
            "source_type": typed[sources[edge_valid]],
            "target_type": typed[targets[edge_valid]],
        }
    )
    counts = node_type[node_type.notna()].value_counts()

    result = (
        pairs.groupby(["source_type", "target_type"])
        .size()
        .reset_index(name="n_edges")
    )
    result["n_source"] = result["source_type"].map(counts).astype(int)
    result["n_target"] = result["target_type"].map(counts).astype(int)
    result["density"] = result["n_edges"] / (result["n_source"] * result["n_target"])
    return result[
        ["source_type", "target_type", "n_source", "n_target", "n_edges", "density"]
    ]


def _score_curve(curve: pd.DataFrame) -> pd.Series:
    """Pick the row minimising |median_log2_ratio|, ties toward higher spearman."""
    return (
        curve.assign(abs_ratio=curve["median_log2_ratio"].abs())
        .sort_values(["abs_ratio", "spearman"], ascending=[True, False])
        .iloc[0]
    )


def calibrate_threshold(
    f1_graph: RegionGraph,
    f1_types: pd.Series,
    f2_connections: pd.DataFrame,
    f2_types: pd.Series,
    *,
    region: str,
    dataset: str = "f2",
    candidates: range = range(1, 16),
    min_neurons: int = 3,
    exclude_types: tuple[str, ...] = (),
    graph_at=None,
) -> ThresholdCalibration:
    """Sweep candidate thresholds for ``dataset`` ("f2" or "f3"); pick the one matching F1's density level.

    For each candidate ``t`` (highest first, so a tie naturally favours the
    stricter, less-noisy threshold): build the F2 region graph at ``t``,
    restrict both graphs' pair densities to ordered cell-type pairs with at
    least ``min_neurons`` neurons on both sides in *both* datasets, and record
    the median log2 density ratio and the Spearman rho between the two
    datasets' log10 densities. A candidate contributing no usable pair (too
    few typed neurons, or the graph itself has no edge reaching ``t``) is
    skipped rather than recorded as a degenerate row. ``exclude_types`` are
    left out of both datasets' densities (their neurons stay in the graphs),
    so the sweep compares only the types the suites can mask.
    ``graph_at(t)``, when given, builds the candidate graph instead (a Phase 3 union,
    ``src/union.py``); ``f2_connections`` is then unused.
    """
    if exclude_types:
        f1_types = f1_types.mask(f1_types.isin(exclude_types))
        f2_types = f2_types.mask(f2_types.isin(exclude_types))
    f1_pairs = type_pair_density(f1_graph, f1_types)

    rows = []
    for t in sorted(set(candidates), reverse=True):
        try:
            f2_graph = (
                graph_at(t) if graph_at is not None
                else build_region_graph(f2_connections, region=region, dataset=dataset, threshold=t)
            )
        except ValueError:
            continue

        f2_pairs = type_pair_density(f2_graph, f2_types)
        merged = f1_pairs.merge(
            f2_pairs, on=["source_type", "target_type"], suffixes=("_f1", "_f2")
        )
        usable = merged[
            (merged["n_source_f1"] >= min_neurons)
            & (merged["n_target_f1"] >= min_neurons)
            & (merged["n_source_f2"] >= min_neurons)
            & (merged["n_target_f2"] >= min_neurons)
        ]
        if usable.empty:
            continue

        log2_ratio = np.log2(usable["density_f2"] / usable["density_f1"])
        if len(usable) >= 2:
            spearman = float(
                stats.spearmanr(
                    np.log10(usable["density_f1"]), np.log10(usable["density_f2"])
                ).statistic
            )
        else:
            spearman = float("nan")

        rows.append(
            {
                "threshold": t,
                "n_edges": f2_graph.n_edges,
                "n_pairs": int(len(usable)),
                "median_log2_ratio": float(np.median(log2_ratio)),
                "spearman": spearman,
            }
        )

    if not rows:
        raise ValueError(
            f"no cell-type pair survives min_neurons={min_neurons} for any "
            f"candidate threshold in {sorted(set(candidates))}; F1 has "
            f"{len(f1_pairs)} typed pair(s) to match against"
        )

    curve = pd.DataFrame(rows)
    best = _score_curve(curve)
    return ThresholdCalibration(
        chosen=int(best["threshold"]),
        curve=curve,
        concordance=float(best["spearman"]),
        n_pairs=int(best["n_pairs"]),
        frozen_at=date.today().isoformat(),
    )
