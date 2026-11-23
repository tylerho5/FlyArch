"""Per-instance scoring task construction for the evaluation harness.

``build_scoring_task`` combines a frozen suite instance (a ``MaskedView``)
with its committed ``NegativeSample`` into a ``ScoringTask``: the arrays a
scorer ranks, the run provenance the results frames are built from, and a
per-positive strata frame (``masked``/``direction``/``type_match``/``siblings``) that
``score_instance`` rolls up into per-stratum metrics.

Nothing reaches a scorer that has not been checked against the committed
artefacts first. A ``ScoringTask`` is built only from an instance of the
frozen ``Suite`` and only when ``NegativeSample.digest()`` reproduces the
committed per-instance digest, so the K=100 pin and the byte-identical
negatives rule hold at the producer boundary rather than
relying on the caller.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from .masks import MaskedView, side_values
from .metrics import hits_at_k, mrr
from .negatives import (
    DEFAULT_SPEC,
    SIDE_SOURCE,
    SIDE_TARGET,
    NegativeSample,
    NegativeSpec,
)
from .suites import Suite

__all__ = [
    "STRATA_AXES",
    "CommittedNegatives",
    "InstanceResult",
    "ScoringTask",
    "build_scoring_task",
    "candidate_pairs",
    "constant_scorer",
    "evaluated_rows",
    "gated_mrr",
    "random_scorer",
    "results_frame",
    "score_instance",
    "strata_frame",
]

STRATA_AXES = ("masked", "direction", "type_match", "siblings")

#: The ladder rungs a dataset may be scored under. A and B are
#: within-F1 (transductive edge-holdout, F1-val masks); C is the
#: cross-individual F2 suite; x is the exploratory cross-sex F3 suite,
#: distinct from C. A row labelled with the wrong rung for its dataset is
#: not the experiment it claims to be.
_LADDER_RUNG_BY_DATASET = {"f1": ("a", "b"), "f2": ("c",), "f3": ("x",)}

#: ``NegativeSpec.alpha`` -> the evaluation setting it samples under.
#: Only the two registered alphas have a setting name, so a task built
#: from any other alpha is rejected rather than silently labelled.
_DEGREE_SETTING_BY_ALPHA = {1.0: "degree_given", 0.0: "degree_free"}

_DIGEST_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class CommittedNegatives:
    """A frozen suite's committed per-instance negative-set digests.

    ``scripts/build_suites.py`` writes one ``<name>.negatives.json`` per suite
    as ``{name, spec: {k, alpha, master_seed}, digests: {instance_id: digest}}``
    (plan 04). This is the artefact that makes the registered byte-identical
    negatives rule checkable: a scorer may only run against a sample whose
    :meth:`NegativeSample.digest` equals the committed value for its instance.
    """

    name: str
    spec: NegativeSpec
    digests: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty suite name")
        if not isinstance(self.spec, NegativeSpec):
            raise ValueError(
                f"spec must be a NegativeSpec, got {type(self.spec).__name__}"
            )
        if self.spec.k != DEFAULT_SPEC.k or self.spec.master_seed != DEFAULT_SPEC.master_seed:
            raise ValueError(
                f"suite {self.name!r} records K={self.spec.k}, "
                f"master_seed={self.spec.master_seed}; both are pinned at "
                f"K={DEFAULT_SPEC.k}, master_seed={DEFAULT_SPEC.master_seed}"
            )
        if self.spec.alpha not in _DEGREE_SETTING_BY_ALPHA:
            raise ValueError(
                f"suite {self.name!r} records alpha={self.spec.alpha!r}, which is "
                f"not a pre-registered degree correction (1.0 degree-given, "
                f"0.0 degree-free)"
            )
        digests = dict(self.digests)
        if not digests:
            raise ValueError(f"suite {self.name!r} has no committed negative digests")
        for instance_id, digest in digests.items():
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError(
                    f"digest keys must be non-empty instance ids, got {instance_id!r}"
                )
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or not _DIGEST_HEX.issuperset(digest)
            ):
                raise ValueError(
                    f"the digest for {instance_id!r} must be a 64-character "
                    f"lowercase sha256 hex string, got {digest!r}"
                )
        object.__setattr__(self, "digests", MappingProxyType(digests))

    @classmethod
    def read(cls, path: str | Path) -> CommittedNegatives:
        """Read one ``<suite>.negatives.json`` written by ``scripts/build_suites.py``."""
        path = Path(path)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            spec = document["spec"]
            return cls(
                name=str(document["name"]),
                spec=NegativeSpec(
                    k=int(spec["k"]),
                    alpha=float(spec["alpha"]),
                    master_seed=int(spec["master_seed"]),
                ),
                digests={str(k): str(v) for k, v in document["digests"].items()},
            )
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path} is not a committed negatives file: {exc}") from exc

    def digest_for(self, instance_id: str) -> str:
        """The committed digest for one suite instance."""
        try:
            return self.digests[instance_id]
        except KeyError:
            raise ValueError(
                f"instance {instance_id!r} has no committed digest in suite "
                f"{self.name!r}; it was not generated from this frozen suite"
            ) from None


@dataclass(frozen=True)
class ScoringTask:
    """One instance's committed negatives, per-positive strata, and provenance."""

    instance_id: str
    suite: str  # the frozen suite name, which is also the split label
    dataset: str
    region: str
    degree_setting: str
    negatives_digest: str
    positives: np.ndarray  # (P, 2) int64, same order as negatives/sides/strata
    negatives: np.ndarray  # (P, K, 2) int64
    sides: np.ndarray  # (P, K) uint8
    strata: pd.DataFrame  # one row per positive; both_masked, direction,
    # source_type, target_type, type_matched, type_trained, siblings


def _validate_sample_arrays(view: MaskedView, sample: NegativeSample) -> None:
    """Reject a sample whose arrays do not fit this graph and its own spec.

    Shape, index range and side values are checked here rather than trusted,
    because ``score_instance`` reshapes whatever it is handed and would
    otherwise emit a normal-looking gate row from a malformed sample.
    """
    positives, negatives, sides = sample.positives, sample.negatives, sample.sides
    if positives.ndim != 2 or positives.shape[1] != 2:
        raise ValueError(f"positives must have shape (P, 2), got {positives.shape}")
    n_positives = positives.shape[0]
    k = sample.spec.k
    if negatives.shape != (n_positives, k, 2):
        raise ValueError(
            f"negatives must have shape ({n_positives}, {k}, 2) for the "
            f"committed K={k}, got {negatives.shape}"
        )
    if sides.shape != (n_positives, k):
        raise ValueError(f"sides must have shape ({n_positives}, {k}), got {sides.shape}")
    if not np.issubdtype(positives.dtype, np.integer) or not np.issubdtype(
        negatives.dtype, np.integer
    ):
        raise ValueError(
            f"positives and negatives must hold integer node indices, got "
            f"{positives.dtype} and {negatives.dtype}"
        )
    if sides.dtype != np.dtype(np.uint8):
        raise ValueError(f"sides must be uint8, got {sides.dtype}")

    n_nodes = view.graph.n_nodes
    for name, array in (("positives", positives), ("negatives", negatives)):
        if array.size and (array.min() < 0 or array.max() >= n_nodes):
            raise ValueError(
                f"{name} holds a node index outside [0, {n_nodes}); the sample "
                f"was not drawn over this graph's nodes"
            )
    if sides.size and not np.isin(sides, (SIDE_TARGET, SIDE_SOURCE)).all():
        raise ValueError(
            f"sides must hold only SIDE_TARGET ({SIDE_TARGET}) or SIDE_SOURCE "
            f"({SIDE_SOURCE}), got {sorted(np.unique(sides).tolist())}"
        )


def build_scoring_task(
    view: MaskedView,
    sample: NegativeSample,
    *,
    suite: Suite,
    negatives: CommittedNegatives,
    types: pd.Series | None = None,
    soma_sides: pd.Series | None = None,
    trained_types: set[str] | None = None,
) -> ScoringTask:
    """Combine a frozen suite instance with its committed negatives.

    Every registered precondition is checked before a scorer runs: the
    instance is one of ``suite``'s frozen instances, the sample was drawn over
    that instance, its spec is the committed spec (so K is the pinned 100), and
    its digest equals the committed per-instance digest (byte-identical
    negatives). A mismatch raises rather than producing a scoreable task.

    ``soma_sides`` supplies L/R/null soma annotations indexed by root id.
    ``trained_types`` contains names seen by the expert, not every name in
    its schema. ``types`` must be in that same vocabulary, F1's names, so an
    F2/F3 caller passes the ``type_crosswalk`` output, never the dataset's own
    names. Sides are read only with ``types``. Omitted sides report ``none``
    unless the type is untrained; omitted trained types leave
    ``type_trained`` unknown and never report ``untrained``.
    """
    if negatives.name != suite.name:
        raise ValueError(
            f"committed negatives belong to suite {negatives.name!r}, but the "
            f"suite is {suite.name!r}; the two artefacts disagree and neither "
            f"may be used"
        )
    suite.validate(view.graph)

    instance_id = view.instance.instance_id
    frozen = {inst.instance_id: inst for inst in suite.resolve(view.graph)}.get(instance_id)
    if frozen is None:
        raise ValueError(
            f"instance {instance_id!r} is not one of suite {suite.name!r}'s "
            f"frozen instances"
        )
    actual = view.instance
    if (
        actual.dataset != frozen.dataset or actual.region != frozen.region
        or actual.regime != frozen.regime or actual.seed != frozen.seed
        or dict(actual.params) != dict(frozen.params)
        or not np.array_equal(actual.masked, frozen.masked)
    ):
        raise ValueError(f"instance {instance_id!r} does not match the frozen mask")
    committed_digest = negatives.digest_for(instance_id)

    if sample.neuropil != instance_id:
        raise ValueError(
            f"sample.neuropil {sample.neuropil!r} is not this view's instance "
            f"{instance_id!r}; the sample was not drawn over this instance"
        )
    if sample.spec != negatives.spec:
        raise ValueError(
            f"sample spec {sample.spec} is not the committed spec "
            f"{negatives.spec} for suite {suite.name!r}; K is pinned at "
            f"{negatives.spec.k}"
        )
    _validate_sample_arrays(view, sample)
    if not np.array_equal(sample.positives, view.hidden_edges):
        raise ValueError(
            "sample.positives does not match view.hidden_edges -- the sample "
            "was not drawn over this view's hidden edges"
        )
    actual_digest = sample.digest()
    if actual_digest != committed_digest:
        raise ValueError(
            f"instance {instance_id!r}: negative sample digest {actual_digest} "
            f"does not match the committed digest {committed_digest}; these are "
            f"not the byte-identical negatives the models were scored against"
        )

    # ``negatives.spec`` is validated at construction, and ``sample.spec`` must
    # equal it, so the alpha is always one of the two pre-registered values.
    degree_setting = _DEGREE_SETTING_BY_ALPHA[sample.spec.alpha]

    source, target = sample.positives[:, 0], sample.positives[:, 1]
    source_masked, target_masked = view.is_masked[source], view.is_masked[target]
    # A both-masked edge has a masked source, so it lands in "out" by design;
    # the "masked" axis reports the both/single split separately (plan 05).
    direction = np.where(source_masked, "out", "in")

    siblings = np.full(view.graph.n_nodes, "none", dtype=object)
    type_trained = None
    if types is None:
        source_type = target_type = type_matched = None
    else:
        values = types.reindex(view.graph.node_ids).to_numpy()
        source_type, target_type = values[source], values[target]
        # type_matched keys on the MASKED endpoint(s): that's the neuron being
        # predicted, so its type is what the OOV path affects.
        type_matched = np.ones(len(sample.positives), dtype=bool)
        type_matched[source_masked] &= ~pd.isna(source_type[source_masked])
        type_matched[target_masked] &= ~pd.isna(target_type[target_masked])
        trained = None
        if trained_types is not None:
            trained = pd.Series(values).isin(trained_types).to_numpy()
            type_trained = np.ones(len(sample.positives), dtype=bool)
            type_trained[source_masked] &= trained[source[source_masked]]
            type_trained[target_masked] &= trained[target[target_masked]]
        if soma_sides is not None:
            node_sides = side_values(view.graph, soma_sides)
            codes, names = pd.factorize(values)
            typed = codes >= 0
            connected = np.zeros(view.graph.n_nodes, dtype=bool)
            connected[view.visible_edges.reshape(-1)] = True
            visible = typed & connected & ~view.is_masked
            counts = np.bincount(codes[typed], minlength=len(names))
            live_counts = np.bincount(codes[visible], minlength=len(names))
            left = np.bincount(codes[visible & (node_sides == "L")], minlength=len(names))
            right = np.bincount(codes[visible & (node_sides == "R")], minlength=len(names))
            nodes = np.flatnonzero(typed & np.isin(node_sides, ("L", "R")))
            group = codes[nodes]
            multiple = counts[group] > 1
            ipsi = np.where(node_sides[nodes] == "L", left[group], right[group])
            contra = np.where(node_sides[nodes] == "L", right[group], left[group])
            siblings[nodes[multiple & (live_counts[group] == 0)]] = "hidden"
            siblings[typed & (counts[codes] > 1) & (live_counts[codes] > 0)] = "unsided"
            siblings[nodes[multiple & (ipsi > 0)]] = "ipsi"
            siblings[nodes[multiple & (contra > 0)]] = "contra"
        # Checked first in the plan's precedence, so it overrides any side
        # label, and it needs no sides: M1 reads an untrained type as OOV.
        if trained is not None:
            siblings[pd.notna(values) & ~trained] = "untrained"

    strata = pd.DataFrame({
        "both_masked": view.both_masked(),
        "direction": direction,
        "source_type": source_type,
        "target_type": target_type,
        "type_matched": type_matched,
        "type_trained": type_trained,
        "siblings": siblings[np.where(source_masked, source, target)],
    })

    return ScoringTask(
        instance_id=instance_id,
        suite=suite.name,
        dataset=suite.dataset,
        region=suite.region,
        degree_setting=degree_setting,
        negatives_digest=committed_digest,
        positives=sample.positives,
        negatives=sample.negatives,
        sides=sample.sides,
        strata=strata,
    )


@dataclass(frozen=True)
class InstanceResult:
    """One instance's provenance, overall metrics, and per-stratum rollup."""

    instance_id: str
    suite: str
    dataset: str
    region: str
    degree_setting: str
    negatives_digest: str
    mrr: float
    recall_at_10: float
    n_positives: int
    strata: pd.DataFrame  # axis, stratum, n_positives, mrr, recall_at_10


def random_scorer(seed: int):
    """A score_fn drawing one uniform score per pair, for sanity-checking the harness."""
    rng = np.random.default_rng(seed)

    def _score(pairs: np.ndarray) -> np.ndarray:
        return rng.uniform(size=pairs.shape[0])

    return _score


def constant_scorer(value: float):
    """A score_fn emitting the same value for every pair -- a collapsed-decoder stand-in."""

    def _score(pairs: np.ndarray) -> np.ndarray:
        return np.full(pairs.shape[0], value, dtype=np.float64)

    return _score


def _stratum_rows(rows, axis, masks_by_stratum, pos_scores, neg_scores):
    for stratum, mask in masks_by_stratum.items():
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "axis": axis,
            "stratum": stratum,
            "n_positives": n,
            "mrr": mrr(pos_scores[mask], neg_scores[mask]),
            "recall_at_10": hits_at_k(pos_scores[mask], neg_scores[mask], k=10),
        })


def evaluated_rows(task: ScoringTask) -> np.ndarray:
    """Rows eligible for the single-masked gate and direction/type rollups."""
    rows = ~task.strata["both_masked"].to_numpy(dtype=bool)
    if not rows.any():
        raise ValueError("instance has no single-masked positive for gated scoring")
    return rows


def candidate_pairs(task: ScoringTask) -> np.ndarray:
    """Each positive followed by its K negatives, as one ``(P*(K+1), 2)`` block."""
    candidates = np.concatenate([task.positives[:, None, :], task.negatives], axis=1)
    return candidates.reshape(-1, 2)


def gated_mrr(task: ScoringTask, scores: np.ndarray) -> float:
    """The gated MRR of ``scores``, one per :func:`candidate_pairs` row."""
    rows = np.asarray(scores, dtype=np.float64).reshape(len(task.positives), -1)
    gated = evaluated_rows(task)
    return mrr(rows[gated, 0], rows[gated, 1:])


def score_instance(task: ScoringTask, score_fn) -> InstanceResult:
    """Score one instance's positives against their committed negatives.

    Stacks positives and negatives into one ``(P*(K+1), 2)`` block and calls
    ``score_fn`` once -- scoring positives and negatives in separate calls
    would let a per-batch-normalising model drift and break the pinned
    sanity numbers.
    """
    gated = evaluated_rows(task)
    flat = candidate_pairs(task)
    scores = np.asarray(score_fn(flat), dtype=np.float64)
    if scores.shape != (flat.shape[0],):
        raise ValueError(
            f"score_fn must return shape {(flat.shape[0],)}, got {scores.shape}"
        )
    scores = scores.reshape(len(task.positives), -1)
    pos_scores = scores[:, 0]
    neg_scores = scores[:, 1:]

    rows: list[dict] = []
    both_masked = task.strata["both_masked"].to_numpy()
    _stratum_rows(
        rows, "masked",
        {"both_masked": both_masked, "single_masked": ~both_masked},
        pos_scores, neg_scores,
    )

    direction = task.strata["direction"].to_numpy()
    _stratum_rows(
        rows, "direction",
        {"out": gated & (direction == "out"), "in": gated & (direction == "in")},
        pos_scores, neg_scores,
    )

    type_matched = task.strata["type_matched"]
    if not type_matched.isna().all():
        matched = type_matched.to_numpy(dtype=bool)
        _stratum_rows(
            rows, "type_match",
            {"matched": gated & matched, "unmatched": gated & ~matched},
            pos_scores, neg_scores,
        )

    siblings = task.strata["siblings"].to_numpy()
    _stratum_rows(
        rows, "siblings",
        {label: gated & (siblings == label) for label in ("contra", "ipsi", "hidden", "unsided", "untrained", "none")},
        pos_scores, neg_scores,
    )

    strata = pd.DataFrame(
        rows, columns=["axis", "stratum", "n_positives", "mrr", "recall_at_10"]
    )

    return InstanceResult(
        instance_id=task.instance_id,
        suite=task.suite,
        dataset=task.dataset,
        region=task.region,
        degree_setting=task.degree_setting,
        negatives_digest=task.negatives_digest,
        mrr=gated_mrr(task, scores),
        recall_at_10=hits_at_k(pos_scores[gated], neg_scores[gated], k=10),
        n_positives=int(gated.sum()),
        strata=strata,
    )


def _run_columns(results: list[InstanceResult], *, model: str, seed: int, regime: str) -> dict:
    """The run-level columns, derived from the results' own provenance.

    Suite, dataset, region and degree setting are facts about the committed
    artefact a result was scored against, so they come from ``InstanceResult``
    rather than from a caller-supplied label. A table relabelled this way --
    F1 rows presented as F2 -- is the wrong-dataset failure the gate exists to
    catch, and it must fail here, at the producer, not at the gate.
    """
    first = results[0]
    for result in results[1:]:
        if (
            result.suite,
            result.dataset,
            result.region,
            result.degree_setting,
        ) != (first.suite, first.dataset, first.region, first.degree_setting):
            raise ValueError(
                "the results span different suites, datasets, regions or degree "
                "settings; build one frame per evaluation context"
            )

    allowed = _LADDER_RUNG_BY_DATASET.get(first.dataset)
    if allowed is None:
        raise ValueError(
            f"dataset {first.dataset!r} has no pre-registered ladder rung"
        )
    if regime not in allowed:
        raise ValueError(
            f"regime {regime!r} is not a registered rung for dataset "
            f"{first.dataset!r}; expected one of {allowed}"
        )

    # regime is the A/B/C ladder value, written verbatim into
    # mask_regime; it is not the per-instance R/T/S regime, which lives in the
    # mask_instance id (see src/gate.py REQUIRED_COLUMNS).
    return {
        "model": model,
        "dataset": first.dataset,
        "region": first.region,
        "split": first.suite,
        "seed": int(seed),
        "degree_setting": first.degree_setting,
        "mask_regime": regime,
    }


_RESULTS_COLUMNS = (
    "model", "dataset", "region", "split", "seed", "degree_setting",
    "mask_regime", "mask_instance", "negatives_digest", "mrr", "recall_at_10",
    "n_positives",
)

_STRATA_COLUMNS = (
    "axis", "stratum", "n_positives", "mrr", "recall_at_10", "mask_instance",
    "negatives_digest", *[
        "model", "dataset", "region", "split", "seed", "degree_setting", "mask_regime",
    ],
)


def results_frame(
    results: list[InstanceResult],
    *,
    model: str,
    seed: int,
    regime: str,
) -> pd.DataFrame:
    """One gate-ready row per InstanceResult -- what evaluate_gate reads.

    ``dataset``, ``region``, ``split`` and ``degree_setting`` are taken from
    each result's provenance; only the fields no artefact can supply (``model``,
    ``seed``, the A/B/C ``regime``) are caller arguments.
    """
    if not results:
        return pd.DataFrame(columns=_RESULTS_COLUMNS)
    run_columns = _run_columns(results, model=model, seed=seed, regime=regime)
    rows = [
        {
            **run_columns,
            "mask_instance": result.instance_id,
            "negatives_digest": result.negatives_digest,
            "mrr": result.mrr,
            "recall_at_10": result.recall_at_10,
            "n_positives": result.n_positives,
        }
        for result in results
    ]
    return pd.DataFrame(rows, columns=_RESULTS_COLUMNS)


def strata_frame(
    results: list[InstanceResult],
    *,
    model: str,
    seed: int,
    regime: str,
) -> pd.DataFrame:
    """One per-stratum reporting row per (instance, axis, stratum)."""
    if not results:
        return pd.DataFrame(columns=_STRATA_COLUMNS)
    run_columns = _run_columns(results, model=model, seed=seed, regime=regime)
    frames = [
        result.strata.assign(
            mask_instance=result.instance_id,
            negatives_digest=result.negatives_digest,
            **run_columns,
        )
        for result in results
    ]
    return pd.concat(frames, ignore_index=True)[list(_STRATA_COLUMNS)]
