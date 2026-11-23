"""The shared pair-feature builder and the F1-train type-pair target encoding.

``pair_features`` emits centroid distance, type-pair target encoding, both
endpoints' NT values, innervation cosine, and side match. The degree-given
setting adds the masked endpoint's total as the final column; the degree-free
setting omits it. Every other value is a post-mask node feature, so the builder
never reads ``view.hidden_edges``.

``fit_type_encoding`` estimates the type-pair encoding on F1-train inputs only.
Fitting it on the graph under evaluation would encode the answers; training
features come from deterministic positive-group folds that exclude their own
labels. F1-train is the F1 graph minus the F1-val masks (phase-1 split table):
an :class:`F1TrainingContext` prunes every edge touching a neuron the F1-val
suite reserves, and fit views are built on that pruned copy with their masks
drawn from the neurons that remain.

``BASELINES`` holds the three pre-registered opponents. The configuration
predictor scores ``out_strength(source) * in_strength(target)`` and learns
nothing; the other two fit a grid-searched sklearn/xgboost estimator over
``pair_features``. Learned baselines require an :class:`F1TrainingContext`
bound to the F1 graph and its F1-val exclusion suite, plus the frozen F2-dev
``Suite`` and its ``CommittedNegatives``. Each grid point is fitted under
three recorded seeds and ranked by mean F2-dev MRR over the gated rows, and
the three winning estimators are retained for per-seed scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol
import tempfile

import numpy as np

from src import negatives
from src.features import FeatureTable
from src.evaluate import CommittedNegatives, ScoringTask, build_scoring_task, evaluated_rows
from src.metrics import mrr
from src.masks import MaskedView, VisibleView, apply_mask, pool_mask
from src.negatives import DEFAULT_SPEC, SIDE_SOURCE, SIDE_TARGET, NegativeSample, NegativeSpec, pack_edges
from src.regions import RegionGraph
from src.suites import Suite, view_negatives

__all__ = [
    "BASELINES",
    "GLOBAL_MEAN_KEY",
    "LOGREG_GRID",
    "XGBOOST_GRID",
    "Baseline",
    "ConfigurationBaseline",
    "F1TrainingContext",
    "LogRegBaseline",
    "TypeEncoding",
    "XGBoostBaseline",
    "fit_type_encoding",
    "pair_features",
]

#: The fallback key in a :data:`TypeEncoding` for an unseen ordered type pair.
GLOBAL_MEAN_KEY: str = "__global__"

TypeEncoding = dict[tuple[int, int] | str, float]

#: Logistic-regression grid, one entry per grid point.
LOGREG_GRID: tuple[dict[str, float], ...] = tuple(
    {"C": float(c)} for c in 10.0 ** np.arange(-3, 4)
)

#: XGBoost grid, the full product of the four registered axes.
XGBOOST_GRID: tuple[dict[str, float | int], ...] = tuple(
    {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "subsample": subsample,
    }
    for max_depth in (3, 6, 9)
    for n_estimators in (200, 600)
    for learning_rate in (0.03, 0.1)
    for subsample in (0.8, 1.0)
)

#: Positives per fit view the target encoding is estimated from.
_TARGET_ENCODING_POSITIVES = 200

_CENTROID_COLUMNS = ("centroid_x", "centroid_y", "centroid_z")
#: The exclusion and tuning suites a fit may use: a region's, or Phase 3's boundary suite on the union.
EXCLUSION_SUITES = frozenset({"f1_val", "phase3_boundary_f1_val"})
TUNING_SUITES = frozenset({"f2_dev", "phase3_boundary_f2_dev"})


@dataclass(frozen=True)
class F1TrainingContext:
    """The F1-train binding a learned baseline fits under.

    ``graph`` is the original F1 graph -- the graph ``exclusion_suite`` pins and
    the one F1-val scoring resolves against. ``train_graph`` is its F1-train
    copy: every edge touching a neuron any F1-val instance masks is pruned, and
    ``fit_pool`` is the neurons that remain, the only ones a fit view may hide
    (phase-1 split table, "minus F1-val masks"). :meth:`draw_fit_view` draws
    from it with the rule M1's gaps use. ``true_edges`` stays the original
    graph's edge set, because the pruned copy no longer holds the validation
    edges and the sampler must keep refusing those pairs as negatives.
    """

    graph: RegionGraph
    exclusion_suite: Suite
    train_graph: RegionGraph = field(init=False, repr=False)
    train_digest: str = field(init=False, repr=False)
    excluded: np.ndarray = field(init=False, repr=False)
    fit_pool: np.ndarray = field(init=False, repr=False)
    true_edges: set[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.graph.dataset != "f1" or self.exclusion_suite.name not in EXCLUSION_SUITES:
            raise ValueError("F1 training context requires the named F1-val exclusion suite")
        self.exclusion_suite.validate(self.graph)
        masks = tuple(
            np.asarray(instance.masked, dtype=np.int64)
            for instance in self.exclusion_suite.resolve(self.graph)
        )
        excluded = np.unique(np.concatenate(masks)) if masks else np.zeros(0, dtype=np.int64)
        if excluded.size == 0:
            raise ValueError(
                "the F1-val exclusion suite reserves no neurons, so F1-train "
                "would be the whole F1 graph"
            )
        train_graph = _pruned_graph(self.graph, excluded)
        object.__setattr__(self, "excluded", excluded)
        object.__setattr__(self, "train_graph", train_graph)
        object.__setattr__(self, "train_digest", train_graph.digest())
        object.__setattr__(
            self, "fit_pool", np.setdiff1d(np.arange(self.graph.n_nodes, dtype=np.int64), excluded)
        )
        object.__setattr__(self, "true_edges", pack_edges(self.graph.edges))

    def draw_fit_view(
        self, rng: np.random.Generator, fraction: float = 0.05
    ) -> MaskedView:
        """One F1-train fit view: a gap drawn from :attr:`fit_pool`.

        The draw rule is :func:`src.masks.pool_mask`, which
        ``train.simulate_gaps`` delegates to, so both training paths sample the
        same distribution. ``complete_graph`` keeps the granted degree-given
        total at the unpruned F1 value a scored view reports.
        """
        return apply_mask(
            self.train_graph,
            pool_mask(self.train_graph, rng, fraction=fraction, pool=self.fit_pool),
            complete_graph=self.graph,
        )

    def validate(self, view: MaskedView) -> None:
        if not isinstance(view, MaskedView) or view.graph.dataset != "f1":
            raise ValueError("fit view does not belong to the F1 graph")
        if view.instance.dataset != "f1" or view.instance.region != self.graph.region:
            raise ValueError("fit view does not belong to the F1 graph")
        if view.graph.digest() != self.train_digest:
            raise ValueError(
                "fit view must be built on the pruned F1-train graph "
                "(F1TrainingContext.train_graph), not the full F1 graph"
            )
        if np.intersect1d(np.asarray(view.instance.masked, dtype=np.int64), self.excluded).size:
            raise ValueError(
                "F1-val neurons are excluded from F1 training views, even when relabelled"
            )


def _pruned_graph(graph: RegionGraph, excluded: np.ndarray) -> RegionGraph:
    """The F1-train copy of ``graph``: no edge touching an excluded neuron.

    Node ids, dense indexing and ``n_nodes`` are unchanged, so masks resolved
    against the original graph keep addressing the same neurons and the feature
    table keeps its row order; only the edge set shrinks. This mirrors
    ``train/train.py``'s ``_training_graph``. The two training modules must not
    import each other (implementation plan 06), so the few lines are repeated
    rather than shared.
    """
    if excluded.size == 0:
        return graph
    flags = np.zeros(graph.n_nodes, dtype=bool)
    flags[excluded] = True
    keep = ~(flags[graph.edges[:, 0]] | flags[graph.edges[:, 1]])
    return replace(graph, edges=graph.edges[keep], weights=graph.weights[keep])


def _visible(view: VisibleView) -> VisibleView:
    if not isinstance(view, VisibleView):
        raise TypeError("baseline scoring requires a VisibleView, not a MaskedView")
    return view


def _n_nodes(view: VisibleView) -> int:
    return view.n_nodes


def _block_index(table: FeatureTable, block: str, name: str) -> int:
    span = table.blocks[block]
    return span.start + table.columns[span].index(name)


def _centroids(table: FeatureTable) -> np.ndarray:
    return table.values[
        :, [_block_index(table, "geometry", name) for name in _CENTROID_COLUMNS]
    ]


def _type_ids(table: FeatureTable, n_nodes: int) -> np.ndarray:
    """The id-encoded type column of ``blocks["types"]``, one id per node.

    ``FeatureSchema`` builds its ``type_vocab`` from the ``cell_type``
    annotation, so this column is the block's only holder of
    ``FeatureSchema.type_id`` values (0 = OOV).
    """
    if table.values.shape[0] != n_nodes:
        raise ValueError(
            f"feature table has {table.values.shape[0]} rows for a view of {n_nodes} nodes"
        )
    return table.values[:, _block_index(table, "types", "cell_type")]


def _encoding_column(
    source_type: np.ndarray, target_type: np.ndarray, type_encoding: TypeEncoding
) -> np.ndarray:
    fallback = float(type_encoding.get(GLOBAL_MEAN_KEY, 0.0))
    pairs = np.column_stack((source_type, target_type)).astype(np.int64)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    values = np.array(
        [type_encoding.get((int(source), int(target)), fallback) for source, target in unique],
        dtype=float,
    )
    return values[inverse.reshape(-1)]


def _side_code(rows: np.ndarray) -> np.ndarray:
    """One integer per row: the set side, or -1 where the side is absent."""
    present = rows.any(axis=1)
    return np.where(present, rows.argmax(axis=1), -1)


def _degree_given(negative_spec: NegativeSpec) -> bool:
    if negative_spec.alpha == 1.0:
        return True
    if negative_spec.alpha == 0.0:
        return False
    raise ValueError(
        "baseline degree setting requires negative_spec.alpha to be 1.0 "
        f"(degree-given) or 0.0 (degree-free), got {negative_spec.alpha}"
    )


def _pair_width(table: FeatureTable, negative_spec: NegativeSpec) -> int:
    nt_span = table.blocks["nt"]
    base = 4 + 2 * (nt_span.stop - nt_span.start)
    return base + int(_degree_given(negative_spec))


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    out = np.zeros(left.shape[0])
    return np.divide(
        np.einsum("ij,ij->i", left, right), denominator, out=out, where=denominator > 0
    )


def pair_features(
    view: VisibleView,
    pairs: np.ndarray,
    table: FeatureTable,
    *,
    type_encoding: TypeEncoding,
    negative_spec: NegativeSpec = DEFAULT_SPEC,
) -> np.ndarray:
    """Score-ready features for candidate pairs: shape ``(M, D)``, all finite.

    ``pairs`` is the candidate list, not a feature source: the builder reads no
    hidden-edge array. The degree-given total is the sole complete-graph input,
    taken from the pair's masked endpoint (its source where masked, else its
    target, which is 0.0 where neither endpoint is masked).
    """
    n_nodes = _n_nodes(_visible(view))
    type_ids = _type_ids(table, n_nodes)
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.size == 0:
        pairs = pairs.reshape(0, 2)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (M, 2), got {pairs.shape}")
    if len(pairs) == 0:
        return np.zeros((0, _pair_width(table, negative_spec)))
    if pairs.min() < 0 or pairs.max() >= n_nodes:
        raise IndexError(f"pairs index outside [0, {n_nodes})")
    source, target = pairs[:, 0], pairs[:, 1]

    centroids = _centroids(table)
    distance = np.linalg.norm(centroids[source] - centroids[target], axis=1)
    encoding = _encoding_column(type_ids[source], type_ids[target], type_encoding)

    types_span = table.blocks["types"]
    side_positions = np.flatnonzero(
        [name.startswith("side_") for name in table.columns[types_span]]
    )
    if side_positions.size == 0:
        raise ValueError("the types block has no side columns")
    sides = table.values[:, types_span.start + side_positions]
    source_side, target_side = _side_code(sides[source]), _side_code(sides[target])
    match = ((source_side >= 0) & (source_side == target_side)).astype(float)

    nt = table.values[:, table.blocks["nt"]]
    innervation_positions = np.flatnonzero(
        [str(name).startswith("innervation_") for name in table.columns]
    )
    innervation = table.values[:, innervation_positions]
    cosine = _cosine(innervation[source], innervation[target])

    columns = [distance, encoding, nt[source], nt[target], cosine, match]
    if _degree_given(negative_spec):
        degree_given = np.asarray(view.degree_given_total(), dtype=float)
        masked = np.asarray(view.is_masked, dtype=bool)
        columns.append(
            np.where(masked[source], degree_given[source], degree_given[target])
        )
    return np.column_stack(columns)


def fit_type_encoding(
    views: Sequence[MaskedView],
    tables: Sequence[FeatureTable],
    *,
    positives_per_view: int,
    context: F1TrainingContext,
    negative_spec: NegativeSpec = DEFAULT_SPEC,
) -> TypeEncoding:
    """The type-pair positive rate on F1-train views, ordered pair as the key.

    Per view the candidates are ``positives_per_view`` hidden edges, sampled
    deterministically, plus masked-endpoint-preserving negatives under
    ``negative_spec``. This matches the candidate distribution used for
    training and ranking.
    """
    views, tables = list(views), list(tables)
    if len(views) != len(tables):
        raise ValueError(f"got {len(views)} views for {len(tables)} tables")
    if not isinstance(context, F1TrainingContext):
        raise ValueError("type-pair encoding requires an F1 training context")
    for view in views:
        context.validate(view)
    positives_per_view = int(positives_per_view)
    if positives_per_view < 1:
        raise ValueError(f"positives_per_view must be >= 1, got {positives_per_view}")
    true_counts: dict[tuple[int, int], float] = {}
    total_counts: dict[tuple[int, int], float] = {}
    for view, table in zip(views, tables):
        positives = _fit_positives(view, positives_per_view)
        negatives_ = _fit_negatives(
            view, positives, spec=negative_spec, true_edges=context.true_edges,
        )
        candidates = np.concatenate([positives, negatives_])
        truth = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives_))])
        type_ids = _type_ids(table, view.graph.n_nodes).astype(np.int64)
        unique, inverse = np.unique(type_ids[candidates], axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        n_true = np.bincount(inverse, weights=truth, minlength=len(unique))
        n_total = np.bincount(inverse, minlength=len(unique))
        for (source_type, target_type), true, total in zip(unique, n_true, n_total):
            key = (int(source_type), int(target_type))
            true_counts[key] = true_counts.get(key, 0.0) + float(true)
            total_counts[key] = total_counts.get(key, 0.0) + float(total)
    if not total_counts:
        raise ValueError("no candidate pairs to fit a type encoding from")
    encoding: TypeEncoding = {
        key: true_counts[key] / total_counts[key] for key in total_counts
    }
    encoding[GLOBAL_MEAN_KEY] = sum(true_counts.values()) / sum(total_counts.values())
    return encoding


def _fit_positives(view: MaskedView, count: int) -> np.ndarray:
    edges = np.asarray(view.hidden_edges, dtype=np.int64)
    if edges.size == 0:
        raise ValueError(f"view {view.instance.instance_id!r} has no hidden edges")
    if count >= len(edges):
        return edges
    rng = negatives._derive_rng(
        DEFAULT_SPEC.master_seed, f"{view.instance.instance_id}-fit-positives"
    )
    return edges[np.sort(rng.choice(len(edges), size=count, replace=False))]


def _fit_negatives(
    view: MaskedView, positives: np.ndarray, *, spec: NegativeSpec,
    true_edges: set[int] | None = None,
) -> np.ndarray:
    sample = view_negatives(
        view, positives=positives, spec=spec, true_edges=true_edges
    )
    return np.asarray(sample.negatives).reshape(-1, 2)


def _pairs_array(pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (M, 2), got {pairs.shape}")
    return pairs


def _check_alignment(views: Sequence, tables: Sequence, *, what: str) -> None:
    if len(views) != len(tables):
        raise ValueError(f"got {len(views)} {what} for {len(tables)} feature tables")


def _predict(estimator, features: np.ndarray, *, iteration_limit: int | None = None) -> np.ndarray:
    kwargs = {} if iteration_limit is None else {"iteration_range": (0, iteration_limit)}
    return np.asarray(estimator.predict_proba(features, **kwargs), dtype=np.float64)[:, 1]


def _candidate_negatives(
    view: MaskedView, *, spec: NegativeSpec,
    pool: np.ndarray | None = None, true_edges: set[int] | None = None,
) -> NegativeSample:
    """The suite negative draw for one view's hidden edges.

    ``pool`` and ``true_edges`` default to the view's own graph. Fitting passes
    the original graph's packed edge set when the view is the pruned F1-train
    copy, so a pruned F1-val edge never comes back as a negative.
    """
    if pool is None and true_edges is None:
        return view_negatives(view, spec=spec)
    return view_negatives(view, spec=spec, pool=pool, true_edges=true_edges)


def _fit_subset(sample: NegativeSample, view: MaskedView, *, seed: int) -> NegativeSample:
    """Draw ten fitting negatives per positive from the generated K=100 sample."""
    rng = negatives._derive_rng(
        sample.spec.master_seed, f"{view.instance.instance_id}-fit-subset-{seed}"
    )
    chosen = np.empty((sample.n_positives, 10), dtype=np.int64)
    for row in range(sample.n_positives):
        sides = sample.sides[row]
        target = np.flatnonzero(sides == SIDE_TARGET)
        source = np.flatnonzero(sides == SIDE_SOURCE)
        if len(target) and len(source):
            indices = np.concatenate([
                rng.choice(target, 5, replace=False),
                rng.choice(source, 5, replace=False),
            ])
        else:
            indices = rng.choice(np.arange(sample.spec.k), 10, replace=False)
        chosen[row] = np.sort(indices)
    indices = np.arange(sample.n_positives)[:, None]
    return replace(
        sample, negatives=sample.negatives[indices, chosen],
        sides=sample.sides[indices, chosen],
        spec=NegativeSpec(k=10, alpha=sample.spec.alpha, master_seed=sample.spec.master_seed),
    )


def _candidate_block(sample: NegativeSample) -> np.ndarray:
    """(P, K+1, 2), the positive first in each row -- ``score_instance``'s layout."""
    return np.concatenate([sample.positives[:, None, :], sample.negatives], axis=1)


def _group_folds(positives: np.ndarray) -> np.ndarray:
    return (positives[:, 0] * 1315423911 + positives[:, 1]) % 5


def _crossfit_encodings(
    pair_types: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> dict[int, TypeEncoding]:
    """Encode each fold from other positive groups and their negatives."""
    result = {}
    for fold in range(5):
        kept = folds != fold
        types = pair_types[kept].reshape(-1, 2)
        truth = labels[kept].reshape(-1)
        if not len(types):
            result[fold] = {GLOBAL_MEAN_KEY: 0.0}
            continue
        unique, inverse = np.unique(types, axis=0, return_inverse=True)
        positives = np.bincount(inverse, weights=truth, minlength=len(unique))
        counts = np.bincount(inverse, minlength=len(unique))
        result[fold] = {
            (int(s), int(t)): float(p / n)
            for (s, t), p, n in zip(unique, positives, counts)
        }
        result[fold][GLOBAL_MEAN_KEY] = float(truth.mean())
    return result


def _training_block(
    views: Sequence[MaskedView],
    tables: Sequence[FeatureTable],
    type_encoding: TypeEncoding,
    negative_spec: NegativeSpec,
    *, context: F1TrainingContext, fit_seed: int = DEFAULT_SPEC.master_seed,
    samples: Sequence[NegativeSample] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fit type-pair rates by positive group before mapping fit features.

    ``context`` is required: a fit view is the pruned F1-train copy, so the
    sampler is handed the original graph's edge set to keep refusing the pruned
    F1-val edges as negatives. Negatives keep the full node pool, as M1's
    per-epoch draw does.
    """
    true_edges = context.true_edges
    if samples is None:
        samples = [
            _candidate_negatives(view, spec=negative_spec, true_edges=true_edges)
            for view in views
        ]
    encoding_types, encoding_labels, encoding_folds = [], [], []
    for view, table, sample in zip(views, tables, samples):
        positives = _fit_positives(view, _TARGET_ENCODING_POSITIVES)
        if np.array_equal(positives, sample.positives):
            capped = sample
        else:
            capped = view_negatives(
                view, positives=positives, spec=negative_spec, true_edges=true_edges,
            )
        # Counts come from the same positives + K negatives draw that
        # fit_type_encoding and the ranked views use, so the encoding column
        # means one thing at fit time and at scoring time. Only the dense
        # training rows below are restricted to the 10-negative subset.
        candidate = _candidate_block(capped)
        ids = _type_ids(table, view.graph.n_nodes).astype(np.int64)
        encoding_types.append(ids[candidate])
        labels = np.zeros(candidate.shape[:2], dtype=float)
        labels[:, 0] = 1.0
        encoding_labels.append(labels)
        encoding_folds.append(_group_folds(positives))
    encodings = _crossfit_encodings(
        np.concatenate(encoding_types), np.concatenate(encoding_labels),
        np.concatenate(encoding_folds),
    )
    features, labels = [], []
    for view, table, sample in zip(views, tables, samples):
        candidates = _candidate_block(_fit_subset(sample, view, seed=fit_seed))
        folds = _group_folds(sample.positives)
        mapped = np.empty((candidates.shape[0], candidates.shape[1], _pair_width(table, negative_spec)), dtype=np.float32)
        for fold in np.unique(folds):
            rows = folds == fold
            mapped[rows] = pair_features(
                view.model_view(), candidates[rows].reshape(-1, 2), table,
                type_encoding=encodings[int(fold)], negative_spec=negative_spec,
            ).reshape(int(rows.sum()), candidates.shape[1], -1)
        features.append(mapped.reshape(-1, mapped.shape[-1]))
        truth = np.zeros(candidates.shape[:2])
        truth[:, 0] = 1.0
        labels.append(truth.reshape(-1))
    return np.vstack(features), np.concatenate(labels)


def _tuning_inputs(
    views: list[MaskedView],
    tables: list[FeatureTable],
    *,
    context: F1TrainingContext,
    val_views: Sequence[MaskedView] | None,
    val_tables: Sequence[FeatureTable] | None,
    val_suite: Suite | None,
    val_negatives: CommittedNegatives | None,
    negative_spec: NegativeSpec,
) -> tuple[list[MaskedView], list[FeatureTable], list[ScoringTask]]:
    if not isinstance(context, F1TrainingContext):
        raise ValueError("learned baselines require an F1 training context")
    for view in views:
        context.validate(view)
    if val_views is None or val_tables is None or val_suite is None or val_negatives is None:
        raise ValueError("learned baselines require the F2-dev suite and committed negatives")
    if val_suite.name not in TUNING_SUITES:
        raise ValueError("learned baselines require the F2-dev suite, not F2-confirm")
    if val_negatives.spec != negative_spec:
        raise ValueError("committed F2-dev negative spec differs from the baseline setting")
    val_views, val_tables = list(val_views), list(val_tables)
    if not val_views:
        raise ValueError("F2-dev validation views must not be empty")
    _check_alignment(val_views, val_tables, what="val views")
    val_suite.validate(val_views[0].graph)
    pool = np.arange(val_views[0].graph.n_nodes, dtype=np.int64)
    true_edges = pack_edges(val_views[0].graph.edges)
    tasks = [build_scoring_task(
        view, _candidate_negatives(
            view, spec=negative_spec, pool=pool, true_edges=true_edges
        ), suite=val_suite, negatives=val_negatives,
    ) for view in val_views]
    return val_views, val_tables, tasks


@dataclass(frozen=True)
class _RankingBlock:
    path: Path
    shape: tuple[int, int]
    gated: np.ndarray


def _ranking_blocks(
    views: Sequence[MaskedView],
    tables: Sequence[FeatureTable],
    tasks: Sequence[ScoringTask],
    type_encoding: TypeEncoding,
    negative_spec: NegativeSpec,
    directory: str,
) -> tuple[_RankingBlock, ...]:
    blocks = []
    for index, (view, table, task) in enumerate(zip(views, tables, tasks)):
        candidates = np.concatenate([task.positives[:, None, :], task.negatives], axis=1)
        features = pair_features(
            view.model_view(), candidates.reshape(-1, 2), table,
            type_encoding=type_encoding, negative_spec=negative_spec,
        ).astype(np.float32)
        path = Path(directory) / f"view-{index}.npy"
        mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=features.shape)
        mapped[:] = features
        mapped.flush()
        mapped._mmap.close()
        del mapped, features, candidates
        blocks.append(_RankingBlock(path, (len(task.positives), task.negatives.shape[1] + 1), evaluated_rows(task)))
    return tuple(blocks)


def _mean_mrr(
    estimator, ranking_blocks: Sequence[_RankingBlock], *, iteration_limit: int | None = None
) -> float:
    """Map one validation view at a time and average its gated MRR."""
    scores = []
    for item in ranking_blocks:
        mapped = np.load(item.path, mmap_mode="r")
        try:
            block = _predict(estimator, mapped, iteration_limit=iteration_limit).reshape(item.shape)
            scores.append(mrr(block[item.gated, 0], block[item.gated, 1:]))
        finally:
            mapped._mmap.close()
    return float(np.mean(scores))


class Baseline(Protocol):
    """The pre-registered opponent contract the running loop consumes.

    ``fit`` trains on F1-train views and ranks learned baselines on F2-dev;
    ``score`` returns one score per candidate pair, higher meaning more likely
    an edge. Context and committed suite inputs prevent F1-val or F2-confirm
    from being used as fit or selection data.
    """

    def fit(
        self,
        views: Sequence[MaskedView],
        tables: Sequence[FeatureTable],
        *,
        context: F1TrainingContext,
        val_views: Sequence[MaskedView],
        val_tables: Sequence[FeatureTable],
        val_suite: Suite,
        val_negatives: CommittedNegatives,
    ) -> Baseline: ...

    def score(
        self, view: VisibleView, table: FeatureTable, pairs: np.ndarray
    ) -> np.ndarray: ...


class ConfigurationBaseline:
    """The configuration-model null: ``out_strength(source) * in_strength(target)``.

    In the degree-given setting a masked endpoint contributes its
    ``degree_given_total``; an unmasked source contributes visible out-strength
    and an unmasked target visible in-strength. Degree-free scoring uses visible
    strengths for every endpoint, so this degree-only null collapses when an
    endpoint is masked. ``fit`` learns nothing, and no feature table is read.
    """

    def __init__(self, *, negative_spec: NegativeSpec = DEFAULT_SPEC) -> None:
        _degree_given(negative_spec)
        self.negative_spec = negative_spec

    def fit(
        self,
        views: Sequence[MaskedView],
        tables: Sequence[FeatureTable],
        *,
        context: F1TrainingContext | None = None,
        val_views: Sequence[MaskedView] | None = None,
        val_tables: Sequence[FeatureTable] | None = None,
        val_suite: Suite | None = None,
        val_negatives: CommittedNegatives | None = None,
    ) -> ConfigurationBaseline:
        return self

    def score(
        self, view: VisibleView, table: FeatureTable, pairs: np.ndarray
    ) -> np.ndarray:
        _visible(view)
        pairs = _pairs_array(pairs)
        if len(pairs) == 0:
            return np.zeros(0, dtype=np.float64)
        masked = np.asarray(view.is_masked, dtype=bool)
        out_strength = np.asarray(view.visible_out_strength(), dtype=np.float64)
        in_strength = np.asarray(view.visible_in_strength(), dtype=np.float64)
        source, target = pairs[:, 0], pairs[:, 1]
        if _degree_given(self.negative_spec):
            total = np.asarray(view.degree_given_total(), dtype=np.float64)
            source_factor = np.where(masked[source], total[source], out_strength[source])
            target_factor = np.where(masked[target], total[target], in_strength[target])
        else:
            source_factor = out_strength[source]
            target_factor = in_strength[target]
        return source_factor * target_factor


class _PairFeatureBaseline:
    """Grid-searched estimator over ``pair_features``, shared by the two learners.

    ``fit`` estimates the type-pair encoding on the training views only --
    never on a view it ranks on -- and then scores every grid point by mean
    gated MRR over the ranking views, averaged across three seeds. Selection
    is deterministic: the champion is the first grid point with the highest
    mean per-seed MRR, so a tie keeps grid order.
    """

    grid: tuple[dict, ...] = ()

    def __init__(self, *, negative_spec: NegativeSpec = DEFAULT_SPEC) -> None:
        _degree_given(negative_spec)
        self.negative_spec = negative_spec

    def _estimator(self, params: dict, seed: int):
        raise NotImplementedError

    def fit(
        self,
        views: Sequence[MaskedView],
        tables: Sequence[FeatureTable],
        *,
        context: F1TrainingContext,
        val_views: Sequence[MaskedView],
        val_tables: Sequence[FeatureTable],
        val_suite: Suite,
        val_negatives: CommittedNegatives,
    ) -> _PairFeatureBaseline:
        views, tables = list(views), list(tables)
        if not views:
            raise ValueError("fit needs at least one view")
        _check_alignment(views, tables, what="fit views")
        rank_views, rank_tables, rank_tasks = _tuning_inputs(
            views, tables, context=context, val_views=val_views,
            val_tables=val_tables, val_suite=val_suite,
            val_negatives=val_negatives, negative_spec=self.negative_spec,
        )
        type_encoding = fit_type_encoding(
            views,
            tables,
            positives_per_view=_TARGET_ENCODING_POSITIVES,
            context=context,
            negative_spec=self.negative_spec,
        )
        samples = [
            _candidate_negatives(
                view, spec=self.negative_spec, true_edges=context.true_edges
            ) for view in views
        ]
        seeds = tuple(int(s) for s in np.random.SeedSequence(DEFAULT_SPEC.master_seed).generate_state(3))
        grid_scores = np.empty((len(self.grid), len(seeds)), dtype=np.float64)
        with tempfile.TemporaryDirectory(prefix="flyarch-baseline-") as directory:
            ranking_blocks = _ranking_blocks(
                rank_views, rank_tables, rank_tasks, type_encoding,
                self.negative_spec, directory,
            )
            for seed_index, seed in enumerate(seeds):
                features, labels = _training_block(
                    views, tables, type_encoding, self.negative_spec,
                    context=context, fit_seed=seed, samples=samples,
                )
                for indices in self._grid_groups():
                    fit_index = max(indices, key=lambda i: self.grid[i].get("n_estimators", 0))
                    estimator = self._estimator(self.grid[fit_index], seed)
                    estimator.fit(features, labels)
                    for index in indices:
                        limit = self._iteration_limit(index)
                        grid_scores[index, seed_index] = _mean_mrr(
                            estimator, ranking_blocks, iteration_limit=limit
                        )
                    del estimator
                del features, labels

            ranking = grid_scores.mean(axis=1)
            champion_index = int(np.argmax(ranking))
            champion_params = dict(self.grid[champion_index])
            seed_models = []
            for seed in seeds:
                features, labels = _training_block(
                    views, tables, type_encoding, self.negative_spec,
                    context=context, fit_seed=seed, samples=samples,
                )
                params = dict(champion_params)
                if self._iteration_limit(champion_index) is not None:
                    params["n_estimators"] = 600
                estimator = self._estimator(params, seed)
                estimator.fit(features, labels)
                seed_models.append(estimator)
                del features, labels

        self.type_encoding = type_encoding
        self.grid_mrr = tuple(float(value) for value in ranking)
        self.grid_seed_mrr = tuple(tuple(float(value) for value in row) for row in grid_scores)
        self.champion_index = champion_index
        self.champion_params = champion_params
        self.training_view_ids = tuple(view.instance.instance_id for view in views)
        self.ranking_view_ids = tuple(view.instance.instance_id for view in rank_views)
        self.seeds = seeds
        self.seed_models = tuple(seed_models)
        self._champion = seed_models[0]
        return self

    def _grid_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple((i,) for i in range(len(self.grid)))

    def _iteration_limit(self, index: int) -> int | None:
        return None

    def score(
        self, view: VisibleView, table: FeatureTable, pairs: np.ndarray
    ) -> np.ndarray:
        _visible(view)
        champion = self._fitted()
        pairs = _pairs_array(pairs)
        if len(pairs) == 0:
            return np.zeros(0, dtype=np.float64)
        features = pair_features(
            view,
            pairs,
            table,
            type_encoding=self.type_encoding,
            negative_spec=self.negative_spec,
        )
        return _predict(champion, features, iteration_limit=self._iteration_limit(self.champion_index))

    def score_seed(
        self, view: VisibleView, table: FeatureTable, pairs: np.ndarray, *, seed: int
    ) -> np.ndarray:
        _visible(view)
        self._fitted()
        try:
            index = self.seeds.index(seed)
        except ValueError:
            raise ValueError(f"seed {seed} was not fitted; expected one of {self.seeds}") from None
        pairs = _pairs_array(pairs)
        if len(pairs) == 0:
            return np.zeros(0, dtype=np.float64)
        features = pair_features(
            view, pairs, table, type_encoding=self.type_encoding,
            negative_spec=self.negative_spec,
        )
        return _predict(self.seed_models[index], features,
                        iteration_limit=self._iteration_limit(self.champion_index))

    def _fitted(self):
        champion = getattr(self, "_champion", None)
        if champion is None or not hasattr(self, "type_encoding"):
            raise RuntimeError("fit must be called before score")
        return champion


class LogRegBaseline(_PairFeatureBaseline):
    """Logistic regression on the pair features, over ``LOGREG_GRID``."""

    grid = LOGREG_GRID

    def _estimator(self, params: dict, seed: int):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed, **params),
                ),
            ]
        )


class XGBoostBaseline(_PairFeatureBaseline):
    """XGBoost on the pair features, over ``XGBOOST_GRID``.

    Grid points that differ only in ``n_estimators`` share one 600-tree fit;
    the 200-tree point is scored from its first 200 iterations, which
    ``tests/test_baselines.py`` pins against a separate 200-tree fit.
    ``xgboost`` is imported inside :meth:`_estimator` so this module and the
    rest of the suite import and run without it installed.
    """

    grid = XGBOOST_GRID

    def _grid_groups(self) -> tuple[tuple[int, ...], ...]:
        groups = {}
        for index, params in enumerate(self.grid):
            key = (params["max_depth"], params["learning_rate"], params["subsample"])
            groups.setdefault(key, []).append(index)
        return tuple(tuple(indices) for indices in groups.values())

    def _iteration_limit(self, index: int) -> int | None:
        return 200 if self.grid[index]["n_estimators"] == 200 else None

    def _estimator(self, params: dict, seed: int):
        from xgboost import XGBClassifier

        return XGBClassifier(
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=1,  # single-threaded: histogram sums stay reproducible
            random_state=seed,
            **params,
        )


#: The pre-registered opponents, keyed by the names the gate uses.
BASELINES: dict[str, type] = {
    "configuration": ConfigurationBaseline,
    "logreg": LogRegBaseline,
    "xgboost": XGBoostBaseline,
}
