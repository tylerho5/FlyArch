"""Neuron-level R/T/S/H masks and the post-mask view used by scorers.

The post-mask API enforces four invariants: (1) every edge incident to a
masked neuron is hidden and edge partitions are exact; (2) masking retains
features by construction because no feature table/input appears in the API;
(3) all graph-derived values use only visible edges; and (4)
``degree_given_total`` is the sole complete-graph exception, read from the
``complete_graph`` a view names (its own graph unless the caller names
another).

``MaskedView`` is evaluator-owned: it holds the full graph, the hidden edges,
and ``both_masked``. A scorer receives ``MaskedView.model_view()`` instead, a
``VisibleView`` carrying only visible structure plus ``degree_given_total``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from . import negatives
from .regions import RegionGraph

__all__ = [
    "GAP_TAG",
    "MASK_REGIMES",
    "RANDOM_FRACTIONS",
    "SPATIAL_FRACTION",
    "SUITE_REGIMES",
    "MaskInstance",
    "MaskedView",
    "VisibleView",
    "apply_mask",
    "pool_mask",
    "random_mask",
    "side_mask",
    "side_values",
    "spatial_mask",
    "spatial_pool_mask",
    "typed_mask",
    "typed_pool_mask",
]

MASK_REGIMES = ("R", "T", "S")
SUITE_REGIMES = MASK_REGIMES + ("H",)
RANDOM_FRACTIONS = (0.01, 0.05, 0.10)
SPATIAL_FRACTION = 0.05

#: The tag a training-only gap mask carries, outside the pinned suite tag set
#: (``R``/``T``/``S``), so a training draw is never mistaken for a scored suite
#: instance and never reuses a committed digest key.
GAP_TAG = "gap"

_MASTER_SEED = negatives.NegativeSpec().master_seed


def _validate_fraction(fraction: float) -> float:
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return fraction


def _instance_id(graph: RegionGraph, tag: str, seed: int) -> str:
    return f"{graph.dataset}-{graph.region.lower()}-{tag}-s{int(seed):03d}"


@dataclass(frozen=True)
class MaskInstance:
    """One frozen selection of dense node indices to hide."""

    instance_id: str
    dataset: str
    region: str
    regime: str
    seed: int
    params: dict[str, Any]
    masked: np.ndarray

    def __post_init__(self) -> None:
        if self.regime not in SUITE_REGIMES:
            raise ValueError(f"regime must be one of {SUITE_REGIMES}, got {self.regime!r}")
        masked = np.asarray(self.masked)
        if masked.ndim != 1 or masked.dtype != np.dtype(np.int32):
            raise ValueError("masked must be a 1-D int32 array")
        if np.any(masked < 0) or np.any(masked[1:] < masked[:-1]):
            raise ValueError("masked indices must be nonnegative and sorted")
        if np.any(masked[1:] == masked[:-1]):
            raise ValueError("masked indices must be unique")
        masked = masked.copy()
        masked.setflags(write=False)
        object.__setattr__(self, "masked", masked)
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    @property
    def n_masked(self) -> int:
        return int(self.masked.size)

    def digest(self) -> str:
        """Stable content digest for suite pinning and reproducibility checks."""
        h = hashlib.sha256()
        h.update(self.instance_id.encode("utf-8"))
        h.update(np.ascontiguousarray(self.masked, dtype=np.int32).tobytes())
        return h.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "dataset": self.dataset,
            "region": self.region,
            "regime": self.regime,
            "seed": int(self.seed),
            "params": dict(self.params),
            "masked": self.masked.astype(np.int32).tolist(),
            "digest": self.digest(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaskInstance:
        raw_masked = np.asarray(data["masked"])
        if raw_masked.ndim != 1:
            raise ValueError("masked must be a 1-D array")
        if raw_masked.size == 0:
            raw_masked = np.asarray([], dtype=np.int32)
        elif not np.issubdtype(raw_masked.dtype, np.integer):
            raise ValueError("masked must contain integer indices")
        if raw_masked.ndim != 1 or np.any(raw_masked < 0) or np.any(raw_masked[1:] < raw_masked[:-1]):
            raise ValueError("masked indices must be 1-D, nonnegative, and sorted")
        if np.any(raw_masked > np.iinfo(np.int32).max):
            raise ValueError("masked indices exceed int32 range")
        instance = cls(
            instance_id=str(data["instance_id"]),
            dataset=str(data["dataset"]),
            region=str(data["region"]),
            regime=str(data["regime"]),
            seed=int(data["seed"]),
            params=dict(data.get("params", {})),
            masked=raw_masked.astype(np.int32),
        )
        if "digest" in data and data["digest"] != instance.digest():
            raise ValueError(
                f"mask {instance.instance_id!r} digest mismatch: "
                f"expected {data['digest']}, got {instance.digest()}"
            )
        return instance


def pool_mask(
    graph: RegionGraph,
    rng: np.random.Generator,
    *,
    fraction: float = 0.05,
    pool: np.ndarray | None = None,
    tag: str = GAP_TAG,
) -> MaskInstance:
    """Draw a fresh random ``fraction`` of ``pool`` to hide.

    ``pool`` is the candidate set and defaults to every node; fitting passes
    the complement of the F1-val masks, so a gap never hides a neuron those
    suites reserved. The count is ``round(fraction * pool.size)``, taken over
    the pool rather than the graph. Indices stay global either way: the
    instance addresses nodes densely against the graph's own ``node_ids``.

    One implementation serves both training paths: M1's per-epoch gaps via
    ``train.simulate_gaps`` and the baseline fit views. The default ``tag`` is
    training-only, so the draw cannot collide with a pinned suite instance.
    """
    pool = _pool(graph, pool)
    n_masked = round(fraction * pool.shape[0])
    masked = rng.choice(pool, size=n_masked, replace=False)
    return _gap_instance(graph, rng, "R", {"fraction": fraction}, masked, tag)


def typed_pool_mask(
    graph: RegionGraph,
    rng: np.random.Generator,
    *,
    types: np.ndarray,
    fraction: float = 0.05,
    pool: np.ndarray | None = None,
    tag: str = GAP_TAG,
) -> MaskInstance:
    """Hide whole cell types, in random order, until about ``fraction`` of ``pool`` is hidden.

    The training counterpart of :func:`typed_mask`: every pool neuron of a
    drawn type goes, so a masked neuron's same-type siblings are hidden with
    it, but several types are drawn so the gap has as many positives as a
    random one. The type that crosses the target is kept only when that lands
    closer to it, or when no type has been taken yet. ``types`` holds one
    integer id per node, with 0 for untyped neurons, which are never drawn.
    """
    pool = _pool(graph, pool)
    types = np.asarray(types, dtype=np.int64)
    if types.shape != (graph.n_nodes,):
        raise ValueError(f"types must have shape ({graph.n_nodes},), got {types.shape}")
    typed = pool[types[pool] > 0]
    if not typed.size:
        raise ValueError("a T gap needs typed neurons, and the pool has none")
    target = round(fraction * pool.shape[0])
    chosen: list[np.ndarray] = []
    hidden = 0
    for cell_type in rng.permutation(np.unique(types[typed])):
        members = typed[types[typed] == cell_type]
        if hidden + members.size >= target:
            if not chosen or hidden + members.size - target <= target - hidden:
                chosen.append(members)
            break
        chosen.append(members)
        hidden += members.size
    masked = np.concatenate(chosen)
    return _gap_instance(
        graph, rng, "T", {"fraction": fraction, "types": len(chosen)}, masked, tag
    )


def spatial_pool_mask(
    graph: RegionGraph,
    rng: np.random.Generator,
    *,
    centroids: np.ndarray,
    fraction: float = 0.05,
    pool: np.ndarray | None = None,
    located: np.ndarray | None = None,
    tag: str = GAP_TAG,
) -> MaskInstance:
    """Hide the ``fraction`` of ``pool`` nearest a randomly drawn pool neuron.

    The training counterpart of :func:`spatial_mask`, grown within the pool.
    ``centroids`` is one row per node in any coordinate frame; distances are
    Euclidean in that frame. ``located`` flags the nodes whose centroid is
    real: only those can be the centre or be hidden, and the gap is still
    ``fraction`` of the whole pool.
    """
    pool = _pool(graph, pool)
    centroids = np.asarray(centroids, dtype=float)
    if centroids.ndim != 2 or centroids.shape[0] != graph.n_nodes:
        raise ValueError(
            f"centroids must have one row per node ({graph.n_nodes}), got {centroids.shape}"
        )
    size = round(fraction * pool.shape[0])
    if located is not None:
        pool = pool[np.asarray(located, dtype=bool)[pool]]
    if pool.size < max(size, 1):
        raise ValueError(
            f"an S gap of {size} neurons needs as many located pool neurons, got {pool.size}"
        )
    centre = int(rng.choice(pool))
    distances = np.linalg.norm(centroids[pool] - centroids[centre], axis=1)
    order = np.lexsort((pool, distances))
    masked = pool[order[:size]]
    return _gap_instance(
        graph, rng, "S", {"fraction": fraction, "centre_node": centre}, masked, tag
    )


def _pool(graph: RegionGraph, pool: np.ndarray | None) -> np.ndarray:
    if pool is None:
        return np.arange(graph.n_nodes, dtype=np.int64)
    return np.asarray(pool, dtype=np.int64)


def _gap_instance(
    graph: RegionGraph,
    rng: np.random.Generator,
    regime: str,
    params: dict[str, Any],
    masked: np.ndarray,
    tag: str,
) -> MaskInstance:
    seed = int(rng.integers(1000))
    return MaskInstance(
        instance_id=f"{graph.dataset}-{graph.region.lower()}-{tag}-s{seed:03d}",
        dataset=graph.dataset,
        region=graph.region,
        regime=regime,
        seed=seed,
        params=params,
        masked=np.sort(masked).astype(np.int32),
    )


def random_mask(
    graph: RegionGraph, *, fraction: float, seed: int, pool: np.ndarray | None = None
) -> MaskInstance:
    """Hide a random ``fraction`` of the graph, or of ``pool`` (dense indices) when given."""
    fraction = _validate_fraction(fraction)
    if abs(fraction * 100 - round(fraction * 100)) > 1e-9:
        raise ValueError(f"fraction must be a whole percent so the R tag is lossless, got {fraction}")
    instance_id = _instance_id(graph, f"R{round(fraction * 100):02d}", seed)
    rng = negatives._derive_rng(_MASTER_SEED, instance_id)
    if pool is None:
        masked = np.sort(rng.choice(graph.n_nodes, round(fraction * graph.n_nodes), replace=False))
    else:
        pool = np.sort(np.asarray(pool, dtype=np.int64))
        masked = np.sort(rng.choice(pool, round(fraction * pool.size), replace=False))
    return MaskInstance(instance_id, graph.dataset, graph.region, "R", int(seed),
                        {"fraction": fraction}, masked.astype(np.int32))


def side_values(graph: RegionGraph, soma_sides: pd.Series) -> np.ndarray:
    """Validated soma sides in node order: ``"L"``, ``"R"``, or ``""`` for unknown."""
    if not isinstance(soma_sides, pd.Series):
        raise ValueError("soma_sides must be a Series indexed by root id")
    if not soma_sides.index.is_unique:
        raise ValueError("soma_sides index has duplicate root ids")
    if not pd.Index(graph.node_ids).isin(soma_sides.index).all():
        raise ValueError("soma_sides is missing one or more graph root ids")
    values = soma_sides.reindex(graph.node_ids)
    if not (values.isna() | values.isin(("L", "R"))).all():
        raise ValueError("soma_sides values must be L, R, or null")
    return values.astype(object).fillna("").to_numpy()


def side_mask(
    graph: RegionGraph, soma_sides: pd.Series, *, side: str, seed: int,
    pool: np.ndarray | None = None,
) -> MaskInstance:
    """Hide every neuron on one soma side (within ``pool`` if given); unknown sides remain visible."""
    if side not in ("L", "R"):
        raise ValueError("side must be L or R")
    values = side_values(graph, soma_sides)
    on_side = values == side
    if pool is not None:
        on_side &= np.isin(np.arange(graph.n_nodes), pool)
    masked = np.flatnonzero(on_side).astype(np.int32)
    if not masked.size:
        raise ValueError(f"no graph node carries soma side {side}")
    return MaskInstance(
        _instance_id(graph, f"H{side}", seed), graph.dataset, graph.region,
        "H", int(seed), {"side": side}, masked,
    )


def typed_mask(
    graph: RegionGraph, types: pd.Series, *, seed: int, exclude: Iterable = ()
) -> MaskInstance:
    """Hide every graph neuron belonging to one uniformly chosen cell type.

    Unmatched neurons (``NaN`` type, e.g. F2's out-of-vocabulary population)
    are never drawn and never masked; the index must still cover every node.

    ``exclude`` drops cell types from the candidate pool before the draw, so a
    suite can hold typed masks that never repeat a type (F2-confirm
    draws types disjoint from F2-dev). It defaults to drawing from every type.
    """
    if not isinstance(types, pd.Series):
        types = pd.Series(types)
    if not types.index.is_unique:
        raise ValueError("types index has duplicate root ids")
    if not pd.Index(graph.node_ids).isin(types.index).all():
        raise ValueError("types is missing one or more graph root ids")
    values = types.reindex(graph.node_ids)
    present = np.asarray(sorted(values.dropna().unique().tolist()), dtype=object)
    if present.size == 0:
        raise ValueError("types has no values for graph nodes")
    exclude_set = set(exclude)
    if exclude_set:
        present = np.asarray([t for t in present if t not in exclude_set], dtype=object)
        if present.size == 0:
            raise ValueError(
                "no cell types remain to draw from after applying exclude; "
                "the graph has too few distinct types for a disjoint suite"
            )
    instance_id = _instance_id(graph, "T", seed)
    chosen = present[
        int(negatives._derive_rng(_MASTER_SEED, instance_id).integers(present.size))
    ]
    masked = np.flatnonzero(values.to_numpy() == chosen).astype(np.int32)
    params = {"cell_type": chosen.item() if hasattr(chosen, "item") else chosen}
    return MaskInstance(instance_id, graph.dataset, graph.region, "T", int(seed), params, masked)


def spatial_mask(
    graph: RegionGraph,
    centroids: np.ndarray,
    *,
    seed: int,
    fraction: float = SPATIAL_FRACTION,
    pool: np.ndarray | None = None,
) -> MaskInstance:
    """Hide a contiguous ball grown around a uniformly selected node.

    With ``pool`` (dense indices), the centre is drawn from it and the ball is
    sized and counted over pool nodes only; other nodes inside it stay visible.
    """
    fraction = _validate_fraction(fraction)
    if fraction != SPATIAL_FRACTION:
        raise ValueError(
            f"spatial fraction must be the registered {SPATIAL_FRACTION} (the S05 tag), got {fraction}"
        )
    centroids = np.asarray(centroids, dtype=float)
    if centroids.ndim != 2 or centroids.shape != (graph.n_nodes, 3):
        raise ValueError(f"centroids must have shape ({graph.n_nodes}, 3), got {centroids.shape}")
    if not np.all(np.isfinite(centroids)):
        raise ValueError("centroids must be finite")
    instance_id = _instance_id(graph, "S05", seed)
    rng = negatives._derive_rng(_MASTER_SEED, instance_id)
    if pool is None:
        pool = np.arange(graph.n_nodes)
        centre_node = int(rng.integers(graph.n_nodes))
    else:
        pool = np.sort(np.asarray(pool, dtype=np.int64))
        centre_node = int(pool[rng.integers(pool.size)])
    distances = np.linalg.norm(centroids[pool] - centroids[centre_node], axis=1)
    order = np.lexsort((pool, distances))
    wanted = max(1, round(fraction * pool.size))
    radius = float(distances[order[wanted - 1]])
    masked = pool[distances <= radius].astype(np.int32)
    return MaskInstance(instance_id, graph.dataset, graph.region, "S", int(seed),
                        {"fraction": fraction, "centre_node": centre_node, "radius": radius}, masked)


def _in_strength(n_nodes: int, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(out, edges[:, 1], weights)
    return out


def _out_strength(n_nodes: int, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(out, edges[:, 0], weights)
    return out


def _input_fraction(edges: np.ndarray, weights: np.ndarray, in_strength: np.ndarray) -> np.ndarray:
    fractions = np.zeros(edges.shape[0], dtype=float)
    if edges.size == 0:
        return fractions
    targets = edges[:, 1]
    nonzero = in_strength[targets] != 0
    fractions[nonzero] = weights[nonzero] / in_strength[targets[nonzero]]
    return fractions


@dataclass(frozen=True)
class MaskedView:
    graph: RegionGraph
    instance: MaskInstance
    visible_edges: np.ndarray
    visible_weights: np.ndarray
    hidden_edges: np.ndarray
    hidden_weights: np.ndarray
    is_masked: np.ndarray
    #: The graph whose complete in+out strength :meth:`degree_given_total`
    #: grants at masked nodes. ``None`` means this view's own graph. Fitting
    #: passes the unpruned graph, so a training row and a scored row report the
    #: same quantity (the degree-given total).
    complete_graph: RegionGraph | None = None

    def visible_in_strength(self) -> np.ndarray:
        return _in_strength(self.graph.n_nodes, self.visible_edges, self.visible_weights)

    def visible_out_strength(self) -> np.ndarray:
        return _out_strength(self.graph.n_nodes, self.visible_edges, self.visible_weights)

    def visible_input_fraction(self) -> np.ndarray:
        return _input_fraction(self.visible_edges, self.visible_weights, self.visible_in_strength())

    def degree_given_total(self) -> np.ndarray:
        complete = self.complete_graph if self.complete_graph is not None else self.graph
        total = np.zeros(self.graph.n_nodes, dtype=np.int64)
        granted = complete.in_strength() + complete.out_strength()
        total[self.is_masked] = granted[self.is_masked]
        return total

    def both_masked(self) -> np.ndarray:
        flags = self.is_masked
        return flags[self.hidden_edges[:, 0]] & flags[self.hidden_edges[:, 1]]

    def model_view(self) -> VisibleView:
        """The leak-free object a scorer receives; MaskedView stays evaluator-owned."""
        return VisibleView(
            n_nodes=self.graph.n_nodes,
            visible_edges=self.visible_edges,
            visible_weights=self.visible_weights,
            is_masked=self.is_masked,
            degree_given=self.degree_given_total(),
        )


@dataclass(frozen=True)
class VisibleView:
    """The model-facing projection of a MaskedView.

    Carries the post-mask visible graph, which nodes are held out, and the one
    complete-graph value the registered rules grant every model (``degree_given_total``).
    It never holds the full graph, the hidden edges/weights, or ``both_masked``,
    so a scorer handed this object cannot read the held-out positives.
    """

    n_nodes: int
    visible_edges: np.ndarray
    visible_weights: np.ndarray
    is_masked: np.ndarray
    degree_given: np.ndarray

    def visible_in_strength(self) -> np.ndarray:
        return _in_strength(self.n_nodes, self.visible_edges, self.visible_weights)

    def visible_out_strength(self) -> np.ndarray:
        return _out_strength(self.n_nodes, self.visible_edges, self.visible_weights)

    def visible_input_fraction(self) -> np.ndarray:
        return _input_fraction(self.visible_edges, self.visible_weights, self.visible_in_strength())

    def degree_given_total(self) -> np.ndarray:
        return self.degree_given


def apply_mask(
    graph: RegionGraph,
    instance: MaskInstance,
    *,
    complete_graph: RegionGraph | None = None,
) -> MaskedView:
    """Mask ``instance`` against ``graph``.

    ``complete_graph`` names the graph the granted degree-given total comes
    from, so a view over a pruned training copy still reports the unpruned
    value a scored view reports. It must share ``graph``'s node ids.
    """
    if instance.dataset != graph.dataset or instance.region != graph.region:
        raise ValueError("mask instance does not belong to this graph")
    if np.any(instance.masked < 0) or np.any(instance.masked >= graph.n_nodes):
        raise ValueError("mask contains a node index outside the graph")
    if complete_graph is not None and (
        complete_graph.n_nodes != graph.n_nodes
        or not np.array_equal(complete_graph.node_ids, graph.node_ids)
    ):
        raise ValueError("complete_graph must share the view graph's node ids")
    flags = np.zeros(graph.n_nodes, dtype=bool)
    flags[instance.masked] = True
    hidden = flags[graph.edges[:, 0]] | flags[graph.edges[:, 1]]
    flags.setflags(write=False)
    return MaskedView(
        graph=graph,
        instance=instance,
        visible_edges=graph.edges[~hidden],
        visible_weights=graph.weights[~hidden],
        hidden_edges=graph.edges[hidden],
        hidden_weights=graph.weights[hidden],
        is_masked=flags,
        complete_graph=complete_graph,
    )
