"""Frozen, digest-pinned mask suites.

A suite is a fixed set of R/T/S or exploratory H instances for one graph, stored as
graph-free JSON (root ids, never dense indices, since dense indices are
meaningless without a graph's node ordering). ``graph_digest`` pins the suite
to the exact graph it was built from, so a regenerated graph is caught by
:meth:`Suite.validate` rather than silently scored against the wrong nodes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .masks import MaskedView, MaskInstance, apply_mask, random_mask, side_mask, spatial_mask, typed_mask
from .negatives import DEFAULT_SPEC, NegativeSample, NegativeSpec, pack_edges, sample_negatives
from .regions import RegionGraph

__all__ = [
    "F1_POOL_SUITES",
    "MASTER_SEED",
    "SIDE_SUITES",
    "SUITE_SIZES",
    "Suite",
    "SuiteInstance",
    "build_suite",
    "f1_train_types",
    "suite_negatives",
    "view_negatives",
]

#: Instances per named suite, including the exploratory F2 side holdouts.
SUITE_SIZES = {"f1_val": 5, "f2_dev": 15, "f2_confirm": 20, "f2_side": 2, "f3_eval": 15}  # f2_side: one per SIDE_SUITES side; f3_eval: f2_dev's mix

#: Recorded on every suite for provenance. Single source of truth: masks and
#: negatives both derive their seeding from this same value, so binding it here
#: to DEFAULT_SPEC keeps the recorded provenance from drifting off the seed the
#: masks were actually built with (it is not threaded into per-instance seeding;
#: see build_suite: the per-instance ``seed`` arg is offset + i, always).
MASTER_SEED = DEFAULT_SPEC.master_seed

#: The preregistered dataset each suite name belongs to: ``f1_val``
#: calibrates on F1, the two F2 suites score on F2. Name and dataset are not
#: independent -- a digest-valid ``f1_val`` artefact built from the F2 graph
#: would calibrate on the wrong individual.
_SUITE_DATASETS = {"f1_val": "f1", "f2_dev": "f2", "f2_confirm": "f2", "f2_side": "f2", "f3_eval": "f3"}

#: Disjoint seed ranges per suite, so dev and confirm instances never collide.
_OFFSETS = {"f1_val": 0, "f2_dev": 1000, "f2_confirm": 2000, "f2_side": 4000, "f3_eval": 3000}

#: Tag cycle every suite spans, in order; i % 5 selects the generator.
_TAGS = ("R01", "R05", "R10", "T", "S05")
_RANDOM_FRACTIONS = {"R01": 0.01, "R05": 0.05, "R10": 0.10}

#: Exploratory side-holdout suites: one H instance per listed soma side, in
#: order, at ``offset + i``. Only these suites may hold H, and they hold
#: nothing else.
SIDE_SUITES = {"f2_side": ("L", "R")}

#: Suites whose typed masks must avoid another suite's, so a confirm instance
#: never duplicates a dev instance the models iterated on.
_DISJOINT_FROM = {"f2_confirm": "f2_dev"}

#: Suites whose typed masks draw only the cell types the region's expert trains
#: on (``f1_train_types``), so a T instance hides a trained type.
F1_POOL_SUITES = frozenset({"f2_dev", "f2_confirm"})


def _assert_disjoint_seed_ranges() -> None:
    """Guard the disjoint-seed invariant against future SUITE_SIZES growth.

    Disjoint seeds are what keep dev and confirm from sharing an instance_id
    (and so byte-identical masks and negatives). The offsets give each suite
    1000 seeds of room; this fails loudly at import if a size ever eats into
    the next suite's range.
    """
    spans = {n: range(_OFFSETS[n], _OFFSETS[n] + SUITE_SIZES[n]) for n in SUITE_SIZES}
    names = list(spans)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ra, rb = spans[names[a]], spans[names[b]]
            if max(ra.start, rb.start) < min(ra.stop, rb.stop):
                raise AssertionError(
                    f"suite seed ranges overlap: {names[a]} {ra} and {names[b]} {rb}"
                )


_assert_disjoint_seed_ranges()


@dataclass(frozen=True)
class SuiteInstance:
    """One mask instance's graph-free record: root ids, not dense indices."""

    instance_id: str
    regime: str
    seed: int
    params: dict[str, Any]
    masked_root_ids: np.ndarray  # (M,) int64, original root ids

    def __post_init__(self) -> None:
        root_ids = np.asarray(self.masked_root_ids, dtype=np.int64)
        root_ids.setflags(write=False)
        object.__setattr__(self, "masked_root_ids", root_ids)
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "regime": self.regime,
            "seed": int(self.seed),
            "params": dict(self.params),
            "masked_root_ids": [int(x) for x in self.masked_root_ids],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuiteInstance:
        return cls(
            instance_id=str(data["instance_id"]),
            regime=str(data["regime"]),
            seed=int(data["seed"]),
            params=dict(data.get("params", {})),
            masked_root_ids=np.asarray(data["masked_root_ids"], dtype=np.int64),
        )


@dataclass(frozen=True)
class Suite:
    """A named, frozen set of mask instances for one graph."""

    name: str
    dataset: str
    region: str
    graph_digest: str
    threshold: int
    master_seed: int
    instances: tuple[SuiteInstance, ...]
    #: Cell types no instance masks (OL's lamina-dependent types). Serialized
    #: only when non-empty, so every other suite keeps its digest.
    mask_exclude_types: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        extra = {"mask_exclude_types": list(self.mask_exclude_types)} if self.mask_exclude_types else {}
        return {
            "name": self.name,
            "dataset": self.dataset,
            "region": self.region,
            "graph_digest": self.graph_digest,
            "threshold": int(self.threshold),
            "master_seed": int(self.master_seed),
            "instances": [inst.to_dict() for inst in self.instances],
            **extra,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Suite:
        return cls(
            name=str(data["name"]),
            dataset=str(data["dataset"]),
            region=str(data["region"]),
            graph_digest=str(data["graph_digest"]),
            threshold=int(data["threshold"]),
            master_seed=int(data["master_seed"]),
            instances=tuple(SuiteInstance.from_dict(d) for d in data["instances"]),
            mask_exclude_types=tuple(sorted(data.get("mask_exclude_types", ()))),
        )

    def digest(self) -> str:
        """Sha256 over the canonical (sorted-key) JSON. Graph-free."""
        payload = json.dumps(self.to_json(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json()) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> Suite:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self, graph: RegionGraph) -> None:
        """Raise if this suite was not built from ``graph``."""
        registered = _SUITE_DATASETS.get(self.name)
        if registered is None:
            raise ValueError(f"unknown suite name {self.name!r}")
        if registered != self.dataset:
            raise ValueError(
                f"suite {self.name!r} is registered to dataset {registered!r}, "
                f"but records dataset {self.dataset!r}"
            )
        if self.dataset != graph.dataset:
            raise ValueError(
                f"suite {self.name!r} dataset {self.dataset!r} does not match "
                f"graph dataset {graph.dataset!r}"
            )
        if self.region != graph.region:
            raise ValueError(
                f"suite {self.name!r} region {self.region!r} does not match "
                f"graph region {graph.region!r}"
            )
        if self.graph_digest != graph.digest():
            raise ValueError(
                f"suite {self.name!r} graph digest mismatch: expected "
                f"{self.graph_digest}, got {graph.digest()}"
            )
        if self.threshold != graph.threshold:
            raise ValueError(
                f"suite {self.name!r} threshold {self.threshold} does not match "
                f"graph threshold {graph.threshold}"
            )
        sides = SIDE_SUITES.get(self.name)
        if sides is not None:
            expected = [("H", _OFFSETS[self.name] + i, {"side": s}) for i, s in enumerate(sides)]
            actual = [(inst.regime, inst.seed, dict(inst.params)) for inst in self.instances]
            if actual != expected:
                raise ValueError(
                    f"suite {self.name!r} must hold only H instances, exactly "
                    f"{expected}; got {actual}"
                )
        elif any(inst.regime == "H" for inst in self.instances):
            raise ValueError(f"H instances belong only to side suites {sorted(SIDE_SUITES)}")
        for inst in self.instances:
            graph.index_of(inst.masked_root_ids)  # raises KeyError on a stranger root id

    def resolve(self, graph: RegionGraph) -> tuple[MaskInstance, ...]:
        self.validate(graph)
        resolved = []
        for inst in self.instances:
            masked = np.sort(graph.index_of(inst.masked_root_ids)).astype(np.int32)
            resolved.append(
                MaskInstance(
                    instance_id=inst.instance_id,
                    dataset=self.dataset,
                    region=self.region,
                    regime=inst.regime,
                    seed=inst.seed,
                    params=dict(inst.params),
                    masked=masked,
                )
            )
        return tuple(resolved)


def _typed_types(graph: RegionGraph, types, name: str, mask_exclude_types=()) -> set:
    """The cell types the named suite's T instances draw, for cross-suite exclusion.

    Replays the suite's typed draws (which depend only on the graph, ``types``,
    and the pinned per-instance seeds) so a disjoint sibling can be built from
    one CLI invocation without sharing state with the sibling's build. It
    replays with the same ``mask_exclude_types`` as the real build.
    """
    offset = _OFFSETS[name]
    return {
        typed_mask(graph, types, seed=offset + i, exclude=mask_exclude_types).params["cell_type"]
        for i in range(SUITE_SIZES[name])
        if _TAGS[i % len(_TAGS)] == "T"
    }


def f1_train_types(graph: RegionGraph, types, val_suite: Suite) -> tuple:
    """The cell types the region's expert trains on, sorted.

    F1-train drops every edge that touches a neuron ``val_suite`` (the region's
    F1-val) masks; a type counts when one of its neurons keeps an edge. This is
    the set ``train.train._trained_type_ids`` leaves out of OOV.
    """
    val_suite.validate(graph)
    hidden = np.isin(graph.node_ids, np.concatenate([i.masked_root_ids for i in val_suite.instances]))
    kept = graph.edges[~(hidden[graph.edges[:, 0]] | hidden[graph.edges[:, 1]])]
    nodes = np.unique(kept.reshape(-1))
    return tuple(sorted(types.reindex(graph.node_ids).iloc[nodes].dropna().unique()))


def build_suite(
    graph: RegionGraph,
    *,
    name: str,
    types=None,
    centroids=None,
    soma_sides=None,
    master_seed: int = MASTER_SEED,
    mask_exclude_types=(),
    draw_types=None,
) -> Suite:
    """Build one named suite's instances at ``seed = offset + i`` directly.

    ``types``/``centroids`` are required once the cycle reaches a T/S05 tag;
    passing ``None`` raises at that point rather than up front, since a
    suite that happens not to need one of them should not be blocked on it.
    A ``SIDE_SUITES`` suite (``f2_side``) requires only ``soma_sides`` and
    holds out each listed side in turn.

    A suite in ``_DISJOINT_FROM`` (F2-confirm) draws its typed masks without
    replacement and disjoint from the sibling suite's (F2-dev), so a confirm
    instance never duplicates a dev instance the models iterated on.

    ``mask_exclude_types`` names cell types no instance masks (OL's
    lamina-dependent types); ``types`` is then required. R and S size over the
    remaining nodes, T never draws those types and H hides only the others.
    Untyped nodes stay eligible.

    ``draw_types`` limits the types T may draw; R, S and H ignore it. Committed
    ``F1_POOL_SUITES`` pass ``f1_train_types`` of the region's F1 build.
    """
    if name not in SUITE_SIZES:
        raise ValueError(f"unknown suite {name!r}; known: {sorted(SUITE_SIZES)}")
    registered = _SUITE_DATASETS[name]
    if graph.dataset != registered:
        raise ValueError(
            f"suite {name!r} is registered to dataset {registered!r}, got graph "
            f"dataset {graph.dataset!r}"
        )
    # master_seed is pinned: masks/negatives always derive from
    # masks._MASTER_SEED/DEFAULT_SPEC, so a different value here would be
    # recorded provenance that doesn't match how the suite was actually built.
    if master_seed != MASTER_SEED:
        raise ValueError(f"master_seed is pinned to {MASTER_SEED}, got {master_seed}")
    sides = SIDE_SUITES.get(name)
    if sides is not None and soma_sides is None:
        raise ValueError(f"suite {name!r}: soma_sides is required")

    banned = set(mask_exclude_types)
    pool = None
    if banned:
        if types is None:
            raise ValueError(f"suite {name!r}: types is required with mask_exclude_types")
        pool = np.flatnonzero(~types.reindex(graph.node_ids).isin(banned).to_numpy())
        if not pool.size:
            raise ValueError(f"suite {name!r}: mask_exclude_types leaves no node to mask")

    undrawable: set = set()
    if draw_types is not None and types is not None:
        undrawable = set(types.reindex(graph.node_ids).dropna().unique()) - set(draw_types)

    disjoint = name in _DISJOINT_FROM
    excluded_types: set = (
        _typed_types(graph, types, _DISJOINT_FROM[name], banned | undrawable)
        if disjoint and types is not None
        else set()
    )

    offset = _OFFSETS[name]
    instances = []
    for i in range(SUITE_SIZES[name]):
        seed = offset + i
        tag = None if sides is not None else _TAGS[i % len(_TAGS)]
        if sides is not None:
            mask = side_mask(graph, soma_sides, side=sides[i], seed=seed, pool=pool)
        elif tag == "T":
            if types is None:
                raise ValueError(f"suite {name!r}: types is required to build a T instance")
            mask = typed_mask(graph, types, seed=seed, exclude=(excluded_types if disjoint else set()) | banned | undrawable)
            if disjoint:
                excluded_types = excluded_types | {mask.params["cell_type"]}
        elif tag == "S05":
            if centroids is None:
                raise ValueError(f"suite {name!r}: centroids is required to build an S05 instance")
            mask = spatial_mask(graph, centroids, seed=seed, fraction=0.05, pool=pool)
        else:
            mask = random_mask(graph, fraction=_RANDOM_FRACTIONS[tag], seed=seed, pool=pool)

        instances.append(
            SuiteInstance(
                instance_id=mask.instance_id,
                regime=mask.regime,
                seed=seed,
                params=dict(mask.params),
                masked_root_ids=graph.node_ids[mask.masked],
            )
        )

    return Suite(
        name=name,
        dataset=graph.dataset,
        region=graph.region,
        graph_digest=graph.digest(),
        threshold=graph.threshold,
        master_seed=master_seed,
        instances=tuple(instances),
        mask_exclude_types=tuple(sorted(banned)),
    )


def view_negatives(
    view: MaskedView,
    *,
    spec: NegativeSpec = DEFAULT_SPEC,
    positives: np.ndarray | None = None,
    neuropil: str | None = None,
    pool: np.ndarray | None = None,
    true_edges: set[int] | None = None,
) -> NegativeSample:
    """Sample negatives for one masked view using the suite construction."""
    graph = view.graph
    return sample_negatives(
        neuropil=view.instance.instance_id if neuropil is None else neuropil,
        positives=(
            np.asarray(view.hidden_edges, dtype=np.int64)
            if positives is None
            else np.asarray(positives, dtype=np.int64)
        ),
        pool=np.arange(graph.n_nodes, dtype=np.int64) if pool is None else pool,
        in_degree=view.visible_in_strength(),
        out_degree=view.visible_out_strength(),
        true_edges=pack_edges(graph.edges) if true_edges is None else true_edges,
        spec=spec,
        masked_nodes=view.is_masked,
    )


def suite_negatives(
    graph: RegionGraph, suite: Suite, *, spec: NegativeSpec = DEFAULT_SPEC
) -> dict[str, NegativeSample]:
    """One personalised negative sample per instance, keyed by instance_id.

    Degrees come from the post-mask visible graph, never the full
    graph, so a masked neuron's true degree never reaches the sampler. Every
    negative keeps one of the positive's masked endpoints, so no decoy can be
    discarded for containing no masked neuron (see ``masked_nodes``).
    """
    result = {}
    pool = np.arange(graph.n_nodes, dtype=np.int64)
    true_edges = pack_edges(graph.edges)
    for instance in suite.resolve(graph):
        view = apply_mask(graph, instance)
        result[instance.instance_id] = view_negatives(
            view, spec=spec, pool=pool, true_edges=true_edges
        )
    return result
