"""Phase 3 union graphs and the boundary census (Phase 3 boundary suites).

``E_r`` is region r's graph, exactly as ``build_region_graph`` builds it. A union
``U`` sums raw counts over its members' labels and thresholds once. A boundary edge
is in ``U`` but in no ``E_r``; a shared neuron is a node of two or more ``E_r``.
Both are audit strata, never model inputs.

:func:`pair_counts` reduces a connections table once to per-pair, per-region
synapse totals, so every graph and census over any region subset reads the same
table. This module lives outside ``train/train.py``'s ``_CODE_FILES`` and builds
``RegionGraph`` objects directly, so adding it changes no Phase 2 run ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from src.regions import DEFAULT_THRESHOLD, REGIONS, ColumnMap, RegionGraph, neuropil_to_region, region_labels

__all__ = ["CODES", "UNIONS", "PairCounts", "boundary_flags", "census", "is_union", "pair_counts", "support_within",
           "union_graph", "union_labels", "union_members", "union_name"]

#: Column order of :attr:`PairCounts.counts`; one more column holds OCG and UNASGD.
CODES = tuple(sorted(REGIONS))

#: Pinned label sets. Any other set is named by its sorted codes joined with ``+``.
UNIONS: dict[str, tuple[str, ...]] = {
    "UNION12": CODES,
    "UNION11": tuple(code for code in CODES if code != "OL"),
}


@dataclass(frozen=True)
class PairCounts:
    node_ids: np.ndarray  # (N,) sorted root ids
    pre: np.ndarray  # (P,) dense indices into node_ids; pairs sorted by (pre, post)
    post: np.ndarray
    counts: np.ndarray  # (P, len(CODES) + 1) int32 synapse totals per region, then excluded labels


def pair_counts(connections: pd.DataFrame, columns: ColumnMap | None = None) -> PairCounts:
    """Per directed pair, the synapse total over each region's labels and over OCG/UNASGD."""
    cols = columns or ColumnMap.resolve(connections.columns)
    groups = {label: (CODES.index(code) if code else len(CODES))
              for label, code in neuropil_to_region().items()}
    labels = connections[cols.neuropil].astype("category")
    if labels.isna().any():  # a null's category code is -1, which would index the last label's group
        raise ValueError(f"{int(labels.isna().sum())} connection rows have no neuropil label")
    unknown = sorted(set(map(str, labels.cat.categories)) - set(groups))
    if unknown:
        raise ValueError(f"labels outside the registry: {unknown}")
    group = np.array([groups[str(c)] for c in labels.cat.categories], dtype=np.int64)[labels.cat.codes.to_numpy()]
    node_ids, inverse = np.unique(
        np.concatenate([connections[cols.pre].to_numpy(np.int64), connections[cols.post].to_numpy(np.int64)]),
        return_inverse=True,
    )
    n_rows, n_nodes = len(connections), len(node_ids)
    keys, pair = np.unique(inverse[:n_rows] * n_nodes + inverse[n_rows:], return_inverse=True)
    width = len(CODES) + 1
    flat = pd.Series(connections[cols.syn_count].to_numpy(np.int64)).groupby(pair * width + group).sum()
    counts = np.zeros(len(keys) * width, dtype=np.int32)
    counts[flat.index.to_numpy()] = flat.to_numpy()
    return PairCounts(node_ids, keys // n_nodes, keys % n_nodes, counts.reshape(len(keys), width))


def _check(regions) -> tuple[str, ...]:
    regions = tuple(sorted(set(regions)))
    if not regions or set(regions) - set(REGIONS):
        raise ValueError(f"regions must be registry codes, got {regions}")
    return regions


def union_name(regions) -> str:
    """The pinned name for this set, else its sorted members joined by ``+``."""
    members = _check(regions)
    return next((name for name, union in UNIONS.items() if union == members), "+".join(members))


def union_members(name: str) -> tuple[str, ...]:
    """Inverse of :func:`union_name`. A union has at least two members, so its name never
    equals a region code and its digest never collides with a region's."""
    members = UNIONS.get(name) or _check(name.split("+"))
    if len(members) < 2 or union_name(members) != name:
        raise ValueError(f"{name!r} is not a union name (two or more sorted codes, or one of {sorted(UNIONS)})")
    return members


def is_union(code: str) -> bool:
    """Whether ``code`` names a union (a pinned name, or two or more sorted codes joined by ``+``)."""
    try:
        union_members(code)
    except ValueError:
        return False
    return True


def union_labels(name: str) -> tuple[str, ...]:
    """Every v783 label of the union's members, in member order."""
    return tuple(label for code in union_members(name) for label in region_labels(code))


def _graph(pc: PairCounts, keep: np.ndarray, weights: np.ndarray, region: str, dataset: str,
           threshold: int) -> RegionGraph:
    """The same graph ``build_region_graph`` returns for these pairs and totals."""
    source, target = pc.pre[keep], pc.post[keep]
    nodes = np.unique(np.concatenate([source, target]))
    edges = np.stack([np.searchsorted(nodes, source), np.searchsorted(nodes, target)], axis=1)
    return RegionGraph(dataset, region, threshold, pc.node_ids[nodes], edges.astype(np.int32),
                       weights[keep].astype(np.int64))


def _strata(pc: PairCounts, regions: tuple, threshold: int, regional_thresholds=None):
    """Per pair: member-region totals, regional and union membership, and the boundary flag.

    ``regional_thresholds`` ({code: t}) sets each region's own graph threshold (F2, where
    every region calibrates its own); by default every region uses ``threshold``.
    """
    c = pc.counts[:, [CODES.index(code) for code in regions]]
    own = np.array([(regional_thresholds or {}).get(code, threshold) for code in regions])
    regional = c >= own
    total = c.sum(axis=1, dtype=np.int64)
    union = total >= threshold
    return c, regional, total, union, union & ~regional.any(axis=1)


def union_graph(pc: PairCounts, name: str, *, dataset: str = "f1",
                threshold: int = DEFAULT_THRESHOLD) -> RegionGraph:
    """The union ``name``: member totals summed per pair, thresholded once."""
    _, _, total, union, _ = _strata(pc, union_members(name), threshold)
    return _graph(pc, union, total, name, dataset, threshold)


def with_isolated(graph: RegionGraph, node_ids) -> RegionGraph:
    """``graph`` plus each of ``node_ids`` it lacks, as a node with no edge."""
    ids = np.union1d(graph.node_ids, np.asarray(node_ids, dtype=np.int64))
    edges = np.searchsorted(ids, graph.node_ids)[graph.edges]
    return RegionGraph(graph.dataset, graph.region, graph.threshold, ids, edges.astype(np.int32), graph.weights)


def boundary_flags(graph: RegionGraph, pc: PairCounts, regional_thresholds=None) -> np.ndarray:
    """Per edge of the union ``graph``, whether it is a boundary edge (in no member region's graph).

    The union is rebuilt from ``pc`` and must carry ``graph``'s digest, so the flags
    are aligned to ``graph.edges`` and come from the same counts as the census. On F2,
    pass every member's frozen ``f2_threshold`` as ``regional_thresholds``: a boundary
    edge is then absent from every F2 region graph at that region's own threshold.
    """
    members = union_members(graph.region)
    if graph.dataset != "f1":
        missing = sorted(set(members) - set(regional_thresholds or {}))
        if missing:
            raise ValueError(f"{graph.dataset} boundary edges need every member's own threshold; missing {missing}")
    _, _, total, union, boundary = _strata(pc, members, graph.threshold, regional_thresholds)
    rebuilt = _graph(pc, union, total, graph.region, graph.dataset, graph.threshold)
    if rebuilt.digest() != graph.digest():
        raise ValueError(f"counts rebuild {graph.region} with digest {rebuilt.digest()}, not {graph.digest()}")
    return boundary[union]


def support_within(graph: RegionGraph, pc: PairCounts, regions) -> np.ndarray:
    """Per edge of the union ``graph``, whether every region contributing a synapse is in ``regions``."""
    members = union_members(graph.region)
    c, _, total, union, _ = _strata(pc, members, graph.threshold)
    if _graph(pc, union, total, graph.region, graph.dataset, graph.threshold).digest() != graph.digest():
        raise ValueError(f"counts do not rebuild {graph.region}")
    outside = [k for k, code in enumerate(members) if code not in set(regions)]
    return ~(c[union][:, outside] >= 1).any(axis=1)


def _members(pc: PairCounts, edges: np.ndarray) -> np.ndarray:
    """(N, k) bool: node is an endpoint of an edge in column j of ``edges``."""
    out = np.zeros((len(pc.node_ids), edges.shape[1]), dtype=bool)
    for j in range(edges.shape[1]):
        out[pc.pre[edges[:, j]], j] = True
        out[pc.post[edges[:, j]], j] = True
    return out


def census(pc: PairCounts, regions, *, threshold: int = DEFAULT_THRESHOLD, dataset: str = "f1",
           top: int | None = None) -> dict:
    """Union, boundary and shared-neuron counts for ``regions``, plus every graph's digest.

    The support-set table holds every support set unless ``top`` limits it to the largest.
    """
    regions = _check(regions)
    c, regional, total, union, boundary = _strata(pc, regions, threshold)
    members = _members(pc, regional)
    in_any = members.any(axis=1)
    shared = members.sum(axis=1) >= 2
    union_nodes = np.zeros(len(pc.node_ids), dtype=bool)
    union_nodes[pc.pre[union]] = union_nodes[pc.post[union]] = True

    pairs = {}
    for (i, a), (j, b) in combinations(enumerate(regions), 2):
        joint = (c[:, i].astype(np.int64) + c[:, j] >= threshold) & ~regional[:, i] & ~regional[:, j]
        pairs[f"{a}-{b}"] = int(joint.sum())

    # Support set: the regions contributing at least one synapse to a boundary edge.
    bits = (c[boundary] >= 1) @ (1 << np.arange(len(regions)))
    weight = total[boundary]
    support = pd.DataFrame({"bits": bits, "synapses": weight}).groupby("bits")["synapses"].agg(["size", "sum"])
    support = support.sort_values(["size", "sum"], ascending=False)
    n_union, n_boundary, boundary_synapses = int(union.sum()), int(boundary.sum()), int(weight.sum())
    union_synapses = int(total[union].sum())
    support_rows = [
        {"regions": "+".join(code for k, code in enumerate(regions) if int(b) >> k & 1),
         "edges": int(row["size"]), "edge_share": int(row["size"]) / n_boundary,
         "synapses": int(row["sum"]), "synapse_share": int(row["sum"]) / boundary_synapses}
        for b, row in (support if top is None else support.head(top)).iterrows()
    ]
    overlap = members.T.astype(np.int64) @ members.astype(np.int64)
    name = union_name(regions)
    out = {
        "regions": list(regions), "threshold": threshold,
        "union": {"name": name, "edges": n_union, "nodes": int(union_nodes.sum()), "synapses": union_synapses,
                  "digest": _graph(pc, union, total, name, dataset, threshold).digest()},
        "boundary": {"edges": n_boundary, "edge_share": n_boundary / n_union if n_union else 0.0,
                     "synapses": boundary_synapses,
                     "synapse_share": boundary_synapses / union_synapses if union_synapses else 0.0,
                     "support_sets": int(len(support)), "support": support_rows},
        "regional": {code: {"edges": int(regional[:, k].sum()), "nodes": int(members[:, k].sum()),
                            "digest": _graph(pc, regional[:, k], c[:, k], code, dataset, threshold).digest()}
                     for k, code in enumerate(regions)},
        "nodes": {"in_any_region_graph": int(in_any.sum()), "shared": int(shared.sum()),
                  "shared_share": int(shared.sum()) / int(in_any.sum()) if in_any.any() else 0.0,
                  "union_only": int((union_nodes & ~in_any).sum())},
        "shared_nodes": {f"{a}-{b}": int(overlap[i, j]) for (i, a), (j, b) in combinations(enumerate(regions), 2)},
        "pairwise_boundary": pairs,
    }
    if regions == CODES:
        every = pc.counts.sum(axis=1, dtype=np.int64) >= threshold
        outside = every & ~regional.any(axis=1)
        out["global"] = {
            "edges": int(every.sum()),
            "in_a_region_graph": int((every & regional.any(axis=1)).sum()),
            "boundary": int(outside.sum()),
            "registered_labels_only": int((outside & union).sum()),
            "needs_excluded_labels": int((outside & ~union).sum()),
            "region_graph_neurons": int(in_any.sum()),
            "in_two_or_more": int(shared.sum()),
        }
    return out
