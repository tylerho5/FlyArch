"""Phase 3 boundary suites, their metric-row selectors and the scorer input contract.

The ``phase3_boundary_*`` family masks a
union graph (``src/union.py``) with the regional mask generators and the negative
sampler. The evaluator still samples every hidden edge; a selector picks the metric
rows: hidden boundary edges with exactly one masked endpoint.

This module sits outside ``train/train.py``'s ``_CODE_FILES``. ``src/suites.py``
registers suite names in module tables it hashes, so the family subclasses
:class:`~src.suites.Suite` here instead of registering there.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json

import numpy as np
import pandas as pd

from src.features import FeatureSchema
from src.masks import MaskInstance, random_mask, spatial_mask, typed_mask
from src.regions import REGIONS, RegionGraph
from src.suites import MASTER_SEED, Suite, SuiteInstance
from src.union import union_labels

__all__ = ["BLOCK", "INPUT_ALLOWLIST", "SUITES", "BoundarySuite", "allowed_columns", "boundary_selector",
           "regional_confirm_digests", "regional_confirm_exclusions",
           "boundary_types", "build_boundary_suite", "check_inputs", "selector_digest"]

#: name -> (dataset, cycle slots, first seed). Seeds from 5000 up, a namespace no regional suite uses.
#: F2 suites skip their T slots (2026-11-10), so they hold 12 and 16 instances.
SUITES = {
    "phase3_boundary_f1_val": ("f1", 5, 5000),
    "phase3_boundary_f2_dev": ("f2", 15, 6000),
    "phase3_boundary_f2_confirm": ("f2", 20, 7000),
}
_TAGS = ("R01", "R05", "R10", "T", "S05")
#: No F1 boundary type survives the regional F2-confirm exclusions, so T is not drawn on F2.
_NO_T = frozenset({"phase3_boundary_f2_dev", "phase3_boundary_f2_confirm"})
_FRACTIONS = {"R01": 0.01, "R05": 0.05, "R10": 0.10}

#: The spec's linked block. Its stratum is the boundary rows whose support set lies inside it.
BLOCK = ("INP", "PENP", "VLNP", "VMNP")

#: What a Phase 3 scorer may read, by column family, with its source. Node-local only:
#: nothing derived from edges, region-graph membership, per-region degree or the boundary
#: label. The one degree value is the VisibleView's granted union in+out total, never a column.
INPUT_ALLOWLIST = {
    "geometry_present": "whether the neuron has synapses on the union's labels",
    "centroid_*, extent_*": "the neuron's own synapse ends on the union's labels, z-scored union-wide",
    "innervation_<label>": "pre+post neuropil counts per union label, as fractions of the neuron's total",
    "superclass_*, class_*, cell_type, side_*, flow_*": "F1 annotations over F1-wide vocabularies",
    "nt_*": "annotated neurotransmitter",
}


class BoundarySuite(Suite):
    """A ``phase3_boundary_*`` suite. Same JSON and digest as :class:`Suite`; its own names."""

    def validate(self, graph: RegionGraph) -> None:
        if self.name not in SUITES:
            raise ValueError(f"unknown boundary suite {self.name!r}; known: {sorted(SUITES)}")
        expected = {
            "dataset": (SUITES[self.name][0], self.dataset), "graph dataset": (graph.dataset, self.dataset),
            "region": (graph.region, self.region), "graph digest": (graph.digest(), self.graph_digest),
            "threshold": (graph.threshold, self.threshold),
        }
        for field, (want, got) in expected.items():
            if want != got:
                raise ValueError(f"suite {self.name!r} {field}: expected {want!r}, got {got!r}")
        if any(inst.regime == "H" for inst in self.instances):
            raise ValueError("boundary suites hold no H instances")
        for inst in self.instances:
            graph.index_of(inst.masked_root_ids)  # raises on a stranger root id


#: T draws only types whose neurons carry at least this many boundary-edge ends.
T_FLOOR = 100


def boundary_ends(graph: RegionGraph, boundary: np.ndarray) -> np.ndarray:
    """Per node, how many boundary edges it is an endpoint of."""
    return np.bincount(graph.edges[boundary].ravel(), minlength=graph.n_nodes)


def boundary_types(graph: RegionGraph, boundary: np.ndarray, types: pd.Series, floor: int = T_FLOOR) -> set:
    """The only types T may draw: those whose neurons carry at least ``floor`` boundary-edge ends."""
    ends = pd.Series(boundary_ends(graph, boundary), index=types.reindex(graph.node_ids).to_numpy())
    per_type = ends[ends.index.notna()].groupby(level=0).sum()
    return set(per_type.index[per_type >= floor])


def boundary_ball(graph: RegionGraph, centroids: np.ndarray, *, seed: int, pool: np.ndarray,
                  centres: np.ndarray) -> MaskInstance:
    """S05 with its centre drawn from ``centres`` and the ball grown over all of ``pool``.

    ``spatial_mask`` (in ``_CODE_FILES``) draws the centre and grows the ball over one pool. It
    draws the centre here, from the boundary neurons, with its own seeded RNG; the ball then
    follows its rule over every eligible neuron, so non-boundary neurons inside it are masked too.
    """
    centre = spatial_mask(graph, centroids, seed=seed, fraction=0.05, pool=centres)
    centre_node = centre.params["centre_node"]
    pool = np.sort(np.asarray(pool, dtype=np.int64))
    distances = np.linalg.norm(centroids[pool] - centroids[centre_node], axis=1)
    order = np.lexsort((pool, distances))
    radius = float(distances[order[max(1, round(0.05 * pool.size)) - 1]])
    return MaskInstance(centre.instance_id, graph.dataset, graph.region, "S", int(seed),
                        {"fraction": 0.05, "centre_node": centre_node, "radius": radius},
                        pool[distances <= radius].astype(np.int32))


def build_boundary_suite(
    graph: RegionGraph, boundary: np.ndarray, *, name: str, types: pd.Series, centroids: np.ndarray,
    mask_exclude_types=(), exclude_roots=(), exclude_types=(), t_types: set | None = None,
) -> BoundarySuite:
    """The named suite at ``seed = first + i`` on the R01/R05/R10/T/S05 cycle, F2 skipping T.

    ``mask_exclude_types`` (OL's lamina types) and ``exclude_types`` are never masked.
    ``exclude_roots`` (every root a regional F2-confirm suite masks, and for F2-confirm
    every F2-dev root) are never masked either, and T never draws a type that holds one.
    T draws only F1 cell types with at least ``T_FLOOR`` F1 boundary-edge ends, and two T
    instances never share a type; S05 centres are neurons with a boundary edge
    (:func:`boundary_ball`). On F1 the type set comes from ``boundary``. F2 suites draw no T,
    so their R and S05 instances keep the seeds their slots had on the full cycle.
    """
    if name not in SUITES:
        raise ValueError(f"unknown boundary suite {name!r}; known: {sorted(SUITES)}")
    dataset, size, first = SUITES[name]
    if graph.dataset != dataset or graph.region in REGIONS:
        raise ValueError(f"{name} needs a {dataset} union graph, got {graph.dataset}/{graph.region}")
    if len(boundary) != graph.n_edges:
        raise ValueError("one boundary flag per union edge is required")
    values = types.reindex(graph.node_ids)
    roots = np.isin(graph.node_ids, np.asarray(list(exclude_roots), dtype=np.int64))
    banned = set(mask_exclude_types) | set(exclude_types)
    pool = np.flatnonzero(~values.isin(banned).to_numpy() & ~roots)
    if t_types is None:
        t_types = boundary_types(graph, boundary, types) if name not in _NO_T else set()
    t_exclude = (set(values.dropna().unique()) - set(t_types)) | banned
    t_exclude |= set(values[roots].dropna().unique())
    centres = np.intersect1d(pool, np.flatnonzero(boundary_ends(graph, boundary)))

    instances = []
    for i in range(size):
        seed, tag = first + i, _TAGS[i % len(_TAGS)]
        if tag == "T" and name in _NO_T:
            continue
        if tag == "T":
            mask = typed_mask(graph, types, seed=seed, exclude=t_exclude)
            t_exclude |= {mask.params["cell_type"]}
        elif tag == "S05":
            mask = boundary_ball(graph, centroids, seed=seed, pool=pool, centres=centres)
        else:
            mask = random_mask(graph, fraction=_FRACTIONS[tag], seed=seed, pool=pool)
        instances.append(SuiteInstance(mask.instance_id, mask.regime, seed, dict(mask.params),
                                       graph.node_ids[mask.masked]))
    return BoundarySuite(name, graph.dataset, graph.region, graph.digest(), graph.threshold, MASTER_SEED,
                         tuple(instances), tuple(sorted(set(mask_exclude_types))))


def boundary_selector(graph: RegionGraph, suite: BoundarySuite, boundary: np.ndarray,
                      block: np.ndarray | None = None) -> dict:
    """Per instance, the metric rows: hidden single-masked boundary edges.

    Rows index the instance's hidden edges in ``apply_mask`` order, which are the
    negative sample's positives, so the evaluator keeps every hidden edge and the
    selector only picks rows. ``block`` (one flag per union edge) adds the block
    stratum's rows. Raises when an instance has no metric row: it is never dropped
    or redrawn.
    """
    if len(boundary) != graph.n_edges or (block is not None and len(block) != graph.n_edges):
        raise ValueError("one flag per union edge is required")
    instances = {}
    for instance in suite.resolve(graph):
        flags = np.zeros(graph.n_nodes, dtype=bool)
        flags[instance.masked] = True
        hidden = flags[graph.edges[:, 0]] | flags[graph.edges[:, 1]]
        both = flags[graph.edges[hidden, 0]] & flags[graph.edges[hidden, 1]]
        rows = boundary[hidden] & ~both
        if not rows.any():
            raise ValueError(f"{suite.name} instance {instance.instance_id} has no single-masked boundary row")
        entry = {"rows": np.flatnonzero(rows).tolist(), "n_rows": int(rows.sum()),
                 "n_positives": int(hidden.sum()), "n_both_masked_boundary": int((boundary[hidden] & both).sum())}
        if block is not None:
            entry["block_rows"] = np.flatnonzero(rows & block[hidden]).tolist()
        instances[instance.instance_id] = entry
    return {"name": suite.name, "suite_digest": suite.digest(), "graph_digest": graph.digest(),
            "boundary_edges": int(boundary.sum()), "block": list(BLOCK) if block is not None else None,
            "instances": instances}


def selector_digest(selector: dict) -> str:
    """Sha256 of the selector as written (indented, sorted keys, trailing newline)."""
    return hashlib.sha256((json.dumps(selector, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()


def allowed_columns(schema: FeatureSchema) -> frozenset[str]:
    """Every column a scorer may read from a union feature table built with ``schema``."""
    if schema.labels != union_labels(schema.region):
        raise ValueError(f"{schema.region} schema labels are not the union's labels")
    allowed = frozenset(schema.column_names) | {"geometry_present"}
    # INPUT_ALLOWLIST documents the families; a column none of them covers is refused, so the two cannot drift.
    families = [p.strip().replace("<label>", "*") for key in INPUT_ALLOWLIST for p in key.split(",")]
    undocumented = sorted(c for c in allowed if not any(fnmatch.fnmatchcase(c, f) for f in families))
    if undocumented:
        raise ValueError(f"schema columns missing from INPUT_ALLOWLIST: {undocumented[:10]}")
    return allowed


def check_inputs(schema: FeatureSchema, columns) -> None:
    """Raise unless every column is on the allowlist (no membership, degree or boundary column)."""
    allowed = allowed_columns(schema)
    refused = [c for c in columns if c not in allowed and c != "root_id"]
    if refused:
        raise ValueError(f"{schema.region} scorer inputs outside the allowlist: {refused[:10]}")


def regional_confirm_digests(registered: dict) -> dict[str, str]:
    """Every region's registered F2-confirm suite digest (``results/registered.json``).
    Raises while any is missing, since the boundary F2 masks must exclude every regional
    confirm mask."""
    import re

    from src.regions import REGIONS

    found = {code: splits.get("f2_confirm", "") for code, splits in registered.get("suites", {}).items()
             if code in REGIONS}
    pending = sorted(code for code in REGIONS if not re.fullmatch(r"[0-9a-f]{64}", found.get(code, "")))
    if pending:
        raise ValueError(f"regional F2-confirm digests are not all committed: {pending} pending")
    return found


def regional_confirm_exclusions(suites_root, digests: dict[str, str]) -> tuple[set[int], set[str]]:
    """The roots every regional F2-confirm suite masks, and the types its T instances draw.

    Each suite is read from ``<suites_root>/f2/<code>/f2_confirm.json`` and must carry its
    committed digest: the exclusion reads the frozen suites, it never rebuilds them.
    """
    from pathlib import Path

    roots, types = set(), set()
    for code, digest in sorted(digests.items()):
        suite = Suite.read(Path(suites_root) / "f2" / code.lower() / "f2_confirm.json")
        if suite.digest() != digest:
            raise ValueError(f"{code} f2_confirm digest {suite.digest()} is not the committed {digest}")
        for inst in suite.instances:
            roots |= set(map(int, inst.masked_root_ids))
            if inst.regime == "T":
                types.add(inst.params["cell_type"])
    return roots, types
