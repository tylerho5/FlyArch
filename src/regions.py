"""Brain-region registry and region-graph extraction.

The registry groups the FlyWire v783 neuropil vocabulary into twelve
super-regions and defines a *region graph* as the directed pairs whose synapse
count, summed over that region's neuropils, clears a threshold. This module is
that registry as data, plus the extraction that follows from it;
``tests/test_regions.py`` checks it.

Two things here are easy to get subtly wrong:

* **The threshold is a within-region pair total.** The published 2,700,513
  connection count is reproduced by a *global* per-pair total at >=5;
  a region graph re-derives the sum over one region's
  neuropils. Neither quantity is per-pair-per-neuropil, which is the reading
  that does not reproduce the paper.
* **Node ids must be densely reindexed.** ``src.negatives.pack_edges`` packs a
  directed pair into one 64-bit integer and rejects ids outside ``[0, 2**32)``.
  Root ids are eighteen digits. :class:`RegionGraph` keeps the originals in
  ``node_ids`` and hands every downstream module dense indices -- it is the
  ``src.graph`` that ``src/negatives.py``'s docstring refers to.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_THRESHOLD",
    "EXCLUDED_BASE",
    "REGIONS",
    "UNPAIRED_NEUROPILS",
    "ColumnMap",
    "Region",
    "RegionGraph",
    "all_labels",
    "build_region_graph",
    "expand",
    "neuropil_to_region",
    "region_labels",
]

#: Within-region pair total at or above which a directed pair is an edge (F1).
#: F2 and F3 thresholds are calibrated rather than assumed -- ``src.harmonize``
#: derives them from cell-type pair density concordance with F1, because the
#: three datasets' synapse detectors do not have comparable recall.
DEFAULT_THRESHOLD = 5


@dataclass(frozen=True)
class Region:
    """One super-region: a fixed set of neuropil *base* names."""

    code: str
    name: str
    base_neuropils: tuple[str, ...]
    model: str


#: The twelve regions in table order; ``model`` is each region's model assignment.
REGIONS: Mapping[str, Region] = {
    r.code: r
    for r in (
        Region("OL", "Optic Lobe", ("LA", "ME", "AME", "LO", "LOP"), "M12"),
        Region("LX", "Lateral Complex", ("BU", "LAL", "GA"), "M5"),
        Region("SNP", "Superior Protocerebrum", ("SLP", "SIP", "SMP"), "M6"),
        Region("INP", "Inferior Protocerebrum", ("CRE", "SCL", "ICL", "IB", "ATL"), "M7"),
        Region("VMNP", "Ventromedial", ("VES", "EPA", "GOR", "SPS", "IPS"), "M8"),
        Region("VLNP", "Ventrolateral", ("AOTU", "AVLP", "PVLP", "PLP", "WED"), "M9"),
        Region("CX", "Central Complex", ("FB", "EB", "PB", "NO"), "M1"),
        Region("MB", "Mushroom Body", ("MB_CA", "MB_PED", "MB_VL", "MB_ML"), "M4"),
        Region("AL", "Antennal Lobe", ("AL",), "M2"),
        Region("LH", "Lateral Horn", ("LH",), "M3"),
        Region("PENP", "Periesophageal", ("SAD", "AMMC", "FLA", "CAN", "PRW"), "M10"),
        Region("GNG", "Gnathal Ganglia", ("GNG",), "M11"),
    )
}

#: Midline structures carrying no _L/_R suffix in v783. The registry's other 35
#: base names are paired, which is what makes the vocabulary come to 79 labels
#: (70 paired + 7 unpaired in-registry + OCG + UNASGD). That arithmetic is
#: asserted by ``test_all_79_v783_labels_are_accounted_for`` and confirmed
#: against the real labels by ``scripts/extract_region.py --verify-labels``.
#: If the audit prints a diff, this set is wrong -- fix it
#: rather than editing the test.
UNPAIRED_NEUROPILS = frozenset({"FB", "EB", "PB", "NO", "GNG", "PRW", "SAD", "OCG"})

#: In the vocabulary but outside the registry: the ocellar
#: ganglion, which is absent from F2 and tiny, and synapses that were never
#: assigned a neuropil. Both are excluded from training and evaluation.
EXCLUDED_BASE = frozenset({"OCG", "UNASGD"})


def expand(base: str) -> tuple[str, ...]:
    """Base neuropil name -> its v783 label(s)."""
    if base == "UNASGD" or base in UNPAIRED_NEUROPILS:
        return (base,)
    return (f"{base}_L", f"{base}_R")


def region_labels(code: str) -> tuple[str, ...]:
    """Every v783 label belonging to region ``code``."""
    if code not in REGIONS:
        raise ValueError(f"unknown region {code!r}; known: {sorted(REGIONS)}")
    return tuple(
        label for base in REGIONS[code].base_neuropils for label in expand(base)
    )


def all_labels() -> tuple[str, ...]:
    """The full v783 label vocabulary: 78 neuropils plus ``UNASGD``."""
    labels = [label for code in REGIONS for label in region_labels(code)]
    labels += [label for base in sorted(EXCLUDED_BASE) for label in expand(base)]
    return tuple(labels)


def neuropil_to_region() -> dict[str, str | None]:
    """Every label -> its region code, or ``None`` where it is excluded."""
    mapping: dict[str, str | None] = {}
    for code in REGIONS:
        for label in region_labels(code):
            mapping[label] = code
    for base in EXCLUDED_BASE:
        for label in expand(base):
            mapping[label] = None
    return mapping


#: Column aliases, in preference order. The exact schema of
#: ``proofread_connections_783.feather`` is recorded nowhere in this repo and the
#: file is 852 MB, so readers resolve names and fail with the columns they
#: actually found rather than raising KeyError on a guess. Confirm the real names
#: on the data box, then prune this table to them.
_ALIASES: Mapping[str, tuple[str, ...]] = {
    "pre": ("pre_pt_root_id", "pre_root_id", "pre_id", "pre"),
    "post": ("post_pt_root_id", "post_root_id", "post_id", "post"),
    "neuropil": ("neuropil", "neuropil_name", "roi", "compartment"),
    "syn_count": ("syn_count", "synapses", "n_syn", "count", "weight"),
}


@dataclass(frozen=True)
class ColumnMap:
    """Which columns of a connections table carry which quantity."""

    pre: str
    post: str
    neuropil: str
    syn_count: str

    @classmethod
    def resolve(cls, columns: Iterable[str]) -> ColumnMap:
        available = list(columns)
        lookup = {str(c).lower(): c for c in available}
        chosen: dict[str, str] = {}
        for field_name, candidates in _ALIASES.items():
            for candidate in candidates:
                if candidate in lookup:
                    chosen[field_name] = lookup[candidate]
                    break
            else:
                raise ValueError(
                    f"could not resolve the {field_name!r} column: tried "
                    f"{candidates}, table has {sorted(map(str, available))}"
                )
        return cls(**chosen)


@dataclass(frozen=True)
class RegionGraph:
    """One region's directed weighted graph, densely indexed.

    ``edges[i] = (s, t)`` are indices into ``node_ids``, and ``weights[i]`` is
    that ordered pair's within-region synapse total. Every downstream module
    addresses neurons by these dense indices; ``node_ids`` is the only place
    original root ids survive.

    Construction copies ``node_ids``, ``edges`` and ``weights`` into read-only
    arrays. The graph owns its data, so ``in_strength()`` and ``out_strength()``
    can hand back a shared cached vector that no caller can write through.
    """

    dataset: str
    region: str
    threshold: int
    node_ids: np.ndarray  # (N,) int64, original root ids, sorted ascending
    edges: np.ndarray  # (E, 2) int32, dense indices into node_ids
    weights: np.ndarray  # (E,) int64

    def __post_init__(self) -> None:
        # The dataclass is frozen but its arrays are not, so keep private
        # copies: freezing the caller's buffers would change their contract,
        # and freezing a view would leave the writable base reachable.
        for name in ("node_ids", "edges", "weights"):
            arr = np.array(getattr(self, name), copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    def __reduce__(self) -> tuple[type, tuple]:
        # Reconstruct through __init__ so pickle and deepcopy re-freeze the
        # copies and start without a strength cache to go stale.
        return (
            self.__class__,
            (
                self.dataset,
                self.region,
                self.threshold,
                self.node_ids,
                self.edges,
                self.weights,
            ),
        )

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    def index_of(self, root_ids: np.ndarray) -> np.ndarray:
        """Dense indices for original root ids. Raises on an unknown id."""
        root_ids = np.asarray(root_ids, dtype=np.int64)
        idx = np.clip(np.searchsorted(self.node_ids, root_ids), 0, self.n_nodes - 1)
        hit = self.node_ids[idx] == root_ids
        if not hit.all():
            missing = root_ids[~hit]
            raise KeyError(
                f"{missing.size} root id(s) are not in the {self.region} graph, "
                f"e.g. {missing[:5].tolist()}"
            )
        return idx.astype(np.int32)

    def _strength(self, column: int) -> np.ndarray:
        out = np.zeros(self.n_nodes, dtype=np.int64)
        np.add.at(out, self.edges[:, column], self.weights)
        out.setflags(write=False)
        return out

    @cached_property
    def _in_strength(self) -> np.ndarray:
        return self._strength(1)

    @cached_property
    def _out_strength(self) -> np.ndarray:
        return self._strength(0)

    def in_strength(self) -> np.ndarray:
        """Within-region incoming synapse total per node. Shared, read-only."""
        return self._in_strength

    def out_strength(self) -> np.ndarray:
        """Within-region outgoing synapse total per node. Shared, read-only."""
        return self._out_strength

    def digest(self) -> str:
        """Content address. Suites pin this, so a regenerated graph is caught.

        Masks, positives and negative digests all derive from a thresholded
        graph, so swapping the graph under a committed suite would invalidate
        every number computed from it. This is what makes that detectable
        rather than deniable.
        """
        h = hashlib.sha256()
        h.update(f"{self.dataset}/{self.region}/t={self.threshold}".encode())
        for arr in (self.node_ids, self.edges, self.weights):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def write(self, directory: str | Path) -> None:
        """Write ``nodes.parquet``, ``edges.parquet`` and ``graph.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"root_id": self.node_ids}).to_parquet(
            directory / "nodes.parquet", index=False
        )
        pd.DataFrame(
            {
                "source": self.edges[:, 0],
                "target": self.edges[:, 1],
                "weight": self.weights,
            }
        ).to_parquet(directory / "edges.parquet", index=False)
        (directory / "graph.json").write_text(
            json.dumps(
                {
                    "dataset": self.dataset,
                    "region": self.region,
                    "threshold": self.threshold,
                    "n_nodes": self.n_nodes,
                    "n_edges": self.n_edges,
                    "digest": self.digest(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, directory: str | Path) -> RegionGraph:
        """Read a graph written by :meth:`write`, verifying its digest."""
        directory = Path(directory)
        meta = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
        nodes = pd.read_parquet(directory / "nodes.parquet")
        edges = pd.read_parquet(directory / "edges.parquet")
        graph = cls(
            dataset=meta["dataset"],
            region=meta["region"],
            threshold=int(meta["threshold"]),
            node_ids=nodes["root_id"].to_numpy(dtype=np.int64),
            edges=edges[["source", "target"]].to_numpy(dtype=np.int32),
            weights=edges["weight"].to_numpy(dtype=np.int64),
        )
        if graph.digest() != meta["digest"]:
            raise ValueError(
                f"{directory} digest mismatch: graph.json says {meta['digest']}, "
                f"the parquet contents hash to {graph.digest()}"
            )
        return graph


def build_region_graph(
    connections: pd.DataFrame,
    *,
    region: str,
    dataset: str,
    threshold: int = DEFAULT_THRESHOLD,
    columns: ColumnMap | None = None,
) -> RegionGraph:
    """Connections table -> one region's graph, per the region registry.

    Parameters
    ----------
    connections
        Long-format table, one row per (pre, post, neuropil) with a synapse
        count. Column names are resolved through :class:`ColumnMap`.
    region
        Registry code, e.g. ``"CX"``.
    dataset
        ``"f1"``, ``"f2"`` or ``"f3"``. Recorded, and folded into the digest,
        so two datasets' graphs can never be mistaken for one another.
    threshold
        Within-region pair total at or above which a pair becomes an edge.
    """
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; known: {sorted(REGIONS)}")
    if threshold < 1:
        raise ValueError(f"threshold must be >= 1, got {threshold}")

    cols = columns or ColumnMap.resolve(connections.columns)
    labels = set(region_labels(region))
    within = connections[connections[cols.neuropil].isin(labels)]
    if within.empty:
        seen = sorted({str(v) for v in connections[cols.neuropil].unique()})[:8]
        raise ValueError(
            f"no rows in region {region!r}; labels searched: {sorted(labels)}, "
            f"table has e.g. {seen}"
        )

    totals = (
        within.groupby([cols.pre, cols.post], sort=True)[cols.syn_count]
        .sum()
        .reset_index()
    )
    kept = totals[totals[cols.syn_count] >= threshold]
    if kept.empty:
        raise ValueError(f"region {region!r} has no pair reaching threshold {threshold}")

    node_ids = np.unique(
        np.concatenate(
            [kept[cols.pre].to_numpy(np.int64), kept[cols.post].to_numpy(np.int64)]
        )
    )
    source = np.searchsorted(node_ids, kept[cols.pre].to_numpy(np.int64))
    target = np.searchsorted(node_ids, kept[cols.post].to_numpy(np.int64))

    return RegionGraph(
        dataset=dataset,
        region=region,
        threshold=threshold,
        node_ids=node_ids,
        edges=np.stack([source, target], axis=1).astype(np.int32),
        weights=kept[cols.syn_count].to_numpy(np.int64),
    )
