"""The shared, dataset-independent node feature schema.

Features describe each neuron's own annotation, synapse locations, and
neuropil innervation.  Graph degree and partner-derived quantities are
deliberately absent: degree-given information is supplied only at scoring time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.regions import RegionGraph, region_labels

NT_CLASSES = ("acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine")
FEATURE_BLOCKS = ("geometry", "types", "nt", "context")
PERTURB_SEED = 20260818
PERTURB_SCALE = 0.01


_ANNOTATION_ALIASES = {
    "root_id": ("root_id", "pt_root_id", "id"),
    "super_class": ("super_class", "superclass"),
    "class": ("class", "cell_class"),
    "cell_type": ("cell_type", "type", "fafb_type"),
    "side": ("side",),
    "flow": ("flow",),
    "nt_type": ("nt_type", "neurotransmitter", "top_nt"),
    "fafb_match_id": ("fafb_match_id", "fafb_root_id", "match_id"),
}

_REQUIRED_ANNOTATION_FIELDS = (
    "root_id", "super_class", "class", "cell_type", "side", "flow", "nt_type",
)

NEUROPIL_ALIASES = ("neuropil", "neuropil_name", "roi", "compartment")


def _nt_probability_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Resolve nt_<class> or <class>_probability, case-insensitively.

    A probability representation requires all six NT_CLASSES columns. Optional
    histamine, tyramine, and other columns contribute their mass to nt_other.
    Each represented row must be finite, nonnegative, and sum to one (1e-6
    absolute tolerance). All-null rows are absent; partial rows are invalid.
    Categorical nt_type/neurotransmitter is used only without these columns.
    """
    columns = {}
    for label in (*NT_CLASSES, "histamine", "tyramine", "other"):
        aliases = (f"nt_{label}", f"{label}_probability")
        matches = [column for column in frame.columns if str(column).lower() in aliases]
        if len(matches) > 1:
            raise ValueError(f"ambiguous NT probability columns for {label}: {matches}")
        if matches:
            columns[label] = matches[0]
    if columns:
        missing = [label for label in NT_CLASSES if label not in columns]
        if missing:
            raise ValueError(f"incomplete NT probability columns; missing: {missing}")
    return columns


def _neurotransmitters(annotations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    columns = _nt_probability_columns(annotations)
    nt = np.zeros((len(annotations), 7))
    if columns:
        frame = annotations[list(columns.values())]
        present = frame.notna().any(axis=1).to_numpy()
        try:
            probabilities = frame.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ValueError("NT probabilities must be numeric") from exc
        supplied = probabilities[present]
        if (not np.isfinite(supplied).all() or (supplied < 0).any()
                or not np.isclose(supplied.sum(axis=1), 1., atol=1e-6, rtol=0).all()):
            raise ValueError("NT probabilities must be complete, finite, nonnegative and sum to one")
        for j, label in enumerate(columns):
            index = NT_CLASSES.index(label) if label in NT_CLASSES else 6
            nt[present, index] += probabilities[present, j]
        return nt, present
    values = _values(annotations, "nt_type")
    present = values.notna().to_numpy()
    lookup = {value: index for index, value in enumerate(NT_CLASSES)}
    for row, value in enumerate(values):
        if pd.notna(value):
            nt[row, lookup.get(str(value).lower(), 6)] = 1.
    return nt, present


def _resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    resolved = {}
    probabilities = _nt_probability_columns(frame)
    for field in _REQUIRED_ANNOTATION_FIELDS:
        if field == "nt_type" and probabilities:
            continue
        resolved[field] = _resolve_column(frame, field)
    return resolved


def _resolve_column(frame: pd.DataFrame, field: str) -> str:
    available = list(frame.columns)
    lookup = {str(column).lower(): column for column in available}
    aliases = _ANNOTATION_ALIASES[field]
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    else:
        raise ValueError(
            f"could not resolve the {field!r} column: tried {aliases}, "
            f"available columns: {sorted(map(str, available))}"
        )


def _values(frame: pd.DataFrame, name: str) -> pd.Series:
    column = _resolve_column(frame, name)
    return frame[column]


def _reject_duplicate_roots(values: pd.Series | pd.Index, label: str) -> None:
    duplicated = values[values.duplicated()]
    if len(duplicated):
        raise ValueError(f"{label} contain duplicate root id(s): {list(duplicated[:5])}")


def _reject_invalid_counts(frame: pd.DataFrame, labels: tuple[str, ...], source: str) -> None:
    """Missing counts (NaN) mean absent; everything present must be finite and >= 0."""
    for label in labels:
        if label in frame.columns:
            values = frame[label].to_numpy(dtype=float, na_value=np.nan)
            if (~(np.isnan(values) | (np.isfinite(values) & (values >= 0)))).any():
                raise ValueError(f"{source} contain nonfinite or negative values")


def _per_node_statistics(
    codes: np.ndarray, coords: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count, mean, and M2 of coords per node from two-pass bincounts."""
    counts = np.bincount(codes, minlength=n_nodes)
    sums = np.column_stack(
        [np.bincount(codes, weights=coords[:, j], minlength=n_nodes) for j in range(3)]
    )
    means = np.zeros((n_nodes, 3))
    active = counts > 0
    means[active] = sums[active] / counts[active, None]
    residual = coords - means[codes]
    m2 = np.column_stack(
        [np.bincount(codes, weights=residual[:, j] ** 2, minlength=n_nodes) for j in range(3)]
    )
    return counts, means, m2


@dataclass(frozen=True)
class FeatureSchema:
    region: str
    superclass_vocab: tuple[str, ...]
    class_vocab: tuple[str, ...]
    side_vocab: tuple[str, ...]
    flow_vocab: tuple[str, ...]
    type_vocab: dict[str, int]
    nt_classes: tuple[str, ...] = NT_CLASSES
    labels: tuple[str, ...] = ()

    @classmethod
    def from_f1(cls, annotations: pd.DataFrame, region: str) -> "FeatureSchema":
        _resolve_columns(annotations)
        def vocab(name: str) -> tuple[str, ...]:
            s = _values(annotations, name).dropna().astype(str)
            return tuple(sorted(set(s)))
        types = vocab("cell_type")
        return cls(region, vocab("super_class"), vocab("class"), vocab("side"),
                   vocab("flow"), {v: i + 1 for i, v in enumerate(types)}, NT_CLASSES,
                   tuple(region_labels(region)))

    @property
    def n_features(self) -> int:
        return len(self.column_names)

    @property
    def column_names(self) -> tuple[str, ...]:
        geometry = ("centroid_x", "centroid_y", "centroid_z", "extent_x", "extent_y", "extent_z")
        types = tuple(f"superclass_{x}" for x in self.superclass_vocab) + ("superclass_OOV",)
        types += tuple(f"class_{x}" for x in self.class_vocab) + ("class_OOV",)
        types += ("cell_type",) + tuple(f"side_{x}" for x in self.side_vocab)
        types += tuple(f"flow_{x}" for x in self.flow_vocab)
        nt = tuple(f"nt_{x}" for x in self.nt_classes) + ("nt_other",)
        geometry += tuple(f"innervation_{x}" for x in self.labels)
        return geometry + types + nt

    def type_id(self, name: Any) -> int:
        return self.type_vocab.get(str(name), 0) if pd.notna(name) else 0

    def to_json(self) -> str:
        return json.dumps({"region": self.region, "superclass_vocab": self.superclass_vocab,
            "class_vocab": self.class_vocab, "side_vocab": self.side_vocab, "flow_vocab": self.flow_vocab,
            "type_vocab": self.type_vocab, "nt_classes": self.nt_classes, "labels": self.labels}, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, source: str | bytes | dict[str, Any]) -> "FeatureSchema":
        obj = json.loads(source) if isinstance(source, (str, bytes)) else source
        return cls(obj["region"], tuple(obj["superclass_vocab"]), tuple(obj["class_vocab"]),
                   tuple(obj["side_vocab"]), tuple(obj["flow_vocab"]),
                   {str(k): int(v) for k, v in obj["type_vocab"].items()}, tuple(obj["nt_classes"]), tuple(obj["labels"]))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass
class FeatureTable:
    values: np.ndarray
    columns: tuple[str, ...]
    blocks: dict[str, slice]
    coverage: pd.DataFrame
    #: Whether each node has synapse statistics; ``None`` if the artifacts predate it.
    geometry_present: np.ndarray | None = None
    #: Optional per-node arrays from ``scripts/build_node_extras.py`` (``extras.npz``),
    #: row-aligned to the graph; opt-in model switches read them.
    extras: dict[str, np.ndarray] | None = None

    def perturb(self, node: int) -> "FeatureTable":
        """Return a copy with one row's geometry coordinates jittered."""
        out = FeatureTable(self.values.copy(), self.columns, dict(self.blocks), self.coverage.copy(),
                           self.geometry_present, self.extras)
        if not 0 <= node < len(out.values):
            raise IndexError(node)
        geom = out.blocks["geometry"]
        if geom.stop > geom.start:
            rng = np.random.default_rng(np.random.SeedSequence([PERTURB_SEED, node]))
            out.values[node, geom] += rng.normal(0.0, PERTURB_SCALE, geom.stop - geom.start)
        return out

    def with_uniform_innervation(self, rows) -> "FeatureTable":
        """A copy with ``rows``' innervation set to the uniform share over the labels.

        A masked node's innervation counts the synapses on its hidden edges, so a
        scorer that must not read them gets this table.
        """
        rows = np.asarray(rows, dtype=np.int64)
        columns = [k for k, name in enumerate(self.columns) if name.startswith("innervation_")]
        if not len(rows):
            return self
        if not columns:
            raise ValueError(f"{len(rows)} rows to ablate, but the table has no innervation column")
        values = self.values.copy()
        values[np.ix_(rows, np.asarray(columns))] = 1.0 / len(columns)
        return FeatureTable(values, self.columns, self.blocks, self.coverage,
                            self.geometry_present, self.extras)


def _one_hot(
    values: pd.Series, vocab: tuple[str, ...], prefix: str, *, include_oov: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.zeros((len(values), len(vocab) + int(include_oov)), dtype=np.float64)
    present = values.notna().to_numpy()
    lookup = {v: i for i, v in enumerate(vocab)}
    for row, value in enumerate(values):
        if pd.notna(value) and str(value) in lookup:
            arr[row, lookup[str(value)]] = 1.0
        elif include_oov:
            arr[row, -1] = 1.0
    return arr, present


def _geometry(graph: RegionGraph, synapses: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """Use labeled in-region locations, or caller-prefiltered label-free data.

    Label-free raw coordinates and preaggregated statistics must already be
    restricted to graph.region; the CLI supplies such reduced statistics.
    """
    n = graph.n_nodes
    raw = np.zeros((n, 6), dtype=float)
    present = np.zeros(n, dtype=bool)
    if synapses is None or synapses.empty:
        return raw, present
    lookup = {str(column).lower(): column for column in synapses.columns}
    neuropil = next((lookup[alias] for alias in NEUROPIL_ALIASES if alias in lookup), None)
    if neuropil is not None:
        synapses = synapses[synapses[neuropil].isin(region_labels(graph.region))]
    ids = _values(synapses, "root_id")
    aggregate_columns = (
        "centroid_x", "centroid_y", "centroid_z",
        "extent_x", "extent_y", "extent_z",
    )
    if set(aggregate_columns) <= set(synapses.columns):
        _reject_duplicate_roots(ids, "preaggregated synapse statistics")
        reduced = synapses.assign(_root_id=ids).set_index("_root_id")
        aligned = reduced.reindex(graph.node_ids)
        present = aligned[list(aggregate_columns)].notna().all(axis=1).to_numpy()
        raw[present] = aligned.loc[present, list(aggregate_columns)].to_numpy(float)
        if present.any():
            for j in range(6):
                values = raw[present, j]
                sd = values.std()
                raw[present, j] = (values - values.mean()) / sd if sd else 0.0
        return raw, present
    if not {"x", "y", "z"}.issubset(synapses.columns):
        return raw, present
    node_index = pd.Index(graph.node_ids)
    in_graph = ids.isin(node_index)
    codes = node_index.get_indexer(ids[in_graph].to_numpy(dtype="int64"))
    coords = synapses.loc[in_graph, ["x", "y", "z"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("synapse coordinates contain non-finite values")
    counts, means, m2 = _per_node_statistics(codes, coords, n)
    present = counts > 0
    raw[present, :3] = means[present]
    raw[present, 3:] = np.sqrt(m2[present] / counts[present, None])
    if present.any():
        for j in range(6):
            vals = raw[present, j]; mean = vals.mean(); sd = vals.std()
            raw[present, j] = (vals - mean) / sd if sd else 0.0
    return raw, present


def build_features(annotations: pd.DataFrame, graph: RegionGraph, schema: FeatureSchema, *, dataset: str,
                   synapses: pd.DataFrame | None = None, innervation: pd.DataFrame | None = None) -> FeatureTable:
    if schema.region != graph.region:
        raise ValueError(f"schema region {schema.region!r} does not match graph region {graph.region!r}")
    if graph.dataset != dataset:
        raise ValueError(
            f"graph dataset {graph.dataset!r} does not match the requested dataset {dataset!r}"
        )
    resolved = _resolve_columns(annotations)
    canonical = annotations.rename(columns={actual: field for field, actual in resolved.items()})
    _reject_duplicate_roots(canonical["root_id"], "annotations")
    ann = canonical.set_index("root_id")
    ann = ann.reindex(graph.node_ids)
    blocks: dict[str, slice] = {}; pieces: list[np.ndarray] = []; coverage_rows = []
    def add(name: str, piece: np.ndarray, present: np.ndarray, labels: list[str]) -> None:
        start = sum(x.shape[1] for x in pieces); pieces.append(piece); blocks[name] = slice(start, start + piece.shape[1])
        coverage_rows.extend({"feature": label, "n_present": int(present.sum()), "n_total": len(present), "fraction_present": float(present.mean())} for label in labels)
    geo, gp = _geometry(graph, synapses)
    if innervation is None:
        frac = np.full((graph.n_nodes, len(schema.labels)), 1.0 / len(schema.labels))
    else:
        _reject_duplicate_roots(innervation.index, "innervation frames")
        _reject_invalid_counts(innervation, schema.labels, "innervation counts")
        frac = np.full((graph.n_nodes, len(schema.labels)), 1.0 / len(schema.labels))
        aligned = innervation.reindex(graph.node_ids)
        label_present = np.zeros((graph.n_nodes, len(schema.labels)), dtype=bool)
        inn = np.zeros_like(frac)
        for j, label in enumerate(schema.labels):
            if label in aligned.columns:
                values = aligned[label]
                label_present[:, j] = values.notna().to_numpy()
                inn[:, j] = values.fillna(0).to_numpy(float)
        totals = inn.sum(axis=1)
        good = totals > 0
        frac[good] = inn[good] / totals[good, None]
        label_present &= good[:, None]
    geo = np.concatenate([geo, frac], axis=1)
    geometry_start = sum(x.shape[1] for x in pieces)
    pieces.append(geo)
    blocks["geometry"] = slice(geometry_start, geometry_start + geo.shape[1])
    geometry_names = ("centroid_x", "centroid_y", "centroid_z", "extent_x", "extent_y", "extent_z")
    for j, label in enumerate(geometry_names):
        coverage_rows.append({"feature": label, "n_present": int(gp.sum()), "n_total": len(gp), "fraction_present": float(gp.mean())})
    for j, label in enumerate(schema.labels):
        present = np.zeros(graph.n_nodes, dtype=bool) if innervation is None else label_present[:, j]
        coverage_rows.append({"feature": f"innervation_{label}", "n_present": int(present.sum()), "n_total": len(present), "fraction_present": float(present.mean())})
    type_pieces = []; type_present = []
    for field, vocab, prefix in (("super_class", schema.superclass_vocab, "superclass"), ("class", schema.class_vocab, "class"), ("cell_type", tuple(schema.type_vocab), "cell_type"), ("side", schema.side_vocab, "side"), ("flow", schema.flow_vocab, "flow")):
        vals = ann[field]
        if field == "cell_type":
            a = np.zeros((len(vals), 1)); a[:, 0] = [schema.type_id(v) for v in vals]; present = vals.notna().to_numpy()
        else:
            a, present = _one_hot(
                vals, vocab, prefix, include_oov=field in {"super_class", "class"}
            )
        type_pieces.append(a); type_present.append((field, present))
    type_array = np.concatenate(type_pieces, axis=1)
    add("types", type_array, np.ones(len(ann), dtype=bool), [])
    for field, present in type_present:
        coverage_rows.append({"feature": field, "n_present": int(present.sum()), "n_total": len(present), "fraction_present": float(present.mean())})
    nt, present = _neurotransmitters(ann)
    add("nt", nt, present, ["nt_type"])
    start = sum(x.shape[1] for x in pieces); blocks["context"] = slice(start, start)
    return FeatureTable(np.concatenate(pieces, axis=1), schema.column_names, blocks, pd.DataFrame(coverage_rows), gp)
