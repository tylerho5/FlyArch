"""Score frozen region experts on a Phase 3 union graph, and fuse them (plan 03, B0).

Three modes:

- ``union``: an expert scores a boundary suite on the masked union graph. Its
  feature table is the union table re-encoded into the expert's schema
  (``src.transfer.reencode_features``), with innervation taken from the union
  table over the expert's own labels and renormalised to sum to 1, and geometry
  recomputed over the expert's labels and z-scored with its region's node
  statistics (:func:`expert_geometry`), the frame it trained in. Only the
  selector's metric rows are scored, against their committed candidates.
- ``home``: an expert scores its own region's suite on its own graph, every
  row, so it can be checked against ``scripts/score_run.py`` row for row.
- ``regional``: a region's suite scored on the union graph. Every union edge of
  each masked root is hidden, the union in+out total is granted, and the rows
  are the suite's committed positives and candidates.

A cache is ``{instance_id: (rows, scores)}``: ``scores[i]`` is row ``rows[i]``'s
positive followed by its K negatives. :func:`fuse` combines caches before ranking.
With ``masked_uniform`` the scorer reads each masked node's innervation as the uniform
share over its labels (the registered ablation, for a leak-free comparison).

This module is outside ``train/train.py``'s ``_CODE_FILES``, so adding it changes
no run ID; the score manifest records its hash instead.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluate import CommittedNegatives, build_scoring_task, candidate_pairs
from src.features import FeatureSchema, FeatureTable
from src.masks import MaskInstance, apply_mask
from src.metrics import ranks_against_negatives
from src.regions import RegionGraph
from src.suites import Suite, suite_negatives
from src.transfer import reencode_features

__all__ = ["GEOMETRY", "MODES", "WEIGHT_GRID", "expert_geometry", "expert_table", "fit_weights", "fuse", "instance_mrr", "load_cache", "loo_b0",
           "registry_label", "save_cache", "score_cache"]

MODES = ("union", "home", "regional")


def registry_label(path, root) -> str:
    """How a registry is recorded in the score manifest: relative to ``root``, else as given."""
    path = Path(path).resolve()
    root = Path(root).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


GEOMETRY = ("centroid_x", "centroid_y", "centroid_z", "extent_x", "extent_y", "extent_z")


def expert_geometry(raw: pd.DataFrame, node_ids: np.ndarray, region_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geometry for ``node_ids`` in a region expert's frame.

    ``raw`` is ``scripts/build_features.stream_synapse_geometry`` over the region's labels
    (``root_id`` plus unscaled :data:`GEOMETRY`). Each column is z-scored with the mean and
    SD over the region graph's nodes (``region_ids``) that have geometry, as
    ``src.features`` z-scored the expert's own table, so a region neuron reads exactly what
    the expert trained on. A neuron with no synapse on the region's labels reads zeros.
    """
    aligned = raw.set_index("root_id").reindex(node_ids)[list(GEOMETRY)]
    present = aligned.notna().all(axis=1).to_numpy()
    values = aligned.to_numpy(float)
    home = present & np.isin(node_ids, region_ids)
    if not home.any():
        raise ValueError("no region neuron has geometry: the raw table does not cover the region")
    mean, sd = values[home].mean(axis=0), values[home].std(axis=0)
    out = np.zeros((len(node_ids), len(GEOMETRY)))
    out[present] = np.divide(values[present] - mean, sd, out=np.zeros_like(values[present]), where=sd > 0)
    return out, present


def expert_table(union_table: FeatureTable, union_schema: FeatureSchema, expert_schema: FeatureSchema, *,
                 geometry: tuple[np.ndarray, np.ndarray]) -> FeatureTable:
    """The union table in the expert's frame: its columns, innervation over its labels, its geometry.

    ``geometry`` is :func:`expert_geometry`'s output for the union's nodes. The union's own
    geometry is z-scored union-wide over the union's labels, a frame no expert trained on.
    """
    table = reencode_features(union_table, union_schema, expert_schema)
    source = [union_schema.column_names.index(f"innervation_{label}") for label in expert_schema.labels]
    raw = union_table.values[:, source]
    total = raw.sum(axis=1, keepdims=True)
    share = np.divide(raw, total, out=np.full_like(raw, 1.0 / len(source)), where=total > 0)
    target = [expert_schema.column_names.index(f"innervation_{label}") for label in expert_schema.labels]
    table.values[:, target] = share
    values, present = geometry
    if values.shape != (len(table.values), len(GEOMETRY)):
        raise ValueError("geometry does not cover the union's nodes")
    table.values[:, [expert_schema.column_names.index(c) for c in GEOMETRY]] = values
    return dataclasses.replace(table, geometry_present=present)


def _predict(model, view, table: FeatureTable, pairs: np.ndarray, width: int) -> np.ndarray:
    scorer = getattr(model, "score_inference", None)
    with torch.no_grad():
        output = (scorer(view, table, pairs, row_width=width) if callable(scorer)
                  else model.score(view, table, pairs, row_width=width))
    logits = output[0] if isinstance(output, tuple) else output
    return np.asarray(logits.detach().cpu().numpy(), dtype=np.float64).reshape(-1, width)


def _tasks(graph: RegionGraph, suite: Suite, negatives: CommittedNegatives, samples=None):
    """Each instance's view and committed scoring task, digests checked."""
    samples = samples if samples is not None else suite_negatives(graph, suite, spec=negatives.spec)
    for instance in suite.resolve(graph):
        view = apply_mask(graph, instance)
        yield view, build_scoring_task(view, samples[instance.instance_id], suite=suite, negatives=negatives)


def score_cache(model, *, mode: str, graph: RegionGraph, table: FeatureTable, suite: Suite,
                negatives: CommittedNegatives, selector: dict | None = None,
                union_graph: RegionGraph | None = None, union_table: FeatureTable | None = None,
                samples=None, masked_uniform: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Score one expert; see the module docstring for ``mode``.

    ``graph``/``table``/``suite``/``negatives`` are the suite's own (the union and its
    boundary suite for ``union``, the region for ``home`` and ``regional``). ``regional``
    also takes the union graph and the union table already in the expert's columns.
    ``samples`` reuses one ``suite_negatives`` result across experts. ``masked_uniform``
    hands the scorer the table with each masked node's innervation made uniform
    (:meth:`src.features.FeatureTable.with_uniform_innervation`).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if masked_uniform and mode == "home":
        raise ValueError("home mode masks nothing, so it has no masked innervation to set uniform")

    def scorer_table(frame: FeatureTable, view) -> FeatureTable:
        return frame.with_uniform_innervation(np.flatnonzero(view.is_masked)) if masked_uniform else frame

    if (mode == "union") != (selector is not None):
        raise ValueError("union mode, and only union mode, scores a selector's rows")
    if mode == "regional" and (union_graph is None or union_table is None):
        raise ValueError("regional mode needs the union graph and the union table")
    if mode == "union" and selector["suite_digest"] != suite.digest():
        raise ValueError("selector was not built from this suite")
    width = negatives.spec.k + 1
    out = {}
    for view, task in _tasks(graph, suite, negatives, samples):
        iid = task.instance_id
        rows = (np.asarray(selector["instances"][iid]["rows"], dtype=np.int64) if mode == "union"
                else np.arange(len(task.positives)))
        pairs = candidate_pairs(task).reshape(len(task.positives), width, 2)[rows]
        if mode == "regional":
            index = pd.Index(union_graph.node_ids)
            masked = np.sort(index.get_indexer(graph.node_ids[view.instance.masked])).astype(np.int32)
            pairs = index.get_indexer(graph.node_ids[pairs.reshape(-1)]).reshape(pairs.shape)
            if (masked < 0).any() or (pairs < 0).any():  # -1 would silently index the last union node
                raise ValueError(f"{iid}: a suite neuron or candidate is not in the union graph")
            instance = view.instance
            union_view = apply_mask(union_graph, MaskInstance(
                iid, union_graph.dataset, union_graph.region, instance.regime, instance.seed,
                dict(instance.params), masked)).model_view()
            scores = _predict(model, union_view, scorer_table(union_table, union_view), pairs.reshape(-1, 2), width)
        else:
            scores = _predict(model, view.model_view(), scorer_table(table, view), pairs.reshape(-1, 2), width)
        out[iid] = (rows, scores)
    return out


def _zscore(scores: np.ndarray) -> np.ndarray:
    """Row z-scores; a zero-variance row becomes 0."""
    mean = scores.mean(axis=1, keepdims=True)
    sd = scores.std(axis=1, keepdims=True)
    return np.divide(scores - mean, sd, out=np.zeros_like(scores), where=sd > 0)


def fuse(groups: dict[str, list[dict]], weights: dict[str, float] | None = None) -> dict:
    """B0's score mixture: seeds combine before ranking.

    ``groups`` maps a region (or any expert name) to its seeds' caches. Within a group
    the seeds' row z-scores are averaged; groups are then mixed with ``weights``
    (equal when omitted). Every cache must hold the same instances and rows.
    """
    weights = weights or {name: 1.0 for name in groups}
    first = next(iter(next(iter(groups.values()))))
    out = {}
    for iid, (rows, _) in first.items():
        total = 0.0
        for name, seeds in groups.items():
            for cache in seeds:
                if not np.array_equal(cache[iid][0], rows):
                    raise ValueError(f"{name}: instance {iid} scores different rows")
            total = total + weights.get(name, 0.0) * np.mean([_zscore(c[iid][1]) for c in seeds], axis=0)
        out[iid] = (rows, np.asarray(total, dtype=np.float64))
    return out


def instance_mrr(cache: dict, rows: dict[str, np.ndarray] | None = None) -> dict[str, float]:
    """Per instance, the MRR over the cache's rows (or the subset ``rows[iid]`` of them)."""
    out = {}
    for iid, (row_ids, scores) in cache.items():
        keep = np.ones(len(row_ids), dtype=bool) if rows is None else np.isin(row_ids, rows[iid])
        if keep.any():
            ranks = ranks_against_negatives(scores[keep, 0], scores[keep, 1:], tie_policy="pessimistic")
            out[iid] = float(np.mean(1.0 / ranks))
    return out


def save_cache(path: Path, cache: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for iid, (rows, scores) in cache.items():
        arrays[f"{iid}/rows"], arrays[f"{iid}/scores"] = rows, scores.astype(np.float32)
    np.savez_compressed(path, **arrays)


def load_cache(path: Path) -> dict:
    with np.load(path) as archive:
        ids = sorted({key.rsplit("/", 1)[0] for key in archive.files})
        return {iid: (archive[f"{iid}/rows"], archive[f"{iid}/scores"].astype(np.float64)) for iid in ids}


#: B0's per-expert weight grid (plan 03: chosen on F1-val boundary instances only).
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def fit_weights(groups: dict[str, list[dict]], instances, grid=WEIGHT_GRID, rounds: int = 3) -> dict[str, float]:
    """Weights maximising mean instance MRR on ``instances``, one expert at a time.

    Starts from equal weights and moves each expert's weight over ``grid`` in turn,
    keeping the best, for ``rounds`` passes or until nothing changes. Ties keep the
    current weight, so the result is deterministic.
    """
    # ponytail: coordinate ascent, not the full grid (6**12 mixtures); exhaustive search if B0 is close to a bar.
    names = sorted(groups)
    weights = {name: 1.0 for name in names}
    z = {name: [{iid: (rows, _zscore(s)) for iid, (rows, s) in c.items() if iid in instances} for c in seeds]
         for name, seeds in groups.items()}

    def value(w):
        scores = instance_mrr(fuse(z, w))
        return float(np.mean([scores[i] for i in instances]))

    best = value(weights)
    for _ in range(rounds):
        changed = False
        for name in names:
            for candidate in grid:
                trial = {**weights, name: candidate}
                if not any(trial.values()):
                    continue
                score = value(trial)
                if score > best + 1e-12:
                    best, weights, changed = score, trial, True
        if not changed:
            break
    return weights


def loo_b0(groups: dict[str, list[dict]], grid=WEIGHT_GRID) -> tuple[dict[str, float], dict[str, dict]]:
    """B0's leave-one-instance-out MRR per instance, and the weights each fold chose."""
    instances = sorted(next(iter(next(iter(groups.values())))))
    held, folds = {}, {}
    for out in instances:
        weights = fit_weights(groups, [i for i in instances if i != out], grid)
        held[out] = instance_mrr(fuse(groups, weights))[out]
        folds[out] = weights
    return held, folds
