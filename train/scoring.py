"""One shared expert scoring path for command-line and transfer analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.evaluate import (
    CommittedNegatives,
    build_scoring_task,
    candidate_pairs,
    results_frame,
    score_instance,
)
from src import negatives as _negatives
from src.metrics import ranks_against_negatives
from src.masks import apply_mask
from src.negatives import DEFAULT_SPEC
from src.regions import RegionGraph
from src.suites import Suite, suite_negatives
from src.features import FeatureTable
from .model import M1

__all__ = ["baseline_fit_views", "score_expert"]

#: The baselines' F1 fit views.
BASELINE_FIT_FRACTION = 0.05
BASELINE_FIT_VIEWS = 3


def baseline_fit_views(context) -> list:
    """The pinned fit views; view ``i`` draws from ``master_seed`` and ``baseline-fit-view-<i>``."""
    return [
        context.draw_fit_view(
            _negatives._derive_rng(DEFAULT_SPEC.master_seed, f"baseline-fit-view-{index}"),
            fraction=BASELINE_FIT_FRACTION,
        )
        for index in range(BASELINE_FIT_VIEWS)
    ]


def score_expert(
    model: M1,
    graph: RegionGraph,
    table: FeatureTable,
    suite: Suite,
    *,
    types: pd.Series,
    negatives: CommittedNegatives,
    rung: str,
    seed: int = 0,
    run_id: str = "inference",
    cohort: str = "inference",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score all suite instances and return gate rows plus one rank row per positive."""
    if table.columns != model.schema.column_names:
        raise ValueError("feature table columns do not match the expert schema")

    def predict(view, pairs):
        scorer = getattr(model, "score_inference", None)
        with torch.no_grad():
            output = (
                scorer(view, table, pairs, row_width=negatives.spec.k + 1)
                if callable(scorer)
                else model.score(view, table, pairs, row_width=negatives.spec.k + 1)
            )
        logits = output[0] if isinstance(output, tuple) else output
        return logits.detach().cpu().numpy()

    return _score_predictor(
        predict,
        graph, suite, types=types, negatives=negatives, rung=rung,
        model_name="gnn_a4", seed=seed, run_id=run_id, cohort=cohort,
        trained_types=trained_type_names(model),
    )


def trained_type_names(model: M1) -> set[str]:
    """The F1 type names the expert has its own embedding row for; the rest read as OOV."""
    trained = model.trained_types.detach().cpu().numpy()
    return {name for name, index in model.schema.type_vocab.items() if trained[index]}


def score_baseline(
    predict,
    graph: RegionGraph,
    table: FeatureTable,
    suite: Suite,
    *,
    types: pd.Series,
    negatives: CommittedNegatives,
    rung: str,
    seed: int,
    run_id: str,
    cohort: str,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one already-fitted baseline seed through the same result path."""
    return _score_predictor(
        lambda view, pairs: predict(view, table, pairs),
        graph, suite, types=types, negatives=negatives, rung=rung,
        model_name=model_name, seed=seed, run_id=run_id,
        cohort=cohort,
    )


def _score_predictor(
    predict,
    graph: RegionGraph,
    suite: Suite,
    *,
    types: pd.Series,
    negatives: CommittedNegatives,
    rung: str,
    model_name: str,
    seed: int,
    run_id: str,
    cohort: str,
    trained_types: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    suite.validate(graph)
    samples = suite_negatives(graph, suite, spec=negatives.spec)
    results, positive_frames = [], []
    for instance in suite.resolve(graph):
        view = apply_mask(graph, instance)
        task = build_scoring_task(
            view, samples[instance.instance_id], suite=suite, negatives=negatives,
            types=types, trained_types=trained_types,
        )

        def score(pairs):
            with torch.no_grad():
                return np.asarray(predict(view.model_view(), pairs), dtype=np.float64)

        candidates = candidate_pairs(task)
        all_scores = np.asarray(score(candidates), dtype=np.float64).reshape(
            len(task.positives), -1
        )
        results.append(score_instance(task, lambda pairs: all_scores.reshape(-1)))
        ranks = ranks_against_negatives(all_scores[:, 0], all_scores[:, 1:])
        masked = view.is_masked
        masked_endpoint = np.where(masked[task.positives[:, 0]], 0, 1)
        endpoints = task.positives[np.arange(len(task.positives)), masked_endpoint]
        frame = pd.DataFrame({
            "mask_instance": task.instance_id,
            "seed": int(seed),
            "model": model_name,
            "dataset": task.dataset,
            "region": task.region,
            "split": task.suite,
            "degree_setting": task.degree_setting,
            "mask_regime": rung,
            "root_id": graph.node_ids[endpoints],
            "positive_index": np.arange(len(task.positives)),
            "reciprocal_rank": 1.0 / ranks,
            "both_masked": view.both_masked(),
            # plan 10 splits the per-type readout by these, so it needs no rescoring
            "type_matched": task.strata["type_matched"].to_numpy(),
            "type_trained": task.strata["type_trained"].to_numpy(),
            "negatives_digest": task.negatives_digest,
            "run_id": run_id,
            "cohort": cohort,
        })
        positive_frames.append(frame)
    rows = results_frame(results, model=model_name, seed=seed, regime=rung)
    rows["run_id"] = run_id
    rows["cohort"] = cohort
    positives = pd.concat(positive_frames, ignore_index=True) if positive_frames else pd.DataFrame()
    return rows, positives
