"""F1-val MRR of several runs' best checkpoints, alone and with their logits averaged.

Selection-only: it reads F1-val and nothing else, scoring each instance the
way ``fit`` does (T instances are reported but left out of the mean).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import candidate_pairs, gated_mrr  # noqa: E402
from src.feature_io import read_feature_artifacts  # noqa: E402
from src.regions import RegionGraph  # noqa: E402
from src.suites import Suite  # noqa: E402
from train.checkpoints import load_expert  # noqa: E402
from train.train import _REPORT_ONLY_REGIMES, validate_m1_inputs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--suites-dir", required=True, type=Path)
    parser.add_argument("--run", required=True, action="append", type=Path)
    args = parser.parse_args(argv)

    graph = RegionGraph.read(args.processed_dir)
    table, schema = read_feature_artifacts(args.processed_dir, graph)
    tasks = validate_m1_inputs(graph, Suite.read(args.suite), args.suites_dir).tasks
    models = [load_expert(run, schema) for run in args.run]
    scores: dict[str, list[float]] = {}
    for view, task in tasks:
        visible, pairs = view.model_view(), candidate_pairs(task)
        width = task.negatives.shape[1] + 1
        logits = [m.score_inference(visible, table, pairs, row_width=width)[0].numpy() for m in models]
        scores[view.instance.instance_id] = [gated_mrr(task, x) for x in logits] + [
            gated_mrr(task, np.mean(logits, axis=0))
        ]
    selected = [
        i for (view, _), i in zip(tasks, scores) if view.instance.regime not in _REPORT_ONLY_REGIMES
    ]
    labels = [run.name for run in args.run] + ["ensemble"]
    for column, label in enumerate(labels):
        mean = np.mean([scores[i][column] for i in selected])
        print(f"{label}: selection MRR {mean:.4f}  "
              + " ".join(f"{i.split('-')[2]}={scores[i][column]:.4f}" for i in scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
