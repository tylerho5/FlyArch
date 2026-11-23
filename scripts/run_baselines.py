"""Fit the registered baselines on F1, select on F2-dev, and score one suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.masks import apply_mask  # noqa: E402
from src.negatives import DEFAULT_SPEC  # noqa: E402
from train.baselines import BASELINES, F1TrainingContext  # noqa: E402
from train.region_config import load_region_config, region_paths  # noqa: E402
from train.scoring import baseline_fit_views, score_baseline  # noqa: E402
from scripts.scoring_common import (  # noqa: E402
    CALIBRATION_STAMP,
    append_results, baseline_fit_digest, confirm_preflight,
    dev_fit_identity, file_digest, inspect_suite_identity, load_scoring_inputs,
    resolve_cohort_manifests, resolve_data_root, rung_for_suite, scoring_commit, scoring_input_identity,
    validate_f1_cohort_inputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--milestone", help="f2_confirm only: a nonempty label recorded on the confirm rows")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--cohort", help="expert cohort digest to associate with these baseline rows")
    args = parser.parse_args(argv)
    try:
        args.data_root = resolve_data_root(args.data_root)
        args.runs_dir = args.runs_dir or args.data_root / "runs"
        config = load_region_config(args.config)
        suite_name, _ = inspect_suite_identity(config, args.suite, args.data_root)
        cohort, manifests = resolve_cohort_manifests(
            args.runs_dir, config, args.cohort, seeds=config.seeds,
        )
        paths = region_paths(config.region, "f2", args.data_root)
        stamp = {}
        if suite_name == "f2_confirm":
            calibration = confirm_preflight(
                region=config.region, milestone=args.milestone,
                delta_path=REPO_ROOT / paths.results / "delta.json", expected_cohort=cohort,
                repo_root=REPO_ROOT,
            )
            stamp = {field: calibration[field] for field in CALIBRATION_STAMP}
            found_seeds = {int(manifest["seed"]) for _, manifest in manifests}
            if found_seeds != set(config.seeds):
                raise ValueError(
                    f"confirm baseline scoring requires configured seed manifests "
                    f"{list(config.seeds)}; found {sorted(found_seeds)}"
                )
        commit = scoring_commit(REPO_ROOT)
        f1_identity = validate_f1_cohort_inputs(config, [manifest for _, manifest in manifests], args.data_root)

        f1_graph, f1_table, _, _, _, f1_suite, _ = f1_identity["inputs"]
        context = F1TrainingContext(f1_graph, f1_suite)
        dev_inputs = load_scoring_inputs(
            config, paths.suites / "f2_dev.json", args.data_root
        )
        dev_graph, dev_table, _, dev_types, _, dev_suite, dev_negatives = dev_inputs
        dev_identity = dev_fit_identity(config, args.data_root, inputs=dev_inputs)
        dev_views = [apply_mask(dev_graph, item) for item in dev_suite.resolve(dev_graph)]
        scoring_inputs = load_scoring_inputs(config, args.suite, args.data_root)
        graph, table, _, types, _, suite, negatives = scoring_inputs
        scoring_identity = scoring_input_identity(
            config, args.suite, args.data_root, inputs=scoring_inputs
        ) | stamp
        fit_views = baseline_fit_views(context)
        shared = {
            "milestone": args.milestone,
            "suite_digest": suite.digest(),
            "negative_file_digest": file_digest(args.suite.with_name(f"{suite.name}.negatives.json")),
            "scoring_commit": commit,
            **scoring_identity,
        }
        rung = rung_for_suite(suite.name)
        output_rows, output_positives = [], []
        for name, cls in BASELINES.items():
            baseline = cls(negative_spec=DEFAULT_SPEC)
            baseline.fit(
                fit_views, [f1_table] * len(fit_views), context=context,
                val_views=dev_views, val_tables=[dev_table] * len(dev_views),
                val_suite=dev_suite, val_negatives=dev_negatives,
            )
            baseline_seeds = getattr(baseline, "seeds", None)
            if baseline_seeds is None:
                baseline_seeds = np.random.SeedSequence(DEFAULT_SPEC.master_seed).generate_state(3)
            if len(baseline_seeds) != 3:
                raise ValueError(f"{name} fitted {len(baseline_seeds)} seeds; exactly three are required")
            for seed in baseline_seeds:
                fit_seed = int(seed)
                predict = (
                    (lambda view, table, pairs, m=baseline: m.score(view, table, pairs))
                    if name == "configuration" else
                    (lambda view, table, pairs, m=baseline, s=fit_seed:
                     m.score_seed(view, table, pairs, seed=s))
                )
                rows, positives = score_baseline(
                    predict, graph, table, suite, types=types, negatives=negatives,
                    rung=rung, seed=fit_seed,
                    run_id=f"baseline-{name}-s{fit_seed}", cohort=cohort,
                    model_name=name,
                )
                stamp = shared | {"baseline_fit_inputs_digest": baseline_fit_digest(
                    cohort, f1_identity, name, fit_seed, dev_identity,
                )}
                output_rows.append(rows.assign(**stamp))
                output_positives.append(positives.assign(**stamp))
        append_results(REPO_ROOT / paths.results, pd.concat(output_rows, ignore_index=True),
                       pd.concat(output_positives, ignore_index=True))
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        print(f"run_baselines failed: {exc}", file=sys.stderr)
        return 2
    print(f"scored three baselines on {suite.name}; cohort={cohort}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
