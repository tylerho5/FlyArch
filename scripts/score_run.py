"""Score one or more frozen experts against a committed region suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train.checkpoints import load_expert  # noqa: E402
from train.cohorts import cohort_digest  # noqa: E402
from train.milestones import _read_manifest  # noqa: E402
from train.region_config import load_region_config  # noqa: E402
from train.scoring import score_expert  # noqa: E402
from scripts.scoring_common import (  # noqa: E402
    CALIBRATION_STAMP,
    append_results, confirm_preflight, inspect_suite_identity,
    file_digest, load_scoring_inputs, manifest_region, resolve_data_root, rung_for_suite, scoring_commit, scoring_input_identity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, action="append", type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--milestone", help="f2_confirm only: a nonempty label recorded on the confirm rows")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    try:
        args.data_root = resolve_data_root(args.data_root)
        config = load_region_config(args.config)
        suite_name, paths = inspect_suite_identity(config, args.suite, args.data_root)
        manifests, cohorts = [], []
        for run_dir in args.run:
            manifest = _read_manifest(run_dir)
            if manifest_region(manifest) != config.region:
                raise ValueError(f"{run_dir} belongs to another region")
            if suite_name == "f2_confirm" and int(manifest["seed"]) not in config.seeds:
                raise ValueError(f"confirm scoring refuses non-gate seed {manifest['seed']}")
            manifests.append(manifest)
            cohorts.append(cohort_digest(manifest))
        if len(set(cohorts)) != 1:
            raise ValueError("requested runs do not share one scoring cohort")
        stamp = {}
        if suite_name == "f2_confirm":
            calibration = confirm_preflight(
                region=config.region, milestone=args.milestone,
                delta_path=REPO_ROOT / paths.results / "delta.json", expected_cohort=cohorts[0],
                repo_root=REPO_ROOT,
            )
            stamp = {field: calibration[field] for field in CALIBRATION_STAMP}
        commit = scoring_commit(REPO_ROOT)
        inputs = load_scoring_inputs(config, args.suite, args.data_root)
        graph, table, schema, types, _, suite, negatives = inputs
        scoring_identity = scoring_input_identity(
            config, args.suite, args.data_root, inputs=inputs
        ) | stamp
        scored_runs = []
        for run_dir, manifest, cohort in zip(args.run, manifests, cohorts, strict=True):
            model = load_expert(run_dir, schema)
            scored, ranked = score_expert(
                model, graph, table, suite, types=types, negatives=negatives,
                rung=rung_for_suite(suite.name), seed=int(manifest["seed"]),
                run_id=run_dir.name, cohort=cohort,
            )
            scored_runs.append((scored, ranked, file_digest(run_dir / "best.pt")))
        shared = {
            "milestone": args.milestone,
            "suite_digest": suite.digest(),
            "negative_file_digest": file_digest(args.suite.with_name(f"{suite.name}.negatives.json")),
            "scoring_commit": commit,
            **scoring_identity,
        }
        rows = [scored.assign(**shared, checkpoint_digest=digest) for scored, _, digest in scored_runs]
        positives = [ranked.assign(**shared, checkpoint_digest=digest) for _, ranked, digest in scored_runs]
        append_results(REPO_ROOT / paths.results, pd.concat(rows, ignore_index=True),
                       pd.concat(positives, ignore_index=True))
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        print(f"score_run failed: {exc}", file=sys.stderr)
        return 2
    print(f"scored {len(args.run)} run(s) on {suite.name}; cohort={cohorts[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
