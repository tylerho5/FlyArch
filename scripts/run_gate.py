"""Calibrate the per-region noise floor or evaluate its pre-registered gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.gate import REQUIRED_COLUMNS, PreregConfig, calibrate_delta, evaluate_gate  # noqa: E402
from train.region_config import load_region_config, region_paths  # noqa: E402
from scripts.scoring_common import (  # noqa: E402
    CALIBRATION_STAMP, load_result_tables, load_verified_results, refuse_recalibration,
    resolve_data_root, verify_tagged_delta,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("mode", choices=("calibrate", "gate"))
    parser.add_argument("--cohort")
    parser.add_argument("--delta", type=float)
    parser.add_argument("--tag", help="tag name to record in delta.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_region_config(args.config)
        data_root = resolve_data_root(args.data_root)
        results_dir = REPO_ROOT / region_paths(config.region, "f2").results
        delta_path = results_dir / "delta.json"
        verify = dict(
            results_dir=results_dir, runs_dir=args.runs_dir or data_root / "runs",
            data_root=data_root, cohort=args.cohort, repo_root=REPO_ROOT,
        )
        if args.mode == "calibrate":
            if delta_path.exists():
                raise ValueError(f"{delta_path} already exists; calibration is frozen")
            refuse_recalibration(config.region, REPO_ROOT)
            if (load_result_tables(results_dir)[0].split == "f2_confirm").any():
                raise ValueError("confirm rows already exist; calibration cannot follow confirm scoring")
            cohort, manifests, rows, _ = load_verified_results(
                config, seeds=config.calibration_seeds, splits=("f1_val",),
                models=("gnn_a4",), **verify,
            )
            s, delta = calibrate_delta(rows, PreregConfig(region=config.region))
            tag = args.tag or f"delta-{config.region.lower()}-{cohort[:8]}"
            document = {
                "region": config.region,
                "s": s,
                "delta": delta,
                "run_ids": [run_dir.name for run_dir, _ in manifests],
                "cohort": cohort,
                "tag": tag,
            }
            results_dir.mkdir(parents=True, exist_ok=True)
            temp = delta_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8", newline="\n")
            temp.replace(delta_path)
            print(json.dumps(document, indent=2, sort_keys=True))
            print(f"Commit delta.json, then tag that commit {tag!r} before any confirm scoring.")
            return 0

        if not args.cohort:
            raise ValueError("gate mode requires --cohort")
        document = verify_tagged_delta(
            region=config.region, expected_cohort=args.cohort,
            delta_path=delta_path, repo_root=REPO_ROOT,
        )
        delta = float(document["delta"])
        if args.delta is not None and args.delta != delta:
            raise ValueError(f"--delta {args.delta} differs from committed delta {delta}")
        _, _, selected, positives = load_verified_results(
            config, seeds=config.seeds, splits=("f2_dev", "f2_confirm"), **verify,
        )
        confirm = selected[selected.split == "f2_confirm"]
        if ("milestone" not in confirm or confirm.milestone.isna().any()
                or confirm.milestone.astype(str).str.strip().eq("").any()):
            raise ValueError("confirm rows must record one nonempty milestone")
        milestones = confirm.milestone.astype(str).unique()
        if len(milestones) != 1:
            raise ValueError("confirm rows must record one nonempty milestone")
        for frame in (confirm, positives[positives.split == "f2_confirm"]):
            for field in CALIBRATION_STAMP:
                if field not in frame or set(frame[field].astype(str)) != {document[field]}:
                    raise ValueError("confirm rows were not scored under the current tagged delta")
        prereg = PreregConfig(delta=delta, region=config.region)
        verdict = evaluate_gate(selected[list(REQUIRED_COLUMNS) + ["mrr"]], prereg)
        print(verdict.summary())
        runs = sorted(confirm.run_id.dropna().astype(str).unique())
        milestone = str(milestones[0])
        print(f"confirm-score record: {config.region}/{milestone}; "
              f"runs={','.join(runs)}; gate={'PASS' if verdict.passed else 'FAIL'}; "
              f"delta={delta:.2f} ({document['calibration_tag_commit']})")
        return 0 if verdict.passed else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"run_gate failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
