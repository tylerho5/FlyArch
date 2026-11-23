"""Per-instance confirm MRR of a region's three registered gate runs, for Phase 3's regional guard.

Reads this machine's stored ``f2_confirm`` rows through the same verifier ``run_gate.py gate`` uses and
makes the same checks before trusting them (the tagged delta, one declared milestone, the calibration
stamp on rows and positives), then checks ``--cohort`` is the reference registry's. It prints the
``reference`` block ``scripts/phase3_gate.py`` expects in its components file::

    python scripts/phase3_reference_mrr.py --config train/configs/al.toml --cohort <cohort> \
        --data-root <data> --runs-dir <data>/runs --out results/phase3/reference/AL.json

Read-only on rows. It scores nothing and opens no new suite; the confirm read must already have run here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.gate import PreregConfig  # noqa: E402
from train.region_config import load_region_config, region_paths  # noqa: E402
from scripts.scoring_common import (  # noqa: E402
    CALIBRATION_STAMP, load_verified_results, resolve_data_root, verify_tagged_delta,
)


def check_confirm_rows(confirm, positives, document: dict) -> str:
    """The checks ``run_gate.py gate`` makes before it trusts confirm rows (one nonempty milestone, rows and
    positives stamped with the frozen delta). Mirrors ``run_gate.py gate``; returns the milestone."""
    if ("milestone" not in confirm or confirm.milestone.isna().any()
            or confirm.milestone.astype(str).str.strip().eq("").any()):
        raise ValueError("confirm rows must record one nonempty milestone")
    milestones = confirm.milestone.astype(str).unique()
    if len(milestones) != 1:
        raise ValueError("confirm rows must record one nonempty milestone")
    for frame in (confirm, positives):
        for field in CALIBRATION_STAMP:
            if field not in frame or set(frame[field].astype(str)) != {document[field]}:
                raise ValueError("confirm rows were not scored under the current tagged delta")
    return str(milestones[0])


def registry_cohort(region: str) -> str:
    """The reference registry's cohort for ``region`` (its gate rows' cohort column); raises unless exactly one."""
    with (REPO_ROOT / "results" / "experts.csv").open(encoding="utf-8", newline="") as handle:
        cohorts = {row["cohort"] for row in csv.DictReader(handle)
                   if row.get("region") == region and row.get("gate_seed") == "true"}
    if len(cohorts) != 1:
        raise ValueError(f"results/experts.csv has {len(cohorts)} gate cohorts for {region}, expected exactly one")
    return cohorts.pop()


def reference_mrr(rows, gnn_model: str, seeds: int) -> dict[str, dict[str, float]]:
    """{run_id: {mask_instance: mrr}} for the gate runs' f2_confirm rows (one value per run and instance)."""
    confirm = rows[(rows.split == "f2_confirm") & (rows.model == gnn_model)]
    if confirm.empty:
        raise ValueError("no f2_confirm rows for the GNN model; the confirm read has not run on this machine")
    out: dict[str, dict[str, float]] = {}
    for run_id, frame in confirm.groupby(confirm.run_id.astype(str)):
        per = frame.groupby("mask_instance")["mrr"]
        if (per.count() != 1).any():
            raise ValueError(f"{run_id} has more than one row for an instance; refusing to average it away")
        out[run_id] = {str(k): float(v) for k, v in per.first().items()}
    if len(out) != seeds:
        raise ValueError(f"expected {seeds} gate runs, found {len(out)}: {sorted(out)}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_region_config(args.config)
        data_root = resolve_data_root(args.data_root)
        results_dir = REPO_ROOT / region_paths(config.region, "f2").results
        document = verify_tagged_delta(region=config.region, expected_cohort=args.cohort,
                                       delta_path=results_dir / "delta.json", repo_root=REPO_ROOT)
        _, _, rows, positives = load_verified_results(
            config, seeds=config.seeds, splits=("f2_confirm",),
            results_dir=results_dir, runs_dir=args.runs_dir or data_root / "runs",
            data_root=data_root, cohort=args.cohort, repo_root=REPO_ROOT,
        )
        milestone = check_confirm_rows(rows[rows.split == "f2_confirm"],
                                       positives[positives.split == "f2_confirm"], document)
        registry = registry_cohort(config.region)
        if registry != args.cohort:
            raise ValueError(f"--cohort {args.cohort[:8]} is not the registry's cohort {registry[:8]} "
                             f"for {config.region}")
        prereg = PreregConfig(region=config.region)
        result = {"region": config.region, "cohort": args.cohort, "milestone": milestone,
                  "reference": reference_mrr(rows, prereg.gnn_model, prereg.n_seeds_required)}
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8", newline="\n")
        else:
            sys.stdout.write(text)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"phase3_reference_mrr failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
