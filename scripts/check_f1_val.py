"""Check that every configured expert seed beats the F2-dev-selected baselines on F1-val."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train.region_config import load_region_config, region_paths  # noqa: E402
from train.milestones import best_f1_val  # noqa: E402
from train.train import _selection_mean  # noqa: E402
from scripts.scoring_common import load_verified_results, resolve_data_root  # noqa: E402
from src.gate import PreregConfig, select_champion_baseline  # noqa: E402


def compare_f1_val(rows, history: dict, expert_by_instance: dict[str, float], region: str):
    """Compare selection MRR against the F2-dev champion on identical F1-val instances."""
    prereg = PreregConfig(region=region)
    champion = select_champion_baseline(rows, prereg)
    selected = list(history.get("val_selection_instances") or expert_by_instance)
    if not selected or not set(selected).issubset(expert_by_instance):
        raise ValueError("expert history does not match its F1-val selection instances")
    baseline = rows[
        (rows.region == region)
        & (rows.split == "f1_val") & (rows.model == champion)
        & rows.mask_instance.isin(selected)
    ]
    expected = set(selected)
    actual = set(baseline.mask_instance)
    if actual != expected:
        raise ValueError("F1-val champion does not cover the expert selection instances")
    if baseline.duplicated(["mask_instance", "seed"]).any():
        raise ValueError("duplicate champion seed/instance F1-val rows")
    counts = baseline.groupby("mask_instance").seed.nunique()
    if set(counts.index) != expected or not counts.eq(3).all():
        raise ValueError("F1-val champion needs exactly three seeds per selected instance")
    baseline_means = baseline.groupby("mask_instance").mrr.mean()
    expert_mrr = float(_selection_mean(history, expert_by_instance))
    baseline_mrr = float(baseline_means.reindex(selected).mean())
    return {
        "champion": champion,
        "expert_mrr": expert_mrr,
        "baseline_mrr": baseline_mrr,
        "passed": expert_mrr > baseline_mrr,
        "instances": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--cohort", help="expert cohort digest to check")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--attempt", default="iteration-1")
    args = parser.parse_args(argv)
    try:
        args.data_root = resolve_data_root(args.data_root)
        args.runs_dir = args.runs_dir or args.data_root / "runs"
        config = load_region_config(args.config)
        cohort, manifests, rows, _ = load_verified_results(
            config, results_dir=REPO_ROOT / region_paths(config.region, "f1").results,
            runs_dir=args.runs_dir, data_root=args.data_root, cohort=args.cohort,
            seeds=config.seeds, splits=("f1_val", "f2_dev"), repo_root=REPO_ROOT,
        )
        run_scores = []
        for run_dir, manifest in manifests:
            history, selected = best_f1_val(run_dir, manifest.get("best_epoch"))
            comparison = compare_f1_val(rows, history, selected, config.region)
            run_scores.append((manifest["seed"], run_dir.name, comparison))
        signatures = {
            (item[2]["champion"], tuple(item[2]["instances"]), item[2]["baseline_mrr"])
            for item in run_scores
        }
        if len(signatures) != 1:
            raise ValueError("expert runs do not share one matched F1-val comparison")
        expert_mean = float(sum(item[2]["expert_mrr"] for item in run_scores) / len(run_scores))
        champion, _, baseline_mrr = next(iter(signatures))
        passed = expert_mean > baseline_mrr
        for seed, run_id, result in run_scores:
            print(f"seed={seed} run={run_id} F1-val MRR={result['expert_mrr']:.6f} "
                  f"{result['champion']}={result['baseline_mrr']:.6f}")
        print(f"F1-val check: {'PASS' if passed else 'FAIL'}; "
              f"cohort expert mean={expert_mean:.6f}; F2-dev champion={champion} "
              f"mean={baseline_mrr:.6f}")
        print(
            f"F1-val check record: - `{config.region.lower()}/{args.attempt}`: "
            f"cohort={cohort}; 3-seed expert F1-val compared with "
            f"F2-dev-selected {champion} (3-seed expert mean={expert_mean:.6f}; "
            f"champion mean={baseline_mrr:.6f}); "
            f"check={'PASS' if passed else 'FAIL'}"
        )
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"check_f1_val failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
