"""Write plan 10's F3 cross-sex readout from the frozen experts' stored scores.

    python scripts/dimorphism_report.py --registry results/experts.csv \\
        --regions OL AL LH VLNP SNP --out f3-dimorphism.md

Reads each region's gate-seed expert rows on ``f3_eval`` and ``f2_dev``
(``score_run.py`` output under ``results/<code>/scored``), the suite type files
under the data root, and the raw F3 annotations for ``f3_type`` and
``dimorphism``. Everything involving F3 is exploratory: the report
states patterns, their intervals and counts, and makes no causal claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_suites import load_column  # noqa: E402
from scripts.scoring_common import load_verified_results, resolve_data_root  # noqa: E402
from src.dimorphism import (  # noqa: E402
    COMPARISONS, UNTYPED, categories, category_table, per_type_frame, per_type_mrr,
)
from src.gate import PreregConfig  # noqa: E402
from src.harmonize import normalize_f3_annotations  # noqa: E402
from train.region_config import config_path, load_region_config, region_paths  # noqa: E402

CONSERVED = ("OL", "AL", "LH")
DIMORPHIC = ("VLNP", "SNP")
SPLITS = {"f3": "f3_eval", "f2": "f2_dev"}


def expert_positives(registry: pd.DataFrame, region: str, args) -> dict[str, pd.DataFrame]:
    """The region's gate-seed expert ranks per dataset, after ``run_gate.py``'s checks.

    ``load_verified_results`` refuses a partly scored suite and rows from another
    cohort or another provenance, so a stale or missing row fails instead of moving
    the region MRR.
    """
    mine = registry[(registry["region"] == region) & (registry["gate_seed"].str.lower() == "true")]
    if mine.empty:
        raise ValueError(f"{region} has no gate-seed rows in the registry")
    if mine["cohort"].nunique() != 1:
        raise ValueError(f"{region}: registry gate-seed rows do not share one cohort")
    config = load_region_config(config_path(region))
    _, _, _, positives = load_verified_results(
        config, results_dir=args.results_root / region.lower(), runs_dir=args.runs_dir,
        data_root=args.data_root, cohort=mine["cohort"].iloc[0], seeds=config.seeds,
        splits=tuple(SPLITS.values()), models=("gnn_a4",),
    )
    return {dataset: positives[positives["split"] == split] for dataset, split in SPLITS.items()}


def region_mrr(positives: pd.DataFrame) -> float:
    """Gated MRR: single-masked positives per instance, then instances, then seeds."""
    single = positives.loc[~positives["both_masked"].astype(bool)]
    per_instance = single.groupby(["seed", "mask_instance"])["reciprocal_rank"].mean()
    return float(per_instance.groupby("seed").mean().mean())


def f3_side_table(path: Path) -> pd.DataFrame:
    """Root id → the release's own ``f3_type`` and its ``dimorphism`` label."""
    frame = normalize_f3_annotations(pd.read_parquet(path))[0]
    return frame.set_index("root_id")[["f3_type", "dimorphism"]]


def build_report(args, prereg: PreregConfig) -> str:
    registry = pd.read_csv(args.registry, dtype=str)
    side = f3_side_table(args.f3_annotations)
    labels = categories(side["f3_type"].dropna(), side.loc[side["f3_type"].notna(), "dimorphism"])
    lines = ["# F3 cross-sex readout", "",
             "Exploratory. Frozen experts, gate seeds only; nothing here selects or tunes.",
             "Intervals resample types (not mask instances, as the gate does) with the gate's bootstrap seed and "
             f"{prereg.n_bootstrap:,} resamples, BCa with the percentile fallback.", ""]
    region_rows, f3_types, ratios = [], [], []
    for region in args.regions:
        ranks = expert_positives(registry, region, args)
        f3, f2 = ranks["f3"], ranks["f2"]
        region_rows.append({"region": region, "f3_eval": region_mrr(f3), "f2_dev": region_mrr(f2)})
        f3_names = load_column(region_paths(region, "f3", args.data_root).interim / "f3_types.parquet", "cell_type")
        f2_names = load_column(region_paths(region, "f2", args.data_root).interim / "f2_types.parquet", "cell_type")
        own = per_type_mrr(per_type_frame(f3, side["f3_type"]))
        f3_types.append(own.assign(region=region))
        # The F3/F2 ratio joins F3's crosswalked name to F2's, never raw names across releases.
        f3_by_name = per_type_mrr(per_type_frame(f3, f3_names))
        f2_by_name = per_type_mrr(per_type_frame(f2, f2_names))
        joined = f3_by_name.merge(f2_by_name, on=["group", "type"], suffixes=("_f3", "_f2"))
        joined = joined.loc[(joined["type"] != UNTYPED) & (joined["mrr_f2"] > 0)]
        ratios.append(joined.assign(region=region, ratio=joined["mrr_f3"] / joined["mrr_f2"]))
    regions = pd.DataFrame(region_rows)

    lines += ["## 1. Conserved regions", ""]
    lines += _region_table(regions.loc[regions["region"].isin(CONSERVED)])
    if "OL" in args.regions:
        lines += ["", "OL's suites on every dataset never mask the lamina-dependent types, so "
                  "F3 and F2 score the same masked population. Lamina neurons stay in F3's graph as "
                  "context, but lack their lamina edges in F2, so the pair is still not like-for-like."]
    lines += ["", "## 2. Dimorphic regions", ""]
    lines += _region_table(regions.loc[regions["region"].isin(DIMORPHIC)])
    dimorphic = [frame for frame in f3_types if frame["region"].iat[0] in DIMORPHIC]
    if dimorphic:
        lines += ["", *_category_section(pd.concat(dimorphic), labels, prereg)]
    lines += ["", "## 3. Stratified readout, all scored regions", ""]
    lines += _category_section(pd.concat(f3_types), labels, prereg)
    lines += ["", "### F3/F2 MRR ratio per type", "",
              "Types scored on both datasets, joined on the crosswalked F1 name.", ""]
    lines += _ratio_table(pd.concat(ratios))
    above = regions.loc[regions["f3_eval"] > regions["f2_dev"], "region"].tolist()
    lines += ["", "## Reading", "",
              f"F3 scores above F2 in {len(above)} of {len(regions)} scored regions"
              f"{' (' + ', '.join(above) + ')' if 0 < len(above) < len(regions) else ''}. "
              "A gap that is uniform across categories is consistent with dataset shift, for example in "
              "how closely each release's connectivity matches F1's, and says nothing about "
              "sex. The sex-relevant readout is within F3, by category. A gap concentrated in dimorphic "
              "and male-specific types is consistent with sex differences. Neither is proof: typing "
              "quality, degree, proofreading or detector differences can differ by category too. "
              "Comparisons hold `type_trained` fixed, because an untrained type reads as OOV and scores "
              "lower whatever its sex. Each number states the types and positives behind it."]
    return "\n".join(lines) + "\n"


def _category_section(per_type: pd.DataFrame, labels: pd.Series, prereg: PreregConfig) -> list[str]:
    # Types are unique per region, so a type scored in two regions counts twice. The untyped
    # bucket goes first: prefixed, it would no longer match UNTYPED and would count as a type.
    per_type = per_type.loc[per_type["type"] != UNTYPED]
    if per_type.empty:
        return ["No typed masked neuron was scored, so there is no category table."]
    keyed = per_type.assign(type=per_type["region"] + ":" + per_type["type"])
    keyed_labels = pd.concat([labels.rename(lambda t, r=r: f"{r}:{t}") for r in per_type["region"].unique()])
    table = category_table(keyed, keyed_labels, prereg)
    lines = ["| regime | category | type trained | types | positives | MRR | 95% CI |",
             "|---|---|---|---|---|---|---|"]
    for row in table.itertuples():
        lines.append(f"| {row.group} | {row.category} | {row.type_trained} | {row.n_types} | "
                     f"{row.n_positives} | {row.mrr:.4f} | {_ci(row)} |")
    lines += ["", "Comparisons (each holds `type_trained` fixed):", ""]
    for left, left_trained, right, right_trained in COMPARISONS:
        for group in ("R/S", "T"):
            pick = lambda c, t: table.loc[(table["group"] == group) & (table["category"] == c)
                                          & (table["type_trained"] == str(t))]
            a, b = pick(left, left_trained), pick(right, right_trained)
            if a.empty or b.empty:
                lines.append(f"- {group}: {left} vs {right} (trained={left_trained}): not scored")
                continue
            a, b = a.iloc[0], b.iloc[0]
            lines.append(f"- {group}: {left} {a.mrr:.4f} ({a.n_types} types, {a.n_positives} positives) vs "
                         f"{right} {b.mrr:.4f} ({b.n_types} types, {b.n_positives} positives), "
                         f"type_trained={left_trained}")
    return lines


def _region_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["None of these regions was requested."]
    return ["| region | F3 f3_eval MRR | F2 f2_dev MRR |", "|---|---|---|",
            *(f"| {r.region} | {r.f3_eval:.4f} | {r.f2_dev:.4f} |" for r in frame.itertuples())]


def _ratio_table(frame: pd.DataFrame) -> list[str]:
    lines = ["| region | regime | types | median F3/F2 |", "|---|---|---|---|"]
    for (region, group), part in frame.groupby(["region", "group"], sort=True):
        lines.append(f"| {region} | {group} | {len(part)} | {part['ratio'].median():.3f} |")
    return lines


def _ci(row) -> str:
    return "n/a (one type)" if pd.isna(row.ci_low) else f"[{row.ci_low:.4f}, {row.ci_high:.4f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--regions", required=True, nargs="+")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--f3-annotations", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.data_root = resolve_data_root(args.data_root)
    args.runs_dir = args.runs_dir or args.data_root / "runs"
    args.f3_annotations = args.f3_annotations or args.data_root / "raw/f3/annotations.parquet"
    try:
        report = build_report(args, PreregConfig())
    except (OSError, ValueError, KeyError) as exc:
        print(f"dimorphism_report failed: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
