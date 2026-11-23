"""Score each region's frozen expert on every other region's suite (exploratory transfer matrix)."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features import FeatureSchema  # noqa: E402
from src.gate import PreregConfig, select_champion_baseline  # noqa: E402
from src.transfer import ablate_geometry, ablate_innervation, masked_oov_fraction, reencode_features  # noqa: E402
from train.checkpoints import load_expert  # noqa: E402
from train.cohorts import cohort_digest  # noqa: E402
from train.milestones import _read_manifest  # noqa: E402
from train.region_config import config_path, load_region_config, region_paths  # noqa: E402
from train.scoring import score_expert  # noqa: E402
from scripts.scoring_common import (  # noqa: E402
    _DATASETS, file_digest, load_scoring_inputs, load_verified_results,
    resolve_data_root, rung_for_suite, scoring_commit, scoring_input_identity,
)

SPLITS = ("f2_dev", "f1_val")


def load_region_expert(registry: pd.DataFrame, code: str, config, runs_dir: Path, data_root: Path):
    """One region's gate-seed models, keyed by seed, with the cohort and run directories."""
    mine = registry[(registry.region == code) & (registry.gate_seed == "true")]
    if sorted(mine.seed.astype(int)) != sorted(config.seeds):
        raise ValueError(f"{code}: registry gate seeds {sorted(mine.seed)} are not {list(config.seeds)}")
    if mine.cohort.nunique() != 1:
        raise ValueError(f"{code}: registry gate-seed rows do not share one cohort")
    schema = FeatureSchema.from_json(
        (region_paths(code, "f1", data_root).processed / "schema.json").read_text(encoding="utf-8")
    )
    models, run_dirs = {}, {}
    for row in mine.itertuples():
        run_dir = runs_dir / row.run_id
        manifest = _read_manifest(run_dir)
        if int(manifest["seed"]) != int(row.seed) or cohort_digest(manifest) != row.cohort:
            raise ValueError(f"{run_dir}: manifest does not match the registry row")
        models[int(row.seed)], run_dirs[int(row.seed)] = load_expert(run_dir, schema), run_dir
    return schema, models, run_dirs, mine.cohort.iloc[0]


def champion_baseline_rows(
    config, split: str, runs_dir: Path, data_root: Path, cohort: str, inputs=None
) -> pd.DataFrame:
    """The region's dev-chosen champion baseline, from its verified stored rows."""
    splits = ("f2_dev",) if split == "f2_dev" else ("f2_dev", "f1_val")
    _, _, rows, _ = load_verified_results(
        config, results_dir=REPO_ROOT / region_paths(config.region, "f2").results,
        runs_dir=runs_dir, data_root=data_root, cohort=cohort, seeds=config.seeds, splits=splits,
        loaded=None if inputs is None else {split: inputs},
    )
    champion = select_champion_baseline(rows, PreregConfig(region=config.region))
    return rows[(rows.model == champion) & (rows.split == split)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--regions", required=True, nargs="+")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError(f"{args.out} exists; refusing to overwrite")
        if len(set(args.regions)) != len(args.regions):
            raise ValueError("--regions repeats a region")
        data_root = resolve_data_root(args.data_root)
        runs_dir = args.runs_dir or data_root / "runs"
        commit = scoring_commit(REPO_ROOT)
        registry = pd.read_csv(args.registry, dtype=str)
        configs = {code: load_region_config(config_path(code)) for code in args.regions}
        experts = {
            code: load_region_expert(registry, code, configs[code], runs_dir, data_root)
            for code in args.regions
        }
        checkpoints = {
            code: {seed: file_digest(run_dir / "best.pt") for seed, run_dir in experts[code][2].items()}
            for code in args.regions
        }
        # Resolve every baseline first so a stale stored row fails before any inference.
        baselines = {
            j: champion_baseline_rows(configs[j], args.split, runs_dir, data_root, experts[j][3])
            for j in args.regions
        }
        frames = []
        for j in args.regions:
            suite_path = region_paths(j, _DATASETS[args.split], data_root).suites / f"{args.split}.json"
            inputs = load_scoring_inputs(configs[j], suite_path, data_root)
            graph, table_j, schema_j, types, _, suite, negatives = inputs
            shared = {
                "suite_digest": suite.digest(),
                "negative_file_digest": file_digest(suite_path.with_name(f"{suite.name}.negatives.json")),
                "scoring_commit": commit,
                **scoring_input_identity(configs[j], suite_path, data_root, inputs=inputs),
            }
            for i in args.regions:
                schema_i, models, run_dirs, cohort_i = experts[i]
                if i == j:
                    if schema_j.digest() != schema_i.digest():
                        raise ValueError(f"{i}: F1 and target schemas differ")
                    ablated = ablate_innervation(table_j, schema_j)
                    variants = {"home": table_j, "ablated": ablated,
                                "ablated_geo": ablate_geometry(ablated, schema_j)}
                else:
                    moved = reencode_features(table_j, schema_j, schema_i)
                    variants = {"transfer": moved, "transfer_geo": ablate_geometry(moved, schema_i)}
                for variant, table in variants.items():
                    for seed, model in models.items():
                        scored, _ = score_expert(
                            model, graph, table, suite, types=types, negatives=negatives,
                            rung=rung_for_suite(args.split), seed=seed,
                            run_id=run_dirs[seed].name, cohort=cohort_i,
                        )
                        frames.append(scored.assign(
                            model=f"gnn_a4@{i}", expert_region=i, variant=variant,
                            masked_oov_frac=scored["mask_instance"].map(
                                masked_oov_fraction(model, table, graph, suite)),
                            checkpoint_digest=checkpoints[i][seed], **shared,
                        ))
            frames.append(baselines[j].assign(variant="baseline", expert_region=j))
        tmp = args.out.with_name(f".{args.out.name}.{uuid.uuid4().hex}.tmp")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        try:
            pd.concat(frames, ignore_index=True).to_csv(tmp, index=False)
            os.replace(tmp, args.out)
        finally:
            tmp.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        print(f"transfer_matrix failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}: {len(args.regions)} region(s) on {args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
