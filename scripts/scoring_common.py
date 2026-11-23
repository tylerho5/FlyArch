"""Shared Phase 2 scoring CLI input, provenance and output rules."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import uuid
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np

from src.evaluate import CommittedNegatives
from src.gate import REQUIRED_COLUMNS
from src.digests import sha256 as file_digest
from src.feature_io import read_feature_artifacts
from src.regions import REGIONS, RegionGraph
from src.run_control import blocking_lock, replace
from src.suites import Suite
from train.cohorts import cohort_digest
from train.milestones import _read_manifest, best_f1_val
from train.region_config import RegionConfig, region_paths

REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASETS = {"f1_val": "f1", "f2_dev": "f2", "f2_confirm": "f2", "f2_side": "f2", "f3_eval": "f3"}
_SUITE_FIELDS = (
    "name", "dataset", "region", "graph_digest", "threshold", "master_seed", "instances",
)
_SUITE_FIELDS_WITH_EXCLUSIONS = _SUITE_FIELDS + ("mask_exclude_types",)
_CODE_DIRS = ("src", "train", "scripts")
#: The published calibrations and deposited score files (2026-11-22 amendment).
RELEASE_EVIDENCE = "results/release-evidence.json"
#: DataFrame.attrs key: a deposited original score file's commit predates this repository's root.
_PINNED_COMMITS = "pinned_scoring_commits"
_CALIBRATION_FIELDS = frozenset({"region", "path", "cohort", "tag", "source_tag_commit", "delta_sha256"})
_SCORING_IDENTITY_FIELDS = (
    "scoring_graph_digest", "scoring_feature_digest", "scoring_schema_digest",
    "scoring_types_digest", "scoring_sides_digest",
)
CALIBRATION_STAMP = ("calibration_tag_commit", "calibration_delta_digest")
BASELINE_MODELS = frozenset({"configuration", "logreg", "xgboost"})
_RUN_KEY = [*REQUIRED_COLUMNS, "cohort"]
_POSITIVE_KEY = [*_RUN_KEY, "positive_index"]


def resolve_data_root(argument: Path | None) -> Path:
    """``--data-root``, then ``FLYARCH_DATA_ROOT``, then the repository's ``data/``."""
    return Path(argument or os.environ.get("FLYARCH_DATA_ROOT") or REPO_ROOT / "data")


def rung_for_suite(name: str) -> str:
    if name == "f1_val":
        return "b"
    if name in {"f2_dev", "f2_confirm", "f2_side"}:
        return "c"
    if name == "f3_eval":
        return "x"
    raise ValueError(f"suite {name!r} has no pre-registered ladder rung")


def inspect_suite_identity(config: RegionConfig, suite_path: Path, data_root=None):
    """Reject aliases and validate suite metadata before opening scoring inputs."""
    suite_path = Path(suite_path)
    name = suite_path.stem
    if name not in _DATASETS or suite_path.name != f"{name}.json":
        raise ValueError(f"unsupported suite path {suite_path}")
    paths = region_paths(config.region, _DATASETS[name], data_root)
    expected = paths.suites / f"{name}.json"
    if suite_path.is_symlink() or suite_path.absolute() != expected.absolute():
        raise ValueError(f"suite path must be the canonical {expected}")
    metadata = _read_suite_document(suite_path)
    expected_dataset = _DATASETS[name]
    if (metadata.get("name") != name or metadata.get("dataset") != expected_dataset
            or metadata.get("region") != config.region):
        raise ValueError(f"{suite_path}: suite name/dataset/region do not match its canonical path")
    return name, paths


def manifest_region(manifest: dict) -> str | None:
    region = manifest.get("region")
    if isinstance(region, str) and region in REGIONS:
        return region
    run_id = manifest.get("run_id", "")
    match = re.match(r"^f1-([a-z]+)-s\d{3}-", str(run_id), re.IGNORECASE)
    return match.group(1).upper() if match else None


def resolve_cohort_manifests(
    runs_dir: Path, config: RegionConfig, cohort: str | None = None, *,
    seeds: tuple[int, ...] | None = None, require_complete=False,
) -> tuple[str, list[tuple[Path, dict]]]:
    expected_seeds = tuple(config.seeds if seeds is None else seeds)
    by_cohort: dict[str, dict[int, tuple[Path, dict]]] = {}
    for path in runs_dir.glob("*/manifest.json"):
        run_dir = path.parent
        try:
            manifest = _read_manifest(run_dir)
            if manifest_region(manifest) != config.region:
                continue
            seed = int(manifest["seed"])
            if seed not in expected_seeds:
                continue
            digest = cohort_digest(manifest)
        except (OSError, ValueError, TypeError):
            continue
        if seed in by_cohort.setdefault(digest, {}):
            raise ValueError(f"multiple {config.region} manifests for gate seed {seed}")
        by_cohort[digest][seed] = (run_dir, manifest)
    if cohort is None:
        if len(by_cohort) != 1:
            raise ValueError(
                f"expected one cohort among the configured gate-seed runs in {runs_dir}, "
                f"found {len(by_cohort)}; train or select the region's expert cohort first"
            )
        cohort = next(iter(by_cohort))
    if cohort not in by_cohort:
        raise ValueError("no matching region manifest establishes the requested cohort")
    selected = by_cohort[cohort]
    if require_complete and set(selected) != set(expected_seeds):
        raise ValueError(
            f"requires configured seed manifests {list(expected_seeds)} "
            f"for cohort {cohort}; found {sorted(selected)}"
        )
    return cohort, [selected[seed] for seed in sorted(selected)]


def validate_f1_cohort_inputs(
    config: RegionConfig, manifests: list[dict], data_root: Path | None = None,
) -> dict:
    """Bind an expert cohort to current F1 graph/features/schema and selection suite."""
    from train.train import _feature_digest, extras_read

    paths = region_paths(config.region, "f1", data_root)
    inputs = load_scoring_inputs(config, paths.suites / "f1_val.json", data_root)
    graph, table, schema, _, _, suite, _ = inputs
    current = {
        "graph_digest": graph.digest(),
        "feature_digest": _feature_digest(table, extras=extras_read(manifests[0]["config"]) if manifests else ()),
        "schema_digest": schema.digest(),
        "suite_digest": suite.digest(),
    }
    for manifest in manifests:
        if manifest_region(manifest) != config.region:
            raise ValueError(f"run {manifest.get('run_id')} does not identify region {config.region}")
        for name, value in current.items():
            if manifest.get(name) != value:
                raise ValueError(
                    f"run {manifest.get('run_id')} {name} does not match current F1 inputs"
                )
    return {
        **current,
        "negative_file_digest": file_digest(paths.suites / "f1_val.negatives.json"),
        "inputs": inputs,
    }


def load_scoring_inputs(config: RegionConfig, suite_path: Path, data_root: Path | None = None):
    """Load graph, schema, table, type labels, suite and committed negatives."""
    name, paths = inspect_suite_identity(config, suite_path, data_root)
    dataset = _DATASETS[name]
    graph = RegionGraph.read(paths.processed)
    table, schema = read_feature_artifacts(paths.processed, graph)
    type_path = paths.interim / f"{dataset}_types.parquet"
    if not type_path.is_file():
        raise FileNotFoundError(f"required suite input is missing: {type_path}")
    type_frame = pd.read_parquet(type_path)
    types = type_frame.set_index("root_id")["cell_type"].reindex(graph.node_ids)
    sides_path = paths.interim / f"{dataset}_sides.parquet"
    sides = (
        pd.read_parquet(sides_path).set_index("root_id")["side"].reindex(graph.node_ids)
        if sides_path.is_file() else None
    )
    try:
        suite = Suite.from_json(_read_suite_document(suite_path))
    except Exception as exc:
        raise ValueError(f"{suite_path}: suite is unreadable: {exc}") from exc
    suite.validate(graph)
    negatives = CommittedNegatives.read(
        suite_path.with_name(f"{suite_path.stem}.negatives.json")
    )
    if negatives.name != name:
        raise ValueError(f"{suite_path}: committed negatives name does not match suite")
    return graph, table, schema, types, sides, suite, negatives


def _read_suite_document(suite_path: Path) -> dict:
    try:
        document = json.loads(
            suite_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_suite_keys
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{suite_path}: suite is unreadable: {exc}") from exc
    if not isinstance(document, dict) or tuple(document) not in (
        _SUITE_FIELDS, _SUITE_FIELDS_WITH_EXCLUSIONS,
    ):
        raise ValueError(f"{suite_path}: suite is not in canonical Suite.write format")
    return document


def _unique_suite_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate suite key {key!r}")
        result[key] = value
    return result


def append_results(
    results_dir: Path, rows: pd.DataFrame, positives: pd.DataFrame
) -> None:
    """Publish one scoring call's rows and ranks after refusing any stored row key."""
    for frame, key in ((rows, _RUN_KEY), (positives, _POSITIVE_KEY)):
        missing = [column for column in key if column not in frame.columns]
        if missing:
            raise ValueError(f"scoring output is missing key columns: {missing}")
        if frame.duplicated(key).any():
            raise ValueError("scoring output contains duplicate result keys")
    _validate_score_positive_pair(rows, positives)
    parts = results_dir / "scored"
    with blocking_lock(parts / ".lock"):
        stored, _ = _read_parts(parts)
        if not stored.empty and stored.merge(rows[_RUN_KEY], on=_RUN_KEY).shape[0]:
            raise ValueError("score rows already exist for this cohort and run; refusing rerun")
        name = uuid.uuid4().hex
        # Scores publish last: a call without its scores file was never stored.
        for frame, kind in ((positives, "positives"), (rows, "scores")):
            target = parts / f"{name}.{kind}.parquet"
            staged = parts / f".{target.name}.tmp"
            frame.to_parquet(staged, index=False)
            replace(staged, target)


def _read_parts(parts: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, positives = [], []
    for path in sorted(parts.glob("*.scores.parquet")):
        rows.append(pd.read_parquet(path))
        positives.append(pd.read_parquet(path.with_name(path.name.replace(".scores.", ".positives."))))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(rows, ignore_index=True), pd.concat(positives, ignore_index=True)


def scoring_commit(repo_root: Path = REPO_ROOT) -> str:
    """The commit whose code scores; refuses uncommitted edits to that code."""
    if _git(repo_root, "status", "--porcelain", "--", *_CODE_DIRS).strip():
        raise ValueError(f"commit or stash changes under {', '.join(_CODE_DIRS)} before scoring")
    return _git(repo_root, "rev-parse", "HEAD").decode().strip()


def scoring_input_identity(config: RegionConfig, suite_path: Path, data_root=None, *, inputs=None):
    from train.train import _feature_digest, extras_read

    name, paths = inspect_suite_identity(config, suite_path, data_root)
    if inputs is None:
        inputs = load_scoring_inputs(config, suite_path, data_root)
    graph, table, schema, _, sides, _, _ = inputs
    dataset = _DATASETS[name]
    types_path = paths.interim / f"{dataset}_types.parquet"
    sides_path = paths.interim / f"{dataset}_sides.parquet"
    return {
        "scoring_graph_digest": graph.digest(),
        "scoring_feature_digest": _feature_digest(table, extras=extras_read(config.train)),
        "scoring_schema_digest": schema.digest(),
        "scoring_types_digest": file_digest(types_path),
        "scoring_sides_digest": file_digest(sides_path) if sides is not None else "absent",
    }


def dev_fit_identity(config: RegionConfig, data_root=None, *, inputs=None) -> dict:
    from train.train import _feature_digest, extras_read

    paths = region_paths(config.region, "f2", data_root)
    if inputs is None:
        inputs = load_scoring_inputs(config, paths.suites / "f2_dev.json", data_root)
    graph, table, schema, _, _, suite, _ = inputs
    return {
        "graph_digest": graph.digest(), "feature_digest": _feature_digest(table, extras=extras_read(config.train)),
        "schema_digest": schema.digest(), "suite_digest": suite.digest(),
        "negative_file_digest": file_digest(paths.suites / "f2_dev.negatives.json"),
    }


def baseline_fit_digest(cohort: str, f1_identity: dict, model: str, seed: int,
                        dev_identity: dict) -> str:
    payload = {
        "cohort": cohort,
        "f1_graph_digest": f1_identity["graph_digest"],
        "f1_feature_digest": f1_identity["feature_digest"],
        "f1_schema_digest": f1_identity["schema_digest"],
        "f1_suite_digest": f1_identity["suite_digest"],
        "f1_negative_file_digest": f1_identity["negative_file_digest"],
        "model": model,
        "seed": int(seed),
        "dev_fit_inputs": dev_identity,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_score_positive_pair(rows: pd.DataFrame, positives: pd.DataFrame) -> None:
    if rows.empty and positives.empty:
        return
    key = [*REQUIRED_COLUMNS, "run_id", "cohort"]
    missing_rows = [column for column in (*key, "mrr", "n_positives", "negatives_digest")
                    if column not in rows]
    missing_positives = [column for column in (*key, "positive_index", "reciprocal_rank",
                                                "both_masked", "negatives_digest")
                         if column not in positives]
    if missing_rows or missing_positives:
        raise ValueError(f"result pair is missing columns: scores={missing_rows}, positives={missing_positives}")
    row_keys = rows[key].drop_duplicates()
    positive_keys = positives[key].drop_duplicates()
    if len(row_keys) != len(rows) or positives.duplicated([*key, "positive_index"]).any():
        raise ValueError("result pair contains duplicate row identities")
    if row_keys.merge(positive_keys, on=key, how="outer", indicator=True)["_merge"].ne("both").any():
        raise ValueError("score rows and positive ranks have different run identities")
    digest_groups = rows.groupby(
        ["cohort", "dataset", "region", "split", "mask_instance"]
    )["negatives_digest"]
    if digest_groups.apply(lambda values: values.isna().any() or values.astype(str).str.strip().eq("").any()
                           or values.astype(str).nunique() != 1).any():
        raise ValueError("compared rows do not share one nonempty negatives digest per instance")
    gated = ~positives.both_masked.astype(bool)
    summary = positives.assign(
        _missing=positives.negatives_digest.isna(),
        _digest=positives.negatives_digest.astype(str),
        _gated=gated,
        _gated_rank=positives.reciprocal_rank.astype(float).where(gated),
    ).groupby(key).agg(
        _missing=("_missing", "any"), _digests=("_digest", "nunique"),
        _digest=("_digest", "first"), _gated=("_gated", "sum"),
        _gated_mrr=("_gated_rank", "mean"),
    ).reset_index()
    merged = rows.merge(summary, on=key, how="left")
    if merged._missing.ne(False).any():
        raise ValueError("score row has no matching positive ranks")
    if (merged._digests.ne(1) | merged._digest.ne(merged.negatives_digest.astype(str))).any():
        raise ValueError("score and positive rows have different negatives digests")
    if merged._gated.ne(merged.n_positives.astype(int)).any() or not np.isclose(
        merged._gated_mrr.astype(float), merged.mrr.astype(float), rtol=0, atol=1e-12
    ).all():
        raise ValueError("score MRR/count does not match persisted single-masked positive ranks")


def load_result_tables(
    results_dir: Path, repo_root: Path = REPO_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every stored scoring call's rows and per-positive ranks."""
    try:
        rows, positives = _read_parts(results_dir / "scored")
    except Exception as exc:
        raise ValueError(f"stored scoring results are unreadable: {exc}") from exc
    if rows.empty:
        raise ValueError(f"{results_dir / 'scored'} holds no scoring results")
    if "expert_region" in rows.columns:
        raise ValueError(f"{results_dir / 'scored'} holds transfer-matrix rows, not scoring results")
    _validate_score_positive_pair(rows, positives)  # exact, on read too: the history tolerance must not loosen it
    pinned = _pinned_scoring_commits(results_dir / "scored", repo_root)
    rows.attrs[_PINNED_COMMITS] = positives.attrs[_PINNED_COMMITS] = pinned
    return rows, positives


def _pinned_scoring_commits(parts: Path, repo_root: Path) -> frozenset[str]:
    """Scoring commits carried only by files whose exact bytes the release evidence pins."""
    pinned = release_evidence(repo_root)["scored_files"]
    if not pinned:
        return frozenset()
    exempt, other = set(), set()
    for path in sorted(parts.glob("*.parquet")):
        try:
            relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            relative = None
        frame = pd.read_parquet(path)
        commits = set(frame["scoring_commit"].astype(str)) if "scoring_commit" in frame else set()
        (exempt if pinned.get(relative) == file_digest(path) else other).update(commits)
    return frozenset(exempt - other)


def validate_scored_suite(
    rows: pd.DataFrame, config: RegionConfig, suite_name: str,
    data_root: Path | None = None, positives: pd.DataFrame | None = None,
    repo_root: Path = REPO_ROOT, *, inputs=None,
) -> dict:
    """Bind stored rows to one current canonical suite and negatives file."""
    suite_path = _suite_path(config, suite_name, data_root)
    if inputs is None:
        inputs = load_scoring_inputs(config, suite_path, data_root)
    suite, negatives = inputs[5], inputs[6]
    identity = scoring_input_identity(
        config, suite_path, data_root, inputs=inputs
    )
    negatives_path = suite_path.with_name(f"{suite_name}.negatives.json")
    expected_instances = {item.instance_id for item in suite.instances}
    if set(negatives.digests) != expected_instances:
        raise ValueError(f"{suite_name} suite and negatives cover different instances")
    required = {"suite_digest", "negative_file_digest", "negatives_digest", "run_id"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"stored {suite_name} rows lack provenance columns: {missing}")
    if rows.empty:
        raise ValueError(f"stored {suite_name} rows are missing")
    _validate_scoring_identity(rows, positives, identity, repo_root)
    if set(rows["dataset"].astype(str)) != {_DATASETS[suite_name]}:
        raise ValueError(f"stored {suite_name} rows have the wrong dataset")
    if set(rows["region"]) != {config.region}:
        raise ValueError(f"stored {suite_name} rows have the wrong region")
    if set(rows["split"].astype(str)) != {suite_name}:
        raise ValueError(f"stored {suite_name} rows have the wrong split")
    if set(rows["suite_digest"].astype(str)) != {suite.digest()}:
        raise ValueError(f"stored {suite_name} rows do not match the frozen suite")
    negative_file_digest = file_digest(negatives_path)
    if set(rows["negative_file_digest"].astype(str)) != {negative_file_digest}:
        raise ValueError(f"stored {suite_name} rows do not match the frozen negatives file")
    for instance_id, group in rows.groupby("mask_instance"):
        if instance_id not in expected_instances:
            raise ValueError(f"stored {suite_name} row names unknown instance {instance_id!r}")
        if set(group["negatives_digest"].astype(str)) != {negatives.digest_for(instance_id)}:
            raise ValueError(f"stored {suite_name} rows have changed negatives for {instance_id}")
    for key, group in rows.groupby(["model", "seed", "run_id"]):
        if set(group["mask_instance"].astype(str)) != expected_instances:
            raise ValueError(f"stored {suite_name} run {key} has incomplete instance coverage")
    return {
        "suite": suite,
        "negatives": negatives,
        "suite_digest": suite.digest(),
        "negative_file_digest": negative_file_digest,
        **identity,
    }


def _validate_scoring_identity(rows: pd.DataFrame, positives: pd.DataFrame | None,
                               expected: dict, repo_root: Path = REPO_ROOT) -> None:
    commits = set()
    for name, frame in (("score rows", rows), ("positive ranks", positives)):
        if frame is None:
            continue
        missing = sorted({*_SCORING_IDENTITY_FIELDS, "scoring_commit"} - set(frame.columns))
        if missing:
            raise ValueError(f"{name} lacks scoring input provenance columns: {missing}")
        for field in _SCORING_IDENTITY_FIELDS:
            if set(frame[field].astype(str)) != {str(expected[field])}:
                raise ValueError(f"{name} does not match current {field}")
        commits |= set(frame["scoring_commit"].astype(str)) - frame.attrs.get(_PINNED_COMMITS, frozenset())
    for commit in commits:
        if not _is_ancestor(repo_root, commit):
            raise ValueError(f"scoring commit {commit} is not an ancestor of HEAD")


#: ``best.pt``'s F1-val history and the stored rows can come from different hosts, whose float
#: summation order can break one near-tied pair differently. On a 565-positive instance 1e-5
#: admits a one-place rank change at rank 13 or lower and refuses one that moves a top-12 positive.
F1_VAL_HISTORY_ATOL = 1e-5


def validate_expert_provenance(
    rows: pd.DataFrame, manifests: list[tuple[Path, dict]], cohort: str, *, schema=None,
) -> None:
    """Bind every expert row to its manifest and checkpoint, and F1-val rows to its history."""
    expert = rows[rows.model == "gnn_a4"]
    by_seed = {int(manifest["seed"]): (run_dir, manifest) for run_dir, manifest in manifests}
    if expert.empty or set(expert.seed.astype(int)) != set(by_seed):
        raise ValueError("expert rows do not cover the configured manifest seeds")
    required = {"checkpoint_digest"}
    missing = sorted(required - set(expert.columns))
    if missing:
        raise ValueError(f"expert rows lack provenance columns: {missing}")
    for seed, (run_dir, manifest) in by_seed.items():
        if cohort_digest(manifest) != cohort:
            raise ValueError(f"run {run_dir.name} does not belong to cohort {cohort}")
        selected = expert[expert.seed.astype(int) == seed]
        if set(selected.run_id.astype(str)) != {run_dir.name}:
            raise ValueError(f"expert seed {seed} rows do not identify run {run_dir.name}")
        checkpoint_path = run_dir / "best.pt"
        if set(selected.checkpoint_digest.astype(str)) != {file_digest(checkpoint_path)}:
            raise ValueError(f"expert seed {seed} rows do not match {checkpoint_path}")
        if schema is not None:
            from train.checkpoints import load_expert
            load_expert(run_dir, schema)
        f1_val = selected[selected.split == "f1_val"]
        if f1_val.empty:
            continue
        _, expected = best_f1_val(run_dir, manifest.get("best_epoch"))
        actual = dict(zip(f1_val.mask_instance.astype(str), f1_val.mrr.astype(float), strict=True))
        if set(actual) != set(expected) or any(
            not np.isclose(actual[name], value, rtol=0, atol=F1_VAL_HISTORY_ATOL)
            for name, value in expected.items()
        ):
            raise ValueError(f"{run_dir}: stored F1-val rows do not match best.pt history")


def validate_baseline_provenance(
    rows: pd.DataFrame, cohort: str, f1_identity: dict, dev_identity: dict,
) -> None:
    """Bind baseline rows to current F1 fit and F2-dev tuning inputs."""
    baseline = rows[rows.model.isin(BASELINE_MODELS)]
    if baseline.empty:
        raise ValueError("baseline rows are missing")
    if "baseline_fit_inputs_digest" not in baseline:
        raise ValueError("baseline rows lack provenance column baseline_fit_inputs_digest")
    split_seeds = baseline.groupby(["model", "split"])["seed"].agg(
        lambda values: frozenset(values.astype(int))
    )
    for model, values in split_seeds.groupby(level="model"):
        if values.nunique() != 1:
            raise ValueError(f"baseline {model} uses different fitted seeds across suites")
    for row in baseline.itertuples(index=False):
        expected_run = f"baseline-{row.model}-s{int(row.seed)}"
        if row.run_id != expected_run:
            raise ValueError(f"baseline row run_id must be {expected_run}")
        expected = baseline_fit_digest(cohort, f1_identity, row.model, int(row.seed), dev_identity)
        if row.baseline_fit_inputs_digest != expected:
            raise ValueError("baseline rows do not match the current F1/F2-dev fit inputs")


def load_verified_results(
    config: RegionConfig, *, results_dir: Path, runs_dir: Path, data_root: Path,
    cohort: str | None, seeds, splits: tuple[str, ...],
    models=("gnn_a4", *BASELINE_MODELS), repo_root: Path = REPO_ROOT, loaded: dict | None = None,
) -> tuple[str, list[tuple[Path, dict]], pd.DataFrame, pd.DataFrame]:
    """One complete cohort's stored rows and ranks after every provenance check.

    ``loaded`` maps a split to its ``load_scoring_inputs`` result, already read by the caller.
    """
    cohort, manifests = resolve_cohort_manifests(
        runs_dir, config, cohort, seeds=seeds, require_complete=True,
    )
    rows, positives = load_result_tables(results_dir, repo_root)

    def keep(frame):
        return frame[
            (frame.cohort == cohort) & (frame.region == config.region)
            & frame.split.isin(splits) & frame.model.isin(models)
            & ((frame.model != "gnn_a4") | frame.seed.astype(int).isin(seeds))
        ]

    rows, positives = keep(rows), keep(positives)
    f1 = validate_f1_cohort_inputs(config, [manifest for _, manifest in manifests], data_root)
    inputs = {**(loaded or {}), "f1_val": f1["inputs"]}
    for split in splits:
        if split not in inputs:
            inputs[split] = load_scoring_inputs(
                config, _suite_path(config, split, data_root), data_root
            )
        validate_scored_suite(
            rows[rows.split == split], config, split, data_root,
            positives[positives.split == split], repo_root, inputs=inputs[split],
        )
    validate_expert_provenance(rows, manifests, cohort, schema=f1["inputs"][2])
    if not BASELINE_MODELS.isdisjoint(models):
        if "f2_dev" not in inputs:
            inputs["f2_dev"] = load_scoring_inputs(
                config, _suite_path(config, "f2_dev", data_root), data_root
            )
        dev_identity = dev_fit_identity(config, data_root, inputs=inputs["f2_dev"])
        validate_baseline_provenance(rows, cohort, f1, dev_identity)
    return cohort, manifests, rows, positives


def _suite_path(config: RegionConfig, name: str, data_root) -> Path:
    return region_paths(config.region, _DATASETS[name], data_root).suites / f"{name}.json"


def phase2_verdict(region: str, repo_root: Path = REPO_ROOT) -> str | None:
    """The region's registered Phase 2 verdict in ``results/registered.json`` at HEAD."""
    try:
        registered = json.loads(_git(repo_root, "show", "HEAD:results/registered.json").decode("utf-8"))
    except (OSError, ValueError):
        return None
    verdict = registered.get("phase2_verdicts", {}).get(region.upper())
    return verdict if verdict in ("PASS", "FAIL") else None


def _delta_tags(region: str, repo_root: Path) -> list[str]:
    return _git(repo_root, "tag", "--list", f"delta-{region.lower()}-*").decode().split()


def _hex(value, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def release_evidence(repo_root: Path = REPO_ROOT) -> dict:
    """``results/release-evidence.json`` at HEAD as {"calibrations": {region: entry},
    "scored_files": {path: sha256}}; empty when the file is absent."""
    working = repo_root / RELEASE_EVIDENCE
    try:
        raw = _git(repo_root, "show", f"HEAD:{RELEASE_EVIDENCE}")
    except ValueError:
        if working.exists():
            raise ValueError(f"{RELEASE_EVIDENCE} is not committed at HEAD") from None
        return {"calibrations": {}, "scored_files": {}}
    if not working.is_file() or working.read_bytes() != raw:
        raise ValueError(f"working {RELEASE_EVIDENCE} differs from the committed HEAD blob")
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{RELEASE_EVIDENCE} is unreadable: {exc}") from exc
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or not {"schema_version", "source_snapshot", "calibrations"} <= set(document)
            or not set(document) <= {"schema_version", "source_snapshot", "calibrations", "scored_files"}
            or not _hex(document["source_snapshot"], 40)
            or not isinstance(document["calibrations"], list)
            or not isinstance(document.get("scored_files", []), list)):
        raise ValueError(f"{RELEASE_EVIDENCE} does not follow schema_version 1")
    calibrations = {}
    for entry in document["calibrations"]:
        region = entry.get("region") if isinstance(entry, dict) else None
        if (set(entry or ()) != _CALIBRATION_FIELDS or not isinstance(region, str)
                or entry["path"] != f"results/{region.lower()}/delta.json"
                or not isinstance(entry["tag"], str)
                or not re.fullmatch(rf"delta-{re.escape(region.lower())}-[0-9A-Za-z._-]+", entry["tag"])
                or not _hex(entry["source_tag_commit"], 40) or not _hex(entry["cohort"], 64)
                or not _hex(entry["delta_sha256"], 64)):
            raise ValueError(f"{RELEASE_EVIDENCE} has a malformed calibration entry: {entry!r}")
        if region in calibrations:
            raise ValueError(f"{RELEASE_EVIDENCE} lists {region} more than once")
        calibrations[region] = entry
    scored = {}
    for entry in document.get("scored_files", []):
        path = entry.get("path") if isinstance(entry, dict) else None
        if (set(entry or ()) != {"path", "sha256"} or not isinstance(path, str)
                or path.startswith("/") or ".." in path.split("/") or not _hex(entry["sha256"], 64)):
            raise ValueError(f"{RELEASE_EVIDENCE} has a malformed scored_files entry: {entry!r}")
        if path in scored:
            raise ValueError(f"{RELEASE_EVIDENCE} lists {path} more than once")
        scored[path] = entry["sha256"]
    return {"calibrations": calibrations, "scored_files": scored}


def refuse_recalibration(region: str, repo_root: Path = REPO_ROOT) -> None:
    """Refuse a new calibration once the region has a local tag or a published calibration."""
    if tags := _delta_tags(region, repo_root):
        raise ValueError(f"{region} already has calibration tag(s) {tags}; calibration is frozen")
    if region in release_evidence(repo_root)["calibrations"]:
        raise ValueError(f"{region}'s calibration is published in {RELEASE_EVIDENCE}; calibration is frozen")


def _git(repo_root: Path, *args) -> bytes:
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.decode().strip()}")
    return result.stdout


def _is_ancestor(repo_root: Path, commit: str) -> bool:
    return not subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=repo_root,
        capture_output=True, check=False,
    ).returncode


def verify_tagged_delta(
    *, region: str, expected_cohort: str, delta_path: Path, repo_root: Path = REPO_ROOT,
) -> dict:
    """Validate live delta bytes against the region's local tag, or with no tag, its release evidence.

    A local ``delta-<code>-*`` tag is checked strictly and must agree with any evidence entry;
    a bad tag never falls back to the evidence. The stamp keeps the original tag commit.
    """
    try:
        raw = delta_path.read_bytes()
        delta = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError(f"calibrated delta is unreadable: {exc}") from exc
    relative = delta_path.resolve().relative_to(repo_root.resolve()).as_posix()
    head_blob = _git(repo_root, "show", f"HEAD:{relative}")
    if raw != head_blob:
        raise ValueError("working delta.json differs from the committed HEAD blob")
    if not isinstance(delta, dict):
        raise ValueError("delta.json must contain an object")
    if delta.get("region") != region:
        raise ValueError("calibrated delta region does not match selected region")
    if delta.get("cohort") != expected_cohort:
        raise ValueError("calibrated delta cohort does not match selected cohort")
    try:
        value = float(delta["delta"])
        spread = float(delta["s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("delta.json must contain numeric s and delta") from exc
    if not math.isfinite(value) or value < 0 or not math.isfinite(spread) or spread < 0:
        raise ValueError("delta.json s and delta must be finite and nonnegative")
    if round(2 * spread, 2) != value:
        raise ValueError("delta.json delta must equal round(2 * s, 2)")
    tag = delta.get("tag")
    if not isinstance(tag, str) or not re.fullmatch(
        rf"delta-{re.escape(region.lower())}-[0-9A-Za-z._-]+", tag
    ):
        raise ValueError(f"delta.json must name a delta-{region.lower()}-* calibration tag")
    digest = hashlib.sha256(raw).hexdigest()
    entry = release_evidence(repo_root)["calibrations"].get(region)
    if entry is not None:
        if entry["path"] != relative:
            raise ValueError(f"{RELEASE_EVIDENCE} lists {region}'s calibration at {entry['path']}, not {relative}")
        if entry["tag"] != tag or entry["cohort"] != delta["cohort"]:
            raise ValueError(f"delta.json tag or cohort differs from {region}'s entry in {RELEASE_EVIDENCE}")
        if entry["delta_sha256"] != digest:
            raise ValueError(f"delta.json bytes differ from {region}'s entry in {RELEASE_EVIDENCE}")
    tags = _delta_tags(region, repo_root)
    if not tags and entry is not None:
        return {**delta, "calibration_tag_commit": entry["source_tag_commit"],
                "calibration_delta_digest": digest, "calibration_source": "release evidence"}
    if tags != [tag]:
        raise ValueError(f"expected one calibration tag {tag!r} for {region}, found {tags}")
    tag = f"refs/tags/{tag}"

    tag_commit = _git(repo_root, "rev-parse", "--verify", f"{tag}^{{commit}}").decode().strip()
    if entry is not None and tag_commit != entry["source_tag_commit"]:
        raise ValueError(f"local tag {tag} conflicts with {region}'s entry in {RELEASE_EVIDENCE}")
    if not _is_ancestor(repo_root, tag_commit):
        raise ValueError("calibration tag is not an ancestor of HEAD")
    tag_blob = _git(repo_root, "show", f"{tag}:{relative}")
    if tag_blob != head_blob:
        raise ValueError("delta.json differs from the named calibration tag")
    return {**delta, "calibration_tag_commit": tag_commit,
            "calibration_delta_digest": digest, "calibration_source": "tag"}


def confirm_preflight(
    *, region: str, milestone: str | None, delta_path: Path,
    expected_cohort: str, repo_root: Path = REPO_ROOT,
) -> dict:
    """Require a milestone label and a frozen delta before confirm inputs open."""
    if not milestone or not str(milestone).strip():
        raise ValueError("f2_confirm scoring requires --milestone")
    return verify_tagged_delta(
        region=region, expected_cohort=expected_cohort, delta_path=delta_path,
        repo_root=repo_root,
    )
