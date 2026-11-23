"""Score registry experts for Phase 3 B0 (plan 03) and cache their per-row scores.

    python scripts/score_union.py --mode union --union UNION12 --suite phase3_boundary_f1_val
    python scripts/score_union.py --mode union --dataset f2 --suite phase3_boundary_f2_dev
    python scripts/score_union.py --mode home --region CX --suite f1_val
    python scripts/score_union.py --mode regional --union UNION12 --region CX --suite f1_val --expert LH
    python scripts/score_union.py --mode union --suite phase3_boundary_f1_val --system-run <run> [--system-run ...]
    python scripts/score_union.py --mode union --dataset f2 --suite phase3_boundary_f2_dev --system-run <run>

``union`` scores a boundary suite's metric rows on the masked union graph. ``home``
scores a region's suite on its own graph, every row (the parity check against
``scripts/score_run.py``). ``regional`` scores a region's suite on the union graph with
every union edge of each masked root hidden; ``--region`` is the suite's region and
``--expert`` the visiting experts (default every registry region). Both union modes read
each expert's geometry in its own region's frame, from ``scripts/expert_geometry.py``. Experts come from ``results/experts.csv``
(all five seeds of each region by default), loaded from ``--runs-dir/<run_id>`` after
their manifest is checked against the registry row. ``--system-run`` scores plan 06's A
instead: runs trained on the union itself, read on the union table as trained (no expert
re-encoding), cached as region ``A`` and never part of B0's mixture.

Writes ``results/phase3/scores/<mode>/<suite region>/<suite>/<run_id>.npz`` (see
``train.union_scoring.save_cache``) and records, per file, the code, schema, feature,
suite, selector, negatives and checkpoint hashes in ``results/phase3/manifest.json``.
``--masked-innervation uniform`` scores the same rows with each masked node's innervation
uniform (the registered ablation); write it to its own ``--out`` tree.

``--dataset`` picks whose union graph, suites and expert geometry are read, and defaults to
F1. F2 reads F2's, so its caches carry their own graph, schema, feature and suite digests.
A's manifests are checked against F1's union graph, features, schema and boundary F1-val
suite whatever ``--dataset`` names; with F2 it scores the F2 union inputs. A's F2 regional
scores read ``<union>/regional/`` (``scripts/build_union.py``), the union plus each member's
F2 neurons it lacks with no edge, so every committed regional row is scorable.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.boundary_suites import BoundarySuite, check_inputs, selector_digest  # noqa: E402
from src.evaluate import CommittedNegatives  # noqa: E402
from src.feature_io import read_feature_artifacts  # noqa: E402
from src.features import FeatureSchema  # noqa: E402
from src.digests import sha256  # noqa: E402
from src.regions import REGIONS, RegionGraph  # noqa: E402
from src.suites import Suite, suite_negatives  # noqa: E402
from train.checkpoints import load_expert  # noqa: E402
from train.cohorts import cohort_digest  # noqa: E402
from train.milestones import _read_manifest  # noqa: E402
from scripts.expert_geometry import load as load_geometry  # noqa: E402
from scripts.scoring_common import phase2_verdict  # noqa: E402
from train.union_scoring import MODES, expert_table, registry_label, save_cache, score_cache  # noqa: E402
from train.train import _VAL_SUITES as VAL_SUITES, _feature_digest, extras_read  # noqa: E402

#: Every module a score depends on: a change to any of them changes the manifest's stamp.
CODE = ("scripts/score_union.py", "scripts/expert_geometry.py", "train/union_scoring.py", "src/transfer.py",
        "src/union.py", "src/boundary_suites.py", "train/model.py", "train/checkpoints.py", "src/evaluate.py",
        "src/negatives.py", "src/masks.py", "src/features.py", "src/feature_io.py", "src/suites.py",
        "src/regions.py")
MANIFEST = REPO_ROOT / "results/phase3/manifest.json"
CONFIRM_SUITES = frozenset({"phase3_boundary_f2_confirm", "f2_confirm"})


def log(message: str) -> None:
    print(f"[score_union] {message}", flush=True)


def registry_rows(path: Path, regions, seeds) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["region"] in regions and int(r["seed"]) in seeds]
    if not rows:
        raise SystemExit(f"{path}: no registry rows for {sorted(regions)} seeds {sorted(seeds)}")
    return rows


def system_rows(runs_dir: Path, run_ids, union: str, *, graph: RegionGraph, table, schema: FeatureSchema,
                val_suite_digest: str) -> list[dict]:
    """Registry-shaped rows for A's runs, checked against its F1 training inputs.

    A run is refused unless it trained on this union's graph, features and schema and
    early-stopped on its boundary F1-val suite: training excludes that suite's roots, so
    a run without it (``suite_digest`` "none") or with another suite trained on them.
    """
    rows = []
    for run_id in run_ids:
        manifest = _read_manifest(runs_dir / run_id)
        if manifest.get("region") != union:
            raise SystemExit(f"{run_id} trained on {manifest.get('region')!r}, not {union}: it is not a union system")
        trained = {"graph_digest": graph.digest(), "schema_digest": schema.digest(),
                   "feature_digest": _feature_digest(table, extras=extras_read(manifest["config"])),
                   "suite_digest": val_suite_digest}
        wrong = [k for k, v in trained.items() if manifest.get(k) != v]
        if wrong:
            raise SystemExit(f"{run_id}: trained on other inputs than {union} and its {VAL_SUITES['union']} "
                             f"({', '.join(wrong)}); it may have trained on the boundary roots")
        seed = int(manifest["seed"])
        rows.append({"run_id": run_id, "region": "A", "seed": seed, "gate_seed": "true" if seed in (0, 1, 2) else "false",
                     "cohort": cohort_digest(manifest), "train": {"dataset": "f1", **trained}})
    return rows


def load_model(row: dict, runs_dir: Path, schema: FeatureSchema):
    run_dir = runs_dir / row["run_id"]
    if not (run_dir / "best.pt").is_file():
        raise SystemExit(f"{run_dir}: checkpoint missing; train the run or download its checkpoint")
    manifest = _read_manifest(run_dir)
    if int(manifest["seed"]) != int(row["seed"]) or cohort_digest(manifest) != row["cohort"]:
        raise SystemExit(f"{run_dir}: manifest does not match its results/experts.csv row")
    return load_expert(run_dir, schema), sha256(run_dir / "best.pt")


def region_inputs(code: str, suite_name: str, data_root: Path, suites_root: Path, dataset: str):
    folder = data_root / "processed" / dataset / code.lower()
    graph = RegionGraph.read(folder)
    table, schema = read_feature_artifacts(folder, graph)
    path = suites_root / dataset / code.lower() / f"{suite_name}.json"
    suite = Suite.read(path)
    negatives = CommittedNegatives.read(path.with_name(f"{suite_name}.negatives.json"))
    return graph, table, schema, suite, negatives, path


def train_inputs(code: str, data_root: Path):
    """A's F1 union inputs, whose digests its run manifests must match."""
    folder = data_root / "processed" / "f1" / code
    graph = RegionGraph.read(folder)
    table, schema = read_feature_artifacts(folder, graph)
    check_inputs(schema, table.columns)
    return graph, table, schema


def phase3_confirm_preflight(args: argparse.Namespace, repo_root: Path) -> dict:
    """A confirm read's milestone label and regional verdict, checked before its inputs open."""
    if args.suite not in CONFIRM_SUITES:
        if args.milestone:
            raise SystemExit("--milestone applies to confirm suites only")
        return {}
    if not args.milestone or not args.milestone.strip():
        raise SystemExit(f"{args.suite} scoring requires --milestone")
    region = args.region[0] if args.mode in ("home", "regional") else None
    if region and not phase2_verdict(region, repo_root):
        raise SystemExit(f"{region} has no accepted Phase 2 verdict in results/registered.json; no regional confirm read")
    return {"milestone": args.milestone}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--union", default="UNION12")
    parser.add_argument("--dataset", choices=("f1", "f2"), default="f1",
                        help="which dataset's union graph, suites and expert geometry to read")
    parser.add_argument("--region", action="append", choices=sorted(REGIONS),
                        help="union: experts to score (default all); home/regional: the suite's region")
    parser.add_argument("--expert", action="append", choices=sorted(REGIONS),
                        help="regional mode: experts to score on --region's suite (default every registry region)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--masked-innervation", choices=("real", "uniform"), default="real",
                        help="uniform scores the registered ablation: each masked node's innervation set to uniform")
    parser.add_argument("--system-run", action="append",
                        help="plan 06's A: F1-trained runs scored on the union table (union or regional mode)")
    parser.add_argument("--milestone", help="confirm only: a nonempty label for this confirm read, recorded on every cache")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "results/experts.csv")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "data/runs")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--suites-root", type=Path, default=REPO_ROOT / "data/suites")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/phase3/scores")
    args = parser.parse_args(argv)
    if args.mode != "union" and (not args.region or len(args.region) != 1):
        parser.error(f"--mode {args.mode} scores one region's suite: pass exactly one --region")
    if args.expert and args.mode != "regional":
        parser.error("--expert applies to --mode regional only")
    if args.system_run and (args.mode == "home" or args.expert or (args.mode == "union" and args.region)):
        parser.error("--system-run scores A in union or regional mode, without --expert (or --region in union mode)")
    if args.masked_innervation == "uniform" and args.mode == "home":
        parser.error("--masked-innervation uniform needs a mode that masks roots")
    confirm_record = phase3_confirm_preflight(args, REPO_ROOT)

    code = args.union.lower()
    record = {"mode": args.mode, "suite": args.suite, "union": args.union, "dataset": args.dataset,
              "masked_innervation": args.masked_innervation, **confirm_record,
              "registry": registry_label(args.registry, REPO_ROOT),
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                       text=True, check=True).stdout.strip(),
              "code": {name: sha256(REPO_ROOT / name) for name in CODE}}
    union_graph = union_table = union_schema = None
    if args.mode in ("union", "regional"):
        union_dir = args.data_root / "processed" / args.dataset / code
        if args.mode == "regional" and args.dataset == "f2" and args.system_run:
            union_dir = union_dir / "regional"
            if not union_dir.is_dir():
                raise SystemExit(f"{union_dir} is missing; run build_union.py --dataset f2 --steps regional")
        union_graph = RegionGraph.read(union_dir)
        union_table, union_schema = read_feature_artifacts(union_dir, union_graph)
        check_inputs(union_schema, union_table.columns)
        record |= {"union_graph": union_graph.digest(), "union_schema": union_schema.digest(),
                   "union_features": sha256(union_dir / "features.parquet")}

    if args.mode == "union":
        path = args.suites_root / args.dataset / code / f"{args.suite}.json"
        graph, suite = union_graph, BoundarySuite.read(path)
        negatives = CommittedNegatives.read(path.with_name(f"{args.suite}.negatives.json"))
        selector = json.loads(path.with_name(f"{args.suite}.selector.json").read_text(encoding="utf-8"))
        record["selector"] = selector_digest(selector)
        regions, table = args.region or sorted(REGIONS), None
    else:
        graph, table, _, suite, negatives, path = region_inputs(
            args.region[0], args.suite, args.data_root, args.suites_root, args.dataset)
        selector = None
        regions = args.region if args.mode == "home" else (args.expert or sorted(REGIONS))
        region_dir = args.data_root / "processed" / args.dataset / args.region[0].lower()
        record |= {"region_graph": graph.digest(), "region_features": sha256(region_dir / "features.parquet")}
    record |= {"suite_region": graph.region, "suite_digest": suite.digest(),
               "negatives_file": sha256(path.with_name(f"{args.suite}.negatives.json"))}
    log(f"sampling {suite.name}'s committed negatives on {graph.region}")
    samples = suite_negatives(graph, suite, spec=negatives.spec)
    uniform = args.masked_innervation == "uniform"

    out = args.out / args.mode / graph.region.lower() / args.suite
    entries = {}
    system_schema = None
    if args.system_run:
        if args.dataset == "f1":
            train_graph, train_table, system_schema = union_graph, union_table, union_schema
        else:
            train_graph, train_table, system_schema = train_inputs(code, args.data_root)
        val = BoundarySuite.read(args.suites_root / "f1" / code / f"{VAL_SUITES['union']}.json")
        rows = system_rows(args.runs_dir, args.system_run, args.union, graph=train_graph, table=train_table,
                           schema=system_schema, val_suite_digest=val.digest())
    else:
        rows = registry_rows(args.registry, set(regions), set(args.seeds))
    for row in rows:
        if args.system_run:  # A reads the scoring union's table in its F1-trained schema
            if tuple(union_table.columns) != system_schema.column_names:
                raise SystemExit(f"{args.dataset} union table does not match A's F1 union schema")
            expert_schema = system_schema
            in_frame = union_table
        else:
            expert_schema = FeatureSchema.from_json(
                (args.data_root / "processed" / "f1" / row["region"].lower() / "schema.json").read_text(encoding="utf-8"))
        model, checkpoint = load_model(row, args.runs_dir, expert_schema)
        common = {"graph": graph, "suite": suite, "negatives": negatives, "samples": samples}
        if args.mode != "home" and not args.system_run:  # the expert reads union geometry in its own region's frame
            geometry = load_geometry(union_dir, row["region"], union_graph.node_ids)
            in_frame = expert_table(union_table, union_schema, expert_schema, geometry=geometry)
        if args.mode == "union":
            cache = score_cache(model, mode="union", table=in_frame, selector=selector, masked_uniform=uniform, **common)
        elif args.mode == "home":
            cache = score_cache(model, mode="home", table=table, **common)
        else:
            cache = score_cache(model, mode="regional", table=table, union_graph=union_graph,
                                union_table=in_frame, masked_uniform=uniform, **common)
        target = out / f"{row['run_id']}.npz"
        save_cache(target, cache)
        key = target.relative_to(REPO_ROOT).as_posix() if target.is_relative_to(REPO_ROOT) else str(target)
        entries[key] = {**record, "region": row["region"], "seed": int(row["seed"]), "run_id": row["run_id"],
                        "gate_seed": row["gate_seed"] == "true", "cohort": row["cohort"], "checkpoint": checkpoint,
                        "expert_schema": expert_schema.digest(), "scores": sha256(target),
                        "system": "A" if args.system_run else None,
                        "expert_geometry": (None if args.mode == "home" or args.system_run else sha256(
                            union_dir / "expert_geometry" / f"{row['region'].lower()}.parquet"))}
        if args.system_run:
            entries[key]["system_train"] = row["train"]
        log(f"{row['run_id']}: {sum(len(r) for r, _ in cache.values())} rows -> {target}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    manifest.update(entries)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
