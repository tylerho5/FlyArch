"""Exploratory regional F2-dev diagnosis of Phase 3's gate-1 regional guard.

    python scripts/phase3_f2dev_diagnosis.py score --region CX --out <dir>/CX.csv
    python scripts/phase3_f2dev_diagnosis.py report --scores <dir> --a-caches <dir> --out results/phase3/f2dev-diagnosis

``score`` scores one region's F2-dev suite (the gate's single-masked rows, 3 gate seeds) under four
conditions: ``ref_real`` (Phase 2 reference as scored), ``ref_unif`` (same models, masked roots'
innervation uniform), ``clean_unif`` (clean cohort, region graph, masked uniform) and ``clean_union``
(clean cohort on the F2 union graph, masked uniform; skipped for a region whose suite nodes are
missing from the plain union). ``report`` joins them with A's regional F2-dev caches, writes the
per-instance table, and prints each step's mean log2 ratio with a 95% BCa interval and a closure
check. Reads F2-dev only; no confirm or F3 suite, no selection, and nothing feeds a gate.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expert_geometry import load as load_geometry  # noqa: E402
from scripts.score_union import load_model, region_inputs  # noqa: E402
from src.boundary_suites import check_inputs  # noqa: E402
from src.evaluate import candidate_pairs  # noqa: E402
from src.feature_io import read_feature_artifacts  # noqa: E402
from src.features import FeatureSchema  # noqa: E402
from src.gate import PreregConfig, _bootstrap_ci  # noqa: E402
from src.regions import REGIONS, RegionGraph  # noqa: E402
from src.suites import suite_negatives  # noqa: E402
from train.union_scoring import _predict, _tasks, expert_table, instance_mrr, load_cache, score_cache  # noqa: E402

SUITE = "f2_dev"
CONDITIONS = ("ref_real", "ref_unif", "clean_unif", "clean_union")
REGISTRY = {"ref": "results/experts.csv", "clean": "results/experts-clean.csv"}
A_RUNS = "f1-union12-s00{seed}-76efea1e"
#: (label, start condition, end condition); every step is mean log2(end / start) over instances.
STEPS = (("input mismatch", "ref_real", "ref_unif"), ("retrain uniform", "ref_unif", "clean_unif"),
         ("clean vs reference", "ref_real", "clean_unif"), ("union graph", "clean_unif", "clean_union"),
         ("A vs clean_union", "clean_union", "A"), ("A vs clean_unif", "clean_unif", "A"),
         ("A vs reference", "ref_real", "A"))


def single_rows(graph, suite, negatives, samples) -> dict[str, np.ndarray]:
    """The gate's rows per instance: positives with exactly one masked endpoint."""
    return {task.instance_id: np.flatnonzero(~task.strata["both_masked"].to_numpy(bool))
            for _, task in _tasks(graph, suite, negatives, samples)}


def score(args: argparse.Namespace) -> int:
    code = args.region.upper()
    data_root = args.data_root
    graph, table, _, suite, negatives, _ = region_inputs(code, SUITE, data_root, data_root / "suites", "f2")
    samples = suite_negatives(graph, suite, spec=negatives.spec)
    width = negatives.spec.k + 1
    single = single_rows(graph, suite, negatives, samples)
    schema = FeatureSchema.from_json(
        (data_root / "processed/f1" / code.lower() / "schema.json").read_text(encoding="utf-8"))
    union_dir = data_root / "processed/f2/union12"
    union = None
    out = []
    for condition in args.conditions:
        registry = REPO_ROOT / REGISTRY["ref" if condition.startswith("ref") else "clean"]
        with open(registry, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["region"] == code and r["gate_seed"] == "true"]
        for row in rows:
            model, _ = load_model(row, data_root / "runs", schema)
            if condition != "clean_union":
                cache = {}
                for view, task in _tasks(graph, suite, negatives, samples):
                    pairs = candidate_pairs(task).reshape(-1, 2)
                    frame = (table if condition == "ref_real"
                             else table.with_uniform_innervation(np.flatnonzero(view.is_masked)))
                    cache[task.instance_id] = (np.arange(len(task.positives)),
                                               _predict(model, view.model_view(), frame, pairs, width))
            else:
                if union is None:
                    ugraph = RegionGraph.read(union_dir)
                    utable, uschema = read_feature_artifacts(union_dir, ugraph)
                    check_inputs(uschema, utable.columns)
                    union = (ugraph, utable, uschema)
                ugraph, utable, uschema = union
                frame = expert_table(utable, uschema, schema,
                                     geometry=load_geometry(union_dir, code, ugraph.node_ids))
                try:
                    cache = score_cache(model, mode="regional", graph=graph, table=table, suite=suite,
                                        negatives=negatives, samples=samples, union_graph=ugraph,
                                        union_table=frame, masked_uniform=True)
                except ValueError as error:
                    print(f"{code} {condition}: skipped ({error})", flush=True)
                    break
            for instance, mrr in instance_mrr(cache, single).items():
                out.append({"region": code, "cond": condition, "seed": int(row["seed"]),
                            "instance": instance, "mrr": mrr})
        print(f"{code} {condition}: done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(args.out, index=False)
    return 0


def a_instance_mrr(code: str, a_caches: Path, data_root: Path) -> pd.Series:
    graph, _, _, suite, negatives, _ = region_inputs(code, SUITE, data_root, data_root / "suites", "f2")
    single = single_rows(graph, suite, negatives, suite_negatives(graph, suite, spec=negatives.spec))
    base = a_caches / "regional" / code.lower() / SUITE
    seeds = [pd.Series(instance_mrr(load_cache(base / f"{A_RUNS.format(seed=s)}.npz"), single)) for s in (0, 1, 2)]
    return pd.concat(seeds, axis=1).mean(axis=1)


def report(args: argparse.Namespace) -> int:
    per_instance, steps, closure = [], [], []
    for code in sorted(REGIONS):
        frame = pd.read_csv(args.scores / f"{code}.csv")
        by = frame.groupby(["cond", "instance"]).mrr.mean().unstack(0).reindex(columns=list(CONDITIONS))  # 3-seed mean
        by["A"] = a_instance_mrr(code, args.a_caches, args.data_root)
        per_instance.append(by.reset_index().assign(region=code))
        mean_log = lambda x, y: float(np.log2(by[y] / by[x]).mean())  # noqa: E731
        for label, start, end in STEPS:
            if by[[start, end]].notna().all(axis=None):
                values = np.log2(by[end] / by[start]).to_numpy()
                low, high, *_ = _bootstrap_ci(values, PreregConfig(region=code))
                steps.append({"region": code, "step": label, "mean": values.mean(), "low": low, "high": high})
        path = [("ref_real", "ref_unif"), ("ref_unif", "clean_unif"), ("clean_unif", "A")]
        closure.append({"region": code, "total": mean_log("ref_real", "A"),
                        "sum_via_region_graph": sum(mean_log(*p) for p in path)})
        if by["clean_union"].notna().all():
            full = path[:2] + [("clean_unif", "clean_union"), ("clean_union", "A")]
            closure[-1]["sum_via_union_graph"] = sum(mean_log(*p) for p in full)
    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["region", "instance", *CONDITIONS, "A"]
    pd.concat(per_instance)[cols].to_csv(args.out / "f2dev-instances.csv", index=False)
    steps = pd.DataFrame(steps)
    steps.to_csv(args.out / "f2dev-steps.csv", index=False)
    means = pd.concat(per_instance).groupby("region")[[*CONDITIONS, "A"]].mean()
    pd.set_option("display.width", 220)
    print(means.round(4).to_string())
    print(steps.pivot(index="region", columns="step", values="mean").round(3).to_string())
    print(pd.DataFrame(closure).round(4).to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("score")
    one.add_argument("--region", required=True, choices=sorted(REGIONS))
    one.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    one.add_argument("--out", type=Path, required=True)
    two = sub.add_parser("report")
    two.add_argument("--scores", type=Path, required=True, help="directory of the per-region score CSVs")
    two.add_argument("--a-caches", type=Path, required=True, help="directory holding regional/<code>/f2_dev/*.npz")
    two.add_argument("--out", type=Path, required=True)
    for p in (one, two):
        p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args(argv)
    return score(args) if args.command == "score" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
