"""Fit the three registered baseline families on the F1 union and score a Phase 3 F2 boundary suite.

    python scripts/score_union_baselines.py --suite phase3_boundary_f2_dev --masked-innervation uniform \
        --out results/phase3/scores-clean
    python scripts/score_union_baselines.py --suite phase3_boundary_f2_confirm --milestone gate-1 \
        --masked-innervation uniform --out results/phase3/scores-clean

Each family in ``train.baselines.BASELINES`` fits once on the union's pinned F1-train views
(``train.scoring.baseline_fit_views``: gap fraction 0.05, three views seeded from ``master_seed``).
The F1 context excludes the roots of ``phase3_boundary_f1_val``, the suite A early-stopped on, and
the learners choose their grid point on ``phase3_boundary_f2_dev``'s F2 union views (every row of
each instance). Each fitted seed then scores the selector's metric rows of ``--suite`` on the F2
union against their committed candidates, the rows ``scripts/score_union.py --mode union`` scores
for an expert, with ``--masked-innervation uniform`` applied to tuning and scoring alike.

Writes ``<out>/union/<union>/<suite>/baseline-<family>-s<seed>.npz`` in
``train.union_scoring.save_cache``'s format and records each in ``results/phase3/manifest.json`` with
``mode: baselines`` (``scripts/b0_report.py`` reads ``union`` entries only, so these never enter B0)
and ``region: baseline:<family>``. A family's per-instance MRR is the mean over its three seeds of
``train.union_scoring.instance_mrr``. F2-dev names the champion family; the gate reads all three.
The scoring code lives here, not in ``score_union.py``, so the experts' and A's code stamps stay put.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.boundary_suites import BoundarySuite, check_inputs, selector_digest  # noqa: E402
from src.evaluate import CommittedNegatives, candidate_pairs  # noqa: E402
from src.feature_io import read_feature_artifacts  # noqa: E402
from src.digests import sha256  # noqa: E402
from src.masks import apply_mask  # noqa: E402
from src.negatives import DEFAULT_SPEC  # noqa: E402
from src.regions import RegionGraph  # noqa: E402
from src.suites import suite_negatives  # noqa: E402
from train.baselines import BASELINES, F1TrainingContext  # noqa: E402
from train.scoring import baseline_fit_views  # noqa: E402
from train.train import _VAL_SUITES as VAL_SUITES  # noqa: E402
from train.union_scoring import _tasks, instance_mrr, save_cache  # noqa: E402
from scripts.score_union import MANIFEST, phase3_confirm_preflight, train_inputs  # noqa: E402

CODE = ("scripts/score_union_baselines.py", "scripts/score_union.py", "train/baselines.py", "train/scoring.py",
        "train/union_scoring.py", "src/boundary_suites.py", "src/evaluate.py", "src/negatives.py", "src/masks.py",
        "src/features.py", "src/feature_io.py", "src/suites.py", "src/regions.py")
TUNING_SUITE = "phase3_boundary_f2_dev"
SUITES = (TUNING_SUITE, "phase3_boundary_f2_confirm")


def log(message: str) -> None:
    print(f"[score_union_baselines] {message}", flush=True)


def boundary_inputs(suites_root: Path, code: str, name: str):
    path = suites_root / "f2" / code / f"{name}.json"
    return BoundarySuite.read(path), CommittedNegatives.read(path.with_name(f"{name}.negatives.json")), path


def manifest_key(path: Path, root: Path = REPO_ROOT) -> str:
    """A cache's manifest key: its POSIX path under ``root``, whatever the OS or the form of ``--out``."""
    path = path.resolve()
    return (path.relative_to(root) if path.is_relative_to(root) else path).as_posix()


def fit_seeds(baseline) -> tuple[int, ...]:
    """The three seeds a fitted family scores with; the configuration null has none of its own."""
    seeds = getattr(baseline, "seeds", None)
    if seeds is None:
        seeds = np.random.SeedSequence(DEFAULT_SPEC.master_seed).generate_state(3)
    seeds = tuple(int(s) for s in seeds)
    if len(seeds) != 3:
        raise SystemExit(f"{type(baseline).__name__} fitted {len(seeds)} seeds; exactly three are required")
    return seeds


def predictor(name: str, baseline, seed: int):
    if name == "configuration":
        return baseline.score
    return lambda view, table, pairs: baseline.score_seed(view, table, pairs, seed=seed)


def score_rows(predict, *, graph: RegionGraph, table, suite, negatives, selector: dict, samples,
               uniform: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The selector's rows of each instance scored by ``predict``, as a ``save_cache`` cache."""
    if selector["suite_digest"] != suite.digest():
        raise SystemExit("selector was not built from this suite")
    width = negatives.spec.k + 1
    out = {}
    for view, task in _tasks(graph, suite, negatives, samples):
        rows = np.asarray(selector["instances"][task.instance_id]["rows"], dtype=np.int64)
        pairs = candidate_pairs(task).reshape(len(task.positives), width, 2)[rows]
        frame = table.with_uniform_innervation(np.flatnonzero(view.is_masked)) if uniform else table
        scores = np.asarray(predict(view.model_view(), frame, pairs.reshape(-1, 2)), dtype=np.float64)
        out[task.instance_id] = (rows, scores.reshape(-1, width))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True, choices=SUITES)
    parser.add_argument("--union", default="UNION12")
    parser.add_argument("--masked-innervation", choices=("real", "uniform"), default="real")
    parser.add_argument("--milestone", help="confirm only: a nonempty label for this confirm read, recorded on every cache")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--suites-root", type=Path, default=REPO_ROOT / "data/suites")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/phase3/scores")
    parser.add_argument("--family", action="append", choices=sorted(BASELINES),
                        help="fit and score only these families (default all three); each is recorded as it finishes")
    args = parser.parse_args(argv)
    confirm_record = phase3_confirm_preflight(argparse.Namespace(**vars(args), mode="union", region=None), REPO_ROOT)
    uniform = args.masked_innervation == "uniform"
    code = args.union.lower()

    f1_graph, f1_table, _ = train_inputs(code, args.data_root)
    fit_path = args.suites_root / "f1" / code / f"{VAL_SUITES['union']}.json"
    context = F1TrainingContext(f1_graph, BoundarySuite.read(fit_path))
    union_dir = args.data_root / "processed" / "f2" / code
    graph = RegionGraph.read(union_dir)
    table, schema = read_feature_artifacts(union_dir, graph)
    check_inputs(schema, table.columns)
    if tuple(table.columns) != tuple(f1_table.columns):
        raise SystemExit("the F2 union table's columns differ from the F1 union's; a baseline fit on one cannot score the other")

    dev_suite, dev_negatives, dev_path = boundary_inputs(args.suites_root, code, TUNING_SUITE)
    dev_views = [apply_mask(graph, instance) for instance in dev_suite.resolve(graph)]
    dev_tables = [table.with_uniform_innervation(np.flatnonzero(v.is_masked)) if uniform else table for v in dev_views]
    suite, negatives, path = boundary_inputs(args.suites_root, code, args.suite)
    selector = json.loads(path.with_name(f"{args.suite}.selector.json").read_text(encoding="utf-8"))
    log(f"sampling {suite.name}'s committed negatives on {graph.region}")
    samples = suite_negatives(graph, suite, spec=negatives.spec)
    fit_views = baseline_fit_views(context)

    record = {"mode": "baselines", "suite": args.suite, "union": args.union, "dataset": "f2",
              "masked_innervation": args.masked_innervation, **confirm_record,
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                       text=True, check=True).stdout.strip(),
              "code": {name: sha256(REPO_ROOT / name) for name in CODE},
              "union_graph": graph.digest(), "union_schema": schema.digest(),
              "union_features": sha256(union_dir / "features.parquet"), "selector": selector_digest(selector),
              "suite_region": graph.region, "suite_digest": suite.digest(),
              "negatives_file": sha256(path.with_name(f"{args.suite}.negatives.json")),
              "fit_graph": f1_graph.digest(), "fit_exclusion_suite": BoundarySuite.read(fit_path).digest(),
              "fit_views": [view.instance.instance_id for view in fit_views],
              "tuning_suite_digest": dev_suite.digest(),
              "tuning_negatives_file": sha256(dev_path.with_name(f"{TUNING_SUITE}.negatives.json"))}
    out = args.out / "union" / graph.region.lower() / args.suite
    for name, cls in BASELINES.items():
        if args.family and name not in args.family:
            continue
        entries = {}
        log(f"fitting {name} on {len(fit_views)} F1 union views, tuned on {TUNING_SUITE}")
        baseline = cls(negative_spec=DEFAULT_SPEC).fit(
            fit_views, [f1_table] * len(fit_views), context=context, val_views=dev_views,
            val_tables=dev_tables, val_suite=dev_suite, val_negatives=dev_negatives)
        fit = {"champion_params": getattr(baseline, "champion_params", None),
               "grid_mrr": list(getattr(baseline, "grid_mrr", ()))}
        for seed in fit_seeds(baseline):
            cache = score_rows(predictor(name, baseline, seed), graph=graph, table=table, suite=suite,
                               negatives=negatives, selector=selector, samples=samples, uniform=uniform)
            run_id = f"baseline-{name}-s{seed}"
            target = out / f"{run_id}.npz"
            save_cache(target, cache)
            key = manifest_key(target)
            entries[key] = {**record, "region": f"baseline:{name}", "seed": seed, "run_id": run_id,
                            "gate_seed": True, "cohort": None, "checkpoint": None, "system": None,
                            "expert_schema": schema.digest(), "expert_geometry": None, "scores": sha256(target),
                            "baseline": fit}
            log(f"{run_id}: mean instance MRR {np.mean(list(instance_mrr(cache).values())):.4f} -> {target}")
        record_entries(entries)
    return 0


def record_entries(entries: dict) -> None:
    """Add one family's entries to the manifest, so a later family's failure keeps them."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    manifest.update(entries)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
