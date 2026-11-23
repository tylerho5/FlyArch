"""B1's report on a boundary suite (Phase 3 plan 05, question 3).

    python scripts/router_report.py --suite phase3_boundary_f1_val --out phase3-router.md

Scores, on the selector's metric rows, from the ``union``-mode caches that
``scripts/score_union.py`` wrote (each region's three gate seeds):

- the innervation-profile router from #151, with each region's seeds combined first;
- the same router with masked neurons' innervation set to uniform in the router (the
  experts' own inputs are unchanged unless their caches were scored that way);
- the hybrid, which hands a row to one region's expert when the masked neuron's profile
  puts at least ``--home-share`` in that region (exploratory on F1: the share is chosen
  on boundary F2-dev only);
- B0 and the bars from ``scripts/b0_report.py``.

It then applies plan 05's admission rule: B1 is built only if B0 (leave-one-instance-out)
trails the best boundary bar by more than B0's seed spread.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from scripts.b0_report import DEFAULT_REGISTRY, GATE, SEED_CONTROL_LOO, check_provenance, methods  # noqa: E402
from src.digests import sha256  # noqa: E402
from src.boundary_suites import BoundarySuite  # noqa: E402
from src.evaluate import CommittedNegatives, candidate_pairs  # noqa: E402
from src.feature_io import read_feature_artifacts  # noqa: E402
from src.regions import RegionGraph  # noqa: E402
from train.union_router import profile_route, region_profile  # noqa: E402
from train.union_scoring import _tasks, _zscore, instance_mrr, load_cache, registry_label  # noqa: E402


def suite_rows(graph, suite, negatives, selector):
    """Per instance: the metric rows' ids, candidate pairs (rows, K+1, 2), masked endpoints and masked nodes."""
    out = {}
    width = negatives.spec.k + 1
    for view, task in _tasks(graph, suite, negatives):
        rows = np.asarray(selector["instances"][task.instance_id]["rows"], dtype=np.int64)
        pairs = candidate_pairs(task).reshape(len(task.positives), width, 2)[rows]
        positives = task.positives[rows]
        masked = np.where(view.is_masked[positives[:, 0]], positives[:, 0], positives[:, 1])
        out[task.instance_id] = (rows, pairs, masked, np.flatnonzero(view.is_masked))
    return out


def admission(rows: dict[str, dict]) -> tuple[bool, str]:
    """Plan 05's rule, from the b0_report rows (means over instances, all metric rows)."""
    mean = {name: float(np.mean(list(by["all"].values()))) for name, by in rows.items()}
    # Like for like: B0's number is leave-one-out, so the seed control's is too (no in-sample bar).
    bars = {n: v for n, v in mean.items()
            if n.endswith("(3 seeds, §2 form)") or n == SEED_CONTROL_LOO}
    best_name = max(bars, key=bars.get)
    spread = float(np.ptp([v for n, v in mean.items() if n.startswith("B0's weights, seed")]))
    b0 = mean["B0, leave-one-instance-out weights"]
    gap = bars[best_name] - b0
    admitted = gap > spread
    return admitted, (f"B0 (leave-one-out) {b0:.4f}; best bar {best_name} {bars[best_name]:.4f}; "
                      f"gap {gap:+.4f} against B0's seed spread {spread:.4f}: "
                      f"B1 {'is' if admitted else 'is not'} admitted.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--union", default="UNION12")
    parser.add_argument("--home-share", type=float, default=0.9)
    parser.add_argument("--masked-innervation", choices=("real", "uniform"), default="real",
                        help="which caches to read: the registered ablation scores with masked innervation uniform")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / DEFAULT_REGISTRY,
                        help="the cohort's registry, as recorded when its caches were scored")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "results/phase3/manifest.json")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--suites-root", type=Path, default=REPO_ROOT / "data/suites")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    code = args.union.lower()
    folder = args.data_root / "processed" / "f1" / code
    graph = RegionGraph.read(folder)
    table, schema = read_feature_artifacts(folder, graph)
    path = args.suites_root / "f1" / code / f"{args.suite}.json"
    suite = BoundarySuite.read(path)
    negatives = CommittedNegatives.read(path.with_name(f"{args.suite}.negatives.json"))
    selector = json.loads(path.with_name(f"{args.suite}.selector.json").read_text(encoding="utf-8"))

    block = {iid: np.asarray(r["block_rows"]) for iid, r in selector["instances"].items()}
    entries = {p: e for p, e in sorted(json.loads(args.manifest.read_text(encoding="utf-8")).items())
               if e["mode"] == "union" and e["suite"] == args.suite and e.get("suite_region", args.union) == args.union
               and e.get("masked_innervation", "real") == args.masked_innervation
               and e.get("registry", DEFAULT_REGISTRY) == registry_label(args.registry, REPO_ROOT)
               and not e.get("system")}  # A (plan 06) is not an expert the router mixes
    check_provenance(entries, selector, REPO_ROOT)  # the same inputs and code as B0's report, or refuse
    live = {"union_graph": graph.digest(), "union_schema": schema.digest(),
            "union_features": sha256(folder / "features.parquet"), "suite_digest": suite.digest(),
            "negatives_file": sha256(path.with_name(f"{args.suite}.negatives.json"))}
    recorded = next(iter(entries.values()))
    changed = sorted(k for k, v in live.items() if recorded.get(k) != v)
    if changed:  # the router reads these live, so they must be the inputs the caches were scored on
        raise SystemExit(f"current inputs differ from the ones the caches were scored on: {changed}")
    caches: dict[str, dict[int, dict]] = defaultdict(dict)
    for p, entry in entries.items():
        caches[entry["region"]][entry["seed"]] = load_cache(REPO_ROOT / p)
    report = dict(methods(dict(caches), {"all": None, "block": block})[0])  # refuses a region missing a gate seed
    regions = sorted(caches)
    profile = region_profile(table, schema, regions)

    names = ("Profile router", "Profile router, masked innervation uniform in the router",
             f"Hybrid, home share >= {args.home_share:g} (exploratory)")
    variants = {name: {} for name in names}
    for iid, (row_ids, pairs, masked, masked_nodes) in suite_rows(graph, suite, negatives, selector).items():
        if any(not np.array_equal(caches[r][s][iid][0], row_ids) for r in regions for s in GATE):
            raise SystemExit(f"{iid}: a cache scores different rows from the selector")
        scores = np.stack([np.mean([_zscore(caches[r][s][iid][1]) for s in GATE], axis=0) for r in regions])
        ablated = profile.copy()
        ablated[masked_nodes] = 1.0 / len(regions)
        fused = (profile_route(scores, pairs, profile), profile_route(scores, pairs, ablated),
                 profile_route(scores, pairs, profile, masked=masked, home_share=args.home_share))
        for name, cache in zip(names, fused, strict=True):
            variants[name][iid] = (row_ids, cache)

    for name, cache in variants.items():
        report[name] = {"all": instance_mrr(cache), "block": instance_mrr(cache, block)}
    _, sentence = admission(report)
    instances = sorted(selector["instances"])

    def mean(values):
        return f"{np.mean(list(values.values())):.4f}" if values else "—"

    lines = [f"# B1 router on {args.suite} (F1, transductive)", "",
             "Per-instance MRR over the selector's metric rows, averaged over instances. F1 is transductive; "
             "selection happens on boundary F2-dev only.", "",
             "| Method | All rows | Block rows | " + " | ".join(f"`{i}`" for i in instances) + " |",
             "|---|---|---|" + "---|" * len(instances)]
    for name, by in report.items():
        lines.append(f"| {name} | {mean(by['all'])} | {mean(by.get('block', {}))} | "
                     + " | ".join(f"{by['all'][i]:.4f}" if i in by["all"] else "—" for i in instances) + " |")
    lines += ["", f"**Admission (plan 05).** {sentence}", ""]
    text = "\n".join(lines)
    print(text) if args.out is None else args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
