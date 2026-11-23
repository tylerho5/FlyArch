"""Write B0's report from cached union scores (Phase 3 plan 03, task 3).

    python scripts/b0_report.py --suite phase3_boundary_f1_val --out phase3-b0.md
    python scripts/b0_report.py --suite phase3_boundary_f1_val --masked-innervation uniform --out <path>
    python scripts/b0_report.py --dataset f2 --suite phase3_boundary_f2_dev --registry results/experts-clean.csv --masked-innervation uniform --out <path>

Reads ``results/phase3/manifest.json`` for every ``union``-mode cache of ``--suite``, and
the suite's selector for the block stratum. Every number is per-instance MRR over the
selector's metric rows, averaged over instances:

- each expert in the per-instance form, the mean of its three gate seeds' MRRs;
- each region's 1-, 2-, 3- and 5-seed score ensembles (seeds combined before ranking);
- B0 at equal weights, B0 with leave-one-instance-out weights (its honest F1 number),
  and B0 with weights fit on every instance (in-sample, labelled so);
- the seed control: B0's fitted weights over each region's five seeds;
- B0 built from one seed per region, for seeds 0, 1 and 2, to show seed spread;
- plan 06's A, when its caches exist (``score_union.py --system-run``): its three gate
  seeds in the per-instance form and as a 3-seed score ensemble. A is never part of B0's mixture.

Every F1-val number is transductive: some experts trained on the masked neurons.
``--masked-innervation uniform`` reads the caches scored with each masked node's innervation
set to uniform (the registered ablation), the leak-free comparison. ``--dataset f2`` reads
the F2 boundary caches and the F2 selector, where the experts trained on another dataset.
It fits B0's weights on the F1-val caches in the same manifest and applies them unchanged;
no F2 instance fits or selects a weight.
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

from src.boundary_suites import selector_digest  # noqa: E402
from src.digests import sha256  # noqa: E402
from train.union_scoring import fit_weights, fuse, instance_mrr, load_cache, loo_b0, registry_label  # noqa: E402

GATE = (0, 1, 2)
#: The registry a cache was scored from, as ``score_union.py`` records it.
DEFAULT_REGISTRY = "results/experts.csv"
#: The seed control's out-of-sample row; plan 05's B1 admission reads it by this name.
SEED_CONTROL_LOO = "Seed control, leave-one-instance-out weights"
#: B0's frozen mixture on F2, fit only on the F1-val boundary suite.
WEIGHTS_SUITE = "phase3_boundary_f1_val"
WEIGHTS_DATASET = "f1"
B0_FROZEN = "B0, F1-val weights"


#: Inputs every cache in one report must share, and those one region's seeds must share.
SHARED = ("union_graph", "union_schema", "union_features", "suite_digest", "negatives_file")
PER_REGION = ("cohort", "expert_schema", "expert_geometry")


def check_provenance(entries: dict[str, dict], selector: dict | None, root: Path, shared=SHARED) -> None:
    """Refuse caches from mixed code or inputs, from another selector, or changed since they were written.

    ``shared`` names the inputs every cache must agree on; ``selector`` is None for suites
    without one (a region's own suite, every row scored).
    """
    if not entries:
        raise SystemExit("no caches in the manifest for this suite")
    stamps = {(e["commit"], json.dumps(e["code"], sort_keys=True)) for e in entries.values()}
    if len(stamps) != 1:
        raise SystemExit(f"caches come from {len(stamps)} different commits or code hashes; rescore them together")
    mixed = [k for k in shared if len({e.get(k) for e in entries.values()}) != 1]
    regions = {e["region"] for e in entries.values()}
    mixed += [f"{r} {k}" for r in sorted(regions) for k in PER_REGION
              if len({e.get(k) for e in entries.values() if e["region"] == r}) != 1]
    if mixed:
        raise SystemExit(f"caches were scored on different inputs ({', '.join(mixed[:5])}); rescore them together")
    stale = [] if selector is None else sorted(
        p for p, e in entries.items() if e.get("selector") != selector_digest(selector))
    changed = sorted(p for p, e in entries.items() if sha256(root / p) != e["scores"])
    if stale or changed:
        raise SystemExit(f"caches built on another selector: {stale[:3]}; caches changed on disk: {changed[:3]}")


def check_with_system(entries: dict[str, dict], selector: dict | None, root: Path) -> None:
    """The experts' caches and A's (plan 06) each pass ``check_provenance`` as their own group.

    A is scored later, so its commit may differ, but it must share the experts' inputs, selector
    and scoring-code hashes; when the scoring code has changed since, rescore the experts too.
    """
    experts = {p: e for p, e in entries.items() if not e.get("system")}
    systems = {p: e for p, e in entries.items() if e.get("system")}
    check_provenance(experts, selector, root)
    if systems:
        check_provenance(systems, selector, root)
        ours, theirs = next(iter(experts.values())), next(iter(systems.values()))
        differ = [k for k in (*SHARED, "code") if ours.get(k) != theirs.get(k)]
        if differ:
            raise SystemExit(f"A's caches differ from the experts' in {differ}; rescore the experts with A's code")


def check_rows(caches: dict[str, dict], selector: dict) -> None:
    """Every cache (experts' and A's) holds exactly the selector's instances and rows, at one candidate width.

    Provenance stamps say which inputs a cache came from, not that it scored every row:
    one cache with rows missing would compare B0 and A on different rows.
    """
    widths = set()
    for path, cache in caches.items():
        if set(cache) != set(selector["instances"]):
            raise SystemExit(f"{path}: instances differ from the selector's")
        for iid, (rows, scores) in cache.items():
            if not np.array_equal(rows, selector["instances"][iid]["rows"]) or scores.shape[0] != len(rows):
                raise SystemExit(f"{path}: {iid} does not score exactly the selector's rows")
            widths.add(scores.shape[1])
    if len(widths) != 1:
        raise SystemExit(f"caches rank against different numbers of candidates: {sorted(widths)}")


def selector_for(suites_root: Path, dataset: str, union: str, suite: str) -> dict:
    path = suites_root / dataset / union.lower() / f"{suite}.selector.json"
    return json.loads(path.read_text(encoding="utf-8"))


def group_entries(manifest: dict, *, suite: str, dataset: str, union: str,
                  masked_innervation: str, registry: str, include_system: bool = True) -> dict[str, dict]:
    return {path: e for path, e in sorted(manifest.items())
            if e["mode"] == "union" and e["suite"] == suite and e.get("suite_region", union) == union
            and e.get("dataset", "f1") == dataset
            and e.get("masked_innervation", "real") == masked_innervation
            and e.get("registry", DEFAULT_REGISTRY) == registry
            and (include_system or not e.get("system"))}


def load_group(entries: dict[str, dict], selector: dict, root: Path):
    loaded = {path: load_cache(root / path) for path in entries}
    check_rows(loaded, selector)
    caches: dict[str, dict[int, dict]] = defaultdict(dict)
    system: dict[int, dict] = {}
    for path, entry in entries.items():
        (system if entry.get("system") else caches[entry["region"]])[entry["seed"]] = loaded[path]
    return dict(caches), system


def check_weight_provenance(entries: dict[str, dict], weight_entries: dict[str, dict]) -> None:
    ours = next(e for e in entries.values() if not e.get("system"))
    theirs = next(iter(weight_entries.values()))
    if ours.get("code") != theirs.get("code"):
        raise SystemExit("F1-val caches were scored with other code than the F2 caches; rescore them")
    for key in ("union_schema", "registry", "masked_innervation"):
        if ours.get(key) != theirs.get(key):
            raise SystemExit(f"F1-val caches have other {key} than the F2 caches; rescore them")

    def stamps(group):
        return {r: {(e.get("cohort"), e.get("expert_schema")) for e in group.values()
                    if not e.get("system") and e["region"] == r}
                for r in {e["region"] for e in group.values() if not e.get("system")}}

    read, weight = stamps(entries), stamps(weight_entries)
    if set(read) != set(weight) or any(len(read[r]) != 1 or read[r] != weight[r] for r in read):
        raise SystemExit("F1-val caches came from another cohort or expert schema; rescore them")


def gate_caches(caches: dict[str, dict[int, dict]]) -> dict[str, list[dict]]:
    partial = sorted(f"{r} (missing {[s for s in GATE if s not in seeds]})"
                     for r, seeds in caches.items() if not all(s in seeds for s in GATE))
    if partial or not caches:
        raise SystemExit(f"regions without all three gate seeds cached: {', '.join(partial) or 'none cached'}")
    return {r: [seeds[s] for s in GATE] for r, seeds in caches.items()}


def frozen_weights(caches: dict[str, dict[int, dict]], selector: dict) -> dict[str, float]:
    return fit_weights(gate_caches(caches), sorted(selector["instances"]))


def _mean(values: dict) -> float:
    return float(np.mean(list(values.values())))


def section2(seeds: list[dict], strata: dict[str, dict | None]) -> dict:
    """The registered form: per instance, the mean of the seeds' MRRs (not a score ensemble)."""
    per_seed = {label: [instance_mrr(c, rows_) for c in seeds] for label, rows_ in strata.items()}
    return {label: {i: float(np.mean([s[i] for s in mrrs if i in s])) for i in mrrs[0]}
            for label, mrrs in per_seed.items()}


def system_rows(system: dict[int, dict], strata: dict[str, dict | None]) -> list:
    """A's rows: its three gate seeds in the per-instance form and as a 3-seed score ensemble."""
    missing = [s for s in GATE if s not in system]
    if missing:
        raise SystemExit(f"A is missing gate seed(s) {missing}")
    seeds = [system[s] for s in GATE]
    ensemble = fuse({"A": seeds})
    return [("A (3 seeds, §2 form)", section2(seeds, strata)),
            ("A 3-seed ensemble", {label: instance_mrr(ensemble, rows_) for label, rows_ in strata.items()})]


def methods(caches: dict[str, dict[int, dict]], strata: dict[str, dict | None],
            weights: dict[str, float] | None = None) -> tuple[list, dict]:
    """``(name, {stratum: {instance: mrr}})`` rows, and B0's weights."""
    gate = gate_caches(caches)
    rows = []

    def add(name, cache):
        rows.append((name, {label: instance_mrr(cache, rows_) for label, rows_ in strata.items()}))

    for region in sorted(gate):
        rows.append((f"{region} expert (3 seeds, §2 form)", section2(gate[region], strata)))
    for region in sorted(caches):
        seeds = [caches[region][s] for s in sorted(caches[region])]
        for k in (1, 2, 3, 5):
            if len(seeds) >= k:
                add(f"{region} {k}-seed ensemble", fuse({region: seeds[:k]}))
    add("B0, equal weights", fuse(gate))
    five = {r: [seeds[s] for s in sorted(seeds)] for r, seeds in caches.items() if len(seeds) == 5}
    if weights is None:
        held, fold_weights = loo_b0(gate)
        rows.append(("B0, leave-one-instance-out weights", {"all": held}))
        weights = fit_weights(gate, sorted(next(iter(gate.values()))[0]))
        add("B0, weights fit on all instances (in-sample)", fuse(gate, weights))
        if set(five) == set(gate):  # a partial control would be a different mixture, so it is not reported
            add("Seed control: B0's weights over five seeds (weights fit in-sample)", fuse(five, weights))
            # B0's own leave-one-out fold weights over five seeds: only the seed count changes
            held5 = {i: instance_mrr(fuse(five, w))[i] for i, w in fold_weights.items()}
            rows.append((SEED_CONTROL_LOO, {"all": held5}))
        for s in GATE:
            add(f"B0's weights, seed {s} only (weights fit in-sample)",
                fuse({r: [seeds[s]] for r, seeds in caches.items() if s in seeds}, weights))
    else:
        if set(weights) != set(gate):
            raise SystemExit(f"B0's frozen weights cover other regions: {sorted(set(weights) ^ set(gate))}")
        add(B0_FROZEN, fuse(gate, weights))
        if set(five) == set(gate):
            add("Seed control: B0's F1-val weights over five seeds", fuse(five, weights))
        for s in GATE:
            add(f"B0's weights, seed {s} only (F1-val weights)",
                fuse({r: [seeds[s]] for r, seeds in caches.items() if s in seeds}, weights))
    return rows, weights


def render(suite: str, rows: list, weights: dict, n_rows: dict, masked_innervation: str = "real",
           dataset: str = "f1", weights_source: str | None = None) -> str:
    instances = sorted(next(iter(rows))[1]["all"])
    if dataset == "f1":
        heading = f"# B0 on {suite} (F1, transductive)"
        provenance = ("Experts trained on F1, some on these neurons, so F1-val is transductive and "
                      "F2-dev is the clean check.")
    else:
        heading = f"# B0 on {suite} (F2, cross-dataset)"
        provenance = ("The experts trained on F1 and these rows are F2's, whose edges no expert saw; "
                      "F1-val is the transductive check they were selected on. B0's weights are frozen "
                      "from the F1-val boundary suite; no F2 instance fits or selects a weight.")
    lines = [heading, "",
             "Every number is a per-instance MRR over the selector's metric rows, averaged over instances. "
             + provenance,
             ""]
    if masked_innervation == "uniform":
        lines.append("Scored with every masked root's innervation read as the uniform share over its labels, "
                     "so masked innervation supplies no hidden-edge profile information. Whether the models "
                     "were also trained that way is their cohort's policy, not this flag's (`--registry`).")
        lines.append("")
    if weights_source:
        lines.append(f"B0's weights are frozen from {weights_source}; no instance of this suite fits or selects a weight.")
        lines.append("")
    missing = [i for i in instances if i not in n_rows["block_instances"]]
    block_note = (f" Block rows average only the {len(instances) - len(missing)} instances with block rows; "
                  f"{', '.join(f'`{i}`' for i in missing)} have none and are left out of that column."
                  if missing else "")
    lines += [f"Metric: the mean of the {len(instances)} instances' MRRs, over {n_rows['all']:,} metric rows "
              f"({n_rows['block']:,} in the INP+PENP+VLNP+VMNP block). Those rows are not independent "
              "replicates; the instance is the resampling unit." + block_note, "",
              "| Method | All rows | Block rows | " + " | ".join(f"`{i}`" for i in instances) + " |",
              "|---|---|---|" + "---|" * len(instances)]
    for name, by in rows:
        cells = [f"{_mean(by['all']):.4f}", f"{_mean(by['block']):.4f}" if by.get("block") else "—"]
        cells += [f"{by['all'][i]:.4f}" if i in by["all"] else "—" for i in instances]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    source = "fit on every instance" if weights_source is None else f"fit on {weights_source} only"
    lines += ["", f"B0's weights ({source}): "
              + ", ".join(f"{r} {w:g}" for r, w in sorted(weights.items())) + ".", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--union", default="UNION12")
    parser.add_argument("--dataset", choices=("f1", "f2"), default="f1",
                        help="whose caches and selector to read")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "results/phase3/manifest.json")
    parser.add_argument("--suites-root", type=Path, default=REPO_ROOT / "data/suites")
    parser.add_argument("--masked-innervation", choices=("real", "uniform"), default="real",
                        help="which caches to read: the registered ablation scores with masked innervation uniform")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / DEFAULT_REGISTRY,
                        help="the cohort's registry, as recorded when its caches were scored")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    label = registry_label(args.registry, REPO_ROOT)
    entries = group_entries(manifest, suite=args.suite, dataset=args.dataset, union=args.union,
                            masked_innervation=args.masked_innervation, registry=label)
    selector = selector_for(args.suites_root, args.dataset, args.union, args.suite) if entries else None
    check_with_system(entries, selector, REPO_ROOT)
    caches, system = load_group(entries, selector, REPO_ROOT)
    frozen = None
    weights_source = None
    if args.dataset == "f2":
        weight_entries = group_entries(manifest, suite=WEIGHTS_SUITE, dataset=WEIGHTS_DATASET,
                                       union=args.union, masked_innervation=args.masked_innervation,
                                       registry=label, include_system=False)
        if not weight_entries:
            raise SystemExit(f"no {WEIGHTS_SUITE} caches in the manifest to fit B0's weights")
        weight_selector = selector_for(args.suites_root, WEIGHTS_DATASET, args.union, WEIGHTS_SUITE)
        check_provenance(weight_entries, weight_selector, REPO_ROOT)
        weight_caches, _ = load_group(weight_entries, weight_selector, REPO_ROOT)
        check_weight_provenance(entries, weight_entries)
        frozen = frozen_weights(weight_caches, weight_selector)
        weights_source = f"`{WEIGHTS_SUITE}` (F1-val)"
    block = {iid: np.asarray(r["block_rows"]) for iid, r in selector["instances"].items()}
    strata = {"all": None, "block": block}
    rows, weights = methods(caches, strata, weights=frozen)
    if system:
        rows += system_rows(system, strata)
    n_rows = {"all": sum(r["n_rows"] for r in selector["instances"].values()),
              "block": sum(len(r["block_rows"]) for r in selector["instances"].values()),
              "block_instances": {iid for iid, r in selector["instances"].items() if len(r["block_rows"])}}
    text = render(args.suite, rows, weights, n_rows, args.masked_innervation, args.dataset, weights_source)
    print(text) if args.out is None else args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
