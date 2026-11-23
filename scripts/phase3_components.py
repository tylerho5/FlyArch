"""Assemble the components file ``scripts/phase3_gate.py`` reads, from cached confirm scores.

    python scripts/phase3_components.py --milestone gate-1 --out results/phase3/confirm-gate-1.json
    python scripts/phase3_gate.py --components results/phase3/confirm-gate-1.json --out phase3-gate.md

Run it where the caches are (they are large; the JSON it writes is small). It reads only; it scores
nothing and opens no suite. Every value is a per-instance MRR taken from a cache the manifest vouches
for, in the registered form (the mean of the three gate seeds' per-instance MRRs):

- ``regional[R].system``: A's three gate realizations on R's ``f2_confirm`` (manifest entries with
  ``mode: regional``, ``system: A``, ``suite_region: R``);
- ``regional[R].reference``: ``results/phase3/reference/<R>.json`` (``scripts/phase3_reference_mrr.py``);
- ``boundary_system``: A on ``phase3_boundary_f2_confirm`` (union mode, system caches);
- ``boundary["expert:<code>"]``: the clean registry's three gate runs, keyed by run ID;
- ``boundary["baseline:<family>"]``: ``mode: baselines`` entries of ``score_union_baselines.py``, three seeds;
- ``boundary["B0"]`` and ``["seed_control"]``: the clean experts mixed with B0's F1-val weights
  (frozen exactly as ``b0_report.py --dataset f2`` freezes them), over three and five seeds.

Every confirm cache it reads must have a manifest entry for ``--milestone``, a scores hash that matches
the file on disk, the selector's rows, and (per family) one scoring-code stamp and one set of inputs; A's
regional caches must share one union graph (the U+ digest). It refuses on any missing region, seed,
family, instance or reference file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from scripts.b0_report import (  # noqa: E402
    GATE, WEIGHTS_DATASET, WEIGHTS_SUITE, check_provenance, check_rows, check_weight_provenance,
    check_with_system, frozen_weights, gate_caches, group_entries, load_group, selector_for,
)
from src.digests import sha256  # noqa: E402
from train.union_scoring import fuse, instance_mrr, load_cache, registry_label  # noqa: E402

SUITE = "phase3_boundary_f2_confirm"
REGIONAL_SUITE = "f2_confirm"
FAMILIES = ("configuration", "logreg", "xgboost")


def seed_mean(per_seed: list[dict[str, float]]) -> dict[str, float]:
    """The section 2 form: each instance's MRR averaged over its seeds; every seed must hold the same instances."""
    if not per_seed:
        raise ValueError("no seeds")
    keys = set(per_seed[0])
    if any(set(m) != keys for m in per_seed):
        raise ValueError("seeds cover different instances")
    return {k: float(np.mean([m[k] for m in per_seed])) for k in sorted(keys)}


def gate_runs(registry: Path) -> dict[str, dict[int, str]]:
    """{region: {seed: run_id}} for the registry's gate seeds."""
    out: dict[str, dict[int, str]] = {}
    with registry.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["gate_seed"] == "true":
                out.setdefault(row["region"], {})[int(row["seed"])] = row["run_id"]
    return out


def require_milestone(entries: dict[str, dict], milestone: str, what: str) -> None:
    """Every confirm cache used must have been scored under the milestone being assembled."""
    bad = sorted(p for p, e in entries.items() if e.get("milestone") != milestone)
    if bad:
        raise ValueError(f"{what}: caches not scored under milestone {milestone!r}: {bad[:3]}")


def require_unchanged(entries: dict[str, dict], root: Path, what: str) -> None:
    """Each cache's bytes must be the ones the manifest recorded."""
    missing = sorted(p for p in entries if not (root / p).is_file())
    changed = sorted(p for p, e in entries.items() if p not in missing and sha256(root / p) != e["scores"])
    if missing or changed:
        raise ValueError(f"{what}: missing {missing[:3]}, changed on disk {changed[:3]}")


def baseline_maps(manifest: dict, selector: dict, root: Path, milestone: str, masked: str = "uniform",
                  suite: str = SUITE) -> dict[str, dict[str, float]]:
    """``baseline:<family>`` maps from ``mode: baselines`` entries: three seeds each, provenance-checked."""
    out = {}
    for family in FAMILIES:
        entries = {p: e for p, e in sorted(manifest.items())
                   if e.get("mode") == "baselines" and e.get("suite") == suite and e.get("dataset") == "f2"
                   and e.get("masked_innervation") == masked and e.get("region") == f"baseline:{family}"}
        if len(entries) != 3:
            raise ValueError(f"baseline:{family} needs exactly three seed entries for {suite}, found {len(entries)}")
        require_milestone(entries, milestone, f"baseline:{family}")
        require_unchanged(entries, root, f"baseline:{family}")
        check_provenance(entries, selector, root)
        loaded = {p: load_cache(root / p) for p in entries}
        check_rows(loaded, selector)
        out[f"baseline:{family}"] = seed_mean([instance_mrr(c) for c in loaded.values()])
    return out


def regional_system(manifest: dict, root: Path, region: str, runs: list[str], milestone: str,
                    masked: str = "uniform") -> tuple[dict[str, float], set]:
    """A's per-instance MRR on ``region``'s f2_confirm (mean of the gate runs) and the union graphs it read."""
    entries = {}
    for run in runs:
        hits = {p: e for p, e in manifest.items()
                if e.get("mode") == "regional" and e.get("suite") == REGIONAL_SUITE and e.get("dataset") == "f2"
                and e.get("suite_region") == region and e.get("system") == "A" and e.get("run_id") == run
                and e.get("masked_innervation") == masked}
        if len(hits) != 1:
            raise ValueError(f"{region}: expected one manifest entry for A's run {run}, found {len(hits)}")
        entries.update(hits)
    require_milestone(entries, milestone, f"A on {region}")
    require_unchanged(entries, root, f"A on {region}")
    maps = [instance_mrr(load_cache(root / p)) for p in entries]
    return seed_mean(maps), {e.get("union_graph") for e in entries.values()}


def assemble(args) -> dict:
    root = REPO_ROOT
    delta = json.loads((root / "results/phase3/delta.json").read_text(encoding="utf-8"))
    runs = sorted(delta["run_ids"])[:3]
    if [r.split("-")[-2] for r in runs] != ["s000", "s001", "s002"]:
        raise ValueError(f"delta.json's first three realizations are not the gate seeds s000..s002: {runs}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    label = registry_label(args.opponents_registry, root)

    entries = group_entries(manifest, suite=SUITE, dataset="f2", union=args.union,
                            masked_innervation="uniform", registry=label)
    require_milestone(entries, args.milestone, f"boundary {SUITE}")
    require_unchanged(entries, root, f"boundary {SUITE}")
    selector = selector_for(args.suites_root, "f2", args.union, SUITE)
    check_with_system(entries, selector, root)
    caches, system = load_group(entries, selector, root)
    gate = gate_caches(caches)
    if sorted(system) != list(GATE):
        raise ValueError(f"A's boundary caches must cover seeds {list(GATE)}, found {sorted(system)}")

    weight_entries = group_entries(manifest, suite=WEIGHTS_SUITE, dataset=WEIGHTS_DATASET, union=args.union,
                                   masked_innervation="uniform", registry=label, include_system=False)
    if not weight_entries:
        raise ValueError(f"no {WEIGHTS_SUITE} caches to freeze B0's weights")
    weight_selector = selector_for(args.suites_root, WEIGHTS_DATASET, args.union, WEIGHTS_SUITE)
    check_provenance(weight_entries, weight_selector, root)
    weight_caches, _ = load_group(weight_entries, weight_selector, root)
    check_weight_provenance(entries, weight_entries)
    weights = frozen_weights(weight_caches, weight_selector)
    if set(weights) != set(gate):
        raise ValueError(f"B0's weights cover other regions: {sorted(set(weights) ^ set(gate))}")
    five = {r: [seeds[s] for s in sorted(seeds)] for r, seeds in caches.items() if len(seeds) == 5}
    if set(five) != set(gate):
        raise ValueError("the seed control needs five seeds for every region")

    registry_runs = gate_runs(args.opponents_registry)
    boundary: dict = {}
    for region, by_seed in sorted(caches.items()):
        if sorted(registry_runs.get(region, {})) != list(GATE):
            raise ValueError(f"{region}: registry gate seeds are not {list(GATE)}")
        boundary[f"expert:{region}"] = {registry_runs[region][s]: instance_mrr(by_seed[s]) for s in GATE}
    boundary.update(baseline_maps(manifest, selector, root, args.milestone))
    boundary["B0"] = instance_mrr(fuse(gate, weights))
    boundary["seed_control"] = instance_mrr(fuse(five, weights))
    boundary_system = seed_mean([instance_mrr(system[s]) for s in GATE])

    reference = {}
    for path in sorted(args.reference_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        reference[doc["region"]] = doc["reference"]
    absent = sorted(set(registry_runs) - set(reference))
    if absent:
        raise ValueError(f"reference files missing for {absent} in {args.reference_dir}; the gate needs all twelve")
    regional, graphs = {}, set()
    for region, ref in sorted(reference.items()):
        mine, region_graphs = regional_system(manifest, root, region, runs, args.milestone)
        graphs |= region_graphs
        ids = {frozenset(m) for m in ref.values()}
        if ids != {frozenset(mine)}:
            raise ValueError(f"{region}: A's confirm instances differ from the reference run's instances")
        regional[region] = {"system": mine, "reference": ref}
    if len(graphs) != 1 or None in graphs:
        raise ValueError(f"A's regional caches must all read one union graph (the U+ digest); "
                         f"found {sorted(map(str, graphs))}")
    return {"milestone": args.milestone, "system": delta["system"], "system_cohort": delta["cohort"],
            "regional": regional, "boundary_system": boundary_system, "boundary": boundary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--milestone", required=True)
    parser.add_argument("--union", default="UNION12")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "results/phase3/manifest.json")
    parser.add_argument("--suites-root", type=Path, default=REPO_ROOT / "data/suites")
    parser.add_argument("--reference-dir", type=Path, default=REPO_ROOT / "results/phase3/reference")
    parser.add_argument("--opponents-registry", type=Path, default=REPO_ROOT / "results/experts-clean.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec = assemble(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"phase3_components failed: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: {len(spec['regional'])} regions, {len(spec['boundary'])} boundary opponents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
