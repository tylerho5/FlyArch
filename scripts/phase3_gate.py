"""Evaluate Phase 3's integrated claim from per-instance MRRs (Phase 3 plan 07).

    python scripts/phase3_gate.py --components results/phase3/confirm-<milestone>.json \
        --out phase3-gate.md

``--components`` is JSON holding only measured values:

    {"milestone": "...", "system": "B1", "system_cohort": "<cohort in results/phase3/delta.json>",
     "regional": {"AL": {"system": {iid: mrr}, "reference": {run_id: {iid: mrr}} x 3}},
     "boundary_system": {iid: mrr},
     "boundary": {"expert:AL": {run_id: {iid: mrr}} x 3, "B0": {iid: mrr}, ...}}

Nothing that decides the verdict comes from that file. The gated regions are the Phase 2
registry's regions (``--reference-registry``), whose cohorts are also what each regional
guard is scored against: the cohort that earned that region's Phase 2 verdict. The
``expert:<code>`` boundary opponents are keyed by ``--opponents-registry``'s three gate
runs, the clean cohort by default, because the registered rules (2026-11-07) make it the
primary Phase 3 opponent. B0, the seed control and the baselines arrive as measured MRR
maps in the components file. Every boundary comparison reads the one
``boundary_system`` map, so the conjunction describes one system. A regional guard's
references are keyed by exactly ``--reference-registry``'s three gate runs, and each
``expert:<code>`` opponent's seeds by exactly ``--opponents-registry``'s three gate runs,
so one seed cannot stand in for three. ``system`` and ``system_cohort`` must match the tagged ``delta_P3`` record,
which names the system it calibrated, so B0 is the system only when that record says so. Every delta is frozen-checked
(``scoring_common.verify_tagged_delta``): ``delta_P3`` from ``results/phase3/delta.json``
at its ``delta-phase3-*`` tag, and each region's ``delta_r`` from
``results/<code>/delta.json`` at its ``delta-<code>-*`` tag, or with no tag, at its entry in
``results/release-evidence.json`` (0 only when the region has no file, no tag and no entry).
The milestone must be a nonempty label. The report lists every component, failures included.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.scoring_common import _delta_tags, _git, release_evidence, verify_tagged_delta  # noqa: E402
from src.integration_gate import _seed_mean, boundary_comparison, integrated_claim, regional_guard  # noqa: E402

PHASE3 = "PHASE3"


#: The Phase 2 registry: the cohorts that earned each region's verdict, which its guard reads.
REFERENCE_REGISTRY = "results/experts.csv"
#: The primary boundary opponents (registered 2026-11-07: the clean, ablated cohort).
OPPONENTS_REGISTRY = "results/experts-clean.csv"


def registry_cohorts(text: str, source: str) -> dict[str, tuple[str, frozenset[str]]]:
    """Each registry region's cohort and three gate run IDs, from the registry committed at HEAD."""
    cohorts: dict[str, set] = {}
    gate: dict[str, set] = {}
    for row in csv.DictReader(io.StringIO(text)):
        cohorts.setdefault(row["region"], set()).add(row["cohort"])
        if row["gate_seed"] == "true":
            gate.setdefault(row["region"], set()).add(row["run_id"])
    if not cohorts or any(len(c) != 1 for c in cohorts.values()):
        raise SystemExit(f"{source}: every region needs exactly one registered cohort")
    if any(len(gate.get(r, ())) != 3 for r in cohorts):
        raise SystemExit(f"{source}: every region needs exactly three gate runs")
    return {region: (c.pop(), frozenset(gate[region])) for region, c in cohorts.items()}


def registries(reference: str, opponents: str, repo_root: Path) -> tuple[dict, dict]:
    """The guard's registry and the boundary opponents' registry, both as committed at HEAD."""
    return tuple(registry_cohorts(_git(repo_root, "show", f"HEAD:{path}").decode("utf-8"), path)
                 for path in (reference, opponents))


def registry_blobs(reference: str, opponents: str, repo_root: Path) -> tuple[str, str]:
    """The Git blob IDs of the two registries the gate read at HEAD."""
    return tuple(_git(repo_root, "rev-parse", f"HEAD:{path}").decode().strip()
                 for path in (reference, opponents))


def registered_seeds(region: str, reference: dict, runs: frozenset[str]) -> list[dict]:
    """A region's reference seeds, refused unless keyed by exactly its three registered gate runs."""
    if set(reference) != runs:
        raise ValueError(f"{region}'s reference must be its registered gate runs {sorted(runs)}, "
                         f"got {sorted(reference)}")
    return [reference[run] for run in sorted(runs)]


def regional_delta(region: str, cohort: str, repo_root: Path) -> float | None:
    """The region's frozen delta, or None (margin 0) when it has no delta.json, tag or evidence entry."""
    path = repo_root / "results" / region.lower() / "delta.json"
    if (not path.is_file() and not _delta_tags(region, repo_root)
            and region not in release_evidence(repo_root)["calibrations"]):
        return None
    return float(verify_tagged_delta(region=region, expected_cohort=cohort, delta_path=path,
                                     repo_root=repo_root)["delta"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--components", required=True, type=Path)
    parser.add_argument("--reference-registry", default=REFERENCE_REGISTRY,
                        help="whose gate cohorts each regional guard is scored against")
    parser.add_argument("--opponents-registry", default=OPPONENTS_REGISTRY,
                        help="whose gate runs key the boundary expert:<code> opponents; the seed control is a measured component")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    root = args.repo_root
    spec = json.loads(args.components.read_text(encoding="utf-8"))
    if not isinstance(spec.get("milestone"), str) or not spec["milestone"].strip():
        raise SystemExit("the components must name a nonempty milestone")
    delta_path = root / "results" / "phase3" / "delta.json"
    if not delta_path.is_file():
        raise SystemExit(f"{delta_path} is missing: calibrate delta_P3 before any boundary comparison")
    try:
        reference_cohorts, opponent_cohorts = registries(args.reference_registry, args.opponents_registry, root)
        if set(reference_cohorts) != set(opponent_cohorts):
            raise ValueError(f"{args.opponents_registry}'s regions differ from {args.reference_registry}'s "
                             f"(missing {sorted(set(reference_cohorts) - set(opponent_cohorts))}, "
                             f"extra {sorted(set(opponent_cohorts) - set(reference_cohorts))})")
        record = verify_tagged_delta(region=PHASE3, expected_cohort=spec["system_cohort"],
                                     delta_path=delta_path, repo_root=root)
        if record.get("system") != spec["system"]:
            raise ValueError(f"delta_P3 was calibrated for {record.get('system')!r}, not {spec['system']!r}")
        delta_p3 = float(record["delta"])
        typed = sorted(r for r, c in spec["regional"].items() if "delta" in c)
        if typed:
            raise ValueError(f"components may not carry a delta ({typed}); delta_r comes from results/<code>/delta.json")
        unknown = sorted(set(spec["regional"]) - set(reference_cohorts))
        if unknown:
            raise ValueError(f"regional components for regions outside {args.reference_registry}: {unknown}")
        regional = [regional_guard(region, c["system"],
                                   registered_seeds(region, c["reference"], reference_cohorts[region][1]),
                                   regional_delta(region, reference_cohorts[region][0], root))
                    for region, c in sorted(spec["regional"].items())]
        system = spec["boundary_system"]  # one map: every opponent faces the same system
        boundary = []
        for name, mrrs in sorted(spec["boundary"].items()):
            if name.startswith("expert:") and name[7:] in opponent_cohorts:  # the per-instance form: mean of its gate seeds
                mrrs = _seed_mean(registered_seeds(name, mrrs, opponent_cohorts[name[7:]][1]))
            boundary.append(boundary_comparison(name, system, mrrs, delta_p3))
        passed, failures = integrated_claim(regional, boundary, gated_regions=sorted(reference_cohorts),
                                            experts=sorted(opponent_cohorts), system_is_b0=spec["system"] == "B0")
    except ValueError as exc:
        raise SystemExit(f"refused: {exc}") from exc
    reference_blob, opponents_blob = registry_blobs(args.reference_registry, args.opponents_registry, root)
    role = "primary" if args.opponents_registry == OPPONENTS_REGISTRY else "labelled sensitivity comparison"
    lines = [f"# Phase 3 gate: {spec['system']} at {spec['milestone']}", "",
             f"**Integrated claim: {'PASS' if passed else 'FAIL'}.** delta_P3 = {delta_p3}.", "",
             f"Reference registry: `{args.reference_registry}` ({reference_blob}); "
             f"opponents registry: `{args.opponents_registry}` ({opponents_blob}, {role}).", "",
             "| Component | Name | mean r_i | 95% CI | Margin | Sign p | Pass |", "|---|---|---|---|---|---|---|"]
    for c in (*regional, *boundary):
        p = "—" if c.p_sign is None else f"{c.p_sign:.4g}"
        lines.append(f"| {c.kind} | {c.name} | {c.r_bar:+.4f} | [{c.ci_low:+.4f}, {c.ci_high:+.4f}] | "
                     f"{c.margin:+.4f} | {p} | {'yes' if c.passed else 'no'} |")
    lines += [""] + [f"- {f}" for f in failures] + [""]
    text = "\n".join(lines)
    print(text) if args.out is None else args.out.write_text(text, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
