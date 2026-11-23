"""Calibrate the Phase 3 noise floor from one system's five realizations, or re-check it.

    python scripts/phase3_calibrate.py calibrate --system-run <run-id> [--system-run ...] \
        --scores-root results/phase3/scores-clean --runs-dir runs --data-root data
    python scripts/phase3_calibrate.py check --scores-root results/phase3/scores-clean

``calibrate`` writes ``results/phase3/delta.json`` from five full
realizations of one system, each read from the score cache ``scripts/score_union.py`` wrote
for it. It resolves the run manifests, checks the five caches against the manifest entries
and refuses any input that is not the system's. ``check`` re-derives the statistic from the
same inputs and refuses a committed record that disagrees, so an approval can read the
number rather than trust it.

Two Phase 3 differences from ``scripts/run_gate.py``: the record names the system it
calibrated (``system``), and the tag namespace is ``delta-phase3-*``. The record's
``system`` names one system; a second system needs its own calibration and record.

Phase 3 stores no row table equivalent to a region's ``results/<code>/scored``, so the
"no confirm scoring yet" guard looks for caches of the confirm suite under the scores root
and under ``results/phase3/scores`` instead of for stored confirm rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.scoring_common import refuse_recalibration  # noqa: E402
from src.boundary_suites import BoundarySuite  # noqa: E402
from src.feature_io import read_feature_artifacts  # noqa: E402
from src.integration_gate import calibrate_delta_p3  # noqa: E402
from src.digests import sha256  # noqa: E402
from src.regions import RegionGraph  # noqa: E402
from train.cohorts import cohort_digest  # noqa: E402
from train.milestones import _read_manifest  # noqa: E402
from train.train import _feature_digest, extras_read  # noqa: E402
from train.union_scoring import instance_mrr, load_cache  # noqa: E402

PHASE3 = "PHASE3"
CONFIRM_SUITE = "phase3_boundary_f2_confirm"
#: Cache-entry fields that must name this calibration's inputs, not another system's.
ENTRY_FIELDS = ("cohort", "system", "masked_innervation", "registry")


def registry_label(value: str | Path, root: Path) -> str:
    """How ``score_union.py`` records a registry path: relative to the repo root, else as given."""
    path, root = Path(str(value).replace("\\", "/")), Path(root).resolve()
    path = (path if path.is_absolute() else root / path).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def union_inputs(union: str, suite: str, data_root: Path, suites_root: Path) -> tuple[object, dict[str, str]]:
    """The union's feature table and the digests a run trained on it records."""
    folder = data_root / "processed" / "f1" / union.lower()
    graph = RegionGraph.read(folder)
    table, schema = read_feature_artifacts(folder, graph)
    return table, {"graph_digest": graph.digest(), "schema_digest": schema.digest(),
                   "suite_digest": BoundarySuite.read(suites_root / "f1" / union.lower() / f"{suite}.json").digest()}


def cache_path(scores_root: Path, union: str, suite: str, run_id: str) -> Path:
    return scores_root / "union" / union.lower() / suite / f"{run_id}.npz"


def manifest_entry(entries: dict, cache: Path, root: Path) -> dict:
    """The score-manifest entry for ``cache``; keys are written relative to the scoring repo."""
    wanted = cache.resolve()
    for key, entry in entries.items():
        path = Path(str(key).replace("\\", "/"))
        if (path if path.is_absolute() else root / path).resolve() == wanted:
            return entry
    raise ValueError(f"{cache} has no entry in the score manifest")


def verified_runs(run_ids: list[str], *, union: str, suite: str, system: str, masked_innervation: str,
                  registry: Path, scores_root: Path, manifest_path: Path, runs_dir: Path,
                  data_root: Path, suites_root: Path, root: Path) -> tuple[str, list[Path]]:
    """Resolve the five runs, refusing any whose recorded inputs are not this system's.

    Returns the shared cohort digest and one cache path per run, in ``run_ids`` order.
    """
    table, digests = union_inputs(union, suite, data_root, suites_root)
    entries = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    label = registry_label(registry, root)
    cohorts, caches = set(), []
    for run_id in run_ids:
        manifest = _read_manifest(runs_dir / run_id)
        if manifest.get("region") != union:
            raise ValueError(f"{run_id} trained on {manifest.get('region')!r}, not {union}: it is not this union system")
        trained = {**digests, "feature_digest": _feature_digest(table, extras=extras_read(manifest["config"]))}
        wrong = sorted(k for k, v in trained.items() if manifest.get(k) != v)
        if wrong:
            raise ValueError(f"{run_id}: recorded {', '.join(wrong)} differ from what current code produces "
                             f"for {union} and its {suite}")
        cohort = cohort_digest(manifest)
        cohorts.add(cohort)
        cache = cache_path(scores_root, union, suite, run_id)
        if not cache.is_file():
            raise ValueError(f"{cache}: score cache missing; score this run before calibrating")
        entry = manifest_entry(entries, cache, manifest_path.parents[2])
        recorded = {**{name: entry.get(name) for name in ENTRY_FIELDS}, "scores": entry.get("scores")}
        recorded["registry"] = registry_label(recorded["registry"], manifest_path.parents[2])
        expected = {"cohort": cohort, "system": system, "masked_innervation": masked_innervation,
                    "registry": label, "scores": sha256(cache)}
        wrong = sorted(name for name, value in expected.items() if recorded[name] != value)
        if wrong:
            raise ValueError(f"{cache}: score manifest entry records other {', '.join(wrong)} than this calibration "
                             f"({', '.join(f'{n}={recorded[n]!r}' for n in wrong)})")
        caches.append(cache)
    if len(cohorts) != 1:
        raise ValueError(f"the five runs are not one cohort: {len(cohorts)} distinct cohort digests")
    return cohorts.pop(), caches


def realizations(caches: list[Path]) -> list[dict[str, float]]:
    return [instance_mrr(load_cache(path)) for path in caches]


def confirm_caches(*roots: Path) -> list[str]:
    return [str(path) for root in roots if root.is_dir() for path in sorted(root.rglob(f"{CONFIRM_SUITE}*"))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("calibrate", "check"))
    parser.add_argument("--system-run", action="append", default=[],
                        help="a full realization of the system under calibration (five for calibrate)")
    parser.add_argument("--system", default="A", help="the system the floor is calibrated for")
    parser.add_argument("--union", default="UNION12")
    parser.add_argument("--suite", default="phase3_boundary_f1_val")
    parser.add_argument("--masked-innervation", choices=("real", "uniform"), default="real")
    parser.add_argument("--tag", help="tag name to record in delta.json")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--suites-root", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--scores-root", type=Path)
    parser.add_argument("--manifest", type=Path, help="the score manifest, by default results/phase3/manifest.json")
    args = parser.parse_args(argv)
    root = args.repo_root
    data_root = args.data_root or root / "data"
    scores_root = args.scores_root or root / "results/phase3/scores"
    delta_path = root / "results" / "phase3" / "delta.json"
    try:
        if args.mode == "calibrate":
            if delta_path.exists():
                raise ValueError(f"{delta_path} already exists; calibration is frozen")
            refuse_recalibration(PHASE3, root)
            if found := confirm_caches(scores_root, root / "results/phase3/scores"):
                raise ValueError(f"confirm-suite caches already exist ({found[0]}); "
                                 "calibration cannot follow confirm scoring")
            if len(set(args.system_run)) != 5:
                raise ValueError(f"calibrate needs five distinct --system-run, got {len(set(args.system_run))}")
            record = {}
        else:
            if not delta_path.is_file():
                raise ValueError(f"{delta_path} is missing: calibrate delta_P3 before checking it")
            record = json.loads(delta_path.read_text(encoding="utf-8"))
            if args.system_run and sorted(args.system_run) != sorted(record.get("run_ids") or []):
                raise ValueError("--system-run must name exactly the runs the record lists")
        run_ids = args.system_run or list(record.get("run_ids") or [])
        if not run_ids:
            raise ValueError("no --system-run given and the record names none")
        cohort, caches = verified_runs(
            run_ids, union=args.union, suite=args.suite, system=args.system,
            masked_innervation=args.masked_innervation, registry=args.registry or root / "results/experts.csv",
            scores_root=scores_root,
            manifest_path=args.manifest or root / "results/phase3/manifest.json",
            runs_dir=args.runs_dir or data_root / "runs", data_root=data_root,
            suites_root=args.suites_root or root / "data/suites", root=root,
        )
        s, delta = calibrate_delta_p3(realizations(caches))
        document = {
            "region": PHASE3, "s": s, "delta": delta, "run_ids": list(run_ids), "cohort": cohort,
            "system": args.system, "tag": args.tag or f"delta-phase3-{cohort[:8]}",
        }
        if args.mode == "check":
            wrong = {k: v for k, v in document.items() if record.get(k) != v}
            if wrong:
                raise ValueError(f"{delta_path} disagrees with the caches: " + ", ".join(
                    f"recorded {k}={record.get(k)!r}, derived {k}={v!r}" for k, v in sorted(wrong.items())))
            print(f"{delta_path} re-derives from its five runs: s={s!r}, delta_P3={delta}.")
            return 0

        delta_path.parent.mkdir(parents=True, exist_ok=True)
        temp = delta_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        temp.replace(delta_path)
        print(json.dumps(document, indent=2, sort_keys=True))
        print(f"Commit delta.json, then tag that commit {document['tag']!r} before any confirm scoring.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"phase3_calibrate failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
