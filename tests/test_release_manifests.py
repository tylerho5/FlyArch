import csv
import json
from pathlib import Path

from train.cohorts import cohort_digest

ROOT = Path(__file__).resolve().parents[1]


def test_every_registry_run_ships_a_manifest_in_its_cohort():
    registry = {}
    for name in ("experts.csv", "experts-clean.csv"):
        with open(ROOT / "results" / name, newline="") as fh:
            registry.update({row["run_id"]: row["cohort"] for row in csv.DictReader(fh)})
    for run_id, cohort in registry.items():
        manifest = json.loads((ROOT / "results" / "manifests" / f"{run_id}.json").read_text())
        assert manifest["run_id"] == run_id
        assert cohort_digest(manifest) == cohort, run_id
