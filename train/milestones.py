"""Deterministic milestone export for completed M1 runs.

One tracked CSV row per run: input and suite digests, configuration identity,
the device and precision it trained under, the F1-val-selected epoch, and
whether the run finished its epoch budget or stopped early. The exporter is a read-only projection of run directories: it
never stages or commits the output (publishing a milestone is the operator's
deliberate act) and never invokes Git. ``best_val_mrr`` is the F1-val
selection statistic, not a confirm score, and ``selection_instances`` names the
F1-val instances it averages (``all`` for a history that predates per-instance
scoring).
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from .train import validation_mrr
from .cohorts import cohort_digest
from .region_config import TRAINING_ORDER, config_path, load_region_config

__all__ = ["best_f1_val", "export_milestones", "validate_completed_run"]

#: The CSV's columns, in write order. One row per run ID.
COLUMNS = (
    "run_id",
    "seed",
    "graph_digest",
    "feature_digest",
    "schema_digest",
    "suite_digest",
    "config_json",
    "best_epoch",
    "best_val_mrr",
    "selection_instances",
    "completion",
    "device",
    "precision",
)

EXPERT_COLUMNS = ("region", "gate_seed", *COLUMNS, "backend", "cohort")

#: What ended the epoch loop of a completed run.
_EPOCH_BUDGET = "epoch_budget"
_EARLY_STOPPED = "early_stopped"

_DIGEST_FIELDS = (
    "graph_digest",
    "feature_digest",
    "schema_digest",
    "suite_digest",
)


def _reject(run_dir: Path, reason: str) -> ValueError:
    return ValueError(f"{run_dir}: {reason}")


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise _reject(run_dir, "no manifest.json; not a run directory") from None
    except (OSError, ValueError) as exc:
        raise _reject(run_dir, f"manifest.json is unreadable ({exc})") from exc
    if not isinstance(manifest, dict):
        raise _reject(run_dir, "manifest.json is not an object")
    if manifest.get("run_id") != run_dir.name:
        raise _reject(
            run_dir,
            f"manifest names run {manifest.get('run_id')!r}, not this directory",
        )
    if not isinstance(manifest.get("seed"), int):
        raise _reject(run_dir, "manifest has no integer seed")
    for field in (*_DIGEST_FIELDS, "device", "precision"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise _reject(run_dir, f"manifest has no {field}")
    config = manifest.get("config")
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("epochs"), int)
        or config["epochs"] < 1
    ):
        raise _reject(run_dir, "manifest has no epoch budget")
    return manifest


def _read_checkpoint(run_dir: Path, name: str) -> dict[str, Any]:
    path = run_dir / name
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise _reject(run_dir, f"no {name}; the run is incomplete") from None
    except Exception as exc:
        # Any checkpoint this process cannot unpickle is a rejected run, not an
        # exporter crash: the operator needs the run's name in the diagnosis.
        raise _reject(run_dir, f"{name} is unreadable ({exc})") from exc
    if not isinstance(checkpoint, dict):
        raise _reject(run_dir, f"{name} is not a checkpoint")
    return checkpoint


def _history(checkpoint: dict[str, Any], run_dir: Path, name: str) -> dict[str, Any]:
    history = checkpoint.get("history")
    if not isinstance(history, dict) or not isinstance(history.get("loss"), list):
        raise _reject(run_dir, f"{name} has no epoch history")
    return history


def _finite_scores(values: Any) -> bool:
    return isinstance(values, list) and all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in values
    )


def _validation(
    history: dict[str, Any], run_dir: Path, last_epoch: int
) -> tuple[list[int], list[float], str]:
    """The history's F1-val-scored epochs and mean MRRs, after checking their shape.

    Histories written before per-instance scoring store the mean as
    ``val_mrr``, one per epoch, and have no ``val_mrr_epoch``.
    """
    missing = _reject(
        run_dir,
        "no F1-val history; a run without the committed suite is a smoke "
        "test and has no milestone row",
    )
    if "val_mrr" not in history and "val_mrr_by_instance" not in history:
        raise missing
    if "val_mrr_by_instance" in history:
        by_instance = history["val_mrr_by_instance"]
        if not isinstance(by_instance, dict) or not by_instance:
            raise _reject(run_dir, "F1-val history has no instances")
        if not all(_finite_scores(values) for values in by_instance.values()):
            raise _reject(run_dir, "F1-val history is not finite")
        selected = history.get("val_selection_instances", list(by_instance))
        if (
            not isinstance(selected, list)
            or not selected
            or not all(isinstance(instance_id, str) for instance_id in selected)
            or len(set(selected)) != len(selected)
            or not set(selected) <= set(by_instance)
        ):
            raise _reject(
                run_dir, "F1-val selection instances are not distinct instances of the history"
            )
        selection = " ".join(selected)
        if len({len(values) for values in by_instance.values()}) != 1:
            raise _reject(
                run_dir, "F1-val instances were scored a different number of times"
            )
        scored = history.get("val_mrr_epoch")
        if not isinstance(scored, list) or not all(
            type(epoch) is int for epoch in scored
        ):
            raise _reject(run_dir, "F1-val scored epochs are not a list of epochs")
        if scored != sorted(set(scored)) or not all(
            0 <= epoch <= last_epoch for epoch in scored
        ):
            raise _reject(
                run_dir, "F1-val scored epochs are not increasing epochs of this run"
            )
    elif not _finite_scores(history.get("val_mrr")):
        raise _reject(run_dir, "F1-val history is not finite")
    else:
        selection = "all"

    scored, values = validation_mrr(history)
    if not values:
        raise missing
    if len(scored) != len(values):
        raise _reject(run_dir, "F1-val history and its scored epochs differ in length")
    return scored, values, selection


def best_f1_val(run_dir: Path, best_epoch: Any) -> tuple[dict[str, Any], dict[str, float]]:
    """``best.pt``'s history and its per-instance F1-val MRR at ``best_epoch``."""
    history = _read_checkpoint(run_dir, "best.pt").get("history")
    if not isinstance(history, dict) or "val_mrr_by_instance" not in history:
        raise _reject(run_dir, "best.pt has no per-instance F1-val history")
    if type(best_epoch) is not int:
        raise _reject(run_dir, "manifest has no integer best epoch")
    scored, _, _ = _validation(history, run_dir, best_epoch)
    if best_epoch not in scored:
        raise _reject(run_dir, f"best epoch {best_epoch} has no F1-val score")
    index = scored.index(best_epoch)
    return history, {
        instance_id: float(values[index])
        for instance_id, values in history["val_mrr_by_instance"].items()
    }


def _row(run_dir: Path) -> tuple[dict[str, str], tuple[str, str, bool]]:
    manifest = _read_manifest(run_dir)
    run_id = manifest["run_id"]
    epochs = manifest["config"]["epochs"]
    last = _read_checkpoint(run_dir, "last.pt")
    best = _read_checkpoint(run_dir, "best.pt")

    history = _history(last, run_dir, "last.pt")
    last_epoch = last.get("epoch")
    if not isinstance(last_epoch, int) or len(history["loss"]) != last_epoch + 1:
        raise _reject(
            run_dir, "last.pt history does not cover every checkpointed epoch"
        )

    scored_epochs, values, selection = _validation(history, run_dir, last_epoch)

    if last.get("stopped_early") is True:
        completion = _EARLY_STOPPED
    elif last_epoch + 1 >= epochs:
        completion = _EPOCH_BUDGET
    else:
        raise _reject(
            run_dir,
            f"interrupted after {last_epoch + 1} of {epochs} epochs with no early "
            f"stop; only completed runs have a milestone row",
        )

    best_epoch = manifest.get("best_epoch")
    if not isinstance(best_epoch, int) or not 0 <= best_epoch < epochs:
        raise _reject(
            run_dir,
            f"manifest best_epoch {best_epoch!r} is not an epoch of this run; the "
            f"run did not finish writing its selection",
        )
    if last.get("best_epoch") != best_epoch or best.get("best_epoch") != best_epoch:
        raise _reject(
            run_dir,
            f"selected epochs disagree: manifest {best_epoch}, last.pt "
            f"{last.get('best_epoch')}, best.pt {best.get('best_epoch')}",
        )

    if best_epoch not in scored_epochs:
        raise _reject(
            run_dir, f"selected best epoch {best_epoch} was never scored on F1-val"
        )
    selected = values[scored_epochs.index(best_epoch)]
    best_value = max(values)
    if selected != best_value:
        raise _reject(
            run_dir,
            f"selected best epoch {best_epoch} is not the history's best "
            f"({selected!r} < {best_value!r})",
        )

    prefix = run_id.split("-", 2)
    id_region = prefix[1].upper() if len(prefix) == 3 and prefix[0] == "f1" else None
    region = manifest.get("region")
    if region is not None and (not isinstance(region, str) or region.upper() != id_region):
        raise _reject(run_dir, f"manifest region {region!r} disagrees with run ID")
    if region is None:
        region = id_region
    if region not in TRAINING_ORDER:
        raise _reject(run_dir, f"manifest/run ID does not identify a known region: {region!r}")
    gate_seeds = load_region_config(config_path(region)).seeds
    return {
        "region": region,
        "gate_seed": str(manifest["seed"] in gate_seeds).lower(),
        "run_id": run_id,
        "seed": str(manifest["seed"]),
        "graph_digest": manifest["graph_digest"],
        "feature_digest": manifest["feature_digest"],
        "schema_digest": manifest["schema_digest"],
        "suite_digest": manifest["suite_digest"],
        "config_json": json.dumps(
            manifest["config"], sort_keys=True, separators=(",", ":")
        ),
        "best_epoch": str(best_epoch),
        "best_val_mrr": repr(float(selected)),
        "selection_instances": selection,
        "completion": completion,
        "device": manifest["device"],
        "precision": manifest["precision"],
    }, _backend(manifest)


def validate_completed_run(run_dir: Path) -> None:
    """Validate a completed run on CPU; raise ValueError on invalid artifacts.

    The manifest and checkpoints must agree on an F1-val-selected epoch, and
    the run must finish its epoch budget or stop early. Run state and locking
    remain the caller's responsibility.
    """
    _row(Path(run_dir))


def _backend(manifest: dict[str, Any]) -> tuple[str, str, bool]:
    """The arithmetic a run trained under; CUDA ordinals are ignored, no AdamW flag is unfused."""
    invocations = manifest.get("invocations")
    first = invocations[0] if isinstance(invocations, list) and invocations else {}
    runtime = first.get("runtime") if isinstance(first, dict) else None
    fused = isinstance(runtime, dict) and runtime.get("fused_adamw") is True
    return manifest["device"].split(":")[0], manifest["precision"], fused


def _describe(backend: tuple[str, str, bool]) -> str:
    device, precision, fused = backend
    return f"{device}/{precision}" + ("/fused-adamw" if fused else "")


def export_milestones(run_dirs: Sequence[Path], output: Path) -> None:
    """Write one deterministic row per completed run to ``output``.

    Rows are sorted by run ID and the file is replaced, never appended, so an
    operator can rerun the export over the same completed directories. A run
    directory that is interrupted, lacks an F1-val history, or whose manifest
    and checkpoints disagree on the selected epoch raises ``ValueError``
    instead of contributing a row; the previous output stays in place. So do
    runs trained on different backends, precisions or AdamW implementations,
    which one result must not mix.
    """
    rows: dict[str, dict[str, str]] = {}
    backends: set[tuple[str, str, bool]] = set()
    for path in run_dirs:
        run_dir = Path(path)
        row, backend = _row(run_dir)
        run_id = row["run_id"]
        if run_id in rows:
            raise _reject(run_dir, f"duplicate run ID {run_id!r}")
        rows[run_id] = row
        backends.add(backend)
    if len(backends) > 1:
        raise ValueError(
            "runs trained under different arithmetic cannot share one milestone "
            "export: " + ", ".join(map(_describe, sorted(backends)))
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            for run_id in sorted(rows):
                writer.writerow({key: rows[run_id][key] for key in COLUMNS})
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, output)


def export_experts(run_dirs: Sequence[Path], output: Path) -> None:
    """Export one five-seed, single-cohort registry per region."""
    rows: dict[str, dict[str, str]] = {}
    backends: dict[str, set[tuple[str, str, bool]]] = {}
    cohorts: dict[str, set[str]] = {}
    seeds: dict[str, list[int]] = {}
    for path in run_dirs:
        run_dir = Path(path)
        row, backend = _row(run_dir)
        run_id = row["run_id"]
        if run_id in rows:
            raise _reject(run_dir, f"duplicate run ID {run_id!r}")
        manifest = _read_manifest(run_dir)
        region = row["region"]
        try:
            cohort = cohort_digest(manifest)
        except ValueError as exc:
            raise _reject(run_dir, f"cannot establish cohort identity: {exc}") from exc
        rows[run_id] = {
            **row,
            "backend": _describe(backend),
            "cohort": cohort,
        }
        backends.setdefault(region, set()).add(backend)
        cohorts.setdefault(region, set()).add(cohort)
        seeds.setdefault(region, []).append(manifest["seed"])

    by_region: dict[str, list[str]] = {}
    for run_id, row in rows.items():
        by_region.setdefault(row["region"], []).append(run_id)
    for region, run_ids in by_region.items():
        if len(backends[region]) != 1:
            raise ValueError(
                f"{region}: runs trained under different arithmetic cannot share a registry: "
                + ", ".join(map(_describe, sorted(backends[region])))
            )
        if len(cohorts[region]) != 1:
            raise ValueError(f"{region}: runs mix scientific config, code or input digests")
        expected = list(load_region_config(config_path(region)).calibration_seeds)
        actual = seeds[region]
        if len(actual) != len(set(actual)):
            raise ValueError(f"{region}: duplicate calibration seed")
        if sorted(actual) != sorted(expected):
            raise ValueError(
                f"{region}: expected calibration seeds {expected}, got {sorted(actual)}"
            )

    if not rows:
        raise ValueError("expert registry needs completed runs")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPERT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for run_id in sorted(rows):
                writer.writerow(rows[run_id])
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, output)
