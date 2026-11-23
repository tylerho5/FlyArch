"""One TOML config per region: the only thing that differs between region runs.

``train/configs/<code>.toml`` holds the region code, its F1 (and, once frozen,
F2) edge threshold, the gate and calibration seeds, and a ``[train]`` table of
:class:`~train.train.TrainConfig` overrides. :func:`region_paths` is the
per-region data layout every pipeline stage reads and writes.
"""

from __future__ import annotations

import argparse
import os
import tomllib
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from src.gate import CALIBRATION_SEEDS
from src.regions import DEFAULT_THRESHOLD, REGIONS, RegionGraph
from src.union import is_union
from src.run_labels import clean_title, overrides

from .train import TrainConfig

__all__ = [
    "CONFIG_DIR",
    "TRAIN_FIELDS",
    "TRAINING_ORDER",
    "RegionConfig",
    "RegionPaths",
    "check_region",
    "config_from_args",
    "config_path",
    "load_region_config",
    "region_paths",
    "resolve_args",
]

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

#: M1 to M12: the planned training order.
TRAINING_ORDER = (
    "CX", "AL", "LH", "MB", "LX", "SNP", "INP", "VMNP", "VLNP", "PENP", "GNG", "OL",
)

_STORAGE_KEYS = frozenset({"run_dir", "suites_dir"})
#: TrainConfig fields a flag or a ``[train]`` table can set.
TRAIN_FIELDS = tuple(
    name for name in TrainConfig.__dataclass_fields__ if name not in _STORAGE_KEYS
)
_TRAIN_KEYS = frozenset(TRAIN_FIELDS)
_TOP_KEYS = frozenset({"region", "f1_threshold", "f2_threshold", "f3_threshold", "seeds",
                       "calibration_seeds", "train", "mask_exclude"})
_MASK_EXCLUDE_RULES = ("lamina_fraction",)


@dataclass(frozen=True)
class RegionConfig:
    region: str
    f1_threshold: int = DEFAULT_THRESHOLD
    f2_threshold: int | None = None
    f3_threshold: int | None = None
    seeds: tuple[int, ...] = (0, 1, 2)
    calibration_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    train: Mapping[str, Any] = field(default_factory=dict)
    mask_exclude: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.region not in REGIONS and not is_union(self.region):
            raise ValueError(f"unknown region {self.region!r}; known: {sorted(REGIONS)} and Phase 3 unions")
        for name in ("f1_threshold", "f2_threshold", "f3_threshold"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "calibration_seeds", tuple(self.calibration_seeds))
        if (len(self.calibration_seeds) != CALIBRATION_SEEDS
                or len(set(self.calibration_seeds)) != CALIBRATION_SEEDS):
            raise ValueError(
                f"calibration_seeds must hold {CALIBRATION_SEEDS} distinct seeds, "
                f"got {list(self.calibration_seeds)}"
            )
        missing = [seed for seed in self.seeds if seed not in self.calibration_seeds]
        if missing:
            raise ValueError(
                f"calibration_seeds must contain every gate seed; missing {missing}"
            )
        refused = sorted(set(self.train) & _STORAGE_KEYS)
        if refused:
            raise ValueError(
                f"[train] cannot set {refused}; pass them to train_config instead"
            )
        unknown = sorted(set(self.train) - _TRAIN_KEYS)
        if unknown:
            raise ValueError(f"[train] keys are not TrainConfig fields: {unknown}")
        if "epochs" not in self.train:
            raise ValueError("[train] must set epochs")
        if self.mask_exclude is not None:
            rule = self.mask_exclude
            if set(rule) != {"rule", "min_fraction"}:
                raise ValueError(f"[mask_exclude] needs exactly rule and min_fraction, got {sorted(rule)}")
            if rule["rule"] not in _MASK_EXCLUDE_RULES:
                raise ValueError(
                    f"[mask_exclude] unknown rule {rule['rule']!r}; known: {list(_MASK_EXCLUDE_RULES)}"
                )
            fraction = rule["min_fraction"]
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
                raise ValueError(f"[mask_exclude] min_fraction must be a number in (0, 1], got {fraction!r}")
            if self.region != "OL":
                raise ValueError("[mask_exclude] lamina_fraction applies to region OL only")
            object.__setattr__(self, "mask_exclude", MappingProxyType(dict(rule)))
        self.train_config(run_dir="data/runs")

    def train_config(
        self, *, run_dir: str | Path, suites_dir: str | Path = "data/suites", **overrides: Any
    ) -> TrainConfig:
        """The file's ``[train]`` values under ``overrides``; a ``None`` override keeps the file's."""
        values = dict(self.train)
        values.update({key: value for key, value in overrides.items() if value is not None})
        if "gap_regimes" in values:
            values["gap_regimes"] = tuple(values["gap_regimes"])
        return TrainConfig(**values, run_dir=run_dir, suites_dir=suites_dir)


def load_region_config(path: str | Path) -> RegionConfig:
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    unknown = sorted(set(data) - _TOP_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    if "region" not in data:
        raise ValueError(f"{path}: missing region")
    return RegionConfig(**data)


def config_path(code: str) -> Path:
    """The committed config for registry region ``code``."""
    return CONFIG_DIR / f"{code.lower()}.toml"


@dataclass(frozen=True)
class RegionPaths:
    """Where one region's artefacts live for one dataset."""

    processed: Path
    interim: Path
    suites: Path
    results: Path

    @property
    def suites_root(self) -> Path:
        """The ``suites_dir`` training reads committed negatives under."""
        return self.suites.parent.parent


def region_paths(code: str, dataset: str, data_root: str | Path | None = None) -> RegionPaths:
    """The per-region layout. ``data_root`` defaults to ``FLYARCH_DATA_ROOT``, then ``data``.

    Suite inputs live under ``interim``, outside ``processed``, so they never
    enter the queue's data fingerprint.
    """
    if code not in REGIONS and not is_union(code):
        raise ValueError(f"unknown region {code!r}; known: {sorted(REGIONS)} and Phase 3 unions")
    root = Path(data_root or os.environ.get("FLYARCH_DATA_ROOT") or "data")
    name = code.lower()
    return RegionPaths(
        processed=root / "processed" / dataset / name,
        interim=root / "interim" / dataset / name,
        suites=root / "suites" / dataset / name,
        results=Path("results") / name,
    )


def resolve_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Resolve every training value: the flag, then the file, then the default.

    ``parser`` produced ``args``, and reports a refusal through ``parser.error``,
    so a caller whose parser is not the single-run one owns how that reads.
    """
    region_config = None
    if args.config is not None:
        try:
            region_config = load_region_config(args.config)
        except (OSError, ValueError) as exc:
            parser.error(f"--config {args.config}: {exc}")
        paths = region_paths(region_config.region, "f1")
        if args.processed_dir is None:
            args.processed_dir = paths.processed
        if args.suite is None:
            # A union early-stops on its boundary suite, whose roots are its whole holdout.
            name = "phase3_boundary_f1_val" if is_union(region_config.region) else "f1_val"
            args.suite = paths.suites / f"{name}.json"
        if args.suites_dir is None:
            args.suites_dir = paths.suites_root
    file_values = dict(region_config.train) if region_config is not None else {}
    flags, baseline = {}, {}
    for name in TRAIN_FIELDS:
        default = TrainConfig.__dataclass_fields__[name].default
        has_baseline = name in file_values or default is not MISSING
        if default is MISSING:
            default = None
        resolved = file_values.get(name, default)
        if has_baseline:
            baseline[name] = resolved
        if getattr(args, name, None) is None:
            setattr(args, name, resolved)
        else:
            flags[name] = getattr(args, name)
    missing = [
        flag for flag, name in (
            ("--processed-dir", "processed_dir"), ("--suite", "suite"),
            ("--suites-dir", "suites_dir"), ("--epochs", "epochs"),
        )
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(f"the following arguments are required without --config: {', '.join(missing)}")
    if hasattr(args, "seeds") and args.seeds is None:
        if region_config is None:
            parser.error("--seeds is required without --config")
        args.seeds = list(region_config.calibration_seeds)
    args.region_config = region_config
    args.title = clean_title(args.title)
    args.overrides = overrides(flags, baseline)
    return args


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """The run's config: the training values ``resolve_args`` set, under its paths."""
    return TrainConfig(
        **{name: getattr(args, name) for name in TRAIN_FIELDS},
        run_dir=args.runs_dir,
        suites_dir=args.suites_dir,
    )


def check_region(args: argparse.Namespace, graph: RegionGraph) -> None:
    """Refuse a graph whose region or threshold disagrees with ``--config``."""
    config = args.region_config
    if config is None:
        return
    if (config.region, config.f1_threshold) != (graph.region, graph.threshold):
        raise ValueError(
            f"{args.config} is for {config.region} at threshold {config.f1_threshold}, "
            f"but {args.processed_dir} holds {graph.region} at threshold {graph.threshold}"
        )
