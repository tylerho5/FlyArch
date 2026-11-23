"""M1's per-epoch gap-simulation training loop.

Each epoch hides a fresh random 5% of the region's neurons feature-only,
supervises the removed edges against freshly corrupted negatives, and adds the
auxiliary ``log1p`` synapse-count loss. This is
the "load-bearing trick" of Phase 1 (Phase 1 design spec, section 7, not
published): training under the same feature-only mask the gate scores under.

A gap mask is stamped with the ``gap`` tag and a draw-dependent seed, so it can
never collide with a committed suite instance or reuse one of its digest keys.
Looking at a validation suite, fitting trains on the graph minus that suite's
masks: every edge touching a neuron any F1-val instance hides is pruned, and
gap masks are drawn only from the neurons that remain. The pinned F2 sets are
never touched here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import platform
import re
import signal
import socket
import subprocess
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePath
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluate import (
    CommittedNegatives,
    ScoringTask,
    build_scoring_task,
    candidate_pairs,
    evaluated_rows,
    gated_mrr,
)
from src.features import FeatureSchema, FeatureTable
from src.masks import (
    MaskInstance,
    MaskedView,
    VisibleView,
    apply_mask,
    pool_mask,
    spatial_pool_mask,
    typed_pool_mask,
)
from src.metrics import mrr as rows_mrr
from src.negatives import DEFAULT_SPEC, pack_edges
from src.regions import REGIONS, RegionGraph
from src.union import is_union
from src.suites import Suite, suite_negatives, view_negatives
from src.run_labels import RunLabels, git_branch, git_subject
from src.run_control import (
    RunLock,
    RunStopped,
    StopSignals,
    clean_reason,
    clear_request,
    read_request,
    read_state,
    write_state,
)

from .baselines import _centroids, _type_ids
from .model import AGGREGATIONS, EDGE_WEIGHTS, EXTRAS_KEYS, M1, MODEL_KNOBS, PreparedPairs
from .run_keys import TEST_RUN_SUFFIX

#: The early-stopping suite each graph kind may use: a region's F1-val, or a union's
#: boundary F1-val (Phase 3 plan 06, whose roots are the union model's whole holdout).
_VAL_SUITES = {"region": "f1_val", "union": "phase3_boundary_f1_val"}

__all__ = [
    "DEVICES",
    "LR_SCHEDULES",
    "MASKED_INNERVATION",
    "SAMPLING_MODES",
    "TrainConfig",
    "TrainResult",
    "ValidationInputs",
    "fit",
    "simulate_gaps",
    "validate_feature_inputs",
    "validate_m1_inputs",
    "validation_mrr",
]

#: Per-epoch probability that a node's ``cell_type`` is replaced by the OOV
#: sentinel during training. The spec names type-feature dropout as the
#: mitigation for F2's unmatched neurons (73% of OL, 1-10% of each other
#: region on v888): without it the encoder never sees the OOV value during F1
#: training.
_TYPE_DROPOUT_P = 0.1

#: F1-val regimes scored and logged but left out of checkpoint selection. T's
#: masked types are held out of F1 training, so it reads OOV types only and its
#: 3 neurons swing the unweighted mean more than the four larger instances.
_REPORT_ONLY_REGIMES = frozenset({"T"})

#: Epochs between train-side MRR probes, or ``eval_every`` if that is longer.
#: The probe is a full-graph fp32 forward pass that only feeds logging.
_TRAIN_MRR_EVERY = 10

#: Suites with at most this many candidate pairs keep them device-resident.
_RESIDENT_PAIRS = 1 << 23


@dataclass(frozen=True)
class _Schedule:
    """A learning-rate schedule after warmup.

    ``multiplier`` maps the fraction of post-warmup epochs already run to a
    factor on ``lr``. A schedule that decays sets the run's length, so
    ``early_stop`` is False and the run trains all ``epochs``.
    """

    multiplier: Callable[[float], float]
    early_stop: bool


#: A config names one entry by ``lr_schedule``.
_LR_SCHEDULES: dict[str, _Schedule] = {
    "constant": _Schedule(lambda progress: 1.0, early_stop=True),
    "cosine": _Schedule(
        lambda progress: 0.5 * (1.0 + math.cos(math.pi * progress)),
        early_stop=False,
    ),
}

#: The schedules ``TrainConfig.lr_schedule`` accepts.
LR_SCHEDULES = tuple(_LR_SCHEDULES)
SAMPLING_MODES = ("full", "neighbor")
#: What ``TrainConfig.masked_innervation`` accepts: the masked roots' innervation regime.
MASKED_INNERVATION = ("uniform", "real")

#: One epoch's gap, drawn from the training rng.
_GapDraw = Callable[[np.random.Generator], MaskInstance]


def _random_gaps(
    graph: RegionGraph, table: FeatureTable, pool: np.ndarray, fraction: float
) -> _GapDraw:
    return lambda rng: pool_mask(graph, rng, fraction=fraction, pool=pool)


def _typed_gaps(
    graph: RegionGraph, table: FeatureTable, pool: np.ndarray, fraction: float
) -> _GapDraw:
    types = _type_ids(table, graph.n_nodes).astype(np.int64)
    if not (types[pool] > 0).any():
        raise ValueError(
            "a T gap needs cell types, and no trainable neuron has one; build the "
            "features with type annotations or drop 'T' from gap_regimes"
        )
    return lambda rng: typed_pool_mask(
        graph, rng, types=types, fraction=fraction, pool=pool
    )


def _spatial_gaps(
    graph: RegionGraph, table: FeatureTable, pool: np.ndarray, fraction: float
) -> _GapDraw:
    # A neuron without synapse geometry sits at the z-scored origin, mid-region.
    located = table.geometry_present
    if located is None:
        raise ValueError(
            "an S gap needs to know which neurons have synapse geometry; rebuild "
            "the features with scripts/build_features.py or drop 'S' from gap_regimes"
        )
    n_located = int(located[pool].sum())
    if n_located < max(round(fraction * pool.size), 1):
        raise ValueError(
            f"an S gap hides {fraction} of the {pool.size} trainable neurons, and "
            f"only {n_located} have synapse geometry; build the features with "
            "synapse statistics or drop 'S' from gap_regimes"
        )
    centroids = _centroids(table)
    return lambda rng: spatial_pool_mask(
        graph, rng, centroids=centroids, fraction=fraction, pool=pool, located=located
    )


#: Per-epoch gaps by regime, each sized to ``gap_fraction`` of the trainable pool.
_GAP_REGIMES: dict[
    str, Callable[[RegionGraph, FeatureTable, np.ndarray, float], _GapDraw]
] = {"R": _random_gaps, "T": _typed_gaps, "S": _spatial_gaps}

#: The repository root, resolved from this file. Paths in the run identity and
#: the git probe resolve against it so both are independent of the process
#: working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The code whose bytes are part of the run identity: the training loop and
#: every ``src`` module in its import closure that shapes a run. Changes to the
#: mask/view semantics, the negative sampler, suite resolution, scoring, the
#: MRR statistic, feature building, or graph construction must open a new run,
#: not append epochs to a checkpoint trained by the old revision.
_CODE_FILES = (
    "train/train.py",
    "train/model.py",
    "train/baselines.py",
    "train/__init__.py",
    "src/masks.py",
    "src/negatives.py",
    "src/suites.py",
    "src/evaluate.py",
    "src/metrics.py",
    "src/features.py",
    "src/regions.py",
)


#: The only field that names a storage location rather than an input to the run.
#: It is excluded from the run identity and from the manifest's ``config``: where
#: artefacts are written does not shape the experiment, so a run moved to
#: another ``FLYARCH_DATA_ROOT`` resumes instead of restarting from zero.
#: ``suites_dir`` is deliberately not here -- it selects which committed
#: negatives the validation path scores against, which is a scientific input.
_STORAGE_FIELDS = ("run_dir",)


@dataclass
class TrainConfig:
    """Training hyperparameters. ``epochs`` is required; the rest are pinned defaults.

    Every field except ``run_dir`` is hashed into the run identity, so raising
    ``epochs`` or ``patience`` opens a new run rather than extending a finished
    one. A preempted run with the budget unchanged resumes and continues.

    ``patience`` is 100 because, at one step per epoch, F1-val MRR peaked near
    epoch 25 and dipped before it climbed; at 50 a run could stop inside the dip
    on a near-chance checkpoint. It counts epochs, so at 8 steps per epoch it
    spans 800 steps.

    ``eval_every`` spaces the F1-val scoring. One F1-val pass scores about six
    times the pairs of a training step, so scoring every epoch spent most of a
    run's time there. The last epoch is always scored, and patience is only
    checked on a scored epoch.

    The learning rate ramps linearly over ``warmup_epochs``, then follows the
    ``lr_schedule`` entry over the remaining ``epochs``. Patience counts from
    the end of warmup at the earliest, and a decaying schedule never stops
    early. The default is ``cosine``, which runs the full ``epochs`` budget;
    ``constant`` keeps early stopping. Either way F1-val selects the best
    checkpoint.

    Each epoch's gap comes from one of ``gap_regimes``, drawn uniformly, so
    listing a regime twice doubles its share. With one regime no draw is made,
    and an R-only run at one step per epoch keeps the gap masks it had before
    regimes existed.

    ``pair_batch`` caps the ranking rows a step trains on: that many of the
    gap's hidden edges, drawn with the training rng, each with its K negatives.
    ``None`` keeps every row. ``sampling`` is ``"full"`` (the whole visible
    graph per step); ``"neighbor"`` is reserved and the config refuses it.
    """

    epochs: int
    hidden: int = 128
    sibling_features: bool = True
    lr: float = 1e-3
    encoder_dropout: float = 0.1
    decoder_dropout: float = 0.1
    steps_per_epoch: int = 8
    patience: int = 100
    gap_fraction: float = 0.05
    gap_regimes: tuple[str, ...] = ("R",)
    aux_weight: float = 0.3
    eval_every: int = 5
    warmup_epochs: int = 0
    lr_schedule: str = "cosine"
    layers: int = 2
    layer_norm: bool = False
    residual: bool = False
    branch_norm: bool = False
    decoder_product_rank: int = 0
    # Opt-in model switches; see ``train.model.MODEL_KNOBS``.
    type_pair_features: bool = False
    pair_geometry: bool = False
    sibling_pooling: bool = False
    aggregation: str = "sum"
    edge_weight: str = "fraction"
    nt_edges: bool = False
    type_pair_shares: bool = False
    annotation_groups: bool = False
    type_profile_rank: int = 0
    nt_pair_features: bool = False
    # Opt-in training switches. ``feature_noise`` is the std of Gaussian noise
    # on the z-scored centroid/extent columns per step. ``type_negatives`` adds
    # that many decoys per row of the true partner's cell type.
    # ``train_negatives`` > 0 draws that many sampler negatives per row instead
    # of the pinned K (F1-val keeps K). ``holdout_probe`` keeps the train-MRR
    # probe's neurons out of training, so train MRR is a held-out estimate.
    # ``ema_decay`` > 0 validates and checkpoints an exponential moving average
    # of the weights.
    feature_noise: float = 0.0
    type_negatives: int = 0
    train_negatives: int = 0
    holdout_probe: bool = False
    ema_decay: float = 0.0
    weight_decay: float = 0.01
    grad_clip: float = 0.0
    sampling: str = "full"
    pair_batch: int | None = None
    # ``uniform`` sets every masked root's innervation to the uniform share, in training
    # and validation, so no step reads the statistics of a hidden edge.
    # ``real`` is the pre-2026-11-07 regime, kept to reproduce those runs.
    masked_innervation: str = "uniform"
    run_dir: str | Path = "data/runs"
    suites_dir: str | Path = "data/suites"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.masked_innervation not in MASKED_INNERVATION:
            raise ValueError(
                f"masked_innervation must be one of {MASKED_INNERVATION}, "
                f"got {self.masked_innervation!r}"
            )
        for name in ("encoder_dropout", "decoder_dropout"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1), got {getattr(self, name)}")
        if self.steps_per_epoch < 1:
            raise ValueError(f"steps_per_epoch must be >= 1, got {self.steps_per_epoch}")
        if self.eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {self.eval_every}")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError(
                f"warmup_epochs must be >= 0 and below epochs ({self.epochs}), "
                f"got {self.warmup_epochs}"
            )
        self.gap_regimes = tuple(self.gap_regimes)
        unknown = [name for name in self.gap_regimes if name not in _GAP_REGIMES]
        if not self.gap_regimes or unknown:
            choices = ", ".join(repr(name) for name in _GAP_REGIMES)
            raise ValueError(
                f"gap_regimes must be a non-empty selection of {choices}, "
                f"got {self.gap_regimes!r}"
            )
        if self.lr_schedule not in _LR_SCHEDULES:
            choices = ", ".join(repr(name) for name in _LR_SCHEDULES)
            raise ValueError(
                f"lr_schedule must be one of {choices}, got {self.lr_schedule!r}"
            )
        if self.layers < 1:
            raise ValueError(f"layers must be >= 1, got {self.layers}")
        if self.sampling not in SAMPLING_MODES:
            choices = ", ".join(repr(name) for name in SAMPLING_MODES)
            raise ValueError(f"sampling must be one of {choices}, got {self.sampling!r}")
        if self.sampling != "full":
            raise ValueError(
                f"sampling={self.sampling!r} is not implemented; only 'full' trains"
            )
        if self.pair_batch is not None and self.pair_batch < 1:
            raise ValueError(f"pair_batch must be >= 1 or None, got {self.pair_batch}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {self.aggregation!r}")
        if self.edge_weight not in EDGE_WEIGHTS:
            raise ValueError(f"edge_weight must be one of {EDGE_WEIGHTS}, got {self.edge_weight!r}")
        if self.feature_noise < 0 or self.type_negatives < 0 or self.train_negatives < 0:
            raise ValueError("feature_noise, type_negatives and train_negatives must be >= 0")
        if self.type_profile_rank < 0:
            raise ValueError(f"type_profile_rank must be >= 0, got {self.type_profile_rank}")
        if not 0 <= self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")


@dataclass(frozen=True)
class ValidationInputs:
    instances: tuple[MaskInstance, ...]
    tasks: list[tuple[MaskedView, ScoringTask]]


@dataclass
class TrainResult:
    """A finished (or resumed) run: weights, curves, and run provenance.

    ``resumed`` is True when the call continued an existing checkpoint,
    including one whose run had already stopped early and trained no further
    epochs.
    """

    state_dict: dict
    history: dict
    best_epoch: int
    manifest: dict
    resumed: bool


def simulate_gaps(
    graph: RegionGraph,
    rng: np.random.Generator,
    fraction: float = 0.05,
    pool: np.ndarray | None = None,
) -> MaskInstance:
    """Draw a fresh random ``fraction`` of the trainable neurons to hide.

    Delegates to :func:`src.masks.pool_mask`, which owns the draw rule M1's
    per-epoch gaps and the baseline fit views share. The identity carries the
    training-only ``gap`` tag, outside the pinned suite tag set, so a gap
    instance is never mistaken for a scored suite instance and never reuses a
    committed digest key.
    """
    return pool_mask(graph, rng, fraction=fraction, pool=pool)


def _table_schema(
    table: FeatureTable, schema: FeatureSchema | None = None
) -> SimpleNamespace:
    """What ``M1`` reads from a schema: input width, column names, type-embedding size.

    With a ``FeatureSchema`` the type-embedding width is the schema's true type
    count, so a checkpoint's type embedding fits ``M1(schema)`` even when the
    table omits the highest-numbered type. Without one, the fine-type
    identity is inferred from the table's ``cell_type`` column -- the same
    column ``build_features`` fills from ``FeatureSchema.type_id`` -- and that
    inference only guarantees checkpoint compatibility when the table happens to
    contain every type.
    """
    if schema is not None:
        # Ids run 1..T with 0 as the OOV sentinel, exactly as
        # FeatureSchema.type_id assigns them, so the type embedding is the
        # same width as a real schema's and a checkpoint built here loads into
        # one.
        return SimpleNamespace(
            n_features=schema.n_features,
            column_names=schema.column_names,
            type_vocab={index: index for index in range(1, len(schema.type_vocab) + 1)},
        )
    names = list(map(str, table.columns))
    if "cell_type" not in names:
        raise ValueError("the feature table has no cell_type column for M1 to embed")
    column = np.asarray(table.values[:, names.index("cell_type")])
    n_types = int(column.max()) + 1 if column.size else 1
    return SimpleNamespace(
        n_features=table.values.shape[1],
        column_names=tuple(map(str, table.columns)),
        type_vocab={index: index for index in range(1, n_types)},
    )


def extras_read(config: TrainConfig | dict) -> tuple[str, ...]:
    """The ``extras.npz`` arrays ``config``'s switches read; a dict is a manifest or TOML config."""
    on = config.get if isinstance(config, dict) else lambda key, default: getattr(config, key, default)
    return tuple(sorted(key for switch, keys in EXTRAS_KEYS.items() if on(switch, False) for key in keys))


def _feature_digest(table: FeatureTable, *, extras: tuple[str, ...]) -> str:
    """Blake2b over the feature values, column names, geometry presence flags and read extras.

    This covers the table the model actually consumes; the schema is hashed
    separately into the run identity, so a change it makes outside the table
    (a widened ``type_vocab``, which sizes the type embedding) opens a new run
    instead of colliding with a checkpoint the model can no longer load.
    ``extras`` names the arrays the config reads (``extras_read``), so an
    unread ``extras.npz`` beside the graph leaves the digest unchanged.
    """
    hasher = hashlib.blake2b()
    hasher.update(np.ascontiguousarray(table.values, dtype=np.float64).tobytes())
    hasher.update(json.dumps(list(map(str, table.columns))).encode("utf-8"))
    if table.geometry_present is not None:
        hasher.update(np.ascontiguousarray(table.geometry_present, dtype=bool).tobytes())
    for name in sorted(extras):
        if table.extras is None or name not in table.extras:
            raise ValueError(f"the config reads extras array {name!r}, which the feature table lacks")
        hasher.update(name.encode("utf-8"))
        hasher.update(np.ascontiguousarray(table.extras[name]).tobytes())
    return hasher.hexdigest()


def _scientific_config(config: TrainConfig) -> dict:
    """The scientific inputs, without the ``run_dir`` storage location, as JSON types."""

    def plain(value):
        if isinstance(value, PurePath):
            return value.as_posix()
        return list(value) if isinstance(value, tuple) else value

    return {
        key: plain(value)
        for key, value in asdict(config).items()
        if key not in _STORAGE_FIELDS
    }


def _code_digest() -> str:
    """Blake2b over the bytes of the training code.

    A run identity without the code lets a bug fix resume the old checkpoint
    and append new-code epochs to old-code ones, returning a curve spanning two
    revisions. ``git rev-parse HEAD`` does not cover that: uncommitted edits
    leave the commit unchanged while the training code has moved. Paths resolve
    against the repo root so the digest is the same from any working directory.
    """
    root = _REPO_ROOT
    hasher = hashlib.blake2b()
    for name in _CODE_FILES:
        hasher.update(name.encode("utf-8"))
        hasher.update((root / name).read_bytes())
    return hasher.hexdigest()


_PRECISION = "fp32"


def _tf32(flags: Any) -> bool:
    """Whether a backend's fp32 ops may run in TF32, under either PyTorch flag API.

    ``flags`` is ``torch.backends.cuda.matmul`` or ``torch.backends.cudnn``.
    Newer PyTorch raises on the legacy ``allow_tf32`` read once the
    ``fp32_precision`` flag has been set, so that flag answers instead.
    """
    try:
        return bool(flags.allow_tf32)
    except RuntimeError:
        return getattr(flags, "fp32_precision", "tf32") == "tf32"


@dataclass(frozen=True)
class _Backend:
    """How training runs on one device.

    ``tensorboard_hint`` is ``None`` when curves are optional. ``empty_cache``
    runs after every epoch; GPUs need it because the allocator cache otherwise
    grows with each epoch's differently sized tensors, until an MPS process
    swaps or an 8 GB CUDA card spills into host memory. ``fused_adamw`` steps
    the optimizer in one fused kernel instead of one launch per parameter.
    ``prefetch`` draws the next epoch's inputs on a worker during the device step.
    ``deterministic`` turns on PyTorch's deterministic algorithms, without which
    CUDA's atomic ``index_add_`` makes two runs of one seed diverge.
    ``rng`` gets and sets the device generator that dropout draws from, which
    a checkpoint carries alongside the CPU one.
    """

    checks: tuple[tuple[Callable[[], bool], str], ...]
    tensorboard_hint: str | None
    empty_cache: Callable[[], None] | None = None
    fused_adamw: bool = False
    prefetch: bool = False
    deterministic: bool = False
    rng: tuple[Callable[[], torch.Tensor], Callable[[torch.Tensor], None]] | None = None


_BACKENDS: dict[str, _Backend] = {
    "cpu": _Backend(
        checks=(),
        tensorboard_hint=None,
    ),
    "cuda": _Backend(
        checks=(
            (
                lambda: torch.cuda.is_available(),
                "torch.cuda.is_available() is False; this build has no visible "
                "GPU, use the 'cpu' default",
            ),
            (
                lambda: not _tf32(torch.backends.cuda.matmul),
                "TF32 matmul is enabled, so CUDA would not train in the fp32 "
                "every device records; call "
                "torch.set_float32_matmul_precision('highest') before fit",
            ),
        ),
        tensorboard_hint=(
            "install the pinned training dependencies from "
            "requirements/training.lock, or run with device='cpu'"
        ),
        empty_cache=torch.cuda.empty_cache,
        fused_adamw=True,
        prefetch=True,
        deterministic=True,
        rng=(torch.cuda.get_rng_state, torch.cuda.set_rng_state),
    ),
    "mps": _Backend(
        checks=(
            (
                lambda: torch.backends.mps.is_available(),
                "torch.backends.mps.is_available() is False; use an "
                "MPS-capable Mac or explicitly select device='cpu'",
            ),
        ),
        tensorboard_hint=(
            "install the 'tensorboard' package in the active Python environment"
        ),
        empty_cache=torch.mps.empty_cache,
        fused_adamw=True,
        prefetch=True,
        rng=(torch.mps.get_rng_state, torch.mps.set_rng_state),
    ),
}

#: The devices ``fit`` accepts.
DEVICES = tuple(_BACKENDS)


def _resolve_device(device: str) -> str:
    """Validate the requested device before a run directory is created."""
    backend = _BACKENDS.get(device)
    if backend is None:
        choices = ", ".join(repr(name) for name in _BACKENDS)
        raise ValueError(f"device must be one of {choices}, got {device!r}")
    for check, problem in backend.checks:
        if not check():
            raise ValueError(f"device={device!r} was requested, but {problem}")
    return device


def _use_deterministic_algorithms() -> None:
    """Make every kernel deterministic for the rest of the process.

    cuBLAS fixes its workspace when PyTorch first uses it, so this must run
    before the process's first CUDA matmul.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def _optimizer(model: M1, config: TrainConfig, device: str) -> torch.optim.AdamW:
    fused = {"fused": True} if _BACKENDS[device].fused_adamw else {}
    return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, **fused)


def _summary_writer_class(device: str):
    """The TensorBoard ``SummaryWriter``, or ``None`` when events are unavailable.

    TensorBoard is pinned in the training image (``requirements/training.lock``)
    but absent from a CPU-only checkout, so the import stays lazy and the
    tiny-graph tests keep running. A GPU run's curves are part of its result:
    without the dependency it fails before creating a run directory rather than
    training with no events.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        hint = _BACKENDS[device].tensorboard_hint
        if hint is not None:
            raise RuntimeError(
                f"device={device!r} requires TensorBoard for its training "
                f"curves; {hint}"
            ) from None
        return None
    return SummaryWriter


def _run_id(
    graph: RegionGraph,
    val_suite: Suite | None,
    config: TrainConfig,
    seed: int,
    feature_digest: str,
    schema: FeatureSchema | None = None,
    device: str = "cpu",
) -> str:
    """Deterministic in every input that shapes the run: graph, features, schema, suite, config, seed, code.

    Hashing config and graph alone would let a run with changed features, a
    changed validation suite, a changed schema, or changed training code reopen
    the old directory and continue from a ``last.pt`` trained on incompatible
    inputs.
    """
    payload = json.dumps(_scientific_config(config), sort_keys=True, default=str).encode("utf-8")
    hasher = hashlib.blake2b(payload)
    hasher.update(graph.digest().encode("utf-8"))
    hasher.update(feature_digest.encode("utf-8"))
    hasher.update((schema.digest() if schema is not None else "none").encode("utf-8"))
    suite_digest = val_suite.digest() if val_suite is not None else "none"
    hasher.update(suite_digest.encode("utf-8"))
    hasher.update(_code_digest().encode("utf-8"))
    # Backends and CPU architectures accumulate in different orders, so a run
    # never resumes a checkpoint trained under other arithmetic.
    hasher.update(f"{device}/{platform.machine()}".encode("utf-8"))
    return f"{graph.dataset}-{graph.region.lower()}-s{seed:03d}-{hasher.hexdigest()[:8]}"


def _git_commit() -> str | None:
    """This checkout's HEAD, regardless of the process working directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=_REPO_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


#: Full Git object IDs and exact image IDs a launcher may supply. The training
#: image probes neither for itself; ``scripts/training_env.py`` collects both on
#: the host under the GPU lock.
_COMMIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_IMAGE_IDENTIFIER_RE = re.compile(r"sha256:[0-9a-f]{64}")


def _utc_now() -> str:
    """Now, in the environment records' stamp format, always microsecond-wide.

    Fixed width keeps lexical order equal to time order for a reader sorting
    the append-only invocations.
    """
    stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    return stamp.replace("+00:00", "Z")


def _gpu_name(device: str) -> str | None:
    """The GPU this invocation ran on, or ``None`` off CUDA.

    A CPU invocation records no GPU even on a host with a visible one: nothing
    in the run touched it, so its name would describe hardware the epochs did
    not use. PyTorch reports no model name for an MPS device.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(device)


def _recorded_device(device: torch.device) -> str:
    """CUDA keeps its ordinal (``cuda:0``); a Mac has one Metal device, recorded as ``mps``."""
    return str(device) if device.type == "cuda" else device.type


def _source_provenance(environ: dict | None = None) -> dict[str, Any]:
    """Launcher-supplied Git and image identity, else explicit unavailability.

    The container mounts the host checkout, so a probe inside it reads that
    checkout at probe time, not the launch the launcher pinned under the GPU
    lock; ``FLYARCH_GIT_STATE`` and ``FLYARCH_IMAGE_IDENTIFIER`` carry the pinned
    values. Direct runs have neither and fall back to probing this process's
    own checkout for the commit, recording what neither source supplied rather
    than inventing it. Compose's metadata-policy variable
    ``FLYARCH_REQUIRE_METADATA``, present as ``"0"`` in the shared anchor and
    ``"1"`` on the trainer, suppresses the probe whenever it is set, so Git is
    recorded unavailable there. This mirrors the source probe in
    ``docker/record_training_environment.py`` without its required-metadata
    failure, which stays the launcher's contract.
    """
    env = os.environ if environ is None else environ
    git: dict[str, Any] = {"commit": None, "dirty": None}
    text = env.get("FLYARCH_GIT_STATE", "")
    if text:
        try:
            state = json.loads(text)
        except (TypeError, ValueError):
            state = None
        if (
            isinstance(state, dict)
            and set(state) == {"commit", "dirty"}
            and isinstance(state["commit"], str)
            and _COMMIT_RE.fullmatch(state["commit"]) is not None
            and type(state["dirty"]) is bool
        ):
            git = {"commit": state["commit"], "dirty": state["dirty"]}
    unavailable = []
    if git["commit"] is None:
        commit = _git_commit() if "FLYARCH_REQUIRE_METADATA" not in env else None
        if commit is None:
            unavailable.append("git")
        else:
            git = {"commit": commit, "dirty": None}
    image = env.get("FLYARCH_IMAGE_IDENTIFIER") or None
    if image is None or _IMAGE_IDENTIFIER_RE.fullmatch(image) is None:
        image = None
        unavailable.append("image_identifier")
    return {
        "git": git,
        "image_identifier": image,
        "metadata_unavailable": unavailable,
    }


def _invocation(
    device: str,
    precision: str,
    *,
    resumed: bool,
    fused_adamw: bool = False,
    title: str | None = None,
) -> dict[str, Any]:
    """One process's environment, as the invocation record stores it."""
    source = _source_provenance()
    commit = source["git"]["commit"]
    git = {
        **source["git"],
        "branch": git_branch(_REPO_ROOT, os.environ),
        "subject": git_subject(_REPO_ROOT, commit, os.environ),
    }
    return {
        "timestamp_utc": _utc_now(),
        "resumed": bool(resumed),
        "device": device,
        "precision": precision,
        "title": title,
        "git": git,
        "image_identifier": source["image_identifier"],
        "metadata_unavailable": source["metadata_unavailable"],
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": _gpu_name(device),
            "tf32": _tf32(torch.backends.cuda.matmul),
            "cudnn_tf32": _tf32(torch.backends.cudnn),
            "fused_adamw": bool(fused_adamw),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
        },
    }


def _first_invocation_fields(record: dict[str, Any]) -> dict[str, Any]:
    """The top-level mirror of the run's first invocation.

    A reader that wants one stack for the run should not have to walk the
    append-only history; the history stays the source of truth for every
    invocation.
    """
    return {
        "git_commit": record["git"]["commit"],
        "image_identifier": record["image_identifier"],
        "metadata_unavailable": list(record["metadata_unavailable"]),
        "torch_version": record["runtime"]["torch"],
        "cuda_version": record["runtime"]["cuda"],
        "gpu_name": record["runtime"]["gpu"],
        "tf32": record["runtime"]["tf32"],
        "device": record["device"],
        "precision": record["precision"],
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    """The previous manifest, or ``None`` when there is not a usable one.

    An unreadable manifest starts a fresh record: losing the run because its
    provenance file cannot be parsed trades a recoverable diagnosis for an
    unrecoverable run. The atomic write keeps this path unreachable for
    manifests this code wrote.
    """
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _invocation_history(
    manifest_path: Path, invocation: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The run's first-invocation fields and its append-only invocation list.

    The existing manifest is read rather than rebuilt, so a resume appends to
    the provenance the run was created with instead of relabelling earlier
    epochs with the resuming process's Git, image, or runtime.
    """
    existing = _read_manifest(manifest_path)
    records: list[dict[str, Any]] = []
    if existing is not None and isinstance(existing.get("invocations"), list):
        records = [
            record for record in existing["invocations"] if isinstance(record, dict)
        ]
    records.append(dict(invocation))
    return _first_invocation_fields(records[0]), records


def _manifest(
    graph: RegionGraph,
    config: TrainConfig,
    seed: int,
    feature_digest: str,
    run_id: str,
    best_epoch: int | None,
    schema: FeatureSchema | None = None,
    val_suite: Suite | None = None,
    *,
    best_val_mrr: float | None = None,
    provenance: dict[str, Any],
    invocations: list[dict[str, Any]],
) -> dict[str, Any]:
    """The run's manifest: input identity, first-invocation provenance, history.

    ``provenance`` mirrors the first invocation so the top level keeps
    describing the stack the first epochs ran under; ``invocations`` is
    append-only and carries each later process's own environment. Only
    ``best_epoch`` and ``best_val_mrr`` move between the two writes of one
    invocation. ``best_val_mrr`` is the F1-val selection statistic at
    ``best_epoch``, null when nothing was scored.
    ``suite_digest`` names the F1-val suite that selected the run, so the
    milestone export reports it without re-reading the suite file.
    """
    return {
        "seed": int(seed),
        "region": graph.region,
        "config": _scientific_config(config),
        "graph_digest": graph.digest(),
        "feature_digest": feature_digest,
        "schema_digest": schema.digest() if schema is not None else "none",
        "suite_digest": val_suite.digest() if val_suite is not None else "none",
        "code_digest": _code_digest(),
        **provenance,
        "invocations": [dict(record) for record in invocations],
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_val_mrr": best_val_mrr,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest through a temporary file and ``os.replace``.

    The manifest is rewritten more often than a run is created, and a
    preemption during a plain write leaves a truncated provenance record that
    no future reader can parse. Resume never reads it, so an interrupted write
    must not damage a previously complete file.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _save_checkpoint(
    path: Path,
    model: M1,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: dict,
    best_epoch: int,
    rng_state: dict,
    stopped_early: bool = False,
    device: str = "cpu",
    ema: dict[str, torch.Tensor] | None = None,
) -> None:
    """Checkpoint everything a resumed run must continue from.

    ``ema`` (the weight average when ``ema_decay`` is on) is saved alongside.

    The generators are part of that: restoring the weights but re-seeding the
    mask draw would make a preempted run train on a different 5% per epoch than
    the uninterrupted run it is meant to reproduce. ``rng_state`` is the
    training generator's state before any draw for the next epoch. The torch
    generators (CPU, and ``device``'s when it has one) are saved as they stand,
    since dropout draws from the device's. ``stopped_early`` records that the
    loop already hit patience, so a re-invocation with identical inputs returns
    the recorded result instead of training more epochs.

    The write goes through a temporary file and ``os.replace``, so a
    preemption mid-pickle leaves the previous checkpoint in place rather than
    a truncated ``last.pt`` that breaks every later resume of the run.
    """
    tmp = path.with_name(path.name + ".tmp")
    device_rng = _BACKENDS[device].rng
    try:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": int(epoch),
                "history": history,
                "best_epoch": int(best_epoch),
                "stopped_early": bool(stopped_early),
                "rng": rng_state,
                "torch_rng": torch.get_rng_state(),
                "device_rng": device_rng[0]() if device_rng is not None else None,
                **({"ema": ema} if ema is not None else {}),
            },
            tmp,
        )
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _training_negatives(
    view: MaskedView,
    true_edges: set[int] | None = None,
    positives: np.ndarray | None = None,
    k: int = 0,
) -> np.ndarray:
    """All K pinned-sampler negatives per removed edge, shape ``(P, K, 2)``.

    ``k`` > 0 draws that many per edge from the same sampler instead of the
    pinned K, for training only.

    Training must rank against the same negative distribution the gate scores
    with (registered: degree-corrected, and every decoy keeping a
    masked endpoint), so this delegates to the sampler the frozen suites are
    built from instead of corrupting endpoints uniformly. The sample is keyed
    on the mask's instance id, so each epoch's fresh mask draws a fresh set.

    ``true_edges`` defaults to the view's own graph. Fitting passes the
    *original* graph's edge set instead, because the training copy has the
    F1-val edges pruned: without them the sampler would count a removed
    validation edge as a non-edge and could hand one back as a negative.

    ``positives`` restricts the sample to those hidden edges; the same rows of
    the same mask draw the same negatives.
    """
    spec = replace(DEFAULT_SPEC, k=k) if k else DEFAULT_SPEC
    sample = view_negatives(view, spec=spec, true_edges=true_edges, positives=positives)
    return np.asarray(sample.negatives, dtype=np.int64)


def _type_negatives(
    view: MaskedView,
    positives: np.ndarray,
    negatives: np.ndarray,
    types: np.ndarray,
    true_edges: set[int],
    count: int,
) -> np.ndarray:
    """``count`` extra decoys per row: the masked end paired with the true partner's type-mates.

    Sampler negatives are mostly easy; these are the confusers a type-level
    rule cannot separate. A row whose partner type has too few non-partner
    members is padded with its own first sampler negative, which only repeats
    a decoy the row already ranks. Seeded by the mask's id, so the batch's
    training rng stream is unchanged.
    """
    rng = np.random.default_rng(zlib.crc32(view.instance.instance_id.encode()))
    masked = np.asarray(view.is_masked, dtype=bool)
    members: dict[int, np.ndarray] = {}
    extra = np.repeat(negatives[:, :1], count, axis=1)
    for row, (source, target) in enumerate(positives):
        source_side = masked[source]
        partner = target if source_side else source
        kind = int(types[partner])
        if kind == 0:
            continue
        if kind not in members:
            members[kind] = np.flatnonzero(types == kind)
        candidates = members[kind][members[kind] != partner]
        if source_side:
            pairs = np.column_stack([np.full(len(candidates), source), candidates])
        else:
            pairs = np.column_stack([candidates, np.full(len(candidates), target)])
        keep = [(int(s) << 32 | int(t)) not in true_edges for s, t in pairs]
        pairs = pairs[keep]
        if len(pairs):
            picked = pairs[rng.choice(len(pairs), size=min(count, len(pairs)), replace=False)]
            extra[row, : len(picked)] = picked
    return extra


def _masked_union(instances: tuple[MaskInstance, ...]) -> np.ndarray:
    """Every dense node index any of ``instances`` hides, sorted and unique."""
    if not instances:
        return np.zeros(0, dtype=np.int64)
    masked = [np.asarray(instance.masked, dtype=np.int64) for instance in instances]
    return np.unique(np.concatenate(masked))


def _training_graph(graph: RegionGraph, excluded: np.ndarray) -> RegionGraph:
    """The F1-train copy of ``graph``: no edge touching an excluded neuron.

    Node ids, dense indexing and ``n_nodes`` are unchanged, so masks resolved
    against the original graph keep addressing the same neurons and the
    feature table keeps its row order; only the visible edge set shrinks.
    """
    if excluded.size == 0:
        return graph
    flags = np.zeros(graph.n_nodes, dtype=bool)
    flags[excluded] = True
    keep = ~(flags[graph.edges[:, 0]] | flags[graph.edges[:, 1]])
    return replace(graph, edges=graph.edges[keep], weights=graph.weights[keep])


def _gap_sampler(
    graph: RegionGraph,
    table: FeatureTable,
    config: TrainConfig,
    pool: np.ndarray | None,
) -> _GapDraw:
    """Bind each of ``config.gap_regimes`` to this fit once, failing before training."""
    pool = np.arange(graph.n_nodes, dtype=np.int64) if pool is None else pool
    regimes = config.gap_regimes
    draws = {
        name: _GAP_REGIMES[name](graph, table, pool, config.gap_fraction)
        for name in dict.fromkeys(regimes)
    }
    if len(regimes) == 1:
        return draws[regimes[0]]
    return lambda rng: draws[regimes[int(rng.integers(len(regimes)))]](rng)


def _ranking_pairs(
    view: MaskedView,
    true_edges: set[int] | None,
    *,
    single_masked: bool = False,
    rows: np.ndarray | None = None,
    k: int = 0,
    type_negatives: int = 0,
    types: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """The view's positives, then each positive's K negatives, as one pair array.

    Intact edges are the encoder's own input edges, and the gated evaluation
    never scores a row without a masked endpoint, so only hidden edges are
    positives. ``single_masked`` also drops the both-masked rows, as the
    gated MRR does. ``rows`` keeps only those hidden edges, and negatives are
    drawn for them alone. Returns ``(pairs, P, K)``; :func:`_ranking_rows`
    undoes the layout.
    """
    positives = np.asarray(view.hidden_edges, dtype=np.int64).reshape(-1, 2)
    if rows is not None:
        if single_masked:
            raise ValueError("rows and single_masked cannot be combined")
        positives = positives[rows]
    negatives = _training_negatives(
        view, true_edges, positives if rows is not None else None, **({"k": k} if k else {})
    )
    if single_masked:
        keep = ~np.asarray(view.both_masked(), dtype=bool)
        positives, negatives = positives[keep], negatives[keep]
    if type_negatives and len(positives):
        if types is None:
            raise ValueError("type negatives need the nodes' cell types")
        known = pack_edges(view.graph.edges) if true_edges is None else true_edges
        negatives = np.concatenate(
            [negatives, _type_negatives(view, positives, negatives, types, known, type_negatives)],
            axis=1,
        )
    pairs = np.concatenate([positives, negatives.reshape(-1, 2)], axis=0)
    return pairs, len(positives), negatives.shape[1]


def _ranking_rows(logits: torch.Tensor, n_positives: int, k: int) -> torch.Tensor:
    """``(P, 1 + K)`` rows, each a positive's logit followed by its negatives'."""
    return torch.cat(
        [logits[:n_positives, None], logits[n_positives:].view(n_positives, k)],
        dim=1,
    )


@dataclass(frozen=True)
class _EpochBatch:
    """One epoch's CPU-side inputs; building one is all the epoch draws from the rng."""

    view: MaskedView
    model_view: VisibleView
    type_dropout: np.ndarray
    pairs: np.ndarray
    n_positives: int
    k: int
    rows: np.ndarray | None = None


def _epoch_batch(
    graph: RegionGraph,
    table: FeatureTable,
    rng: np.random.Generator,
    config: TrainConfig,
    *,
    draw_gap: _GapDraw | None = None,
    true_edges: set[int] | None = None,
    complete_graph: RegionGraph | None = None,
) -> _EpochBatch:
    """One epoch's gap and type dropout from ``rng``; its negatives never draw from ``rng``."""
    if draw_gap is None:
        draw_gap = _gap_sampler(graph, table, config, None)
    view = apply_mask(graph, draw_gap(rng), complete_graph=complete_graph)
    # Drawn from the checkpointed training rng, so a resumed run reproduces the
    # same dropout masks as the uninterrupted one it is meant to continue.
    type_dropout = rng.random(graph.n_nodes) < _TYPE_DROPOUT_P
    rows = None
    n_hidden = len(view.hidden_edges)
    if config.pair_batch is not None and config.pair_batch < n_hidden:
        rows = np.sort(rng.choice(n_hidden, size=config.pair_batch, replace=False))
    extra: dict[str, Any] = {}
    if config.train_negatives:
        extra["k"] = config.train_negatives
    if config.type_negatives:
        extra["type_negatives"] = config.type_negatives
        extra["types"] = _type_ids(table, graph.n_nodes).astype(np.int64)
    pairs, n_positives, k = _ranking_pairs(view, true_edges, rows=rows, **extra)
    return _EpochBatch(
        view, view.model_view(), type_dropout, pairs, n_positives, k, rows
    )


def _scorer_table(table: FeatureTable, view: MaskedView, config: TrainConfig) -> FeatureTable:
    """The table a step or a val score may read for ``view``.

    Under ``masked_innervation="uniform"`` each masked root's innervation is replaced by
    the uniform share, so nothing the model reads counts a hidden edge's synapses.
    """
    if config.masked_innervation != "uniform":
        return table
    return table.with_uniform_innervation(np.flatnonzero(view.is_masked))


def _epoch_loss(
    model: M1, table: FeatureTable, batch: _EpochBatch, config: TrainConfig
) -> torch.Tensor:
    view, n_positives, k = batch.view, batch.n_positives, batch.k
    device = model._device
    table = _scorer_table(table, view, config)
    logits, counts = model.score(
        batch.model_view, table, batch.pairs, type_dropout=batch.type_dropout,
        feature_noise=config.feature_noise,
    )
    # Each positive competes with its own K negatives, the row the gate ranks.
    # Pointwise BCE at 1:K is minimized by scoring every pair negative.
    rows = _ranking_rows(logits, n_positives, k)
    loss = F.cross_entropy(
        rows, torch.zeros(n_positives, dtype=torch.long, device=device)
    )
    if n_positives:
        weights = np.asarray(view.hidden_weights)
        if batch.rows is not None:
            weights = weights[batch.rows]
        targets = torch.log1p(
            torch.as_tensor(weights, dtype=torch.float32, device=device)
        )
        loss = loss + config.aux_weight * F.smooth_l1_loss(counts[:n_positives], targets)
    return loss


def _validation_negatives(
    graph: RegionGraph, suite: Suite, suites_dir: str | Path
) -> tuple[dict, CommittedNegatives]:
    """The suite's negative samples and the digests they must match.

    The committed digests come from the frozen artefact
    ``scripts/build_suites.py`` writes at
    ``<suites_dir>/<dataset>/<region>/<name>.negatives.json``. A missing
    artefact raises rather than deriving digests from the samples just drawn,
    which would make ``build_scoring_task``'s byte-identical check unfalsifiable.
    """
    samples = suite_negatives(graph, suite, spec=DEFAULT_SPEC)
    path = (
        Path(suites_dir)
        / suite.dataset
        / suite.region.lower()
        / f"{suite.name}.negatives.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"committed negatives not found at {path}; generate them with "
            f"scripts/build_suites.py before training against {suite.name!r}"
        )
    committed = CommittedNegatives.read(path)
    return samples, committed


def _build_validation_tasks(
    graph: RegionGraph,
    val_suite: Suite,
    instances: tuple[MaskInstance, ...],
    samples: dict,
    committed: CommittedNegatives,
) -> list[tuple[MaskedView, ScoringTask]]:
    """Resolve and verify the suite's scoring tasks once, before the epoch loop.

    Suite, graph, and committed negatives are fixed for the duration of a fit,
    so re-running ``resolve``/``apply_mask``/``build_scoring_task`` per epoch
    only repeats work -- including the graph-digest check and the per-sample
    digest verification that ``build_scoring_task`` performs. Those checks
    still run, once per instance. The caller resolves the instances against
    the original graph, which is the graph ``Suite.validate`` pins.
    """
    tasks = []
    for instance in instances:
        view = apply_mask(graph, instance)
        task = build_scoring_task(
            view, samples[instance.instance_id], suite=val_suite, negatives=committed
        )
        tasks.append((view, task))
    return tasks


def validate_feature_inputs(table: FeatureTable, config: TrainConfig) -> None:
    """Refuse a config whose switches read per-node extras the region does not have."""
    for switch, keys in EXTRAS_KEYS.items():
        if not getattr(config, switch):
            continue
        if table.extras is None:
            raise ValueError(
                f"{switch} reads the per-node {', '.join(keys)} from extras.npz, which "
                f"this region's processed directory does not have; build it with "
                f"scripts/build_node_extras.py"
            )
        missing = [key for key in keys if key not in table.extras]
        if missing:
            raise ValueError(
                f"{switch} reads the per-node {', '.join(missing)} from extras.npz, "
                f"which this region's file does not carry; rebuild it with "
                f"scripts/build_node_extras.py"
            )


def validate_m1_inputs(
    graph: RegionGraph, suite: Suite, suites_dir: str | Path
) -> ValidationInputs:
    """Verify the frozen F1-val suite and its committed negatives for M1."""
    union = is_union(graph.region)
    if graph.dataset != "f1" or (graph.region not in REGIONS and not union):
        raise ValueError(
            f"M1 input preflight requires an F1 region or union graph; got "
            f"{graph.dataset}/{graph.region}"
        )
    permitted = _VAL_SUITES["union" if union else "region"]
    if suite.name != permitted:
        raise ValueError(
            f"M1 input preflight requires the permitted {permitted!r} suite; got "
            f"{suite.name!r}"
        )
    instances = suite.resolve(graph)
    samples, committed = _validation_negatives(graph, suite, suites_dir)
    tasks = _build_validation_tasks(graph, suite, instances, samples, committed)
    return ValidationInputs(instances=instances, tasks=tasks)


class _Validation:
    """The F1-val tasks, their model-facing views, and (for small suites) resident pairs."""

    def __init__(self, tasks: list[tuple[MaskedView, ScoringTask]]) -> None:
        self.tasks = tasks
        self.views: list[tuple[VisibleView, int]] | None = None
        self.device: torch.device | None = None
        self.batches: list[PreparedPairs] | None = None
        self.n_pairs = sum(
            len(task.positives) * (task.negatives.shape[1] + 1) for _, task in tasks
        )

    def prepared(self) -> list[tuple[VisibleView, int]]:
        """Each task's visible view and ranking-row width, cached across scored epochs."""
        if self.views is None:
            self.views = [
                (view.model_view(), task.negatives.shape[1] + 1)
                for view, task in self.tasks
            ]
        return self.views

    def resident(self, model: M1) -> list[PreparedPairs] | None:
        """Every task's pairs prepared on ``model``'s device, or None above the pair cap.

        Larger suites build one instance's pairs per scoring call instead
        (about 450 MB of host memory at OL R10 if all were held).
        """
        if self.n_pairs > _RESIDENT_PAIRS:
            return None
        if self.batches is None or model._device != self.device:
            self.device = model._device
            self.batches = [
                model.prepare(visible, candidate_pairs(task), row_width=width)
                for (_, task), (visible, width) in zip(self.tasks, self.prepared(), strict=True)
            ]
        return self.batches


def _selection_mrr(task: ScoringTask, logits: torch.Tensor) -> float:
    """``score_instance``'s gated MRR, without the strata and recall@10 ``fit`` never reads."""
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("scores must all be finite; got NaN or inf")
    return gated_mrr(task, logits.cpu().numpy())


def _validation_scores(
    model: M1, table: FeatureTable, validation: _Validation, config: TrainConfig
) -> dict[str, float]:
    """F1-val MRR per instance, from resident pairs when the suite is small."""
    model.eval()
    scores = {}
    try:
        resident = validation.resident(model)
        if resident is not None:
            with torch.no_grad():
                type_ids = model.type_ids(table)
                for (view, task), batch in zip(validation.tasks, resident, strict=True):
                    # The ablation changes only the masked roots' rows, so the instance's
                    # pairs stay resident and just the node matrix is rebuilt for it.
                    x = model._node_features(_scorer_table(table, view, config))
                    logits = model.score_prepared(x, batch, type_ids)[0]
                    scores[view.instance.instance_id] = _selection_mrr(task, logits)
                    del logits, x
            return scores
        for (view, task), (visible, row_width) in zip(
            validation.tasks, validation.prepared(), strict=True
        ):
            pairs = candidate_pairs(task)
            logits = model.score_inference(
                visible, _scorer_table(table, view, config), pairs, row_width=row_width
            )[0]
            scores[view.instance.instance_id] = _selection_mrr(task, logits)
            del logits, pairs
    finally:
        model.train()
    return scores


def _on_stride(epoch: int, stride: int, config: TrainConfig) -> bool:
    return epoch % stride == 0 or epoch == config.epochs - 1


def _is_eval_epoch(epoch: int, config: TrainConfig) -> bool:
    """Whether ``epoch`` is scored: every ``eval_every``-th epoch and the last one."""
    return _on_stride(epoch, config.eval_every, config)


def _is_probe_epoch(epoch: int, config: TrainConfig) -> bool:
    return _on_stride(epoch, max(config.eval_every, _TRAIN_MRR_EVERY), config)


def _lr_at(epoch: int, config: TrainConfig) -> float:
    """The learning rate for ``epoch``; a pure function, so a resume needs no state."""
    if epoch < config.warmup_epochs:
        return config.lr * (epoch + 1) / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
    return config.lr * _LR_SCHEDULES[config.lr_schedule].multiplier(progress)


def _patience_ran_out(epoch: int, best_epoch: int, config: TrainConfig) -> bool:
    if not _LR_SCHEDULES[config.lr_schedule].early_stop:
        return False
    return epoch - max(best_epoch, config.warmup_epochs) >= config.patience


def _selection_mean(history: dict[str, Any], by_instance: dict[str, float]) -> float:
    """The mean MRR ``fit`` selects on: over ``val_selection_instances``, or every instance."""
    selected = history.get("val_selection_instances") or list(by_instance)
    return float(np.mean([by_instance[instance_id] for instance_id in selected]))


def validation_mrr(history: dict[str, Any]) -> tuple[list[int], list[float]]:
    """The F1-val-scored epochs of a run's history and their selection MRR.

    The mean is recomputed from ``val_mrr_by_instance`` exactly as ``fit``
    selected on it. Histories written before per-instance scoring store the
    mean as ``val_mrr``, one per epoch when they have no ``val_mrr_epoch``.
    """
    if "val_mrr_by_instance" in history:
        by_instance = history["val_mrr_by_instance"]
        columns = zip(*by_instance.values())
        means = [_selection_mean(history, dict(zip(by_instance, column))) for column in columns]
    else:
        means = list(history.get("val_mrr", []))
    return list(history.get("val_mrr_epoch", range(len(means)))), means


def _init_count_head(model: M1, graph: RegionGraph) -> None:
    """Start the count head at the mean ``log1p`` synapse count it regresses on.

    From a zero bias every positive's Huber gradient pushes the shared decoder
    the same way, and at initialization that gradient is larger than the
    ranking loss's.
    """
    if not len(graph.weights):
        return
    with torch.no_grad():
        model.decoder.count_head.bias.fill_(float(np.log1p(graph.weights).mean()))


def _trained_type_ids(graph: RegionGraph, table: FeatureTable, model: M1) -> np.ndarray:
    """Type ids of the neurons with an edge in ``graph``; every other type reads as OOV."""
    _, type_ids = model._static_features(table)
    if type_ids.shape[0] != graph.n_nodes:
        raise ValueError(
            f"feature table has {type_ids.shape[0]} rows, but the graph has "
            f"{graph.n_nodes} nodes; the table is not in this graph's node order"
        )
    nodes = torch.as_tensor(np.unique(graph.edges.reshape(-1)), device=type_ids.device)
    return type_ids[nodes].cpu().numpy()


def _probe(
    graph: RegionGraph,
    seed: int,
    config: TrainConfig,
    pool: np.ndarray | None,
    true_edges: set[int] | None,
    complete_graph: RegionGraph,
) -> tuple[MaskedView, np.ndarray, int, int] | None:
    """One fixed F1-train gap mask and its single-masked rows, for the train-side MRR.

    Drawn from its own generator so the training rng stream, and with it a
    resumed run, is the same with or without the probe. It is always an R
    mask, so the reference stays fixed whatever ``gap_regimes`` trains on.
    ``None`` when the mask hides no single-masked edge: the probe is logging
    only and must not abort the fit.
    """
    instance = simulate_gaps(
        graph, np.random.default_rng([seed, 1]), config.gap_fraction, pool
    )
    view = apply_mask(graph, instance, complete_graph=complete_graph)
    if not len(view.hidden_edges) or view.both_masked().all():
        return None
    return (view, *_ranking_pairs(view, true_edges, single_masked=True))


def _prepare_probe(
    _model: M1, probe: tuple[MaskedView, np.ndarray, int, int] | None, table: FeatureTable,
    config: TrainConfig,
) -> tuple[VisibleView, np.ndarray, int, int, FeatureTable] | None:
    if probe is None:
        return None
    view, pairs, n_positives, k = probe
    return view.model_view(), pairs, n_positives, k, _scorer_table(table, view, config)


def _train_mrr(
    model: M1,
    probe: tuple[VisibleView, np.ndarray, int, int, FeatureTable] | None,
) -> float:
    """MRR on the probe's rows, the same stratum and the same input regime as ``val_mrr``.

    Beside ``val_mrr`` it separates underfit from overfit.
    """
    if probe is None:
        return float("nan")
    view, pairs, n_positives, k, table = probe
    model.eval()
    try:
        # The probe's pairs are all positives then all negatives, so a chunk never holds a row.
        logits = model.score_inference(view, table, pairs, row_width=1)[0]
        rows = _ranking_rows(logits, n_positives, k).cpu().double().numpy()
        return rows_mrr(rows[:, 0], rows[:, 1:])
    finally:
        model.train()


_log = logging.getLogger(__name__)


def fit(
    graph: RegionGraph,
    table: FeatureTable,
    val_suite: Suite | None,
    *,
    seed: int,
    config: TrainConfig,
    schema: FeatureSchema | None = None,
    device: str = "cpu",
    test_run: bool = False,
    on_run: Callable[[Path], None] | None = None,
    labels: RunLabels | None = None,
) -> TrainResult:
    """Train M1 under per-epoch gap simulation; early-stop on F1-val MRR.

    F1 only: the F2 regions are the held-out
    generalization test and F3 is exploratory, so neither may train. The only
    permitted early-stopping suite is ``f1_val`` for a region graph and
    ``phase3_boundary_f1_val`` for a union graph; the F2 suites are reserved
    for evaluation.

    ``schema`` sizes the type embedding from the schema's true type count, so a
    checkpoint's type embedding fits ``M1(schema)``. When omitted, the width is
    inferred from the types observed in ``table`` and is only guaranteed to
    match when the table contains every type. The rest of the architecture
    follows ``config``: a loader rebuilds ``M1`` from the manifest's config.
    A config without ``sibling_features``, ``encoder_dropout``, ``decoder_dropout``
    or ``steps_per_epoch`` predates them and means off, 0.0, 0.0 and 1, not the
    current ``TrainConfig`` defaults.

    ``device`` is where the model and its tensors live. ``"cuda"`` requires a
    visible GPU with TF32 matmul off, ``"mps"`` requires Apple Metal, and
    ``"cpu"`` is the default. Every device trains and scores in fp32 without
    autocast (``COMPUTE.md``'s training environment).

    ``test_run`` appends ``-test`` to the run ID, so a test run never shares a
    directory or checkpoint with a real one.

    Every call appends its invocation's provenance (UTC timestamp, device,
    precision, Git/image identity, runtime) to the run's ``manifest.json`` and
    leaves the first invocation's top-level record in place, so a resume cannot
    attribute the resuming host's stack to epochs that ran before it.

    ``labels`` carries the launcher's title and changed flags into ``run_start``;
    the branch and commit subject are resolved here, next to the commit. Without
    it ``title`` is null and ``overrides`` is empty.

    ``on_run`` receives the run directory as soon as it is chosen, before
    anything that can fail inside the run.
    """
    with contextlib.ExitStack() as locks:
        return _fit(
            graph, table, val_suite, seed=seed, config=config, schema=schema,
            device=device, test_run=test_run, locks=locks, on_run=on_run,
            labels=labels,
        )


class RunCancelled(RuntimeError):
    """The run directory holds a cancelled run; it does not train again."""


def _write_state_safely(run_dir: Path, status: str, **fields: Any) -> None:
    try:
        write_state(run_dir, status, **fields)
    except Exception:
        _log.warning("could not write %s state for %s", status, run_dir, exc_info=True)


def _fit(
    graph: RegionGraph,
    table: FeatureTable,
    val_suite: Suite | None,
    *,
    seed: int,
    config: TrainConfig,
    schema: FeatureSchema | None = None,
    device: str = "cpu",
    test_run: bool = False,
    locks: contextlib.ExitStack,
    on_run: Callable[[Path], None] | None = None,
    labels: RunLabels | None = None,
) -> TrainResult:
    device = _resolve_device(device)
    permitted = _VAL_SUITES["union" if is_union(graph.region) else "region"]
    if val_suite is not None and val_suite.name != permitted:
        raise ValueError(
            f"val_suite must be the permitted early-stopping split {permitted!r} "
            f"(the registered rules reserve the F2 splits for evaluation); got "
            f"{val_suite.name!r}"
        )
    if graph.dataset != "f1":
        raise ValueError(
            f"fit trains only on F1 (F2 is the "
            f"generalization test and F3 is exploratory); got "
            f"graph.dataset={graph.dataset!r}"
        )
    validate_feature_inputs(table, config)
    writer_class = _summary_writer_class(device)
    validation_inputs = (
        validate_m1_inputs(graph, val_suite, config.suites_dir)
        if val_suite is not None
        else None
    )
    if _BACKENDS[device].deterministic:
        _use_deterministic_algorithms()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = M1(
        _table_schema(table, schema),
        hidden=config.hidden,
        layers=config.layers,
        encoder_dropout=config.encoder_dropout,
        decoder_dropout=config.decoder_dropout,
        sibling_features=config.sibling_features,
        layer_norm=config.layer_norm,
        residual=config.residual,
        branch_norm=config.branch_norm,
        decoder_product_rank=config.decoder_product_rank,
        **{name: getattr(config, name) for name in MODEL_KNOBS},
    ).to(device)
    optimizer = _optimizer(model, config, device)

    # F1-train is the CX graph minus the F1-val masks (phase-1 split table):
    # prune every edge touching a neuron any F1-val instance hides, and draw
    # gap masks only from the rest. Validation keeps the original graph, whose
    # digest the suite pins, and so does the manifest.
    train_graph, train_pool, train_true_edges = graph, None, None
    val_instances: tuple[MaskInstance, ...] = ()
    if val_suite is not None:
        assert validation_inputs is not None
        val_instances = validation_inputs.instances
        excluded = _masked_union(val_instances)
        train_graph = _training_graph(graph, excluded)
        train_pool = np.setdiff1d(
            np.arange(graph.n_nodes, dtype=np.int64), excluded
        )
        # The pruned copy no longer holds the validation edges, so the sampler
        # needs the original edge set to keep refusing them as negatives.
        train_true_edges = pack_edges(graph.edges)
    # The probe is drawn from the graph and pool before any holdout, so it is
    # the same mask with or without one; a holdout then keeps its neurons out
    # of every training step.
    probe_graph, probe_pool = train_graph, train_pool
    if config.holdout_probe:
        held = simulate_gaps(
            train_graph, np.random.default_rng([seed, 1]), config.gap_fraction, train_pool
        ).masked
        held = np.asarray(held, dtype=np.int64)
        train_graph = _training_graph(train_graph, held)
        base_pool = (
            np.arange(graph.n_nodes, dtype=np.int64) if train_pool is None else train_pool
        )
        train_pool = np.setdiff1d(base_pool, held)
        if train_true_edges is None:
            train_true_edges = pack_edges(graph.edges)
    draw_gap = _gap_sampler(train_graph, table, config, train_pool)
    _init_count_head(model, train_graph)
    model.set_trained_types(_trained_type_ids(train_graph, table, model))

    feature_digest = _feature_digest(table, extras=extras_read(config))
    run_id = _run_id(graph, val_suite, config, seed, feature_digest, schema, device)
    if test_run:
        run_id += TEST_RUN_SUFFIX
    run_dir = Path(config.run_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if on_run is not None:
        on_run(run_dir)
    locks.enter_context(RunLock(run_dir))
    if (read_state(run_dir) or {}).get("status") == "cancelled":
        raise RunCancelled(
            f"run {run_dir} was cancelled; move or delete the directory to train "
            f"this configuration again"
        )
    clear_request(run_dir)
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    manifest_path = run_dir / "manifest.json"

    probe: tuple | None = None
    probe_built = False
    history: dict[str, Any] = {"loss": [], "train_mrr": [], "train_mrr_epoch": []}
    if val_suite is not None:
        assert validation_inputs is not None
        validation = _Validation(validation_inputs.tasks)
        history["val_mrr_epoch"] = []
        history["val_mrr_by_instance"] = {
            instance.instance_id: [] for instance in validation_inputs.instances
        }
        history["val_selection_instances"] = [
            instance.instance_id for instance in validation_inputs.instances
            if instance.regime not in _REPORT_ONLY_REGIMES
        ] or list(history["val_mrr_by_instance"])
        val_positives = {
            view.instance.instance_id: int(evaluated_rows(task).sum())
            for view, task in validation_inputs.tasks
        }
    best_epoch = config.epochs - 1
    best_mrr = -math.inf
    start_epoch = 0
    stopped_early = False

    # Resume only when last.pt exists under this run's computed id, i.e. when
    # graph, features, validation suite, config and seed all match.
    resumed = last_path.exists()
    if resumed:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint.get("history", history)
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        rng.bit_generator.state = checkpoint["rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if _BACKENDS[device].rng is not None:
            _BACKENDS[device].rng[1](checkpoint["device_rng"])
        _, means = validation_mrr(history)
        if means:
            best_mrr = max(means)
        # A run that hit patience is finished: return its recorded result
        # rather than training epochs the caller already spent.
        stopped_early = bool(checkpoint.get("stopped_early", False))
    # The weight average F1-val scores and best.pt keeps when ema_decay is on.
    ema: dict[str, torch.Tensor] | None = None
    if config.ema_decay:
        saved = checkpoint.get("ema") if resumed else None
        source = saved if saved is not None else model.state_dict()
        ema = {key: value.detach().clone().to(device) for key, value in source.items()}
    writer = None
    status, error, reason, resume_epoch = "crashed", None, None, None
    next_epoch = start_epoch
    stops = None
    try:
        stops = locks.enter_context(StopSignals())
        write_state(run_dir, "running", pid=os.getpid(), host=socket.gethostname())

        # This process's provenance, appended to whatever the run already recorded:
        # a resume keeps the first invocation's top-level stack rather than
        # attributing its own to epochs that ran before it.
        invocation = _invocation(
            _recorded_device(next(model.parameters()).device),
            _PRECISION,
            resumed=resumed,
            fused_adamw=bool(optimizer.param_groups[0]["fused"]),
            title=labels.title if labels else None,
        )
        provenance, invocations = _invocation_history(manifest_path, invocation)

        # Rewritten after every run; the initial write records "no best yet" so a
        # preempted first run never claims an epoch it did not reach, and both
        # writes carry the same single invocation for this process.
        manifest = _manifest(
            graph,
            config,
            seed,
            feature_digest,
            run_id,
            best_epoch if resumed else None,
            schema,
            val_suite,
            best_val_mrr=best_mrr if resumed and math.isfinite(best_mrr) else None,
            provenance=provenance,
            invocations=invocations,
        )
        _write_manifest(manifest_path, manifest)

        # Events live under the run directory the TensorBoard service reads. A
        # resume purges from the first epoch this invocation retrains: events at or
        # after it belong to an epoch whose checkpoint never landed.
        if writer_class is not None:
            writer = writer_class(
                str(run_dir), purge_step=start_epoch if resumed else None
            )

        rng_state = rng.bit_generator.state
        prefetcher = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="flyarch-prefetch")
            if _BACKENDS[device].prefetch
            else None
        )
        try:
            def next_batch() -> _EpochBatch:
                return _epoch_batch(
                    train_graph, table, rng, config, draw_gap=draw_gap,
                    true_edges=train_true_edges, complete_graph=graph,
                )

            prefetched: Future[_EpochBatch] | None = None
            for epoch in range(start_epoch, config.epochs):
                if stopped_early:
                    break
                model.train()
                lr = _lr_at(epoch, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                losses = []
                for draw in range(config.steps_per_epoch):
                    optimizer.zero_grad()
                    # A worker error surfaces here, in the epoch it was drawn for.
                    batch = prefetched.result() if prefetched is not None else next_batch()
                    # Taken before the next submit: checkpoints never hold a prefetch draw.
                    rng_state = rng.bit_generator.state
                    prefetched = (
                        prefetcher.submit(next_batch)
                        if prefetcher is not None
                        and (epoch + 1 < config.epochs or draw + 1 < config.steps_per_epoch)
                        else None
                    )
                    loss = _epoch_loss(model, table, batch, config)
                    loss.backward()
                    if config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    if ema is not None:
                        with torch.no_grad():
                            for key, value in model.state_dict().items():
                                if value.is_floating_point():
                                    ema[key].lerp_(value, 1 - config.ema_decay)
                                else:
                                    ema[key].copy_(value)
                    losses.append(float(loss.detach()))
                epoch_loss = float(np.mean(losses))
                history["loss"].append(epoch_loss)
                if writer is not None:
                    writer.add_scalar("loss", epoch_loss, epoch)
                    writer.add_scalar("lr", lr, epoch)
                if _is_probe_epoch(epoch, config):
                    if not probe_built:
                        probe = _prepare_probe(model, _probe(
                            probe_graph, seed, config, probe_pool, train_true_edges, graph
                        ), table, config)
                        probe_built = True
                    train_mrr = _train_mrr(model, probe)
                    history["train_mrr"].append(train_mrr)
                    history["train_mrr_epoch"].append(epoch)
                    if writer is not None:
                        writer.add_scalar("train_mrr", train_mrr, epoch)
                best_epoch = epoch if val_suite is None else best_epoch
                improved = False

                scored = _is_eval_epoch(epoch, config)
                # Validate, and keep as best, the averaged weights; training
                # resumes from the live ones once the best checkpoint is written.
                live_weights = None
                if ema is not None and val_suite is not None and scored:
                    live_weights = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                    model.load_state_dict(ema)
                if val_suite is not None and scored:
                    by_instance = _validation_scores(model, table, validation, config)
                    mrr = _selection_mean(history, by_instance)
                    history["val_mrr_epoch"].append(epoch)
                    for instance_id, value in by_instance.items():
                        history["val_mrr_by_instance"][instance_id].append(value)
                    all_mrr = float(np.mean(list(by_instance.values())))
                    if writer is not None:
                        writer.add_scalar("val_mrr", all_mrr, epoch)
                        writer.add_scalar("val_mrr/selection", mrr, epoch)
                        pooled = sum(by_instance[i] * n for i, n in val_positives.items())
                        writer.add_scalar(
                            "val_mrr/pooled", pooled / max(sum(val_positives.values()), 1), epoch
                        )
                        for instance_id, value in by_instance.items():
                            writer.add_scalar(f"val_mrr/{instance_id}", value, epoch)
                    if mrr > best_mrr:
                        best_mrr = mrr
                        best_epoch = epoch
                        improved = True
                stopped_early = (
                    val_suite is not None
                    and scored
                    and _patience_ran_out(epoch, best_epoch, config)
                )
                with stops.deferred():
                    if improved:
                        _save_checkpoint(
                            best_path, model, optimizer, epoch, history, best_epoch,
                            rng_state, device=device,
                        )
                    if live_weights is not None:
                        model.load_state_dict(live_weights)
                    _save_checkpoint(
                        last_path, model, optimizer, epoch, history, best_epoch,
                        rng_state, stopped_early, device=device, ema=ema,
                    )
                    next_epoch = epoch + 1
                    if _BACKENDS[device].empty_cache is not None:
                        _BACKENDS[device].empty_cache()
                # A request seen after the last epoch would undo a run that already finished.
                request = (
                    read_request(run_dir)
                    if not stopped_early and epoch + 1 < config.epochs
                    else None
                )
                if request is not None:
                    stops.ignore()
                    clear_request(run_dir)
                    raise RunStopped(
                        request["status"], reason=clean_reason(request.get("reason"))
                    )
                if stopped_early:
                    break
        finally:
            # However the loop ended, its outcome is recorded; a signal now is ignored.
            with contextlib.suppress(RunStopped):
                stops.ignore()
            try:
                if writer is not None:
                    writer.close()
            finally:
                if prefetcher is not None:
                    prefetcher.shutdown(wait=True, cancel_futures=True)

        if not best_path.exists():
            with stops.deferred():
                _save_checkpoint(
                    best_path, model, optimizer, best_epoch, history, best_epoch,
                    rng_state, device=device,
                )

        manifest = _manifest(
            graph,
            config,
            seed,
            feature_digest,
            run_id,
            best_epoch,
            schema,
            val_suite,
            best_val_mrr=best_mrr if math.isfinite(best_mrr) else None,
            provenance=provenance,
            invocations=invocations,
        )
        with stops.deferred():
            _write_manifest(manifest_path, manifest)

        # The loop keeps stepping after the best-val epoch, through the rest of the
        # patience window, so the live model is not what the run selected.
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
        result = TrainResult(
            state_dict={
                key: value.detach().clone()
                for key, value in checkpoint["model"].items()
            },
            history=history,
            best_epoch=best_epoch,
            manifest=manifest,
            resumed=resumed,
        )
        status = "stopped_early" if stopped_early else "finished"
        _write_state_safely(run_dir, status)
        return result
    except (RunStopped, KeyboardInterrupt) as exc:
        if stops is not None:
            stops.ignore()
        stop = exc if isinstance(exc, RunStopped) else RunStopped("paused", signum=signal.SIGINT)
        status, reason = stop.status, stop.reason
        stop.resume_epoch, stop.run_dir = next_epoch, run_dir
        resume_epoch = next_epoch if status == "paused" else None
        _write_state_safely(run_dir, status, resume_epoch=resume_epoch, reason=reason)
        if stop is exc:
            raise
        raise stop from None
    except BaseException as exc:
        try:
            if stops is not None:
                stops.ignore()
        except RunStopped:
            pass
        error = type(exc).__name__
        _write_state_safely(run_dir, "crashed", error=error)
        raise
    finally:
        if stops is not None:
            stops.ignore()
