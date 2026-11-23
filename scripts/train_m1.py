"""Train one M1 model from processed graph, feature, and F1-val artifacts."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Before the torch import. A full-graph union step allocates and frees about a gigabyte per
# step, and its backward then asks for a contiguous ~800 MiB block that a fragmented 8 GiB
# allocator cannot serve. Windows has no expandable segments, so it collects its cache instead.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "garbage_collection_threshold:0.6" if os.name == "nt" else "expandable_segments:True",
)

from src.feature_io import read_feature_artifacts  # noqa: E402
from src.regions import RegionGraph  # noqa: E402
from src.union import is_union  # noqa: E402
from src.run_labels import TITLE_WARNING, RunLabels  # noqa: E402
from src.run_control import RunStopped  # noqa: E402
from src.boundary_suites import SUITES as BOUNDARY_SUITES, BoundarySuite  # noqa: E402
from src.suites import Suite  # noqa: E402
from train.region_config import (  # noqa: E402
    check_region,
    config_from_args,
    resolve_args,
)
from train.train import fit, validation_mrr  # noqa: E402

from scripts._train_cli import build_argparser  # noqa: E402

#: Device memory left unclaimed for the display and the driver.
_CUDA_ALLOCATOR_MARGIN = 512 * 2**20
#: A union graph's step churns a gigabyte at a time and its backward then wants one contiguous
#: block, so it needs more of an 8 GiB card than this margin leaves it.
_UNION_ALLOCATOR_MARGIN = 256 * 2**20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` and resolve every training value: flag, then file, then default."""
    parser = build_argparser()
    return resolve_args(parser.parse_args(argv), parser)


def _cap_cuda_allocator(device: str, margin: int = _CUDA_ALLOCATOR_MARGIN) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    free, total = torch.cuda.mem_get_info(device)
    if total <= 0:
        raise ValueError(f"CUDA reported invalid total memory: {total} bytes")
    fraction = (free - margin) / total
    if not 0 < fraction <= 1:
        raise ValueError(
            f"CUDA has insufficient free memory for the allocator cap: {free} of {total} bytes"
        )
    index = torch.device(device).index
    torch.cuda.set_per_process_memory_fraction(
        fraction, torch.cuda.current_device() if index is None else index
    )


def _report_stop(stop: RunStopped, argv: list[str]) -> int:
    print(f"run_id: {stop.run_dir.name}")
    print(f"run_path: {stop.run_dir}")
    print(f"status: {stop.status}")
    paused = stop.status == "paused"
    if paused:
        print(f"resume_epoch: {stop.resume_epoch}")
    if stop.reason:
        print(f"reason: {stop.reason}")
    if paused:
        command = [sys.executable, __file__, *argv]
        join = subprocess.list2cmdline if os.name == "nt" else shlex.join
        print(f"resume: {join(command)}")
        return 128 + stop.signum if stop.signum else 3
    return 4


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv)
    if args.title is None and not args.test_run:
        print(TITLE_WARNING, file=sys.stderr)
    started: list[Path] = []

    def on_run(run_dir: Path) -> None:
        started.append(run_dir)
        print(f"run_id: {run_dir.name}", flush=True)

    try:
        config = config_from_args(args)
        graph = RegionGraph.read(args.processed_dir)
        _cap_cuda_allocator(args.device, _UNION_ALLOCATOR_MARGIN if is_union(graph.region)
                            else _CUDA_ALLOCATOR_MARGIN)
        check_region(args, graph)
        table, schema = read_feature_artifacts(args.processed_dir, graph)
        suite = Suite.read(args.suite)
        if suite.name in BOUNDARY_SUITES:
            suite = BoundarySuite.read(args.suite)
        result = fit(
            graph,
            table,
            suite,
            seed=args.seed,
            config=config,
            schema=schema,
            device=args.device,
            test_run=args.test_run,
            on_run=on_run,
            labels=RunLabels(title=args.title, overrides=args.overrides),
        )
    except RunStopped as stop:
        return _report_stop(stop, argv)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        if started:
            print(f"run_path: {started[0]}")
        print(f"M1 training failed: {exc}", file=sys.stderr)
        return 2

    print(f"run_id: {result.manifest['run_id']}")
    print(f"run_path: {args.runs_dir / result.manifest['run_id']}")
    print(f"best_epoch: {result.best_epoch}")
    scored, scores = validation_mrr(result.history)
    print(f"best_val_mrr: {scores[scored.index(result.best_epoch)]:.4f}")
    print(f"final_val_mrr: {scores[-1]:.4f}")
    print(f"resumed: {result.resumed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
