"""Measure whether a region config fits a device before training its sweep.

Builds the model and F1-train graph the way ``fit`` does, then runs several
training draws (forward, backward, optimizer step) at the config's
``pair_batch`` and the complete F1-val suite as ``fit`` scores it. Reports peak
memory for each, the largest gap's positive count, the highest active type id,
step time, and headroom against the card. Nothing is written to a run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
CUDA_ALLOCATOR_MARGIN = 512 * 2**20

from src.feature_io import read_feature_artifacts  # noqa: E402
from src.negatives import pack_edges  # noqa: E402
from src.regions import RegionGraph  # noqa: E402
from src.boundary_suites import BoundarySuite  # noqa: E402
from src.suites import Suite  # noqa: E402
from src.union import is_union  # noqa: E402
from train import train as training  # noqa: E402
from train.model import M1, MODEL_KNOBS  # noqa: E402
from train.region_config import config_path, load_region_config, region_paths  # noqa: E402


class _Memory:
    """Peak device memory where the backend reports it, else the process high-water mark.

    On CUDA the peak is reserved memory, the quantity the allocator cap limits.
    """

    def __init__(self, device: str) -> None:
        self.kind = torch.device(device).type
        self.device = device

    def reset(self) -> None:
        if self.kind == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak(self) -> int:
        if self.kind == "cuda":
            torch.cuda.synchronize(self.device)
            return int(torch.cuda.max_memory_reserved(self.device))
        if self.kind == "mps":
            torch.mps.synchronize()
            return int(torch.mps.driver_allocated_memory())
        import resource  # POSIX only; CPU probes run on the Mac or Linux

        scale = 1 if sys.platform == "darwin" else 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale

    def peak_allocated(self) -> int | None:
        if self.kind != "cuda":
            return None
        torch.cuda.synchronize(self.device)
        return int(torch.cuda.max_memory_allocated(self.device))

    def total(self) -> int | None:
        if self.kind == "cuda":
            return int(torch.cuda.get_device_properties(self.device).total_memory)
        if self.kind == "mps":
            return int(torch.mps.recommended_max_memory())
        return None


def _gib(n: int | None) -> str:
    return "n/a" if n is None else f"{n / 2**30:.2f} GiB"


def _cap_cuda_allocator(device: str) -> dict[str, int]:
    if not device.startswith("cuda"):
        return {}
    free, total = torch.cuda.mem_get_info(device)
    cap = free - CUDA_ALLOCATOR_MARGIN
    if total <= 0 or cap <= 0:
        raise ValueError(f"CUDA has insufficient free memory for the 512 MiB allocator margin "
                         f"({free} free of {total} bytes)")
    index = torch.device(device).index
    torch.cuda.set_per_process_memory_fraction(
        cap / total, torch.cuda.current_device() if index is None else index)
    return {"free_at_start": int(free), "allocator_cap": int(cap)}


def probe(
    graph: RegionGraph, table, schema, suite: Suite, config: training.TrainConfig,
    *, device: str, draws: int, seed: int = 0, seed_set: bool = False,
) -> dict:
    device = training._resolve_device(device)
    memory = _Memory(device)
    if training._BACKENDS[device].deterministic:
        training._use_deterministic_algorithms()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    inputs = training.validate_m1_inputs(graph, suite, config.suites_dir)
    excluded = training._masked_union(inputs.instances)
    train_graph = training._training_graph(graph, excluded)
    pool = np.setdiff1d(np.arange(graph.n_nodes, dtype=np.int64), excluded)
    draw_gap = training._gap_sampler(train_graph, table, config, pool)
    true_edges = pack_edges(graph.edges)
    model = M1(
        training._table_schema(table, schema),
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
    optimizer = training._optimizer(model, config, device)
    trained = training._trained_type_ids(train_graph, table, model)
    model.set_trained_types(trained)

    steps = []
    model.train()
    for _ in range(draws):
        batch = training._epoch_batch(
            train_graph, table, rng, config, draw_gap=draw_gap,
            true_edges=true_edges, complete_graph=graph,
        )
        memory.reset()
        started = time.monotonic()
        optimizer.zero_grad()
        loss = training._epoch_loss(model, table, batch, config)
        loss.backward()
        optimizer.step()
        peak = memory.peak()
        step = {
            "hidden_edges": len(batch.view.hidden_edges),
            "ranked_rows": batch.n_positives,
            "seconds": time.monotonic() - started,
            "peak": peak,
            "peak_allocated": memory.peak_allocated(),
        }
        if seed_set:
            step["seed_set"] = int(np.unique(batch.pairs).size)
        steps.append(step)

    memory.reset()
    started = time.monotonic()
    training._validation_scores(model, table, training._Validation(inputs.tasks), config)
    validation = {"seconds": time.monotonic() - started, "peak": memory.peak(),
                  "peak_allocated": memory.peak_allocated()}
    return {
        "steps": steps,
        "validation": validation,
        "max_type_id": int(trained.max()) if trained.size else 0,
        "total": memory.total(),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, help="registry code, e.g. OL")
    parser.add_argument("--device", required=True, choices=training.DEVICES)
    parser.add_argument("--config", type=Path, help="defaults to train/configs/<code>.toml")
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--pair-batch", type=int, help="override the config's pair_batch")
    parser.add_argument("--seed-set", action="store_true", help="report distinct endpoints in each draw")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--suites-dir", type=Path, help="defaults to the region's data/suites root")
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--json", action="store_true", help="print the measurement as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    lock = None
    if args.gpu_lock is not None:
        from src.run_control import RunLock, RunLocked

        try:
            lock = RunLock(args.gpu_lock.parent, args.gpu_lock.name)
            lock.__enter__()
        except RunLocked as exc:
            parser.error(f"GPU is busy: {exc}")
    try:
        return _run_probe_command(args, parser)
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)


def _run_probe_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        cuda_memory = _cap_cuda_allocator(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    region_config = load_region_config(args.config or config_path(args.region))
    if args.config is not None and args.region.upper() != region_config.region:
        parser.error(
            f"--region {args.region} disagrees with {args.config}, "
            f"which is for {region_config.region}"
        )
    paths = region_paths(region_config.region, "f1", args.data_root)
    suites_dir = args.suites_dir or paths.suites_root
    config = region_config.train_config(
        run_dir="unused", suites_dir=suites_dir.as_posix(), pair_batch=args.pair_batch
    )
    graph = RegionGraph.read(paths.processed)
    table, schema = read_feature_artifacts(paths.processed, graph)
    if is_union(region_config.region):
        suite = BoundarySuite.read(suites_dir / "f1" / region_config.region.lower() / "phase3_boundary_f1_val.json")
    else:
        suite = Suite.read(suites_dir / "f1" / region_config.region.lower() / "f1_val.json")
    result = probe(
        graph, table, schema, suite, config, device=args.device, draws=args.draws,
        seed_set=args.seed_set,
    )

    if args.json:
        print(json.dumps({
            "region": region_config.region,
            "device": args.device,
            "pair_batch": config.pair_batch,
            "config": training._scientific_config(config),
            "code_digest": training._code_digest(),
            "graph_digest": graph.digest(),
            "feature_digest": training._feature_digest(table, extras=training.extras_read(config)),
            "schema_digest": schema.digest(),
            "suite_digest": suite.digest(),
            **cuda_memory,
            **result,
        }, sort_keys=True))
        return 0

    steps = result["steps"]
    peak = max([step["peak"] for step in steps] + [result["validation"]["peak"]])
    total = result["total"]
    print(f"region: {region_config.region}  device: {args.device}  pair_batch: {config.pair_batch}")
    print(f"nodes: {graph.n_nodes}  edges: {len(graph.edges)}  max_type_id: {result['max_type_id']}")
    print(f"largest gap: {max(step['hidden_edges'] for step in steps)} positives")
    for i, step in enumerate(steps):
        print(f"draw {i}: rows {step['ranked_rows']}  {step['seconds']:.2f} s  "
              f"peak {_gib(step['peak'])}")
        if args.seed_set:
            print(f"  distinct endpoints: {step['seed_set']}")
    print(f"f1_val: {result['validation']['seconds']:.2f} s  "
          f"peak {_gib(result['validation']['peak'])}")
    print(f"peak: {_gib(peak)}  card: {_gib(total)}  "
          f"headroom: {_gib(None if total is None else total - peak)}")
    if cuda_memory:
        print(f"free at start: {_gib(cuda_memory['free_at_start'])}  "
              f"allocator cap: {_gib(cuda_memory['allocator_cap'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
