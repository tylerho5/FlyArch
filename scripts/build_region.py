"""Build one region's graph, features, suite inputs and suites for one dataset.

    python3.13 scripts/build_region.py --config train/configs/<code>.toml --dataset f1
    python3.13 scripts/build_region.py --config train/configs/<code>.toml --dataset f2 --calibrate
    python3.13 scripts/build_region.py --config train/configs/<code>.toml --dataset f2
    python3.13 scripts/build_region.py --config train/configs/<code>.toml --dataset f3 --calibrate
    python3.13 scripts/build_region.py --config train/configs/<code>.toml --dataset f3

For CX, ``<code>`` is ``cx``. F1 comes first, since F2's features take the
region's F1 schema. F2 then goes in three steps: ``--calibrate`` reads F2,
prints the chosen threshold and its curve, and writes nothing; a person commits
the value as ``f2_threshold`` in the config; then the F2 build
runs. The F2 build refuses to start while ``f2_threshold`` is absent. F3 (the
male CNS, plan 08) is the same procedure with ``f3_threshold``: ``--calibrate``
compares F3 to F1 for harmonization only, and nothing tunes on ``f3_eval``.

Raw files sit under ``--raw-dir`` (default ``<data-root>/raw``) at the fixed
names in ``RAW_FILES``: F1's at the top, where ``sync_flywire_dataset.py``
writes them, F2's under ``banc/``, where ``sync_banc_dataset.py`` does, and
F3's under ``f3/``, where ``fetch_f3.py`` writes them already in registry
labels, population-filtered and autapse-free, with annotations normalized to
F1's values (``normalize_f3_annotations``).
Outputs go where ``train.region_config.region_paths`` puts them. ``--report``
prints the pre-build table after the suite inputs and before any suite is
written; given without ``--steps`` it runs no step.

The steps call the existing CLIs' ``main(argv)`` in-process, so F1/CX runs the
same code as Phase 1. F1's ``extras`` step writes ``extras.npz``, the per-node
annotation groups the ``annotation_groups`` model switch reads; F2 and F3 skip
it, since it aligns F1 root ids from the FlyWire annotation table. F1 builds
``f1_val``; F2 builds ``f2_dev``, ``f2_confirm``,
and ``f2_side`` whenever the graph's nodes carry both soma sides; F3 builds only the exploratory ``f3_eval`` (no H
suite, though ``f3_sides.parquet`` is still written). Beside each F2 or F3
suite it writes ``<name>.same_name.json``: per T instance, the drawn type,
its size and single-masked positives, and how many nodes whose own BANC
``cell_type`` is that name the mask leaves visible (nodes whose FAFB match
carries another type). For F3 the own name is the release's ``type``. It is
committed with the suite. F2's T instances draw only the cell types the
region's expert trains on (its F1 graph minus F1-val's masked neurons' edges),
so the F1 build must exist first.

A config with ``[mask_exclude]`` (OL) keeps every neuron in every graph but
never masks F1's lamina-dependent types. Each build of that region, on every
dataset, writes ``excluded_types.json`` beside the graph (the types, the rule,
and the hashes of the files it read) and logs how many graph neurons carry one.
The suites step reads that file, checks its ``min_fraction`` against the config,
and passes the types to ``build_suite``; ``--calibrate`` leaves those types out
of the density comparison on both sides.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import build_features, build_node_extras, build_suites, prepare_suite_inputs  # noqa: E402
from scripts.build_features import (  # noqa: E402
    _canonicalize, f2_annotations, f3_annotations, read_banc_table, read_table,
)
from src.harmonize import banc_neuron_ids, calibrate_threshold, lamina_dependent_types  # noqa: E402
from src.digests import sha256  # noqa: E402
from src.regions import RegionGraph, build_region_graph, region_labels  # noqa: E402
from src.suites import Suite, build_suite  # noqa: E402
from train.region_config import RegionConfig, load_region_config, region_paths  # noqa: E402

RAW_FILES = {
    "f1": {
        "connections": "proofread_connections_783.feather",
        "annotations": "Supplemental_file1_neuron_annotations.tsv",
        "synapses": "flywire_synapses_783.feather",
        "innervation": (
            "per_neuron_neuropil_count_pre_783.feather",
            "per_neuron_neuropil_count_post_783.feather",
        ),
    },
    "f2": {
        "connections": "banc/banc_888_edgelist_neuropil_v2.feather",
        "annotations": "banc/banc_888_meta.feather",
        "synapses": "banc/banc_888_synapses_v2_enriched.parquet",
        "innervation": ("banc/banc_888_synapses_per_neuron_neuropil_side_v2.parquet",),
    },
    "f3": {
        "connections": "f3/connections.parquet",
        "annotations": "f3/annotations.parquet",
        "synapses": "f3/synapses.parquet",
        "innervation": ("f3/innervation.parquet",),
    },
}
STEPS = ("extract", "features", "extras", "suite-inputs", "suites")
SUITES = {"f1": ("f1_val",), "f2": ("f2_dev", "f2_confirm"), "f3": ("f3_eval",)}


def log(message: str) -> None:
    print(f"[build_region] {message}", flush=True)


def _step(main, argv: list) -> None:
    code = main([str(arg) for arg in argv])
    if code:
        raise SystemExit(code)


def read_f2_connections(raw: Path, region: str, meta: pd.DataFrame) -> pd.DataFrame:
    """BANC's edgelist on ``region``: proofread-tier neurons at both ends, no autapses.

    F1's connections table holds proofread neurons only and no autapses; BANC's
    covers every segment, so it is cut to the same population.
    """
    neurons = banc_neuron_ids(meta)
    edges = read_banc_table(
        raw / RAW_FILES["f2"]["connections"], region, ("pre_root_888", "post_root_888")
    ).rename(columns={"pre_root_888": "pre_root_id", "post_root_888": "post_root_id"})
    keep = (
        edges["pre_root_id"].isin(neurons)
        & edges["post_root_id"].isin(neurons)
        & (edges["pre_root_id"] != edges["post_root_id"])
    )
    return edges[keep]


def lamina_exclusion(
    config: RegionConfig, raw: Path, f1_annotations: pd.DataFrame, extra: tuple[str, ...] = ()
) -> tuple[tuple[str, ...], dict]:
    """The F1 types ``config.mask_exclude`` never masks, and the ``excluded_types.json`` record.

    Reads F1 only: its annotations and its per-neuron pre and post neuropil counts.
    ``extra`` names further raw files whose hashes the record carries (F2's annotations).
    """
    rule = config.mask_exclude
    files = [raw / RAW_FILES["f1"]["annotations"], *(raw / name for name in RAW_FILES["f1"]["innervation"])]
    pre, post = (pd.read_feather(path) for path in files[1:])
    f1_types = _canonicalize(f1_annotations, ("root_id", "cell_type")).set_index("root_id")["cell_type"]
    types = lamina_dependent_types(pre, post, f1_types, min_fraction=rule["min_fraction"])
    record = {
        "rule": rule["rule"], "min_fraction": rule["min_fraction"],
        "inputs": {path.name: sha256(path) for path in [*files, *(raw / name for name in extra)]},
        "types": list(types),
    }
    return types, record


def type_ids(annotations: pd.DataFrame, types: tuple[str, ...]) -> np.ndarray:
    """Root ids whose ``cell_type`` is in ``types``: F1 annotations, or F2's crosswalked frame."""
    table = _canonicalize(annotations, ("root_id", "cell_type"))
    return table.loc[table["cell_type"].isin(types), "root_id"].to_numpy(dtype=np.int64)


def write_exclusion(config: RegionConfig, dataset: str, raw: Path, out: Path, graph: RegionGraph) -> None:
    """Write ``excluded_types.json`` and log how many graph neurons it keeps unmasked."""
    f1_annotations = read_table(raw / RAW_FILES["f1"]["annotations"])
    extra = () if dataset == "f1" else (RAW_FILES[dataset]["annotations"],)
    types, record = lamina_exclusion(config, raw, f1_annotations, extra)
    path = out / "excluded_types.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    frame = f1_annotations if dataset == "f1" else crosswalked_annotations(raw, dataset, f1_annotations)
    carried = int(np.isin(graph.node_ids, type_ids(frame, types)).sum())
    log(f"{len(types)} F1 type(s) are never masked: {', '.join(types)}")
    log(f"{carried} of {graph.n_nodes} graph neuron(s) carry one -> {path}")
    if dataset != "f1":
        own = own_names(raw, graph)
        crosswalked = frame.set_index("root_id")["cell_type"].reindex(graph.node_ids)
        residual = int((own.isin(types) & ~crosswalked.isin(types)).sum())
        source = "BANC" if dataset == "f2" else "release"
        log(f"{residual} {dataset.upper()} neuron(s) carry an excluded {source} name but crosswalk to a "
            "kept type; they stay eligible")


def mask_exclude_types(config: RegionConfig, folder: Path) -> tuple[str, ...]:
    """The types ``folder``'s ``excluded_types.json`` lists; refuses a missing file or another ``min_fraction``."""
    if config.mask_exclude is None:
        return ()
    path = folder / "excluded_types.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; rerun with --steps extract")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["min_fraction"] != config.mask_exclude["min_fraction"]:
        raise SystemExit(
            f"{path} has min_fraction {record['min_fraction']}, the config asks for "
            f"{config.mask_exclude['min_fraction']}; rerun with --steps extract"
        )
    return tuple(record["types"])


def read_f3_connections(raw: Path, region: str) -> pd.DataFrame:
    """F3's connections on ``region``'s labels; ``fetch_f3.py`` already cut the population."""
    return pd.read_parquet(
        raw / RAW_FILES["f3"]["connections"], filters=[("neuropil", "in", list(region_labels(region)))]
    )


def read_connections(raw: Path, dataset: str, region: str) -> pd.DataFrame:
    if dataset == "f1":
        return pd.read_feather(raw / RAW_FILES["f1"]["connections"])
    if dataset == "f3":
        return read_f3_connections(raw, region)
    return read_f2_connections(raw, region, read_table(raw / RAW_FILES["f2"]["annotations"]))


def crosswalked_annotations(raw: Path, dataset: str, f1_annotations: pd.DataFrame) -> pd.DataFrame:
    """F2's or F3's annotations with ``cell_type`` crosswalked into F1's vocabulary."""
    meta = read_table(raw / RAW_FILES[dataset]["annotations"])
    return f2_annotations(meta, f1_annotations)[0] if dataset == "f2" else f3_annotations(meta, f1_annotations)


def extract(config: RegionConfig, dataset: str, raw: Path, out: Path, threshold: int) -> None:
    connections = read_connections(raw, dataset, config.region)
    graph = build_region_graph(connections, region=config.region, dataset=dataset, threshold=threshold)
    graph.write(out)
    log(
        f"{dataset}/{config.region} at threshold {threshold}: {graph.n_nodes} nodes, "
        f"{graph.n_edges} edges -> {out} (digest {graph.digest()[:16]}...)"
    )
    if config.mask_exclude is not None:
        write_exclusion(config, dataset, raw, out, graph)


def calibrate(config: RegionConfig, dataset: str, raw: Path, root: Path) -> None:
    """Print the F2 or F3 threshold sweep for one region; the config is never written."""
    f1 = region_paths(config.region, "f1", root)
    f1_graph = RegionGraph.read(f1.processed)
    if f1_graph.threshold != config.f1_threshold:
        raise SystemExit(
            f"{f1.processed} is at threshold {f1_graph.threshold}, the config's f1_threshold is "
            f"{config.f1_threshold}; rebuild F1 first"
        )
    f1_types = build_suites.load_column(f1.interim / "f1_types.parquet", "cell_type")
    f1_meta = read_table(raw / RAW_FILES["f1"]["annotations"])
    typed = crosswalked_annotations(raw, dataset, f1_meta)
    exclude = ()
    if config.mask_exclude is not None:
        exclude, _ = lamina_exclusion(config, raw, f1_meta)
        log(f"the density comparison leaves out {len(exclude)} F1 type(s): {', '.join(exclude)}")
    result = calibrate_threshold(
        f1_graph, f1_types, read_connections(raw, dataset, config.region),
        typed.set_index("root_id")["cell_type"], region=config.region, dataset=dataset,
        exclude_types=exclude,
    )
    print(result.curve.to_string(index=False))
    log(
        f"chosen {dataset}_threshold for {config.region}: {result.chosen} (spearman "
        f"{result.concordance:.3f} over {result.n_pairs} type pairs). Commit it to the config."
    )


def own_names(raw: Path, graph: RegionGraph) -> pd.Series:
    """Each node's own type name before the crosswalk: BANC ``cell_type``, F3 ``type``."""
    meta = read_table(raw / RAW_FILES[graph.dataset]["annotations"])
    keys = ("bodyId", "type") if graph.dataset == "f3" else ("root_id", "cell_type")
    return meta.set_index(keys[0])[keys[1]].reindex(graph.node_ids)


def typed_draws(graph: RegionGraph, suite: Suite, names: pd.Series | None = None) -> list[dict]:
    """Each T instance's type, size and single-masked positives.

    Given ``names`` (F2's own names, ``own_names``), also how many nodes named
    the drawn type stay visible: a node whose FAFB match carries another type
    keeps that type (partial removal).
    """
    rows = []
    for instance in suite.instances:
        if instance.regime != "T":
            continue
        masked = np.isin(graph.node_ids, instance.masked_root_ids)
        row = {
            "instance": instance.instance_id, "cell_type": instance.params["cell_type"],
            "size": int(masked.sum()),
            "single_masked_positives": int((masked[graph.edges[:, 0]] != masked[graph.edges[:, 1]]).sum()),
        }
        if names is not None:
            named = names.eq(row["cell_type"]).fillna(False).to_numpy(dtype=bool)
            row["same_name_unmasked"] = int((named & ~masked).sum())
        rows.append(row)
    return rows


def f1_pool(config: RegionConfig, root: Path) -> list:
    """``--f1-graph``/``--f1-types``/``--f1-val`` for the region's F1 build: F2's T draw pool."""
    f1 = region_paths(config.region, "f1", root)
    if RegionGraph.read(f1.processed).threshold != config.f1_threshold:
        raise SystemExit(f"{f1.processed} is not at f1_threshold {config.f1_threshold}; rebuild F1 first")
    return ["--f1-graph", f1.processed, "--f1-types", f1.interim / "f1_types.parquet",
            "--f1-val", f1.suites / "f1_val.json"]


def report(
    config: RegionConfig, graph: RegionGraph, folder: Path, interim: Path, names: pd.Series | None = None,
    pool: list | None = None,
) -> None:
    """Type sizes, F2's type-match rate, and every typed draw the suites will make."""
    dataset = graph.dataset
    types = build_suites.load_column(interim / f"{dataset}_types.parquet", "cell_type").reindex(graph.node_ids)
    centroids = build_suites.load_centroids(interim / f"{dataset}_centroids.parquet", graph)
    sizes = types.value_counts()
    bins = pd.cut(sizes, [0, 1, 3, 10, 100, 1000, np.inf],
                  labels=["1", "2-3", "4-10", "11-100", "101-1000", ">1000"])
    print(f"\n--- {dataset}/{graph.region} pre-build report ---")
    print(f"{graph.n_nodes} nodes, {int(types.notna().sum())} typed, {len(sizes)} cell types")
    if dataset in ("f2", "f3"):
        print(f"type-match rate: {types.notna().mean():.1%} of nodes carry an F1 type")
    print("cell types by size: " + ", ".join(f"{k}: {v}" for k, v in bins.value_counts(sort=False).items()))
    columns = ["instance", "cell_type", "size", "single_masked_positives"]
    columns += ["same_name_unmasked"] if names is not None else []
    print("\t".join(["suite", *columns]))
    draw_types = build_suites.draw_pool(*pool[1::2]) if pool else None
    for name in SUITES[dataset]:
        suite = build_suite(
            graph, name=name, types=types, centroids=centroids,
            mask_exclude_types=mask_exclude_types(config, folder), draw_types=draw_types,
        )
        for row in typed_draws(graph, suite, names):
            print("\t".join(str(value) for value in [name, *(row[c] for c in columns)]))


def suites(
    config: RegionConfig, graph_dir: Path, interim: Path, out: Path, dataset: str, graph: RegionGraph,
    names: pd.Series | None = None, pool: list | None = None,
) -> None:
    types = interim / f"{dataset}_types.parquet"
    exclude = []
    if mask_exclude_types(config, graph_dir):
        exclude = ["--exclude-types", graph_dir / "excluded_types.json"]
    for name in SUITES[dataset]:
        _step(build_suites.main, [
            "--graph", graph_dir, "--name", name, "--types", types,
            "--centroids", interim / f"{dataset}_centroids.parquet", "--out", out, *exclude, *(pool or []),
        ])
        if names is not None:
            rows = typed_draws(graph, Suite.read(out / f"{name}.json"), names)
            path = out / f"{name}.same_name.json"
            record = {"name": name, "instances": {row.pop("instance"): row for row in rows}}
            path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            unmasked = sum(row["same_name_unmasked"] for row in rows)
            log(f"{name}: its T instances leave {unmasked} same-name node(s) visible -> {path}")
    if dataset != "f2":
        return  # F3 holds no H suite
    sides = build_suites.load_column(interim / "f2_sides.parquet", "side").reindex(graph.node_ids)
    if {"L", "R"} <= set(sides.dropna()):
        _step(build_suites.main, [
            "--graph", graph_dir, "--name", "f2_side", "--sides", interim / "f2_sides.parquet", "--out", out,
            *(["--types", types, *exclude] if exclude else []),
        ])
    else:
        log("f2_side not built: the graph's nodes do not carry both soma sides")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="train/configs/<code>.toml")
    parser.add_argument("--dataset", required=True, choices=sorted(RAW_FILES))
    parser.add_argument("--raw-dir", type=Path, help="default <data-root>/raw")
    parser.add_argument("--data-root", type=Path, help="default $FLYARCH_DATA_ROOT, then data")
    parser.add_argument("--steps", nargs="+", choices=STEPS, help="default: every step")
    parser.add_argument("--report", action="store_true", help="print the pre-build table before any suite")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="f2 or f3, alone: print the chosen threshold and its curve; writes nothing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    try:
        config = load_region_config(args.config)
    except (OSError, ValueError) as exc:
        parser.error(f"--config {args.config}: {exc}")
    root = Path(args.data_root or os.environ.get("FLYARCH_DATA_ROOT") or "data")
    raw = args.raw_dir or root / "raw"

    if args.calibrate:
        if args.dataset == "f1" or args.steps is not None or args.report:
            parser.error("--calibrate runs alone, with --dataset f2 or f3")
        calibrate(config, args.dataset, raw, root)
        return 0

    threshold = getattr(config, f"{args.dataset}_threshold")
    if threshold is None:
        parser.error(
            f"{args.config} has no {args.dataset}_threshold. Run --calibrate, commit the threshold with "
            f"its ingest-verification row, then build"
        )
    steps = args.steps if args.steps is not None else (() if args.report else STEPS)
    paths = region_paths(config.region, args.dataset, root)
    files = RAW_FILES[args.dataset]
    f1_extra = []
    if args.dataset != "f1":
        f1_extra = ["--f1-annotations", raw / RAW_FILES["f1"]["annotations"]]

    if "extract" in steps:
        extract(config, args.dataset, raw, paths.processed, threshold)
    graph = RegionGraph.read(paths.processed)
    if (graph.dataset, graph.region, graph.threshold) != (args.dataset, config.region, threshold):
        raise SystemExit(
            f"{paths.processed} holds {graph.dataset}/{graph.region} at threshold {graph.threshold}, "
            f"not {args.dataset}/{config.region} at {threshold}; rerun with --steps extract"
        )
    if "features" in steps:
        schema = []
        if args.dataset != "f1":
            schema = ["--schema", region_paths(config.region, "f1", root).processed / "schema.json"]
            if not schema[1].exists():
                raise SystemExit(f"{schema[1]} is missing; build F1 for {config.region} first")
        _step(build_features.main, [
            "--graph", paths.processed, "--annotations", raw / files["annotations"],
            "--dataset", args.dataset, "--out", paths.processed, *schema, *f1_extra,
            "--synapses", raw / files["synapses"],
            "--innervation", *(raw / name for name in files["innervation"]),
        ])
    if "extras" in steps:
        if args.dataset == "f1":
            _step(build_node_extras.main, [
                "--annotations", raw / files["annotations"], paths.processed,
            ])
        else:
            log(f"extras skipped: extras.npz aligns F1 root ids from the FlyWire annotation "
                f"table, so the {args.dataset} build has none")
    if "suite-inputs" in steps:
        _step(prepare_suite_inputs.main, [
            "--graph", paths.processed, "--annotations", raw / files["annotations"],
            "--synapses", raw / files["synapses"], "--out", paths.interim, *f1_extra,
        ])
    names = None
    if args.dataset != "f1" and (args.report or "suites" in steps):
        names = own_names(raw, graph)
    pool = f1_pool(config, root) if args.dataset == "f2" and (args.report or "suites" in steps) else None
    if args.report:
        report(config, graph, paths.processed, paths.interim, names, pool)
    if "suites" in steps:
        suites(config, paths.processed, paths.interim, paths.suites, args.dataset, graph, names, pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
