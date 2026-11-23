"""Feasibility report for a built Phase 3 boundary suite (Phase 3 plan 02, task 2).

    python scripts/boundary_feasibility.py --union UNION12 --suite phase3_boundary_f1_val \
        --out phase3-boundary-feasibility.md

Per instance: masked roots, hidden edges, metric rows (single-masked boundary edges), the
block stratum's rows and both-masked boundary rows. Then the share of union edges left
trainable once every root the suite masks is held out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.boundary_suites import SUITES, BoundarySuite, selector_digest  # noqa: E402
from src.regions import RegionGraph  # noqa: E402


def report(graph: RegionGraph, suite: BoundarySuite, selector: dict) -> str:
    if selector["suite_digest"] != suite.digest() or selector["graph_digest"] != graph.digest():
        raise ValueError("selector was not built from this suite and graph")
    lines = [
        f"# {suite.name} feasibility ({graph.region})", "",
        f"Suite `{suite.digest()}`, selector `{selector_digest(selector)}`, graph `{graph.digest()}` "
        f"({graph.n_nodes:,} nodes, {graph.n_edges:,} edges, {selector['boundary_edges']:,} boundary).", "",
        "| Instance | Regime | Masked roots | Hidden edges | Metric rows | Block rows | Both-masked boundary |",
        "|---|---|---|---|---|---|---|",
    ]
    held = np.zeros(graph.n_nodes, dtype=bool)
    for inst in suite.resolve(graph):
        held[inst.masked] = True
        row = selector["instances"][inst.instance_id]
        lines.append(f"| `{inst.instance_id}` | {inst.regime} | {inst.masked.size:,} | {row['n_positives']:,} | "
                     f"{row['n_rows']:,} | {len(row.get('block_rows', [])):,} | {row['n_both_masked_boundary']:,} |")
    total = sum(r["n_rows"] for r in selector["instances"].values())
    block = sum(len(r.get("block_rows", [])) for r in selector["instances"].values())
    trainable = ~(held[graph.edges[:, 0]] | held[graph.edges[:, 1]])
    lines += ["", f"Metric rows: {total:,} ({block:,} in the block stratum).", "",
              f"Holding out every root this suite masks ({int(held.sum()):,} neurons) leaves "
              f"{int(trainable.sum()):,} of {graph.n_edges:,} union edges trainable "
              f"({trainable.mean():.1%}).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--union", required=True)
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    code, dataset = args.union.lower(), SUITES[args.suite][0]
    graph = RegionGraph.read(args.data_root / "processed" / dataset / code)
    folder = args.data_root / "suites" / dataset / code
    suite = BoundarySuite.read(folder / f"{args.suite}.json")
    selector = json.loads((folder / f"{args.suite}.selector.json").read_text(encoding="utf-8"))
    text = report(graph, suite, selector)
    print(text) if args.out is None else args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
