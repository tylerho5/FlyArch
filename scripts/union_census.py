"""Phase 3 boundary census from the F1 connections feather (Phase 3 plan 01).

    python scripts/union_census.py --connections data/raw/proofread_connections_783.feather
    python scripts/union_census.py --connections <file> --regions INP PENP VLNP VMNP --out <json>

Without ``--regions`` it censuses UNION12, UNION11 and the spec's linked block (with and
without CX), checks every number the spec cites and every regional digest against
``results/experts.csv`` (and, for regions not registered yet, ``--reference-dir``'s
``graph.json``), prints PASS or FAIL per gate, and writes the committed record. A failing
gate is reported, never patched: exit 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pyarrow.feather as feather  # noqa: E402

from src.digests import sha256  # noqa: E402
from src.regions import REGIONS  # noqa: E402
from src.union import UNIONS, census, pair_counts  # noqa: E402

OUT = REPO_ROOT / "results/phase3/census_f1.json"
BLOCK = ("INP", "PENP", "VLNP", "VMNP")
#: Numbers the Phase 3 design spec cited for the F1 boundary census.
GLOBAL = {"edges": 2_700_513, "in_a_region_graph": 2_589_381, "boundary": 111_132,
          "registered_labels_only": 110_373, "needs_excluded_labels": 759,
          "region_graph_neurons": 133_844, "in_two_or_more": 33_047}
PAIRS = {"INP-VLNP": 15_274, "VLNP-VMNP": 14_846, "INP-VMNP": 11_404, "PENP-VMNP": 5_672}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def read_connections(path: Path):
    table = feather.read_table(path, columns=["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"])
    return table.set_column(2, "neuropil", table.column("neuropil").dictionary_encode()).to_pandas()


def registry_digests(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    digests = {}
    for row in rows:
        if digests.setdefault(row["region"], row["graph_digest"]) != row["graph_digest"]:
            raise ValueError(f"{path}: {row['region']} has more than one graph_digest")
    return digests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--connections", required=True, type=Path)
    parser.add_argument("--regions", nargs="+", choices=sorted(REGIONS), help="one ad hoc set; no gates")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "results/experts.csv")
    parser.add_argument("--reference-dir", type=Path, help="processed f1 dirs for regions not in the registry")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    pc = pair_counts(read_connections(args.connections))
    print(f"[census] {len(pc.node_ids)} neurons, {len(pc.pre)} directed pairs", flush=True)
    if args.regions:
        text = json.dumps(census(pc, args.regions), indent=2) + "\n"
        print(text) if args.out is None else args.out.write_text(text, encoding="utf-8")
        return 0

    sets = {"UNION12": UNIONS["UNION12"], "UNION11": UNIONS["UNION11"],
            "block": BLOCK, "block+CX": BLOCK + ("CX",)}
    results = {name: census(pc, regions) for name, regions in sets.items()}
    gates = []

    def gate(label: str, expected, found) -> None:
        gates.append({"gate": label, "expected": expected, "found": found, "pass": expected == found})
        print(f"[census] {'PASS' if expected == found else 'FAIL'} {label}: expected {expected}, found {found}")

    u12, u11, block = results["UNION12"], results["UNION11"], results["block"]
    for key, expected in GLOBAL.items():
        gate(f"global {key}", expected, u12["global"][key])
    gate("UNION12 edges", 2_699_754, u12["union"]["edges"])
    gate("UNION12 nodes", 133_910, u12["union"]["nodes"])
    gate("UNION12 boundary = global registered-labels-only", u12["global"]["registered_labels_only"],
         u12["boundary"]["edges"])
    gate("UNION11 edges", 1_232_549, u11["union"]["edges"])
    gate("UNION11 nodes", 48_586, u11["union"]["nodes"])
    gate("block union edges", 666_840, block["union"]["edges"])
    gate("block boundary edges", 45_492, block["boundary"]["edges"])
    gate("CX adds boundary edges", 745, results["block+CX"]["boundary"]["edges"] - block["boundary"]["edges"])
    for pair, expected in PAIRS.items():
        gate(f"pairwise boundary {pair}", expected, u12["pairwise_boundary"][pair])

    registry = registry_digests(args.registry)
    for code, row in u12["regional"].items():
        if code in registry:
            gate(f"{code} digest = {args.registry.name} graph_digest", registry[code], row["digest"])
            continue
        meta = args.reference_dir / code.lower() / "graph.json" if args.reference_dir else None
        if meta is not None and meta.is_file():
            gate(f"{code} digest = {meta.parent.name}/graph.json (not registered)",
                 json.loads(meta.read_text(encoding="utf-8"))["digest"], row["digest"])
        else:
            print(f"[census] no reference digest for {code}")

    record = {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "inputs": {args.connections.name: {"bytes": args.connections.stat().st_size,
                                           "sha256": sha256(args.connections)}},
        "gates": gates,
        "sets": results,
    }
    out = args.out or OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    failed = [row["gate"] for row in gates if not row["pass"]]
    print(f"[census] {len(gates) - len(failed)}/{len(gates)} gates pass -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
