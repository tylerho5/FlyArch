"""Build ``extras.npz``: per-neuron annotation groups for ``--annotation-groups``.

For each processed region directory, aligned to its graph's node order:

- ``supertype``/``hemilineage``: integer group ids from the FlyWire annotation table
  (``supertype``, ``ito_lee_hemilineage``), 1-based per region, 0 where missing.

Synapse positions are deliberately not used: a masked neuron's own synapse sites include the
hidden edges' sites, and the registered protocol only discloses the masked neuron's total degree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.regions import RegionGraph  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--annotations", type=Path,
        default=REPO_ROOT / "data/raw/Supplemental_file1_neuron_annotations.tsv",
    )
    parser.add_argument("processed_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)

    annotations = pd.read_csv(
        args.annotations, sep="\t", usecols=["root_id", "supertype", "ito_lee_hemilineage"],
        low_memory=False,
    ).drop_duplicates("root_id").set_index("root_id")
    for directory in args.processed_dirs:
        graph = RegionGraph.read(directory)
        aligned = annotations.reindex(graph.node_ids)
        arrays = {}
        for name, column in (("supertype", "supertype"), ("hemilineage", "ito_lee_hemilineage")):
            codes, _ = pd.factorize(aligned[column])
            arrays[name] = (codes + 1).astype(np.int64)
        np.savez_compressed(directory / "extras.npz", **arrays)
        print(f"{graph.region}: wrote {directory / 'extras.npz'} "
              f"(supertype {np.mean(arrays['supertype'] > 0):.0%}, "
              f"hemilineage {np.mean(arrays['hemilineage'] > 0):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
