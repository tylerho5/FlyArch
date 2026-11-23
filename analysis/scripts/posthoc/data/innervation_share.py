"""Per region: share of a neuron's in-region synapse ends (pre+post innervation counts, the features' source)
that sit on region-graph edges (the granted degree-given total). Median over graph nodes."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__, lambda ap: ap.add_argument("which", choices=["f1", "f2"]))
EV, DATA, OUT = A.repo_root, A.data_root, A.out
from src.regions import RegionGraph, REGIONS, region_labels
from scripts.build_features import read_innervation
which = A.which
rows = []
codes = [c for c in REGIONS if not (which == "f2" and c == "OL")]
for code in codes:
    g = RegionGraph.read(DATA / "processed" / which / code.lower())
    labels = region_labels(code)
    if which == "f1":
        inn = read_innervation([DATA / "raw/per_neuron_neuropil_count_pre_783.feather", DATA / "raw/per_neuron_neuropil_count_post_783.feather"], labels)
    else:
        inn = read_innervation([DATA / "raw/banc/banc_888_synapses_per_neuron_neuropil_side_v2.parquet"], labels, banc_region=code)
    tot = inn.reindex(columns=[l for l in labels if l in inn.columns]).sum(axis=1).reindex(g.node_ids).fillna(0).to_numpy()
    granted = (g.in_strength() + g.out_strength()).astype(float)
    ok = tot > 0
    share = granted[ok] / tot[ok]
    rows.append(dict(region=code, dataset=which, n_labels=len(labels), nodes=g.n_nodes, nodes_with_innervation=int(ok.sum()),
                     median_share_on_edges=round(float(np.median(share)), 3), p10=round(float(np.quantile(share, .1)), 3),
                     p90=round(float(np.quantile(share, .9)), 3), share_gt1=int((share > 1.0001).sum()),
                     total_edge_synapse_ends=int(granted.sum()), total_innervation=int(tot.sum())))
    print(rows[-1], flush=True)
    del inn
pd.DataFrame(rows).to_csv(OUT / f"innervation_share_{which}.csv", index=False)
