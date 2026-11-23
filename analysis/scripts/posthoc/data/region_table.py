"""Per-region graph stats for F1/F2 from local processed data; checks digests against committed suites."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root, A.out
from src.regions import RegionGraph, REGIONS, region_labels
rows = []; members = {}
for code in REGIONS:
    c = code.lower()
    for ds in ("f1", "f2"):
        g = RegionGraph.read(DATA / "processed" / ds / c)
        suite = "f1_val" if ds == "f1" else "f2_confirm"
        sj = json.loads((EV / "data/suites" / ds / c / f"{suite}.json").read_text())
        types = pd.read_parquet(DATA / "interim" / ds / c / f"{ds}_types.parquet").set_index("root_id")["cell_type"].reindex(g.node_ids)
        sides = pd.read_parquet(DATA / "interim" / ds / c / f"{ds}_sides.parquet").set_index("root_id")["side"].reindex(g.node_ids)
        members[(ds, code)] = set(g.node_ids.tolist())
        rows.append(dict(region=code, dataset=ds, threshold=g.threshold, neurons=g.n_nodes, edges=g.n_edges,
                         synapses_on_edges=int(g.weights.sum()), edges_per_neuron=round(g.n_edges / g.n_nodes, 2),
                         median_edge_weight=float(np.median(g.weights)),
                         n_types=int(types.nunique()), typed_frac=round(float(types.notna().mean()), 4),
                         soma_side_known_frac=round(float(sides.notna().mean()), 4),
                         digest_matches_suite=g.digest() == sj["graph_digest"], labels=" ".join(region_labels(code))))
df = pd.DataFrame(rows)
df.to_csv(OUT / "region_table.csv", index=False)
print(df.drop(columns="labels").to_string(index=False))
# overlap: neurons in >1 regional graph, per dataset
for ds in ("f1", "f2"):
    from collections import Counter
    cnt = Counter()
    for code in REGIONS:
        cnt.update(members[(ds, code)])
    k = pd.Series(cnt)
    print(f"\n{ds}: unique neurons across 12 graphs {len(k)}, sum of graph sizes {int(k.sum())}, in >=2 graphs {int((k>=2).sum())} ({(k>=2).mean():.2%}), max graphs per neuron {int(k.max())}")
    print("distribution of #graphs per neuron:", k.value_counts().sort_index().to_dict())
    per = []
    for code in REGIONS:
        m = members[(ds, code)]
        shared = sum(1 for n in m if cnt[n] >= 2)
        per.append((code, len(m), shared, round(shared / len(m), 3)))
    print("per region: (code, n, n also in another region graph, frac)", per)
    # pairwise jaccard-ish matrix (counts)
    mat = pd.DataFrame(0, index=list(REGIONS), columns=list(REGIONS))
    for a in REGIONS:
        for b in REGIONS:
            mat.loc[a, b] = len(members[(ds, a)] & members[(ds, b)])
    mat.to_csv(OUT / f"region_overlap_{ds}.csv")
    print(mat.to_string())
