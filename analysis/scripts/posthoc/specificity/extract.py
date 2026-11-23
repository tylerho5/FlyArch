"""Post hoc (new, exploratory) extraction for the transfer-matrix confound analysis.

Reads only: analysis/data/transfer_matrix/ and committed F2-dev suites,
processed graphs/features (<data-root>/processed), and the gate-seed checkpoints'
`trained_types` buffer (<data-root>/runs). Writes CSVs to --out.
No model inference is run.
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root, A.out
ORDER = ["CX", "AL", "LH", "MB", "LX", "SNP", "INP", "VMNP", "VLNP", "PENP", "GNG", "OL"]
NT = ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine", "other"]

# ---- 1. matrices, champion baselines and cells from the committed tables ----
import shutil
for f in sorted((EV / "analysis/data/transfer_matrix").glob("*.csv")):
    shutil.copy(f, OUT / f.name)

# ---- 2. trained type vocabularies from gate checkpoints ----
reg = pd.read_csv(EV / "results/experts.csv", dtype=str)
trained = {}
for r in ORDER:
    sets = []
    for run in reg[(reg.region == r) & (reg.gate_seed == "true")].run_id:
        t = torch.load(DATA / "runs" / run / "best.pt", map_location="cpu", weights_only=True)["model"]["trained_types"]
        sets.append(frozenset(np.flatnonzero(t.numpy())) - {0})
    assert len(sets) == 3 and len(set(sets)) == 1, r  # identical across the three gate seeds
    trained[r] = sets[0]

# ---- 3. per-region graph, type, NT stats ----
def load(ds, r):
    p = DATA / "processed" / ds / r.lower()
    nodes = pd.read_parquet(p / "nodes.parquet").root_id.to_numpy()
    edges = pd.read_parquet(p / "edges.parquet")
    feat = pd.read_parquet(p / "features.parquet").set_index("root_id").reindex(nodes)
    return nodes, edges, feat

G = {}
rows = []
for r in ORDER:
    for ds in ("f1", "f2"):
        nodes, edges, feat = load(ds, r)
        tid = np.rint(feat.cell_type.to_numpy()).astype(int)
        typed = tid > 0
        sizes = pd.Series(tid[typed]).value_counts()
        ntm = feat[[f"nt_{n}" for n in NT]].to_numpy()
        has_nt = ntm.sum(1) > 0
        comp = ntm[has_nt].mean(0)
        w = np.bincount(edges.source.to_numpy(), weights=edges.weight.to_numpy(), minlength=len(nodes))
        compw = (ntm[has_nt] * w[has_nt, None]).sum(0) / w[has_nt].sum()
        n, e = len(nodes), len(edges)
        G[(ds, r)] = dict(nodes=nodes, tid=tid, edges=edges)
        p = sizes / sizes.sum()
        row = dict(region=r, dataset=ds, neurons=n, edges=e, density=e / (n * (n - 1)), mean_out_degree=e / n,
                   typed_frac=typed.mean(), n_types=len(sizes), type_size_median=sizes.median(),
                   type_size_mean=sizes.mean(), type_size_max=sizes.max(), singleton_type_frac=(sizes == 1).mean(),
                   neurons_in_singleton_types=sizes[sizes == 1].sum() / n, eff_n_types=float(np.exp(-(p * np.log(p)).sum())),
                   nt_present_frac=has_nt.mean(),
                   **{f"nt_{x}": c for x, c in zip(NT, comp)}, **{f"ntw_{x}": c for x, c in zip(NT, compw)})
        if ds == "f2":
            row["home_trained_coverage"] = np.isin(tid, list(trained[r])).mean()
        else:
            row["n_trained_types"] = len(trained[r])
        rows.append(row)
stats = pd.DataFrame(rows)
stats.to_csv(OUT / "region_stats.csv", index=False)

# ---- 4. pairwise predictors (row i = expert/source, column j = target) ----
suites = {r: json.loads((EV / f"data/suites/f2/{r.lower()}/f2_dev.json").read_text(encoding="utf-8")) for r in ORDER}
pairs = []
for i in ORDER:
    Ti = np.array(sorted(trained[i]))
    for j in ORDER:
        f2 = G[("f2", j)]
        inv = np.isin(f2["tid"], Ti)
        e = f2["edges"]
        both = inv[e.source.to_numpy()] & inv[e.target.to_numpy()]
        idx = pd.Index(f2["nodes"])
        per_inst = [1 - inv[idx.get_indexer(inst["masked_root_ids"])].mean() for inst in suites[j]["instances"]]
        Tj_f2 = set(np.unique(f2["tid"][f2["tid"] > 0]))
        pairs.append(dict(
            expert=i, target=j,
            masked_oov_recomputed=float(np.mean(per_inst)),
            node_coverage=inv.mean(),                     # share of j's F2 neurons whose type i trained on
            edge_both_inv=both.mean(),                    # share of j's F2 edges with both ends in i's vocab
            type_jaccard_trained=len(trained[i] & trained[j]) / len(trained[i] | trained[j]),
            target_types_covered=len(Tj_f2 & trained[i]) / len(Tj_f2),
            shared_f2_frac_of_target=np.isin(f2["nodes"], G[("f2", i)]["nodes"]).mean(),
            shared_f1_frac_of_target=np.isin(G[("f1", j)]["nodes"], G[("f1", i)]["nodes"]).mean(),
            shared_f1_count=int(np.isin(G[("f1", j)]["nodes"], G[("f1", i)]["nodes"]).sum()),
        ))
pd.DataFrame(pairs).to_csv(OUT / "pair_predictors.csv", index=False)
print("done")
