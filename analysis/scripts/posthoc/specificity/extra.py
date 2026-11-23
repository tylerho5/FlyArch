"""Follow-up checks on cell_table.csv (post hoc): collinearity, shared-neuron vs coverage partials, geo-ablated ratio."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root, A.out
ORDER = ["CX", "AL", "LH", "MB", "LX", "SNP", "INP", "VMNP", "VLNP", "PENP", "GNG", "OL"]
rng = np.random.default_rng(1); NPERM = 2000
pp = pd.read_csv(OUT / "cell_table.csv")
raw = pd.read_csv(OUT / "matrix_raw.csv", index_col=0); base = pd.read_csv(OUT / "baselines.csv").set_index("region").baseline_mrr
geo = pd.read_csv(OUT / "matrix_geo_ratio.csv", index_col=0)
pp["geo_ratio"] = [geo.loc[i, j] for i, j in zip(pp.expert, pp.target)]
ri = np.array([ORDER.index(x) for x in pp.expert]); ci = np.array([ORDER.index(x) for x in pp.target])
X0 = np.column_stack([np.ones(len(pp)), (ri[:, None] == np.arange(1, 12)).astype(float), (ci[:, None] == np.arange(1, 12)).astype(float)])
def resid(y, X): return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
rank = stats.rankdata
def mat(col):
    m = np.full((12, 12), np.nan); m[ri, ci] = pp[col].to_numpy(); return m
def partial(y, x, z):
    D = np.column_stack([X0] + [rank(v) for v in z])
    return stats.pearsonr(resid(rank(y), D), resid(rank(x), D)).statistic
def qap_partial(ycol, xcol, zcols):
    y = pp[ycol].to_numpy(); xm = mat(xcol); z = [pp[c].to_numpy() for c in zcols]
    obs = partial(y, xm[ri, ci], z); ge = 1
    for _ in range(NPERM):
        p = rng.permutation(12); xp = xm[np.ix_(p, p)]
        if abs(partial(y, xp[ri, ci], z)) >= abs(obs) - 1e-12: ge += 1
    return obs, ge / (NPERM + 1)
lines = []
P = lambda *a: (print(*a), lines.append(" ".join(map(str, a))))
preds = ["oov_report", "node_coverage", "edge_both_inv", "type_jaccard_trained", "shared_f2_frac_of_target", "shared_f1_frac_of_target"]
P("Spearman intercorrelation of pair predictors (132 cells):")
P(pp[preds].corr(method="spearman").round(3).to_string())
for y in ["ratio", "raw"]:
    for x, z in (("shared_f2_frac_of_target", ["node_coverage"]), ("node_coverage", ["shared_f2_frac_of_target"]),
                 ("shared_f1_frac_of_target", ["node_coverage"]), ("nt_sim_f1_synapse", ["node_coverage", "shared_f2_frac_of_target"]),
                 ("nt_sim_f1_neuron", ["node_coverage", "shared_f2_frac_of_target"])):
        r, p = qap_partial(y, x, z)
        P(f"{y}: partial (row/col removed) of {x} | {z}: {r:.3f}, QAP p {p:.4f}")
P("\ngeometry+innervation-ablated ratio (transfer_geo / ablated_geo home):")
for x in ["oov_report", "node_coverage", "shared_f2_frac_of_target", "nt_sim_f1_neuron"]:
    P(f"  Spearman(geo_ratio, {x}) = {stats.spearmanr(pp.geo_ratio, pp[x]).statistic:.3f}")
P("  geo_ratio vs raw ratio Spearman:", round(stats.spearmanr(pp.geo_ratio, pp.ratio).statistic, 3))
off = pp
tie = [(i, j) for i, j in zip(off.expert, off.target) if abs(raw.loc[i, j] - base[j]) < 1e-9]
P("\noff-diagonal cells tied with the baseline at 3 dp:", tie, [(raw.loc[i, j], base[j]) for i, j in tie])
P("cells with ratio >= 0.80:", off[off.ratio >= 0.8][["expert", "target", "ratio", "oov_report", "node_coverage"]].round(3).values.tolist())
(OUT / "extra_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
