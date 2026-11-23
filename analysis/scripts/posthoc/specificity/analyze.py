"""Post hoc (new, exploratory) analysis of the stored F2-dev transfer matrix. Reads extract.py's CSVs."""
import re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__, lambda ap: ap.add_argument("--skeleton", type=Path, help="manuscript supplement .tex holding tab:s-transfer (optional cross-check)"))
EV, DATA, OUT = A.repo_root, A.data_root, A.out
ORDER = ["CX", "AL", "LH", "MB", "LX", "SNP", "INP", "VMNP", "VLNP", "PENP", "GNG", "OL"]
NT = ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine", "other"]
rng = np.random.default_rng(0)
NPERM = 2000
log = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

M = {k: pd.read_csv(OUT / f"matrix_{k}.csv", index_col=0) for k in ["raw", "home_norm", "ablated_norm", "base_norm", "geo", "oov"]}
base = pd.read_csv(OUT / "baselines.csv").set_index("region").baseline_mrr.reindex(ORDER)
cells = pd.read_csv(OUT / "cells.csv")
abl = cells[cells.variant == "ablated"].set_index("target").mrr.reindex(ORDER)
abl_geo = cells[cells.variant == "ablated_geo"].set_index("target").mrr.reindex(ORDER)
rs = pd.read_csv(OUT / "region_stats.csv")
f1 = rs[rs.dataset == "f1"].set_index("region").reindex(ORDER)
f2 = rs[rs.dataset == "f2"].set_index("region").reindex(ORDER)
pp = pd.read_csv(OUT / "pair_predictors.csv")
raw = M["raw"]; home = pd.Series(np.diag(raw), index=ORDER)
off = ~np.eye(12, dtype=bool)

# ---------- 1. verify the stated numbers ----------
p("== 1. Verification ==")
above3 = (raw.values > base.values[None, :]) & off
abovebn = (M["base_norm"].values > 0) & off
ties = (np.isclose(raw.values, base.values[None, :])) & off
p("off-diagonal cells with raw MRR > target champion baseline (3-dp values):", above3.sum(), "of", off.sum(), "; ties:", ties.sum())
p("off-diagonal cells with base_norm > 0 (full-precision sign):", abovebn.sum())
p("cells above baseline:", [(ORDER[i], ORDER[j]) for i, j in zip(*np.nonzero(abovebn))])
p("per target column, # visitors above baseline:", dict(zip(ORDER, abovebn.sum(0))))
p("per expert row, # targets where it beats baseline:", dict(zip(ORDER, abovebn.sum(1))))
hn = M["home_norm"].values[off]
recomputed = (raw.values / home.values[None, :])[off]
p(f"home-normalised ratio off-diag: min {hn.min():.3f} max {hn.max():.3f} median {np.median(hn):.4f} mean {hn.mean():.3f}")
p(f"  recomputed raw/home from 3-dp raw: min {recomputed.min():.3f} max {recomputed.max():.3f} median {np.median(recomputed):.4f}; max |diff| vs table {np.abs(recomputed - hn).max():.4f}")
p("  argmin:", ORDER[np.argwhere((M['home_norm'].values == hn.min()) & off)[0][0]], "->", ORDER[np.argwhere((M['home_norm'].values == hn.min()) & off)[0][1]],
  "; argmax:", ORDER[np.argwhere((M['home_norm'].values == hn.max()) & off)[0][0]], "->", ORDER[np.argwhere((M['home_norm'].values == hn.max()) & off)[0][1]])
# skeleton table (check against the manuscript's LaTeX source; skipped without --skeleton)
if A.skeleton:
    tex = A.skeleton.read_text(encoding="utf-8")
    blk = tex[tex.index(r"\label{tab:s-transfer}"):]
    blk = blk[:blk.index(r"\end{tabular}")]
    sk = {}
    for line in blk.splitlines():
        m = re.match(r"^(\w+) & (.*)\\\\", line.strip())
        if m and m.group(1) in ORDER:
            sk[m.group(1)] = [float(x) for x in m.group(2).split("&")]
    sk = pd.DataFrame(sk, index=ORDER).T.reindex(ORDER)
    d = (sk - M["home_norm"].round(2)).abs()
    p("skeleton tab:s-transfer vs report home_norm rounded to 2 dp: cells differing:", int((d > 0.0051).sum().sum()), "max |diff|", float(d.values.max()))
    bad = [(i, j, sk.loc[i, j], M['home_norm'].loc[i, j]) for i in ORDER for j in ORDER if d.loc[i, j] > 1e-9]
    p("  cells where skeleton != round(report,2):", bad)

# ---------- 2. diagonal dominance ----------
p("\n== 2. Diagonal dominance (column-wise: is the home model best on its own target?) ==")
dom = []
for k, j in enumerate(ORDER):
    col = raw[j].drop(j)
    gcol = M["geo"][j].drop(j)
    oc = M["oov"][j]
    lowest = oc.drop(j).idxmin()
    dom.append(dict(target=j, home=home[j], ablated_home=abl[j], best_visitor=col.idxmax(), best_visitor_mrr=col.max(),
                    margin=home[j] - col.max(), margin_vs_ablated_home=abl[j] - col.max(),
                    geo_home=abl_geo[j], geo_best_visitor=gcol.idxmax(), geo_best=gcol.max(), geo_margin=abl_geo[j] - gcol.max(),
                    home_masked_oov=oc[j], lowest_oov_visitor=lowest, lowest_visitor_oov=oc[lowest],
                    lowest_oov_visitor_ratio=M["home_norm"].loc[lowest, j], baseline=base[j]))
dom = pd.DataFrame(dom); dom.to_csv(OUT / "diagonal_dominance.csv", index=False)
p(dom.round(3).to_string(index=False))
p("home > every visitor in", int((dom.margin > 0).sum()), "of 12 columns; ablated (uniform-innervation) home > every visitor in", int((dom.margin_vs_ablated_home > 0).sum()))
p("geometry+innervation-ablated home (ablated_geo) > every transfer_geo visitor in", int((dom.geo_margin > 0).sum()), "of 12 columns")
geo_ratio = M["geo"].values / abl_geo.values[None, :]
p(f"transfer_geo / ablated_geo home, off-diag: min {geo_ratio[off].min():.3f} median {np.median(geo_ratio[off]):.3f} max {geo_ratio[off].max():.3f}; cells >= 1: {(geo_ratio[off] >= 1).sum()}")
p("  cells >= 0.9:", [(ORDER[i], ORDER[j], round(geo_ratio[i, j], 3)) for i, j in zip(*np.nonzero((geo_ratio >= 0.9) & off))])
pd.DataFrame(geo_ratio, index=ORDER, columns=ORDER).to_csv(OUT / "matrix_geo_ratio.csv")
p("innervation ablation alone, ablated/home on diagonal:", dict(zip(ORDER, (abl / home).round(3))))

# ---------- 3. build the 132-cell table ----------
pp = pp[pp.expert != pp.target].copy()
pp["ratio"] = [M["home_norm"].loc[i, j] for i, j in zip(pp.expert, pp.target)]
pp["raw"] = [raw.loc[i, j] for i, j in zip(pp.expert, pp.target)]
pp["minus_base"] = pp.raw - base.reindex(pp.target).values
pp["base_norm"] = [M["base_norm"].loc[i, j] for i, j in zip(pp.expert, pp.target)]
pp["oov_report"] = [M["oov"].loc[i, j] for i, j in zip(pp.expert, pp.target)]
pp["target_baseline"] = base.reindex(pp.target).values
pp["target_home"] = home.reindex(pp.target).values
pp["source_home"] = home.reindex(pp.expert).values
pp["target_home_minus_base"] = pp.target_home - pp.target_baseline
for s, who in (("source", "expert"), ("target", "target")):
    pp[f"{s}_log_neurons_f2"] = np.log10(f2.neurons.reindex(pp[who]).values)
    pp[f"{s}_density_f2"] = f2.density.reindex(pp[who]).values
    pp[f"{s}_mean_degree_f2"] = f2.mean_out_degree.reindex(pp[who]).values
pp["source_n_trained_types"] = f1.n_trained_types.reindex(pp.expert).values
pp["target_home_masked_oov"] = [M["oov"].loc[j, j] for j in pp.target]

def comp(df, prefix):
    v = df[[f"{prefix}_{n}" for n in NT]].to_numpy(); return v / v.sum(1, keepdims=True)
def jsd_sim(a, b):
    return 1 - np.sqrt(0.5 * (stats.entropy(a, (a + b) / 2, base=2) + stats.entropy(b, (a + b) / 2, base=2)))
NTS = {}
for name, df, pref in (("nt_sim_f1_neuron", f1, "nt"), ("nt_sim_f1_synapse", f1, "ntw"), ("nt_sim_f2_neuron", f2, "nt"), ("nt_sim_f2_synapse", f2, "ntw")):
    c = comp(df, pref)
    S = np.array([[jsd_sim(c[a], c[b]) for b in range(12)] for a in range(12)])
    NTS[name] = S
    pd.DataFrame(S, index=ORDER, columns=ORDER).to_csv(OUT / f"matrix_{name}.csv")
    pp[name] = [S[ORDER.index(i), ORDER.index(j)] for i, j in zip(pp.expert, pp.target)]
pd.DataFrame({f"F1_{n}": comp(f1, "nt")[:, k] for k, n in enumerate(NT)} | {f"F2_{n}": comp(f2, "nt")[:, k] for k, n in enumerate(NT)},
             index=ORDER).to_csv(OUT / "nt_composition_neuron.csv")
pd.DataFrame({f"F1_{n}": comp(f1, "ntw")[:, k] for k, n in enumerate(NT)} | {f"F2_{n}": comp(f2, "ntw")[:, k] for k, n in enumerate(NT)},
             index=ORDER).to_csv(OUT / "nt_composition_synapse_weighted.csv")
p("\nF1 vs F2 per-region NT composition similarity (1 - JS distance, neuron-level):",
  dict(zip(ORDER, [round(jsd_sim(a, b), 3) for a, b in zip(comp(f1, "nt"), comp(f2, "nt"))])))
pp.to_csv(OUT / "cell_table.csv", index=False)

# ---------- helpers: matrices, additive fit, QAP ----------
ri = np.array([ORDER.index(x) for x in pp.expert]); ci = np.array([ORDER.index(x) for x in pp.target])
X0 = np.column_stack([np.ones(len(pp)), (ri[:, None] == np.arange(1, 12)).astype(float), (ci[:, None] == np.arange(1, 12)).astype(float)])
Xr = X0[:, :12]; Xc = np.column_stack([X0[:, :1], X0[:, 12:]])
def resid(y, X): return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
def r2(y, X): return 1 - resid(y, X).var() / y.var()
def as_mat(col):
    m = np.full((12, 12), np.nan); m[ri, ci] = pp[col].to_numpy(); return m
def from_mat(m, perm=None):
    if perm is not None: m = m[np.ix_(perm, perm)]
    return m[ri, ci]
def rank(v): return stats.rankdata(v)
def qap(y, xmat, f, n=NPERM):
    obs = f(y, from_mat(xmat)); ge = 1
    for _ in range(n):
        if abs(f(y, from_mat(xmat, rng.permutation(12)))) >= abs(obs) - 1e-12: ge += 1
    return obs, ge / (n + 1)
sp = lambda a, b: stats.spearmanr(a, b).statistic
def sp_within(a, b):  # Spearman after removing row and column main effects from both ranks
    return stats.pearsonr(resid(rank(a), X0), resid(rank(b), X0)).statistic

# ---------- 4. two-way decomposition ----------
p("\n== 4. Two-way additive decomposition over 132 off-diagonal cells ==")
dec = []
for col in ["ratio", "raw", "minus_base", "base_norm"]:
    for tr in ("identity", "log"):
        if tr == "log" and (pp[col] <= 0).any(): continue
        y = pp[col].to_numpy(); y = np.log(y) if tr == "log" else y
        a, rr, cc = r2(y, X0), r2(y, Xr), r2(y, Xc)
        dec.append(dict(outcome=col, transform=tr, R2_source_only=rr, R2_target_only=cc, R2_additive=a,
                        unique_source=a - cc, unique_target=a - rr, interaction_residual=1 - a))
dec = pd.DataFrame(dec); dec.to_csv(OUT / "two_way_decomposition.csv", index=False)
p(dec.round(3).to_string(index=False))
# permutation null for additive R2 of ratio is meaningless (row/col exist by construction); report residual SD
y = pp.ratio.to_numpy(); e = resid(y, X0)
p(f"ratio: SD {y.std():.4f}, additive-residual SD {e.std():.4f}; residual max |e| {np.abs(e).max():.3f}")
pp["ratio_additive_resid"] = e
pp["log_ratio_additive_resid"] = resid(np.log(y), X0)
top = pp.assign(ae=np.abs(e)).sort_values("ae", ascending=False).head(10)[["expert", "target", "ratio", "ratio_additive_resid", "oov_report", "shared_f2_frac_of_target", "nt_sim_f1_neuron"]]
p("largest interaction residuals (ratio):\n" + top.round(3).to_string(index=False))
# row / column effects
rowm = pd.Series([pp[pp.expert == r].ratio.mean() for r in ORDER], index=ORDER)
colm = pd.Series([pp[pp.target == r].ratio.mean() for r in ORDER], index=ORDER)
p("source (row) mean ratio:", rowm.round(3).to_dict())
p("target (column) mean ratio:", colm.round(3).to_dict())

# ---------- 5. correlations ----------
p("\n== 5. Spearman correlations over the 132 off-diagonal cells (non-independent; QAP p for pair-level predictors) ==")
pair_preds = ["oov_report", "node_coverage", "edge_both_inv", "type_jaccard_trained", "target_types_covered",
              "shared_f2_frac_of_target", "shared_f1_frac_of_target", "nt_sim_f1_neuron", "nt_sim_f1_synapse", "nt_sim_f2_neuron", "nt_sim_f2_synapse"]
region_preds = ["target_baseline", "target_home", "target_home_minus_base", "target_home_masked_oov", "source_home",
                "source_n_trained_types", "source_log_neurons_f2", "target_log_neurons_f2", "source_density_f2", "target_density_f2",
                "source_mean_degree_f2", "target_mean_degree_f2"]
rows = []
for out in ["ratio", "raw", "minus_base"]:
    for x in pair_preds:
        rho, pq = qap(pp[out].to_numpy(), as_mat(x), sp)
        rw, pw = qap(pp[out].to_numpy(), as_mat(x), sp_within)
        rows.append(dict(outcome=out, predictor=x, level="pair", n=132, spearman=rho, qap_p=pq,
                         within_row_col_rho=rw, within_qap_p=pw))
    for x in region_preds:
        rho = sp(pp[out], pp[x])
        rows.append(dict(outcome=out, predictor=x, level="region (n_eff=12)", n=132, spearman=rho, qap_p=np.nan,
                         within_row_col_rho=np.nan, within_qap_p=np.nan))
corr = pd.DataFrame(rows); corr.to_csv(OUT / "correlations_cells.csv", index=False)
p(corr.round(3).to_string(index=False))

p("\n-- region-level (n=12) Spearman: column mean of off-diagonal outcome vs target property; row mean vs source property --")
reg = pd.DataFrame(index=ORDER)
for out in ["ratio", "raw", "minus_base"]:
    reg[f"col_mean_{out}"] = [pp[pp.target == r][out].mean() for r in ORDER]
    reg[f"row_mean_{out}"] = [pp[pp.expert == r][out].mean() for r in ORDER]
reg["baseline"] = base; reg["home"] = home; reg["home_minus_base"] = home - base; reg["ablated_home"] = abl
reg["home_masked_oov"] = np.diag(M["oov"].values); reg["f2_typed_frac"] = f2.typed_frac
reg["log_neurons_f2"] = np.log10(f2.neurons); reg["density_f2"] = f2.density; reg["mean_degree_f2"] = f2.mean_out_degree
reg["n_types_f2"] = f2.n_types; reg["n_trained_types"] = f1.n_trained_types; reg["eff_n_types_f2"] = f2.eff_n_types
reg["mean_cover_as_target"] = [pp[pp.target == r].node_coverage.mean() for r in ORDER]
reg["mean_cover_as_source"] = [pp[pp.expert == r].node_coverage.mean() for r in ORDER]
reg.to_csv(OUT / "region_table.csv")
rrows = []
for out in ["ratio", "raw", "minus_base"]:
    for side, props in (("col", ["baseline", "home", "home_minus_base", "home_masked_oov", "f2_typed_frac", "log_neurons_f2", "density_f2",
                                 "mean_degree_f2", "n_types_f2", "mean_cover_as_target"]),
                        ("row", ["home", "n_trained_types", "log_neurons_f2", "density_f2", "mean_degree_f2", "mean_cover_as_source"])):
        for x in props:
            r_ = stats.spearmanr(reg[f"{side}_mean_{out}"], reg[x])
            rrows.append(dict(outcome=f"{side}_mean_{out}", predictor=x, n=12, spearman=r_.statistic, p=r_.pvalue))
rcorr = pd.DataFrame(rrows); rcorr.to_csv(OUT / "correlations_region.csv", index=False)
p(rcorr.round(3).to_string(index=False))

# ---------- 6. OOV extrapolation within target column ----------
p("\n== 6. Can masked-OOV alone account for the home advantage? (within-column fit, extrapolated to home OOV) ==")
for xname in ["oov_report"]:
    Xf = np.column_stack([Xc, pp[xname].to_numpy()])
    for out in ["ratio", "raw"]:
        coef = np.linalg.lstsq(Xf, pp[out].to_numpy(), rcond=None)[0]
        beta = coef[-1]
        pred_home = []
        for k, j in enumerate(ORDER):
            row = np.zeros(Xf.shape[1]); row[0] = 1
            if k > 0: row[k] = 1
            row[-1] = M["oov"].loc[j, j]
            pred_home.append(row @ coef)
        pred_home = np.array(pred_home)
        actual = np.ones(12) if out == "ratio" else home.values
        p(f"{out} ~ target FE + {xname}: slope {beta:.3f}; R2 {r2(pp[out].to_numpy(), Xf):.3f} (target FE only {r2(pp[out].to_numpy(), Xc):.3f})")
        p(f"   predicted {out} at each target's home OOV vs actual home: " +
          ", ".join(f"{j} {a:.2f}/{b:.2f}" for j, a, b in zip(ORDER, pred_home, actual)))
        p(f"   mean predicted/actual at home OOV: {np.mean(pred_home / actual):.3f}")
lo = pp[pp.oov_report <= 0.30]
p("visitor cells with masked OOV <= 0.30:", len(lo)); p(lo[["expert", "target", "oov_report", "target_home_masked_oov", "ratio", "raw", "target_baseline"]].round(3).to_string(index=False))

# ---------- 7. asymmetry ----------
p("\n== 7. Asymmetry ==")
R = M["home_norm"].values; Rw = M["raw"].values
iu = np.triu_indices(12, 1)
p(f"Spearman(ratio[i,j], ratio[j,i]) over 66 unordered pairs: {sp(R[iu], R.T[iu]):.3f}; raw: {sp(Rw[iu], Rw.T[iu]):.3f}")
asym = R - R.T
Cv = as_mat("node_coverage"); np.fill_diagonal(Cv, 1)
Ov = M["oov"].values
p(f"mean |ratio_ij - ratio_ji| = {np.abs(asym[iu]).mean():.3f}; max {np.abs(asym[iu]).max():.3f}")
for nm, A in (("coverage C_ij - C_ji", Cv - Cv.T), ("masked OOV oov_ij - oov_ji", Ov - Ov.T),
              ("target home h_j - h_i", home.values[None, :] - home.values[:, None]),
              ("target baseline b_j - b_i", base.values[None, :] - base.values[:, None]),
              ("source home h_i - h_j", home.values[:, None] - home.values[None, :])):
    p(f"  Spearman(ratio_ij - ratio_ji, {nm}) over 66 pairs: {sp(asym[iu], A[iu]):.3f}")
pairs_tbl = pd.DataFrame([dict(a=ORDER[a], b=ORDER[b], ratio_ab=R[a, b], ratio_ba=R[b, a], raw_ab=Rw[a, b], raw_ba=Rw[b, a],
                               oov_ab=Ov[a, b], oov_ba=Ov[b, a]) for a, b in zip(*iu)])
pairs_tbl.to_csv(OUT / "asymmetry_pairs.csv", index=False)
p("spec 'first look' pairs:\n" + pairs_tbl[pairs_tbl.apply(lambda r: {r.a, r.b} in ({"AL", "LH"}, {"CX", "LX"}), axis=1)].round(3).to_string(index=False))

# ---------- 8. NT similarity, controlling for vocabulary overlap ----------
p("\n== 8. NT-composition similarity (post hoc) ==")
def partial_within(y, xm, zm, perm=None):
    x = from_mat(xm, perm); z = from_mat(zm)
    D = np.column_stack([X0, rank(z)])
    return stats.pearsonr(resid(rank(y), D), resid(rank(x), D)).statistic
for out in ["ratio", "raw"]:
    for nm in NTS:
        xm = NTS[nm]
        a, pa = qap(pp[out].to_numpy(), xm, sp)
        b, pb = qap(pp[out].to_numpy(), xm, sp_within)
        yv = pp[out].to_numpy(); zm = as_mat("node_coverage")
        obs = partial_within(yv, xm, zm); ge = 1
        for _ in range(NPERM):
            if abs(partial_within(yv, xm, zm, rng.permutation(12))) >= abs(obs) - 1e-12: ge += 1
        oz = as_mat("oov_report"); obs2 = partial_within(yv, xm, oz)
        p(f"{out} vs {nm}: rho {a:.3f} (QAP p {pa:.3f}); within row/col {b:.3f} (p {pb:.3f}); "
          f"within row/col + coverage partial {obs:.3f} (QAP p {ge/(NPERM+1):.3f}); + masked-OOV partial {obs2:.3f}")
p("correlation between NT similarity and type-vocabulary coverage (pairs):",
  {nm: round(sp(pp[nm], pp.node_coverage), 3) for nm in NTS})

(OUT / "analysis_output.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
