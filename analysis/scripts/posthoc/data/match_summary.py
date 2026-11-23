"""Summarize the match-table transfer baseline (post hoc, unregistered, 2026-11-17)."""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root, A.out
SCORED = A.results_root
from src.gate import _bootstrap_ci, PreregConfig
from scipy import stats
PRE = PreregConfig(delta=0.0)
REGIONS = ["OL", "LX", "SNP", "INP", "VMNP", "VLNP", "CX", "MB", "AL", "LH", "PENP", "GNG"]
dev_model = pd.read_csv(EV / "results/phase3/f2dev-diagnosis/f2dev-instances.csv").set_index("instance")["ref_real"]
def ci(r):
    lo, hi, _, _ = _bootstrap_ci(np.asarray(r, float), PRE); pos, ties = int((r > 0).sum()), int((r == 0).sum())
    return f"{r.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", pos
summ, restr, regime = [], [], []
for code in REGIONS:
    t = pd.read_csv(OUT / f"match_{code.lower()}.csv")
    conf_model = pd.DataFrame(json.loads((EV / f"results/phase3/reference/{code}.json").read_text())["reference"]).mean(axis=1)
    t["model"] = np.where(t.suite == "f2_confirm", t.instance.map(conf_model), t.instance.map(dev_model))
    c, d = t[t.suite == "f2_confirm"], t[t.suite == "f2_dev"]; assert len(c) == 20 and len(d) == 15
    w = c.gated_rows / c.gated_rows.sum()
    s1, w1 = ci(np.log2(c.model / c.m1_mrr).to_numpy()); sc, wc = ci(np.log2(c.model / c.comb_mrr).to_numpy())
    summ.append(dict(region=code, dev_M1=round(d.m1_mrr.mean(), 3), dev_M2=round(d.m2_mrr.mean(), 3), dev_comb=round(d.comb_mrr.mean(), 3), dev_model=round(d.model.mean(), 3),
                     conf_M1=round(c.m1_mrr.mean(), 3), conf_M2=round(c.m2_mrr.mean(), 3), conf_comb=round(c.comb_mrr.mean(), 3), conf_model=round(c.model.mean(), 3),
                     log2_model_vs_M1=s1, wins_vs_M1=w1, log2_model_vs_comb=sc, wins_vs_comb=wc,
                     u_matched=round(float((c.u_matched * w).sum()), 3), v_matched=round(float((c.v_matched * w).sum()), 3),
                     both_matched=round(float((c.both_matched * w).sum()), 3), true_pair_has_F1_synapse=round(float((c.true_pair_f1_synapse_gt0 * w).sum()), 3)))
    for isT, g in c.groupby(c.regime == "T"):
        regime.append(dict(region=code, T=isT, M1=round(g.m1_mrr.mean(), 3), comb=round(g.comb_mrr.mean(), 3), model=round(g.model.mean(), 3),
                           log2_model_vs_M1=round(float(np.log2(g.model / g.m1_mrr).mean()), 3), log2_model_vs_comb=round(float(np.log2(g.model / g.comb_mrr).mean()), 3),
                           M1_beats_model=int((g.m1_mrr > g.model).sum()), comb_beats_model=int((g.comb_mrr > g.model).sum()), n=len(g)))
    R = pd.read_parquet(OUT / f"match_rows_{code.lower()}.parquet"); both = R.u_matched & R.v_matched
    row = dict(region=code, rows=len(R), both_matched_rows=int(both.sum()),
               M1_all=round(R.rr_m1.mean(), 3), M1_both=round(R[both].rr_m1.mean(), 3), comb_both=round(R[both].rr_comb.mean(), 3), H1_both=round(R[both].rr_h1.mean(), 3))
    fs = glob.glob(str(SCORED / code.lower() / "scored" / "*.positives.parquet"))
    if fs:
        pos = pd.concat(pd.read_parquet(f, columns=["mask_instance", "positive_index", "model", "reciprocal_rank", "both_masked", "split"]) for f in fs)
        pos = pos[(pos.split == "f2_confirm") & ~pos.both_masked]
        m = pos.groupby(["mask_instance", "positive_index", "model"]).reciprocal_rank.mean().unstack("model").reset_index()
        J = R.merge(m, on=["mask_instance", "positive_index"], how="inner"); assert len(J) == len(R)
        jb = J[J.u_matched & J.v_matched]
        row.update(model_all=round(J.gnn_a4.mean(), 3), model_both=round(jb.gnn_a4.mean(), 3), xgboost_both=round(jb.xgboost.mean(), 3),
                   logreg_both=round(jb.logreg.mean(), 3), configuration_both=round(jb.configuration.mean(), 3),
                   rows_M1_beats_model=round(float((jb.rr_m1 > jb.gnn_a4).mean()), 3), rows_model_beats_M1=round(float((jb.gnn_a4 > jb.rr_m1).mean()), 3))
    restr.append(row)
S, Rr, G = pd.DataFrame(summ), pd.DataFrame(restr), pd.DataFrame(regime)
S.to_csv(OUT / "match_summary.csv", index=False); Rr.to_csv(OUT / "match_restricted_rows.csv", index=False); G.to_csv(OUT / "match_by_regime.csv", index=False)
pd.set_option("display.width", 260)
print(S.drop(columns=["log2_model_vs_comb", "wins_vs_comb"]).to_string(index=False)); print()
print(S[["region", "log2_model_vs_comb", "wins_vs_comb"]].to_string(index=False)); print()
print(Rr.to_string(index=False)); print(); print(G.to_string(index=False))
