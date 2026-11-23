"""Summarize the type-mate heuristic against the gate cohort (and champion where per-instance rows are local)."""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root, A.out
SCORED = DEV_SCORED = A.results_root
DEV_SCORED = A.results_root
from src.gate import _bootstrap_ci, PreregConfig
from scipy import stats
PRE = PreregConfig(delta=0.0)
REGIONS = ["OL", "LX", "SNP", "INP", "VMNP", "VLNP", "CX", "MB", "AL", "LH", "PENP", "GNG"]
CHAMP = {r: ("logreg" if r == "GNG" else "xgboost") for r in REGIONS}
# Registered confirm-score record: mean log2(model/champion) on F2-confirm (documented)
RBAR = dict(CX=0.7526, LH=0.4750, LX=0.9474, SNP=0.6314, VMNP=1.0564, PENP=0.5076, AL=0.8313, MB=0.6712, INP=0.8677, GNG=0.9660, VLNP=0.5815, OL=0.4123)
DEV_CHAMP_DOC = dict(AL=0.313, CX=0.386, GNG=0.302, INP=0.326, LH=0.315, LX=0.261, MB=0.304, OL=0.449, PENP=0.361, SNP=0.360, VLNP=0.389, VMNP=0.304)
dev_model = pd.read_csv(EV / "results/phase3/f2dev-diagnosis/f2dev-instances.csv").set_index("instance")["ref_real"]

def ci(r):
    lo, hi, m, _ = _bootstrap_ci(np.asarray(r, float), PRE)
    pos, ties = int((r > 0).sum()), int((r == 0).sum())
    p = stats.binomtest(pos, len(r) - ties, 0.5, alternative="greater").pvalue
    return f"{r.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", pos, p

def champ_rows(code, root, split):
    fs = glob.glob(str(root / code.lower() / "scored" / "*.scores.parquet"))
    if not fs: return None
    s = pd.concat(pd.read_parquet(f) for f in fs)
    s = s[(s.split == split) & (s.model == CHAMP[code])]
    return s.groupby("mask_instance").mrr.mean() if len(s) else None

rows, reg_rows = [], []
for code in REGIONS:
    t = pd.read_csv(OUT / f"typemate_{code.lower()}.csv")
    ref = json.loads((EV / f"results/phase3/reference/{code}.json").read_text())["reference"]
    conf_model = pd.DataFrame(ref).mean(axis=1)
    t["model"] = np.where(t.suite == "f2_confirm", t.instance.map(conf_model), t.instance.map(dev_model))
    cc = champ_rows(code, SCORED, "f2_confirm"); cd = champ_rows(code, DEV_SCORED, "f2_dev")
    t["champion"] = np.nan
    if cc is not None: t.loc[t.suite == "f2_confirm", "champion"] = t.instance.map(cc)
    if cd is not None: t.loc[t.suite == "f2_dev", "champion"] = t.instance.map(cd)
    t["isT"] = t.regime == "T"
    rows.append(t)
    c = t[t.suite == "f2_confirm"]; d = t[t.suite == "f2_dev"]
    assert len(c) == 20 and len(d) == 15 and c.model.notna().all() and d.model.notna().all()
    r1 = np.log2(c.model / c.h1_mrr).to_numpy(); r2 = np.log2(c.model / c.h2_mrr).to_numpy()
    s1, w1, p1 = ci(r1); s2, w2, p2 = ci(r2)
    row = dict(region=code,
               dev_H1=round(d.h1_mrr.mean(), 3), dev_H2=round(d.h2_mrr.mean(), 3), dev_model=round(d.model.mean(), 3),
               dev_champion=round(d.champion.mean(), 3) if d.champion.notna().all() else f"{DEV_CHAMP_DOC[code]} (doc)",
               conf_H1=round(c.h1_mrr.mean(), 3), conf_H2=round(c.h2_mrr.mean(), 3), conf_model=round(c.model.mean(), 3),
               conf_champion=round(c.champion.mean(), 3) if c.champion.notna().all() else "n/a",
               log2_model_vs_H1=s1, wins_vs_H1=w1, p_sign_H1=f"{p1:.2g}", log2_model_vs_H2=s2, wins_vs_H2=w2,
               rbar_model_vs_champ=RBAR[code], H1_vs_champ_implied=round(RBAR[code] - r1.mean(), 3),
               H1_share_of_margin=round((RBAR[code] - r1.mean()) / RBAR[code], 2),
               conf_rows_with_typemate=round((c.share_rows_with_typemate * c.gated_rows).sum() / c.gated_rows.sum(), 3),
               conf_T_H1=round(c[c.isT].h1_mrr.mean(), 3), conf_T_model=round(c[c.isT].model.mean(), 3),
               conf_nonT_H1=round(c[~c.isT].h1_mrr.mean(), 3), conf_nonT_model=round(c[~c.isT].model.mean(), 3),
               conf_T_log2_model_vs_H1=round(float(np.log2(c[c.isT].model / c[c.isT].h1_mrr).mean()), 3),
               conf_nonT_log2_model_vs_H1=round(float(np.log2(c[~c.isT].model / c[~c.isT].h1_mrr).mean()), 3))
    if c.champion.notna().all():
        rc = np.log2(c.champion / c.h1_mrr).to_numpy(); sc_, wc, _ = ci(rc)
        row["log2_champ_vs_H1_direct"] = sc_; row["champ_wins_vs_H1"] = wc
        row["log2_model_vs_champ_check"] = round(float(np.log2(c.model / c.champion).mean()), 4)
    reg_rows.append(row)
allr = pd.concat(rows); allr.to_csv(OUT / "typemate_instances_all.csv", index=False)
summ = pd.DataFrame(reg_rows); summ.to_csv(OUT / "typemate_summary.csv", index=False)
pd.set_option("display.width", 250)
for cols in (["region", "dev_H1", "dev_H2", "dev_model", "dev_champion", "conf_H1", "conf_H2", "conf_model", "conf_champion"],
             ["region", "log2_model_vs_H1", "wins_vs_H1", "p_sign_H1", "log2_model_vs_H2", "wins_vs_H2", "rbar_model_vs_champ", "H1_vs_champ_implied", "H1_share_of_margin"],
             ["region", "conf_rows_with_typemate", "conf_T_H1", "conf_T_model", "conf_nonT_H1", "conf_nonT_model", "conf_T_log2_model_vs_H1", "conf_nonT_log2_model_vs_H1"],
             [c for c in ["region", "log2_champ_vs_H1_direct", "champ_wins_vs_H1", "log2_model_vs_champ_check"] if c in summ]):
    print(summ[cols].to_string(index=False)); print()
