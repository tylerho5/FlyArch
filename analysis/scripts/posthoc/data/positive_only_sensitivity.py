"""Post hoc, unregistered (2026-11-17): drop F2-confirm gated rows whose true visible partner has zero visible
strength on the replaced side (it can never be drawn as a decoy) and recompute model/baseline MRR and the gate statistic.
Reads the scored per-positive rows of the gate reads (CX, LH, LX, MB) from <results-root>/<r>/scored."""
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
from src.regions import RegionGraph
from src.suites import Suite
from src.masks import apply_mask
from src.evaluate import CommittedNegatives
from src.gate import _bootstrap_ci, PreregConfig
from scipy import stats
PRE = PreregConfig(delta=0.0)
CHAMPION = "xgboost"  # registered champion for CX, LH, LX, MB (registered confirm-score record)

def gate_stat(model, champ):
    r = np.log2(model / champ)
    lo, hi, method, _ = _bootstrap_ci(r, PRE)
    pos, ties = int((r > 0).sum()), int((r == 0).sum())
    p = stats.binomtest(pos, len(r) - ties, 0.5, alternative="greater").pvalue if len(r) > ties else 1.0
    return dict(mean_log2=round(float(r.mean()), 4), ci_low=round(lo, 4), ci_high=round(hi, 4), ci_method=method, wins=pos, n=len(r), p_sign=float(p))

summary, inst_rows = [], []
for code in ("CX", "LH", "LX", "MB"):
    c = code.lower()
    g = RegionGraph.read(DATA / "processed/f2" / c)
    suite = Suite.read(EV / "data/suites/f2" / c / "f2_confirm.json")
    committed = CommittedNegatives.read(EV / "data/suites/f2" / c / "f2_confirm.negatives.json")
    pos = pd.concat([pd.read_parquet(f) for f in glob.glob(str(SCORED / c / "scored" / "*.positives.parquet"))])
    sc = pd.concat([pd.read_parquet(f) for f in glob.glob(str(SCORED / c / "scored" / "*.scores.parquet"))])
    pos = pos[pos.split == "f2_confirm"]; sc = sc[sc.split == "f2_confirm"]
    assert set(pos.scoring_graph_digest) == {g.digest()}, "graph digest"
    for iid, d in pos.groupby("mask_instance").negatives_digest.unique().items():
        assert list(d) == [committed.digest_for(iid)], f"negatives digest {iid}"
    flags = {}
    for inst in suite.resolve(g):
        v = apply_mask(g, inst); e = v.hidden_edges; m = v.is_masked
        w = np.where(m[e[:, 0]], v.visible_in_strength()[e[:, 1]], v.visible_out_strength()[e[:, 0]])
        flags[inst.instance_id] = (w == 0) & ~v.both_masked()
    pos["zero_partner"] = [flags[i][j] for i, j in zip(pos.mask_instance, pos.positive_index)]
    gated = pos[~pos.both_masked]
    agg = gated.groupby(["model", "seed", "mask_instance"]).apply(lambda f: pd.Series({
        "mrr_all": f.reciprocal_rank.mean(),
        "mrr_drop": f.reciprocal_rank[~f.zero_partner].mean(),
        "mrr_flagged": f.reciprocal_rank[f.zero_partner].mean() if f.zero_partner.any() else np.nan,
        "n_rows": len(f), "n_flagged": int(f.zero_partner.sum())}), include_groups=False).reset_index()
    chk = agg.merge(sc[["model", "seed", "mask_instance", "mrr"]], on=["model", "seed", "mask_instance"])
    max_diff = float((chk.mrr_all - chk.mrr).abs().max())
    inst = agg.groupby(["model", "mask_instance"])[["mrr_all", "mrr_drop", "mrr_flagged", "n_rows", "n_flagged"]].mean().reset_index()
    inst["region"] = code; inst_rows.append(inst)
    wide = {k: inst.pivot(index="mask_instance", columns="model", values=k) for k in ("mrr_all", "mrr_drop", "mrr_flagged")}
    a, b = gate_stat(wide["mrr_all"]["gnn_a4"].to_numpy(), wide["mrr_all"][CHAMPION].to_numpy()), gate_stat(wide["mrr_drop"]["gnn_a4"].to_numpy(), wide["mrr_drop"][CHAMPION].to_numpy())
    fl = gated[gated.zero_partner].groupby("model").reciprocal_rank.mean()
    n_rows = int(inst[inst.model == "gnn_a4"].n_rows.sum()); n_fl = int(inst[inst.model == "gnn_a4"].n_flagged.sum())
    row = dict(region=code, gated_rows=n_rows, flagged_rows=n_fl, flagged_frac=round(n_fl / n_rows, 4), recompute_max_abs_diff=max_diff)
    for model in ("gnn_a4", "xgboost", "logreg", "configuration"):
        row[f"{model}_mrr_all"] = round(float(wide["mrr_all"][model].mean()), 4)
        row[f"{model}_mrr_drop"] = round(float(wide["mrr_drop"][model].mean()), 4)
        row[f"{model}_mrr_on_flagged_rows"] = round(float(fl.get(model, np.nan)), 4)
    row.update({f"gate_all_{k}": v for k, v in a.items()}); row.update({f"gate_drop_{k}": v for k, v in b.items()})
    summary.append(row); print(json.dumps(row), flush=True)
pd.DataFrame(summary).to_csv(OUT / "positive_only_sensitivity_summary.csv", index=False)
pd.concat(inst_rows).to_csv(OUT / "positive_only_sensitivity_instances.csv", index=False)
