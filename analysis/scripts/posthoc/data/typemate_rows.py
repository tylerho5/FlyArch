"""Row-level comparison on F2-confirm for CX, LH, LX, MB: heuristic H1/H2 reciprocal ranks joined to the gate's
scored per-positive rows (model, xgboost, logreg, configuration), split by whether u has >=1 visible type-mate."""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from typemate_heuristic import *  # noqa
from src.metrics import ranks_against_negatives
SCORED = A.results_root
out = []
for code in ("CX", "LH", "LX", "MB"):
    c = code.lower(); g = RegionGraph.read(DATA / "processed/f2" / c)
    root, tid = type_ids_for(code)
    suite = Suite.read(EV / "data/suites/f2" / c / "f2_confirm.json")
    committed = CommittedNegatives.read(EV / "data/suites/f2" / c / "f2_confirm.negatives.json")
    pool = np.arange(g.n_nodes, dtype=np.int64); te = pack_edges(g.edges)
    hrows = []
    for inst in suite.resolve(g):
        view = apply_mask(g, inst)
        task = build_scoring_task(view, view_negatives(view, spec=committed.spec, pool=pool, true_edges=te), suite=suite, negatives=committed)
        h1, h2, live = scores_for(view, candidate_pairs(task), tid)
        P = len(task.positives); h1 = h1.reshape(P, -1); h2 = h2.reshape(P, -1)
        hrows.append(pd.DataFrame({"mask_instance": inst.instance_id, "positive_index": np.arange(P),
                                   "rr_h1": 1 / ranks_against_negatives(h1[:, 0], h1[:, 1:]),
                                   "rr_h2": 1 / ranks_against_negatives(h2[:, 0], h2[:, 1:]),
                                   "live": live.reshape(P, -1)[:, 0], "both": view.both_masked()}))
    H = pd.concat(hrows); H = H[~H.both]
    pos = pd.concat(pd.read_parquet(f, columns=["mask_instance", "positive_index", "model", "seed", "reciprocal_rank", "both_masked", "split"]) for f in glob.glob(str(SCORED / c / "scored" / "*.positives.parquet")))
    pos = pos[(pos.split == "f2_confirm") & ~pos.both_masked]
    m = pos.groupby(["mask_instance", "positive_index", "model"]).reciprocal_rank.mean().unstack("model").reset_index()
    J = H.merge(m, on=["mask_instance", "positive_index"], how="inner"); assert len(J) == len(H)
    for stratum, sel in (("all gated rows", np.ones(len(J), bool)), ("u has >=1 visible type-mate", J.live.to_numpy()), ("u has no visible type-mate", ~J.live.to_numpy())):
        s = J[sel]
        out.append(dict(region=code, stratum=stratum, rows=len(s), share=round(len(s) / len(J), 3),
                        H1=round(s.rr_h1.mean(), 3), H2=round(s.rr_h2.mean(), 3), gnn=round(s.gnn_a4.mean(), 3),
                        xgboost=round(s.xgboost.mean(), 3), logreg=round(s.logreg.mean(), 3), configuration=round(s.configuration.mean(), 3)))
        print(out[-1], flush=True)
pd.DataFrame(out).to_csv(OUT / "typemate_rows_confirm_4regions.csv", index=False)
