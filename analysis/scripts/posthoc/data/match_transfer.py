"""Post hoc, unregistered (2026-11-17): match-table transfer baseline.
For candidate pair (s, t) in F2 (always scored as the directed pair s->t), S, T = F1 root ids of the reviewed
FAFB matches (banc meta `fafb_match`). M1 = F1 within-region synapses S->T (no threshold, 0 if unmatched or none);
M2 = 1[M1 >= 5] (edge of the F1 region graph); C = M1 + 1e-3*H1 (lexicographic: M1 first, type-mate H1 second).
Tie-break on all: + 1e-9 (C) or 1e-6 (M1, M2) * v's visible strength on the replaced side / view max."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from typemate_heuristic import *  # noqa: F401,F403 (EV, DATA, repo imports, type_ids_for, scores_for)
from typemate_heuristic import lookup
from scripts.build_features import read_table
from src.metrics import ranks_against_negatives
SHIFT = np.int64(1 << 18)
IDS = np.load(OUT / "f1_ids_sorted.npy")
META = read_table(DATA / "raw/banc/banc_888_meta.feather")[["root_id", "fafb_match"]]
_fm = META.set_index("root_id")["fafb_match"].dropna().astype(str)
_fm = _fm[_fm.str.fullmatch(r"\d+")]
MATCH = pd.Series(pd.array(_fm.to_numpy().astype(np.int64), dtype="Int64"), index=_fm.index)  # nullable: reindex must not go via float64  # exact string->int64, never via float64

def f1_index(node_ids):
    m = MATCH.reindex(node_ids)
    out = np.full(len(node_ids), -1, np.int64)
    ok = m.notna().to_numpy()
    vals = m[ok].astype("int64").to_numpy()
    pos = np.clip(np.searchsorted(IDS, vals), 0, len(IDS) - 1)
    hit = IDS[pos] == vals
    idx = np.flatnonzero(ok); out[idx[hit]] = pos[hit]
    return out

def run(code, suite_name, rows_out):
    c = code.lower(); g = RegionGraph.read(DATA / "processed/f2" / c)
    root, tid = type_ids_for(code); assert np.array_equal(root, g.node_ids)
    f1i = f1_index(g.node_ids)
    z = np.load(OUT / f"f1_pairs_{c}.npz"); keys, tot = z["keys"], z["totals"].astype(float)
    suite = Suite.read(EV / "data/suites/f2" / c / f"{suite_name}.json")
    committed = CommittedNegatives.read(EV / "data/suites/f2" / c / f"{suite_name}.negatives.json")
    pool = np.arange(g.n_nodes, dtype=np.int64); te = pack_edges(g.edges); res = []
    for inst in suite.resolve(g):
        t0 = time.time()
        view = apply_mask(g, inst)
        task = build_scoring_task(view, view_negatives(view, spec=committed.spec, pool=pool, true_edges=te), suite=suite, negatives=committed)
        pairs = candidate_pairs(task); P = len(task.positives)
        h1, _, _ = scores_for(view, pairs, tid)
        m = view.is_masked; s, t = pairs[:, 0], pairs[:, 1]; u_is_src = m[s]; v = np.where(u_is_src, t, s)
        vin, vout = view.visible_in_strength().astype(float), view.visible_out_strength().astype(float)
        tie = np.where(u_is_src, vin[v] / max(vin.max(), 1), vout[v] / max(vout.max(), 1))
        h1_core = h1 - 1e-6 * tie
        fs, ft = f1i[s], f1i[t]; both = (fs >= 0) & (ft >= 0)
        m1 = np.where(both, lookup(keys, tot, np.where(both, fs * SHIFT + ft, -1)), 0.0)
        sc = {"m1": m1 + 1e-6 * tie, "m2": (m1 >= 5).astype(float) + 1e-6 * tie, "comb": m1 + 1e-3 * h1_core + 1e-9 * tie}
        gated = evaluated_rows(task)
        r = dict(region=code, suite=suite_name, instance=inst.instance_id, regime=inst.regime, gated_rows=int(gated.sum()))
        for k, x in sc.items():
            r[f"{k}_mrr"] = gated_mrr(task, x)
        pos = task.positives; um = np.where(m[pos[:, 0]], pos[:, 0], pos[:, 1]); vm = np.where(m[pos[:, 0]], pos[:, 1], pos[:, 0])
        um_ok, vm_ok = f1i[um] >= 0, f1i[vm] >= 0
        r.update(u_matched=float(um_ok[gated].mean()), v_matched=float(vm_ok[gated].mean()), both_matched=float((um_ok & vm_ok)[gated].mean()),
                 true_pair_f1_synapse_gt0=float((m1.reshape(P, -1)[:, 0] > 0)[gated].mean()), secs=round(time.time() - t0, 1))
        res.append(r); print(r, flush=True)
        if suite_name == "f2_confirm":
            def rr(x):
                x = x.reshape(P, -1); return 1 / ranks_against_negatives(x[:, 0], x[:, 1:])
            h1r = h1.reshape(P, -1)
            rows_out.append(pd.DataFrame({"region": code, "mask_instance": inst.instance_id, "regime": inst.regime,
                                          "positive_index": np.arange(P)[gated], "u_matched": um_ok[gated], "v_matched": vm_ok[gated],
                                          "rr_m1": rr(sc["m1"])[gated], "rr_m2": rr(sc["m2"])[gated], "rr_comb": rr(sc["comb"])[gated],
                                          "rr_h1": (1 / ranks_against_negatives(h1r[:, 0], h1r[:, 1:]))[gated]}))
        del task, pairs, h1, sc
    return res

if __name__ == "__main__":
    codes = A.codes.split(",") if A.codes else list(REGIONS)
    for code in codes:
        out = OUT / f"match_{code.lower()}.csv"
        if out.exists():
            continue
        rows_out = []
        res = run(code, "f2_dev", rows_out) + run(code, "f2_confirm", rows_out)
        pd.concat(rows_out).to_parquet(OUT / f"match_rows_{code.lower()}.parquet", index=False)
        pd.DataFrame(res).to_csv(out, index=False)
