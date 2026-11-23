"""Post hoc, unregistered (2026-11-17): 'copy the visible type-mates' heuristic on F2-dev and F2-confirm.

For a gated row with masked endpoint u (type k, the GNN's type id: crosswalked F1 type, untrained types -> OOV)
and candidate v: M_k = unmasked type-k nodes with >=1 visible edge (the GNN sibling group).
H1 (S1): |{m in M_k : visible m->v}| / |M_k| for u->v rows, |{m : visible v->m}| / |M_k| for v->u rows.
H2: summed visible synapses m->v (or v->m) over M_k, divided by |M_k|.
Both + EPS * v's visible strength on the replaced side / max strength in the view (tie-breaker only).
Scored through the repo path: view_negatives + build_scoring_task (committed digest check) + gated_mrr (pessimistic).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__, lambda ap: ap.add_argument("codes", nargs="?", help="comma-separated region codes (default: all)"))
EV, DATA, OUT = A.repo_root, A.data_root, A.out
from src.regions import RegionGraph, REGIONS
from src.suites import Suite, view_negatives, f1_train_types
from src.masks import apply_mask
from src.negatives import pack_edges
from src.evaluate import CommittedNegatives, build_scoring_task, candidate_pairs, gated_mrr, evaluated_rows
from src.features import FeatureSchema
EPS = 1e-6

def type_ids_for(code):
    c = code.lower()
    f = pd.read_parquet(DATA / "processed/f2" / c / "features.parquet", columns=["root_id", "cell_type"])
    schema = FeatureSchema.from_json((DATA / "processed/f2" / c / "schema.json").read_text(encoding="utf-8"))
    g1 = RegionGraph.read(DATA / "processed/f1" / c)
    t1 = pd.read_parquet(DATA / "interim/f1" / c / "f1_types.parquet").set_index("root_id")["cell_type"]
    trained = {schema.type_vocab[n] for n in f1_train_types(g1, t1, Suite.read(EV / "data/suites/f1" / c / "f1_val.json")) if n in schema.type_vocab}
    ids = np.rint(f["cell_type"].to_numpy()).astype(np.int64)
    ids[~np.isin(ids, list(trained))] = 0
    return f["root_id"].to_numpy(np.int64), ids

def lookup(keys, vals, q):
    pos = np.clip(np.searchsorted(keys, q), 0, max(len(keys) - 1, 0))
    hit = (keys[pos] == q) if len(keys) else np.zeros(len(q), bool)
    return np.where(hit, vals[pos] if len(keys) else 0, 0.0)

def scores_for(view, pairs, tid):
    n = view.graph.n_nodes; m = view.is_masked
    e, w = view.visible_edges.astype(np.int64), view.visible_weights.astype(np.float64)
    s, t = e[:, 0], e[:, 1]
    connected = np.zeros(n, bool); connected[e.reshape(-1)] = True
    group = np.bincount(tid[connected & ~m], minlength=tid.max() + 1).astype(float)
    ok_s, ok_t = tid[s] > 0, tid[t] > 0
    kout = tid[s[ok_s]] * n + t[ok_s]; ko, inv = np.unique(kout, return_inverse=True)
    out_c, out_w = np.bincount(inv, minlength=len(ko)).astype(float), np.bincount(inv, weights=w[ok_s], minlength=len(ko))
    kin = tid[t[ok_t]] * n + s[ok_t]; ki, inv = np.unique(kin, return_inverse=True)
    in_c, in_w = np.bincount(inv, minlength=len(ki)).astype(float), np.bincount(inv, weights=w[ok_t], minlength=len(ki))
    vin, vout = view.visible_in_strength().astype(float), view.visible_out_strength().astype(float)
    src, tgt = pairs[:, 0], pairs[:, 1]
    u_is_src = m[src]
    u = np.where(u_is_src, src, tgt); v = np.where(u_is_src, tgt, src); k = tid[u]
    q = k * n + v
    c = np.where(u_is_src, lookup(ko, out_c, q), lookup(ki, in_c, q))
    sw = np.where(u_is_src, lookup(ko, out_w, q), lookup(ki, in_w, q))
    gsize = np.where(k > 0, group[k], 0.0)
    live = (k > 0) & (gsize > 0)
    s1 = np.where(live, c / np.maximum(gsize, 1), 0.0); h2 = np.where(live, sw / np.maximum(gsize, 1), 0.0)
    tie = np.where(u_is_src, vin[v] / max(vin.max(), 1), vout[v] / max(vout.max(), 1))
    return s1 + EPS * tie, h2 + EPS * tie, live

def run(code, suite_name):
    c = code.lower()
    g = RegionGraph.read(DATA / "processed/f2" / c)
    root, tid0 = type_ids_for(code)
    assert np.array_equal(root, g.node_ids)
    suite = Suite.read(EV / "data/suites/f2" / c / f"{suite_name}.json")
    committed = CommittedNegatives.read(EV / "data/suites/f2" / c / f"{suite_name}.negatives.json")
    pool = np.arange(g.n_nodes, dtype=np.int64); true_edges = pack_edges(g.edges)
    rows = []
    for inst in suite.resolve(g):
        t0 = time.time()
        view = apply_mask(g, inst)
        sample = view_negatives(view, spec=committed.spec, pool=pool, true_edges=true_edges)
        task = build_scoring_task(view, sample, suite=suite, negatives=committed)
        pairs = candidate_pairs(task)
        h1, h2, live = scores_for(view, pairs, tid0)
        gated = evaluated_rows(task)
        live_rows = live.reshape(len(task.positives), -1)[:, 0]
        rows.append(dict(region=code, suite=suite_name, instance=inst.instance_id, regime=inst.regime,
                         gated_rows=int(gated.sum()), share_rows_with_typemate=float(live_rows[gated].mean()),
                         h1_mrr=gated_mrr(task, h1), h2_mrr=gated_mrr(task, h2), secs=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
        del sample, task, pairs, h1, h2, live
    return rows

if __name__ == "__main__":
    codes = A.codes.split(",") if A.codes else list(REGIONS)
    for code in codes:
        out = OUT / f"typemate_{code.lower()}.csv"
        if out.exists():
            continue
        rows = run(code, "f2_dev") + run(code, "f2_confirm")
        pd.DataFrame(rows).to_csv(out, index=False)
