"""Wiring-only rewiring null: frozen logic.

Statistics S1-S4, the N0 degree-preserving swap null, the N1 weight-permutation null, and the
decision rule (z, empirical p, saturation, Holm, F2 replication). Only source/target/weight of a
region's edges.parquet are read. Self-loops are dropped on load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

STATS = ("S1_reciprocity", "S2_transitivity", "S3_spread3", "S4_spectral_radius")
N_SEED_SETS = 200
SAT_SD = 1e-9
DENSE_MAX_N = 2000


def load_graph(root, split, region):
    """Return src, dst, w, n, n_loops. All source==target edges are dropped (prereg rule);
    n counts node ids before the drop."""
    e = pd.read_parquet(Path(root) / "processed" / split / region / "edges.parquet")
    src, dst, w = e.source.to_numpy(np.int64), e.target.to_numpy(np.int64), e.weight.to_numpy()
    n = int(max(src.max(), dst.max()) + 1)
    keep = src != dst
    return src[keep], dst[keep], w[keep], n, int((~keep).sum())


def _check(src, dst, n):
    assert (src != dst).all(), "self-loop entered the pipeline"
    assert len(np.unique(src * n + dst)) == len(src), "duplicate edges"


def make_seed_sets(n, seed_seq, n_sets=N_SEED_SETS):
    """Built once per (region, dataset); the same array goes to every stats() call."""
    k = max(1, int(round(0.05 * n)))
    rng = np.random.default_rng(seed_seq)
    return np.stack([rng.choice(n, k, replace=False) for _ in range(n_sets)])


def swap_rewire(src, dst, n, rng, factor=10, batch=200_000):
    """Degree-preserving directed double-edge swaps (a,b),(c,d)->(a,d),(c,b).
    Each edge keeps its weight and its source node, so out-strength is preserved and
    in-strength is not (callers keep the weight array aligned with the returned dst).
    Runs until factor*|E| ACCEPTED swaps; attempts capped at 200*|E| -> 'null unreachable'."""
    _check(src, dst, n)
    src, dst = src.copy(), dst.copy()
    m = len(src)
    keys = src * n + dst
    accepted, attempts, target, cap = 0, 0, factor * m, 200 * m
    batch = min(batch, max(1, m // 8))
    while accepted < target:
        if attempts >= cap:
            raise RuntimeError(f"null unreachable: {accepted}/{target} accepted swaps after {attempts} attempts")
        k = batch
        attempts += k
        i = rng.integers(0, m, k)
        j = rng.integers(0, m, k)
        both = np.concatenate([i, j])
        _, first, cnt = np.unique(both, return_index=True, return_counts=True)
        single = np.zeros(2 * k, bool)
        single[first[cnt == 1]] = True
        ok = single[:k] & single[k:] & (i != j)
        i, j = i[ok], j[ok]
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        n1, n2 = a * n + d, c * n + b
        ok = (a != d) & (c != b) & (n1 != n2)
        sorted_keys = np.sort(keys)
        for new in (n1, n2):
            pos = np.minimum(np.searchsorted(sorted_keys, new), m - 1)
            ok &= sorted_keys[pos] != new
        i, j, n1, n2 = i[ok], j[ok], n1[ok], n2[ok]
        allnew = np.concatenate([n1, n2])
        _, inv, cnt = np.unique(allnew, return_inverse=True, return_counts=True)
        clash = cnt[inv] > 1
        ok = ~(clash[: len(n1)] | clash[len(n1):])
        i, j = i[ok], j[ok]
        dst_i, dst_j = dst[i].copy(), dst[j].copy()
        dst[i], dst[j] = dst_j, dst_i
        keys[i] = src[i] * n + dst[i]
        keys[j] = src[j] * n + dst[j]
        accepted += len(i)
    return src, dst


def spectral_radius(src, dst, w, n):
    """|largest-magnitude eigenvalue of W|, raw. Raises rather than returning NaN."""
    wm = sp.csr_matrix((w.astype(np.float64), (src, dst)), shape=(n, n))
    try:
        return float(abs(eigs(wm, k=1, which="LM", return_eigenvectors=False)[0]))
    except Exception:
        if n > DENSE_MAX_N:
            raise
        return float(np.abs(np.linalg.eigvals(wm.toarray())).max())


def stats(src, dst, w, n, seed_sets, s4_only=False):
    _check(src, dst, n)
    s4 = spectral_radius(src, dst, w, n)
    if s4_only:
        return {"S4_spectral_radius": s4}
    keys = src * n + dst
    s1 = float(np.isin(dst * n + src, keys).mean())
    a = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    b = ((a + a.T) > 0).astype(np.float64).tocsr()
    deg = np.asarray(b.sum(1)).ravel()
    wedges = float((deg * (deg - 1)).sum())
    tri = float(b.multiply(b @ b).sum())
    s2 = tri / wedges if wedges else 0.0
    cur = np.zeros((n, len(seed_sets)), np.float32)
    for c, s in enumerate(seed_sets):
        cur[s, c] = 1
    at = a.T.tocsr().astype(np.float32)
    for _ in range(3):
        cur = np.minimum(1.0, cur + at @ cur)
    return {"S1_reciprocity": s1, "S2_transitivity": s2, "S3_spread3": float(cur.mean()),
            "S4_spectral_radius": s4}


REGIONS = ("al", "cx", "gng", "inp", "lh", "lx", "mb", "ol", "penp", "snp", "vlnp", "vmnp")
SPLITS = ("f1", "f2")


def derive_seed(master, region, split):
    """One seed per (region, dataset) from the recorded master seed (prereg: Compute)."""
    ss = np.random.SeedSequence([int(master), REGIONS.index(region.lower()), SPLITS.index(split)])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def draw_rng(master, kind, idx):
    """Per-draw stream = SeedSequence(master) child (spawn_key), independent of sharding."""
    return np.random.default_rng(np.random.SeedSequence(master, spawn_key=(1, kind, idx)))


def run_draw(src, dst, w, n, seed_sets, master, idx):
    """One N0 draw (all stats) and one N1 draw (S4 only). Returns two row dicts."""
    s2, d2 = swap_rewire(src, dst, n, draw_rng(master, 0, idx))
    assert (np.bincount(s2, minlength=n) == np.bincount(src, minlength=n)).all()
    assert (np.bincount(d2, minlength=n) == np.bincount(dst, minlength=n)).all()
    n0 = {"kind": "N0", "perm": idx, **stats(s2, d2, w, n, seed_sets)}
    wp = draw_rng(master, 1, idx).permutation(w)
    n1 = {"kind": "N1", "perm": idx, **stats(src, dst, wp, n, seed_sets, s4_only=True)}
    return n0, n1


def real_row(src, dst, w, n, seed_sets):
    return {"kind": "real", "perm": -1, **stats(src, dst, w, n, seed_sets)}


# ---- decision rule -------------------------------------------------------------------------

def z_and_p(t_real, null):
    """z and empirical two-sided p; saturated (SD < 1e-9) -> z=p=nan, status 'undefined'."""
    null = np.asarray(null, float)
    if len(null) < 2:
        return float("nan"), float("nan"), "undefined"
    mu, sd = null.mean(), null.std(ddof=1)
    if not sd >= SAT_SD:
        return float("nan"), float("nan"), "undefined"
    dev = abs(t_real - mu)
    p = (1 + int((np.abs(null - mu) >= dev).sum())) / (len(null) + 1)
    return float((t_real - mu) / sd), float(p), "defined"


def holm(p):
    """Holm step-down adjusted p-values, same order as input."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    return out


def test_table(df, expected_b=None):
    """One row per (region, null, stat) from a CSV with kind real/N0/N1 rows.
    If expected_b is given, a null with a different number of draws (a missing or crashed shard)
    is flagged 'incomplete' and cannot enter the Holm family or a verdict."""
    rows = []
    for region, g in df.groupby("region"):
        real = g[g.kind == "real"].iloc[0]
        unreachable = (g.kind == "N0_unreachable").any()
        for kind in ("N0", "N1"):
            null = g[g.kind == kind]
            for s in STATS if kind == "N0" else STATS[3:]:
                r = {"region": region, "null": kind, "stat": s, "T_real": real[s], "B": len(null)}
                if unreachable and kind == "N0":
                    r.update(status="null unreachable", z=np.nan, p=np.nan)
                elif expected_b is not None and len(null) != expected_b:
                    r.update(status="incomplete", z=np.nan, p=np.nan)
                else:
                    z, p, st = z_and_p(real[s], null[s].to_numpy())
                    r.update(status=st, z=z, p=p, null_mean=null[s].mean() if len(null) else np.nan)
                rows.append(r)
    return pd.DataFrame(rows)


def decide(f1, f2, z_thr=3.0, alpha=0.05, rep_z=2.0, expected_b=None):
    """f1, f2: raw CSV frames. Holm over defined F1 N0 tests; F2 replication on its own N0.
    expected_b: required draws per null (the CLI passes 10 000); incomplete nulls are flagged."""
    t1, t2 = test_table(f1, expected_b), test_table(f2, expected_b)
    fam = (t1["null"] == "N0") & (t1.status == "defined")
    t1["p_holm"] = np.nan
    t1.loc[fam, "p_holm"] = holm(t1.loc[fam, "p"])
    t1["differs"] = fam & (t1.z.abs() > z_thr) & (t1.p_holm < alpha)
    key = ["region", "null", "stat"]
    t2 = t2.rename(columns={"z": "z_F2", "status": "status_F2"})[key + ["z_F2", "status_F2"]]
    out = t1.merge(t2, on=key, how="left")
    out["replicates"] = (out.differs & (out.status_F2 == "defined")
                         & (np.sign(out.z) == np.sign(out.z_F2)) & (out.z_F2.abs() > rep_z))
    return out
