"""Confirmatory logic for the rate-model rewiring test (signed rate model, stratified rewiring).

Everything here follows that document; section names in comments refer to it. Frozen at the merge
commit of the pre-registration. Parameters are constants, not options of the confirmatory run.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

REGIONS = ("al", "lx", "cx")  # region index = position (al=0, lx=1, cx=2)
SPLITS = ("f1", "f2")  # dataset index = position
# Extension. Indices continue the original ones, so every original
# draw keeps its seed: lh=3, mb=4, penp=5; f3 (male CNS) = 2.
EXT_REGIONS = ("lh", "mb", "penp")
ALL_REGIONS = REGIONS + EXT_REGIONS
ALL_SPLITS = SPLITS + ("f3",)
B_EXT, B_SENS_EXT = 2000, 500  # LH/MB/PENP: main draws, sensitivity draws
# Direction for Part A, frozen: the registered LX FlyWire z from the original result (#243, results.csv).
LX_F1_Z_REGISTERED = -2.184644436199633
PRIMARY = ("al", "lx")
DEFAULT_SEED = 20261122
B_MAIN = 5000
K_FLIPS = 10
FLIP_FRACTION = 0.10
B_SENS = 2000  # draws 0..B_SENS-1 are reused for sign-flip and amplitude sensitivity
AMPLITUDES = (0.5, 2.0)
TARGET_RHO = 0.9
SWAP_FACTOR = 10
CAP_FACTOR = 200
T_STEPS = 30
LEAK, GAIN = 0.8, 0.2
MIN_NODES = 10
RHO_MIN = 1e-12
SAT_REL, SAT_MEAN = 1e-6, 1e-12

GROUP_A, GROUP_E, GROUP_I = 0, 1, 2

# SHA-256 of every input file and graph.json "digest" (doc section Inputs): (split, region) -> hashes.
EXPECTED = {
    ("f1", "al"): ("ea6accd5f9d2bb1c21d958523ae09927c1d54c6ecab48e95e581b237f83e20de",
                   "0a15825b3aebd455203391d13d289b5832d3e44f879fd34207e0279ab6596206",
                   "3e89ab66cc8f8e4323ab3e61918a58d7bc62fcfa40bc6fee28e2707f9dbeec8e"),
    ("f1", "lx"): ("c70c5af81ee019e133cc56d4e47367e1d691530d6c1ea6de58a5d3df32e0cf63",
                   "4f93d491722a9c10ec436115341f9fcaa45c275a703d9ac9b268933e3ad90dde",
                   "412cff170e832b3c6cf5bc5e195f5e7b4d2f26078302f2620c0a815249cb8eda"),
    ("f1", "cx"): ("b1ebb06dd08cc2bdc3684d35b738a4960bf370df362fa36c7a2387b92f13710b",
                   "a3a3940b5119a7b85cf3a3806b6d0e4d3e9f551b8bfe7ff8028ef3391057d14f",
                   "9ea04d04c25cedbe1d4fe6bc1527976e784c8078dce4f993f3335e9d1935be6f"),
    ("f2", "al"): ("0730714ff2f6ed7361b08632d0a4d3fb18059a99da8b7e650b58f72b2e1d9ab7",
                   "74803bb327b6cffd324c89074d944f266aab8f8b5286939d469ad42dc9328698",
                   "ad224861bf640bad5c0c2fb77f3163008b8070485552b023681d3ca3273fbb46"),
    ("f2", "lx"): ("6d7e026bd9d54476e5e64762c2af750a2fde9d7bd8e195b4995a0af46a899b2f",
                   "b846dfae541b3dcd402920d0d75b239b3d30c1e9bbea9ce0c431285665c475f9",
                   "39c2816e3754797bc25e4ad274959e02581826643663800b52986d692b802ea7"),
    ("f2", "cx"): ("71a0e96a61f151a28ec6ef8ba5b11664916e64123adeba9c01f872bb4a651442",
                   "f1cd05847a41483f03ee00492ccb022193ccaf3f3b3af2c7b52a494dcead5ac7",
                   "df849f3ca3e68effbea8800a3c7c2262b3ca12eeea225f25b932bb5793c7dd95"),
    # extension inputs
    ("f3", "lx"): ("3d6e271d573ad11cea2b3482f81c62693b6efc6de589940965cac4945b88138f",
                   "c395e8dc26576f975d1213282f647bcd21480335ffda2ad540236620093cd293",
                   "bf401a2556e0f4a1dd6dc743b1fc57779ed87c6cc9dd9666229a19789c39ba92"),
    ("f1", "lh"): ("edb4c6e52d4509f15f1bc3fad85db536aac40bbf80cc873445c9e36187695634",
                   "fe20c40d58d1da57364ac564f24c3541bc5b34e55330ebdc7a611a50b9a9a1a1",
                   "f90da2447e3a118f54c166f9766818d8d23c1115feb3d1cb7b437bb75e0c1319"),
    ("f2", "lh"): ("c3697394e8c2698b69d9e4531d11a73bf49e154f8e7d578ab0d670a1db2e54b5",
                   "2efc4ac96db76d7d2099fa0edbdf3174c9d0131f7fcbc6ccd9372c54d77c3a37",
                   "b71861da673c33542cd71329a1c39bb051444521dc831db133c9f6854e61efc7"),
    ("f1", "mb"): ("9cc8198d67a21498060f53a088672190ecfe65cfba22ad2216e7133e1e0d08d2",
                   "1d04fa24642e3c3b6cf671880fc962edf8dc98fb99e9a81d33b38b268da4ed50",
                   "ea930e432b351c5cfe35e3ccceb51affa4738fc4d78cb3acc11eb14d2cef163e"),
    ("f2", "mb"): ("19fdcf03582cd7d0a49065188006ea3415ae9f47fc35969edbb09c200ee1357b",
                   "f338f1a7a17170f1d657c31b72b985002f23baf149840b2268b27ab528bd19a8",
                   "13f5ac6a9c95c71d7f59d8bbaa198b27d566ef3c0d3ea90cbd2d8d88d6a0abed"),
    ("f1", "penp"): ("1747eef4742c51a43d08a93d2d332b60e31b0adc73b8d2f470b63e91f07c854b",
                   "1870a863edb97588f873c49666d9e3deaf75484a69e1e38fbd566fa8086643d1",
                   "9fa0b8cc7067e9de858e4ee724168c591469a437b20b944c1dc051db854a3cc2"),
    ("f2", "penp"): ("9e4d5404aafa6b4536bb55c5c1a0aa970c7c870748578f36a562f7fe6c749273",
                   "03e4e0ea59bbb4b6869a2d28fcae27f6bfa0628dab787ba1ccdeb2036ae5bebc",
                   "00bce54a412a971b683b4eb2cf54451845c2fd80198a65c4d42f2ab656046ada"),
}


# ---- inputs ---------------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(root, split, region):
    """Re-check the registered hashes; raises RuntimeError on any mismatch. Returns the recorded hashes."""
    d = Path(root) / "processed" / split / region
    from src.regions import RegionGraph  # recomputes the content digest and checks graph.json
    got = (_sha256(d / "edges.parquet"), _sha256(d / "features.parquet"), RegionGraph.read(d).digest())
    if got != EXPECTED[(split, region)]:
        raise RuntimeError(f"input hash mismatch for {split}/{region}: {got}")
    return {"edges_sha256": got[0], "features_sha256": got[1], "graph_digest": got[2]}


def masks_and_signs(feat: pd.DataFrame, region: str):
    """A, E boolean masks, group array (0=A,1=E,2=I) and source sign (+1/-1) per the doc's exact rules."""
    def col(name):
        return feat[name].to_numpy() > 0 if name in feat.columns else np.zeros(len(feat), bool)

    a = col("superclass_sensory") | col("superclass_ascending") | col("superclass_sensory_ascending")
    if region == "al":
        e = col("class_ALPN") | col("class_ALON")
    else:
        e = col("superclass_descending") | col("superclass_endocrine") | col("superclass_motor")
    assert not (a & e).any(), "A and E overlap"
    grp = np.full(len(feat), GROUP_I, np.int8)
    grp[a] = GROUP_A
    grp[e] = GROUP_E
    sign = np.ones(len(feat))
    neg = col("nt_gaba") | col("nt_glutamate")
    sign[neg] = -1.0
    pos = col("nt_acetylcholine") | col("nt_dopamine") | col("nt_serotonin") | col("nt_octopamine")
    n_default = int((~neg & ~pos).sum())  # nt_other or no label -> +1
    return a, e, grp, sign, n_default


@dataclass
class Graph:
    src: np.ndarray
    dst: np.ndarray
    w: np.ndarray
    n: int
    a_idx: np.ndarray
    e_idx: np.ndarray
    grp: np.ndarray
    sign: np.ndarray
    meta: dict


def build_graph(edges: pd.DataFrame, feat: pd.DataFrame, region: str) -> Graph:
    src = edges.source.to_numpy(np.int64)
    dst = edges.target.to_numpy(np.int64)
    w = edges.weight.to_numpy(np.float64)
    n = len(feat)
    assert src.min() >= 0 and max(src.max(), dst.max()) < n, "edge id outside node range"
    keep = src != dst
    n_loops = int((~keep).sum())
    src, dst, w = src[keep], dst[keep], w[keep]
    assert len(np.unique(src * n + dst)) == len(src), "duplicate (source, target) pairs"
    a, e, grp, sign, n_default = masks_and_signs(feat, region)
    meta = {"n_nodes": n, "n_edges": len(src), "n_loops_dropped": n_loops,
            "n_A": int(a.sum()), "n_E": int(e.sum()), "n_sign_defaulted": n_default}
    return Graph(src, dst, w, n, np.flatnonzero(a), np.flatnonzero(e), grp, sign, meta)


def load_graph(root, split, region, check_hashes=True):
    d = Path(root) / "processed" / split / region
    hashes = verify_inputs(root, split, region) if check_hashes else {}
    g = build_graph(pd.read_parquet(d / "edges.parquet"), pd.read_parquet(d / "features.parquet"), region)
    g.meta.update(hashes)
    return g


def eligible(g: Graph) -> bool:
    return g.meta["n_A"] >= MIN_NODES and g.meta["n_E"] >= MIN_NODES


# ---- seeds ----------------------------------------------------------------------------------

def _entropy(master, region, split):
    return [int(master), ALL_REGIONS.index(region), ALL_SPLITS.index(split)]


def draw_rng(master, region, split, i):
    return np.random.default_rng(np.random.SeedSequence(_entropy(master, region, split), spawn_key=(1, int(i))))


def flip_rng(master, region, split, k):
    return np.random.default_rng(np.random.SeedSequence(_entropy(master, region, split), spawn_key=(2, int(k))))


def flip_sets(g: Graph, master, region, split, k_flips=K_FLIPS, frac=FLIP_FRACTION):
    """Per replicate k: sorted node ids flipped = round(frac x n_src) of the nodes with an out-edge,
    chosen uniformly without replacement (rng.choice on the sorted source-node list)."""
    nodes = np.unique(g.src)
    size = int(round(frac * len(nodes)))
    return [np.sort(flip_rng(master, region, split, k).choice(nodes, size=size, replace=False))
            for k in range(k_flips)]


# ---- stratified swap null -------------------------------------------------------------------

def swap_rewire(src, dst, grp, n, rng, factor=SWAP_FACTOR, cap_factor=CAP_FACTOR, batch=200_000):
    """Degree-preserving directed double-edge swaps (a,b),(c,d) -> (a,d),(c,b), group-constrained.

    Accept iff a != d, c != b, neither new edge exists, the two new edges differ, and
    (grp[b] == grp[d] or grp[a] == grp[c]). Proposals are drawn in batches; within a batch only
    proposals whose two edges are used by no other proposal are kept, and new edges are checked
    against the edge set at the start of the batch and against each other, so the accepted swaps
    are independent and valid in any order. Runs until exactly factor*|E| ACCEPTED swaps (the last
    batch is truncated); attempts (every proposal drawn) are capped at cap_factor*|E|.
    Returns (dst_new, accepted, attempts, cap_hit). src (and weights, by position) are unchanged.
    """
    src = np.asarray(src, np.int64)
    dst = np.array(dst, np.int64)
    m = len(src)
    keys = src * n + dst
    target, cap = factor * m, cap_factor * m
    batch = min(batch, max(1, m // 8))
    accepted = attempts = 0
    while accepted < target and attempts < cap:
        k = min(batch, cap - attempts)
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
        ok = (a != d) & (c != b) & (n1 != n2) & ((grp[b] == grp[d]) | (grp[a] == grp[c]))
        sorted_keys = np.sort(keys)
        for new in (n1, n2):
            pos = np.minimum(np.searchsorted(sorted_keys, new), m - 1)
            ok &= sorted_keys[pos] != new
        i, j, n1, n2 = i[ok], j[ok], n1[ok], n2[ok]
        allnew = np.concatenate([n1, n2])
        _, inv, cnt = np.unique(allnew, return_inverse=True, return_counts=True)
        clash = cnt[inv] > 1
        ok = ~(clash[: len(n1)] | clash[len(n1):])
        need = target - accepted
        i, j = i[ok][:need], j[ok][:need]
        dst_i, dst_j = dst[i].copy(), dst[j].copy()
        dst[i], dst[j] = dst_j, dst_i
        keys[i] = src[i] * n + dst[i]
        keys[j] = src[j] * n + dst[j]
        accepted += len(i)
    return dst, accepted, attempts, accepted < target


# ---- model ----------------------------------------------------------------------------------

def signed_matrix(src, dst, w, n, sign):
    """W[v,u] = s_u * w / out_u for edge u -> v (out_u = total out weight of u)."""
    out = np.bincount(src, weights=w, minlength=n)
    vals = sign[src] * w / out[src]
    return sp.csr_matrix((vals, (dst, src)), shape=(n, n))


def spectral_radius(wm):
    """Returns (rho, used_dense_fallback). rho may be non-finite/tiny; caller applies the undefined rule."""
    n = wm.shape[0]
    rho, dense = float("nan"), False
    try:
        v = eigs(wm, k=1, which="LM", v0=np.ones(n) / np.sqrt(n), tol=1e-8, maxiter=5000,
                 return_eigenvectors=False)
        rho = float(abs(v[0]))
    except Exception:
        pass
    if not np.isfinite(rho):
        dense = True
        rho = float(np.max(np.abs(np.linalg.eigvals(wm.toarray()))))
    return rho, dense


def equalise(wm):
    """Rescale to spectral radius 0.9. Returns (W_scaled or None, rho_raw, dense_used, ok)."""
    rho, dense = spectral_radius(wm)
    if not np.isfinite(rho) or rho < RHO_MIN:
        return None, rho, dense, False
    return wm * (TARGET_RHO / rho), rho, dense, True


def simulate(wm, a_idx, e_idx, amp=1.0, steps=T_STEPS):
    """T = sum_{t=1..30} mean_{j in E} r_j[t]; r[0]=0; r[t+1] = max(0, .8 r + .2 tanh(W r + u[t]));
    u[0] = amp on A, u[t>=1] = 0."""
    n = wm.shape[0]
    r = np.zeros(n)
    u = np.zeros(n)
    u[a_idx] = amp
    total = 0.0
    for t in range(steps):
        x = wm @ r
        if t == 0:
            x = x + u
        r = np.maximum(0.0, LEAK * r + GAIN * np.tanh(x))
        total += float(r[e_idx].mean())
    return total


def evaluate(g: Graph, dst, flips, sens: bool):
    """All endpoints for one topology (dst = real or rewired). Returns a dict.
    Main T; if sens: T under each sign-flip replicate (own equalisation each) and under the two
    amplitudes (same equalised W as main)."""
    out = {"T": np.nan, "rho": np.nan, "dense": 0, "rho_undefined": 0,
           "T_flip": [np.nan] * len(flips), "flip_dense": 0, "flip_undefined": 0,
           "T_amp": {a: np.nan for a in AMPLITUDES}}
    wm, rho, dense, ok = equalise(signed_matrix(g.src, dst, g.w, g.n, g.sign))
    out["rho"], out["dense"], out["rho_undefined"] = rho, int(dense), int(not ok)
    if ok:
        out["T"] = simulate(wm, g.a_idx, g.e_idx)
    if not sens:
        return out
    if ok:
        for a in AMPLITUDES:
            out["T_amp"][a] = simulate(wm, g.a_idx, g.e_idx, amp=a)
    for k, fl in enumerate(flips):
        s = g.sign.copy()
        s[fl] *= -1.0
        wk, _, dk, okk = equalise(signed_matrix(g.src, dst, g.w, g.n, s))
        out["flip_dense"] += int(dk)
        if okk:
            out["T_flip"][k] = simulate(wk, g.a_idx, g.e_idx)
        else:
            out["flip_undefined"] += 1
    return out


DRAW_COLUMNS = (["region", "split", "draw", "T", "rho", "dense_fallback", "rho_undefined",
                 "accepted", "attempts", "cap_hit"]
                + [f"T_flip{k}" for k in range(K_FLIPS)]
                + ["flip_dense", "flip_undefined", "T_amp0.5", "T_amp2.0", "rewire_sec", "sim_sec", "code_commit"])


def run_draw(g: Graph, region, split, master, i, flips, commit="", b_sens=B_SENS):
    """One draw row (dict). Draw i depends only on (master, region, split, i)."""
    t0 = time.perf_counter()
    dst, acc, att, cap_hit = swap_rewire(g.src, g.dst, g.grp, g.n, draw_rng(master, region, split, i))
    t1 = time.perf_counter()
    row = {"region": region, "split": split, "draw": i, "accepted": acc, "attempts": att,
           "cap_hit": int(cap_hit), "code_commit": commit}
    if cap_hit:  # null unreachable: nothing is simulated for this draw
        row.update(rewire_sec=t1 - t0, sim_sec=0.0)
        return row
    ev = evaluate(g, dst, flips, sens=i < b_sens)
    row.update(T=ev["T"], rho=ev["rho"], dense_fallback=ev["dense"], rho_undefined=ev["rho_undefined"],
               flip_dense=ev["flip_dense"], flip_undefined=ev["flip_undefined"],
               rewire_sec=t1 - t0, sim_sec=time.perf_counter() - t1)
    row["T_amp0.5"], row["T_amp2.0"] = ev["T_amp"][0.5], ev["T_amp"][2.0]
    for k, v in enumerate(ev["T_flip"]):
        row[f"T_flip{k}"] = v
    return row


def real_result(g: Graph, region, split, master):
    ev = evaluate(g, g.dst, flip_sets(g, master, region, split), sens=True)
    return {"T": ev["T"], "rho_raw": ev["rho"], "dense": ev["dense"], "undefined": bool(ev["rho_undefined"]),
            "T_flip": ev["T_flip"], "T_amp": ev["T_amp"]}


# ---- statistics and decision rule -----------------------------------------------------------

def z_stats(t_real, null):
    """z, p, d from null draws (ddof=1). status 'undefined' if the saturation rule fires or the
    real value is missing; the null summary is always returned."""
    null = np.asarray(null, float)
    mu = float(null.mean()) if len(null) else float("nan")
    sd = float(null.std(ddof=1)) if len(null) > 1 else float("nan")
    res = {"mean_null": mu, "sd_null": sd, "min_null": float(null.min()) if len(null) else np.nan,
           "max_null": float(null.max()) if len(null) else np.nan,
           "frac_zero": float((null == 0).mean()) if len(null) else np.nan,
           "z": np.nan, "p": np.nan, "d": np.nan, "status": "defined"}
    if not np.isfinite(t_real) or not (mu >= SAT_MEAN) or not (sd >= SAT_REL * mu):
        res["status"] = "undefined"
        return res
    dev = abs(t_real - mu)
    res["p"] = (1 + int((np.abs(null - mu) >= dev).sum())) / (len(null) + 1)
    res["z"] = (t_real - mu) / sd
    res["d"] = (t_real - mu) / mu
    return res


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    return out


def _sign(x):
    return int(np.sign(x)) if np.isfinite(x) else 0


def _sens_z(df, col, t_real, bs):
    """z over draws 0..bs-1 of column col. None if those draws are not all present (incomplete);
    z = NaN (that replicate undefined, the cell stays complete) if any present value is non-finite."""
    d = df[df.draw < bs].sort_values("draw")
    if col not in d or len(d) != bs or not np.array_equal(d.draw.to_numpy(), np.arange(bs)):
        return None
    v = d[col].to_numpy(float)
    if not np.isfinite(v).all() or not np.isfinite(t_real):
        return {"z": float("nan"), "undefined": True}
    return z_stats(t_real, v)


def cell_from_draws(df, real, is_eligible, b=B_MAIN, bs=B_SENS):
    """Per (region, dataset) summary from its draw rows (all shards concatenated) and the real result.
    Status precedence: excluded, undefined (real), null unreachable, undefined (null-draw rho),
    incomplete, undefined (saturation), defined."""
    df = df.reindex(columns=DRAW_COLUMNS).drop_duplicates("draw").copy()
    cell = {"status": "defined", "reason": "", "n_draws": 0, "z": np.nan, "p": np.nan, "d": np.nan,
            "sens_complete": False, "z_flip": [np.nan] * K_FLIPS, "z_amp": {a: np.nan for a in AMPLITUDES}}
    if not is_eligible:
        cell.update(status="excluded", reason="fewer than 10 nodes in A or E")
        return cell
    if real["undefined"]:
        cell.update(status="undefined", reason="real graph rho non-finite or < 1e-12")
        return cell
    df = df[(df.draw >= 0) & (df.draw < b)]
    good = df[(df.cap_hit == 0) & (df.rho_undefined == 0) & np.isfinite(df["T"])]
    cell["n_draws"] = len(good)
    cell["n_dense_fallback"] = int(df.dense_fallback.fillna(0).sum())
    if (df.cap_hit == 1).any():
        cell.update(status="null unreachable", reason=f"{int((df.cap_hit == 1).sum())} draw(s) hit the attempt cap")
    elif (df.rho_undefined == 1).any():
        cell.update(status="undefined", reason=f"{int((df.rho_undefined == 1).sum())} null draw(s) with undefined rho")
    elif len(good) != b or not np.array_equal(np.sort(good.draw.to_numpy()), np.arange(b)):
        cell.update(status="incomplete", reason=f"{len(good)} of {b} valid draws")
    st = z_stats(real["T"], good["T"].to_numpy() if len(good) else [])
    cell.update({k: st[k] for k in ("mean_null", "sd_null", "min_null", "max_null", "frac_zero")})
    if cell["status"] == "defined":
        if st["status"] == "undefined":
            cell.update(status="undefined", reason="saturation rule (SD_null < 1e-6 mean or mean < 1e-12)")
        else:
            cell.update(z=st["z"], p=st["p"], d=st["d"])
    cell["T_real"], cell["rho_raw_real"] = real["T"], real["rho_raw"]
    # sensitivity (reported whenever the first bs draws are complete; never gates anything)
    zs = []
    for k in range(K_FLIPS):
        s = _sens_z(df, f"T_flip{k}", real["T_flip"][k], bs)
        zs.append(None if s is None else s)
    cell["sens_complete"] = all(s is not None for s in zs)
    cell["z_flip"] = [np.nan if s is None else s["z"] for s in zs]
    za = {}
    for a, col in ((0.5, "T_amp0.5"), (2.0, "T_amp2.0")):
        s = _sens_z(df, col, real["T_amp"][a], bs)
        za[a] = np.nan if s is None else s["z"]
    cell["z_amp"] = za
    return cell


def _sens_labels(cell, has_effect):
    z0 = cell["z"]
    zf = np.array(cell["z_flip"], float)
    agree = [bool(np.isfinite(z) and _sign(z) == _sign(z0) and abs(z) >= 3) for z in zf]
    cell["n_flip_agree"] = int(sum(agree)) if np.isfinite(z0) else 0
    fin = zf[np.isfinite(zf)]
    cell["z_flip_median"] = float(np.median(fin)) if len(fin) else np.nan
    cell["z_flip_min"] = float(fin.min()) if len(fin) else np.nan
    cell["z_flip_max"] = float(fin.max()) if len(fin) else np.nan
    amp_ok = all(np.isfinite(cell["z_amp"][a]) and _sign(cell["z_amp"][a]) == _sign(z0)
                 and abs(cell["z_amp"][a]) >= 3 for a in AMPLITUDES)
    if cell["status"] != "defined":
        cell["sign_label"] = cell["amp_label"] = ""
    elif not cell["sens_complete"]:
        cell["sign_label"] = cell["amp_label"] = "sensitivity incomplete" if has_effect else ""
    elif has_effect:
        cell["sign_label"] = "sign-robust" if cell["n_flip_agree"] >= 8 else "sign-sensitive"
        cell["amp_label"] = "amplitude-robust" if amp_ok else "amplitude-sensitive"
    else:
        cell["sign_label"] = "sensitivity-only, exploratory" if cell["n_flip_agree"] >= 8 else ""
        cell["amp_label"] = ""


def decide(cells):
    """cells: {(region, split): cell dict from cell_from_draws}. Applies family/Holm/effect/replication/AL
    rule and sensitivity labels. Returns a DataFrame, one row per cell."""
    fam = [(r, "f1") for r in PRIMARY if (r, "f1") in cells and cells[(r, "f1")]["status"] == "defined"]
    adj = dict(zip(fam, holm([cells[k]["p"] for k in fam]))) if fam else {}
    rows = []
    effect, f2_ok = {}, {}
    for region in REGIONS:
        c1, c2 = cells.get((region, "f1")), cells.get((region, "f2"))
        if c1 is None:
            continue
        c1["p_holm"] = adj.get((region, "f1"), np.nan)
        if region == "cx":  # secondary: family of one (its own p), never in the primary Holm family
            c1["p_holm"] = c1["p"] if c1["status"] == "defined" else np.nan
        e1 = bool(c1["status"] == "defined" and c1["p_holm"] < 0.05 and abs(c1["z"]) >= 3 and abs(c1["d"]) >= 0.05)
        effect[region] = e1
        rep = strict = False
        if e1 and c2 is not None and c2["status"] == "defined":
            rep = bool(_sign(c2["z"]) == _sign(c1["z"]) and abs(c2["z"]) >= 2 and abs(c2["d"]) >= 0.05)
            strict = bool(abs(c2["z"]) >= 3 and c2["p"] < 0.05)
        declared = bool(e1 and rep and (strict if region == "al" else True))
        if not e1:
            verdict = "no F1 effect" if c1["status"] == "defined" else f"F1 {c1['status']}"
        elif region == "al":
            verdict = "effect declared" if declared else "AL: F1 only, not confirmed"
        else:
            verdict = "effect declared" if declared else "not replicated"
        if region == "cx":
            verdict = "secondary: " + verdict
        c1["effect_f1"], c1["replicated"], c1["declared"], c1["verdict"] = e1, rep, declared, verdict
        for split, c in (("f1", c1), ("f2", c2)):
            if c is None:
                continue
            has = e1 if split == "f1" else declared
            if split == "f2":
                c.update(effect_f1="", replicated="", declared="", verdict="", p_holm=np.nan)
            _sens_labels(c, has)
            rows.append({"region": region, "split": split, "secondary": region == "cx", **{
                k: v for k, v in c.items() if k not in ("z_flip", "z_amp")},
                **{f"z_flip{k}": z for k, z in enumerate(c["z_flip"])},
                "z_amp0.5": c["z_amp"][0.5], "z_amp2.0": c["z_amp"][2.0]})
    return pd.DataFrame(rows)


# ---- extension decision rules --------------------------

def decide_ext(cells, lx_f1_z=LX_F1_Z_REGISTERED):
    """cells: {(region, split): cell} for ("lx", "f3") and EXT_REGIONS x SPLITS. lx_f1_z: the registered LX
    FlyWire z from the original test (its sign is the direction to confirm). Returns a DataFrame."""
    rows = []

    def emit(region, split, c, family, verdict, has_effect, **extra):
        c.update(family=family, verdict=verdict, **extra)
        _sens_labels_ext(c, has_effect)
        rows.append({"region": region, "split": split,
                     **{k: v for k, v in c.items() if k not in ("z_flip", "z_amp", "p_flip", "p_amp")},
                     **{f"z_flip{k}": z for k, z in enumerate(c["z_flip"])},
                     "z_amp0.5": c["z_amp"][0.5], "z_amp2.0": c["z_amp"][2.0]})

    # Part A: LX in the male CNS, one test, conventional level, direction fixed by the FlyWire result.
    c = cells.get(("lx", "f3"))
    if c is not None:
        ok = bool(c["status"] == "defined" and c["p"] < 0.05 and _sign(c["z"]) == _sign(lx_f1_z)
                  and abs(c["d"]) >= 0.05)
        v = ("male-CNS LX directional test passed" if ok else "male-CNS LX directional test not passed") \
            if c["status"] == "defined" \
            else f"male CNS LX {c['status']}"
        emit("lx", "f3", c, "A: LX male CNS", v, ok, confirmed=ok)

    # Part B: LH, MB, PENP. FlyWire family with Holm at 0.05, replication in BANC.
    fam = [(r, "f1") for r in EXT_REGIONS if (r, "f1") in cells and cells[(r, "f1")]["status"] == "defined"]
    adj = dict(zip(fam, holm([cells[k]["p"] for k in fam]))) if fam else {}
    for region in EXT_REGIONS:
        c1, c2 = cells.get((region, "f1")), cells.get((region, "f2"))
        if c1 is None:
            continue
        ph = adj.get((region, "f1"), np.nan)
        e1 = bool(c1["status"] == "defined" and ph < 0.05 and abs(c1["d"]) >= 0.05)
        rep = bool(e1 and c2 is not None and c2["status"] == "defined" and _sign(c2["z"]) == _sign(c1["z"])
                   and c2["p"] < 0.05 and abs(c2["d"]) >= 0.05)
        if c1["status"] != "defined":
            v = f"F1 {c1['status']}"
        else:
            v = "effect declared" if rep else ("not replicated" if e1 else "no F1 effect")
        emit(region, "f1", c1, "B: extension", v, e1, p_holm=ph, effect_f1=e1, declared=rep)
        if c2 is not None:
            emit(region, "f2", c2, "B: extension", "", rep, p_holm=np.nan)
    return pd.DataFrame(rows)


def sens_p_ext(df, real, bs):
    """Empirical p of each sensitivity replicate over draws 0..bs-1 (extension labels). NaN if incomplete/undefined."""
    df = df.reindex(columns=DRAW_COLUMNS).drop_duplicates("draw")
    pf = []
    for k in range(K_FLIPS):
        s = _sens_z(df, f"T_flip{k}", real["T_flip"][k], bs)
        pf.append(np.nan if s is None or "p" not in s else s["p"])
    pa = {}
    for a, col in ((0.5, "T_amp0.5"), (2.0, "T_amp2.0")):
        s = _sens_z(df, col, real["T_amp"][a], bs)
        pa[a] = np.nan if s is None or "p" not in s else s["p"]
    return pf, pa


def _sens_labels_ext(cell, has_effect):
    """Extension labels: a replicate agrees if its z has the main sign AND its empirical p < 0.05 (no |z| rule,
    matching the extension's decision rules). sign-robust: >= 8 of 10 agree; amplitude-robust: both agree."""
    z0 = cell["z"]
    zf = np.array(cell["z_flip"], float)
    pf = np.array(cell.get("p_flip", [np.nan] * K_FLIPS), float)
    agree = [bool(np.isfinite(z) and np.isfinite(q) and _sign(z) == _sign(z0) and q < 0.05) for z, q in zip(zf, pf)]
    cell["n_flip_agree"] = int(sum(agree)) if np.isfinite(z0) else 0
    fin = zf[np.isfinite(zf)]
    cell["z_flip_median"] = float(np.median(fin)) if len(fin) else np.nan
    cell["z_flip_min"] = float(fin.min()) if len(fin) else np.nan
    cell["z_flip_max"] = float(fin.max()) if len(fin) else np.nan
    pa = cell.get("p_amp", {a: np.nan for a in AMPLITUDES})
    amp_ok = all(np.isfinite(cell["z_amp"][a]) and np.isfinite(pa[a]) and _sign(cell["z_amp"][a]) == _sign(z0)
                 and pa[a] < 0.05 for a in AMPLITUDES)
    for a in AMPLITUDES:
        cell[f"p_amp{a}"] = pa[a]
    if cell["status"] != "defined":
        cell["sign_label"] = cell["amp_label"] = ""
    elif not cell["sens_complete"]:
        cell["sign_label"] = cell["amp_label"] = "sensitivity incomplete" if has_effect else ""
    elif has_effect:
        cell["sign_label"] = "sign-robust" if cell["n_flip_agree"] >= 8 else "sign-sensitive"
        cell["amp_label"] = "amplitude-robust" if amp_ok else "amplitude-sensitive"
    else:
        cell["sign_label"] = "sensitivity-only, exploratory" if cell["n_flip_agree"] >= 8 else ""
        cell["amp_label"] = ""
