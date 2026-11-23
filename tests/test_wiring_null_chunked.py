"""Exact equivalence of the chunked S2/stats with the frozen src/wiring_null.py."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src import wiring_null as wn
from src import wiring_null_chunked as wc

LX = Path(os.environ.get("FLYARCH_LX_DATA_ROOT", "data"))


def graph(n, m, seed, hub=False):
    rng = np.random.default_rng(seed)
    s, d = rng.integers(0, n, m), rng.integers(0, n, m)
    if hub:  # node 0 linked to/from many nodes
        h = rng.integers(1, n, n)
        s, d = np.concatenate([s, np.zeros(n, int), h]), np.concatenate([d, h, np.zeros(n, int)])
    keys = np.unique(s * n + d)
    s, d = keys // n, keys % n
    k = s != d
    return s[k], d[k], n


def same(c, f):
    """S1-S3 bit-exact; S4 is ARPACK with a random start vector (the frozen code differs from
    itself at ~1e-15 between calls), so it is compared to 1e-9 relative."""
    assert c.keys() == f.keys()
    for k in c:
        if k == "S4_spectral_radius":
            assert c[k] == pytest.approx(f[k], rel=1e-9)
        else:
            assert c[k] == f[k], k


def frozen_tri(src, dst, n):
    a = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    b = ((a + a.T) > 0).astype(np.float64).tocsr()
    return float(b.multiply(b @ b).sum()), b


@pytest.mark.parametrize("n,m,seed,hub", [(60, 300, 1, False), (200, 1500, 2, False),
                                          (150, 400, 3, True), (300, 3000, 4, True)])
@pytest.mark.parametrize("block_rows", [1, 7, None])
def test_stats_exactly_equal(n, m, seed, hub, block_rows):
    src, dst, n = graph(n, m, seed, hub)
    w = np.random.default_rng(seed).random(len(src))
    tri_f, b = frozen_tri(src, dst, n)
    tri_c, _ = wc._tri_and_wedges(b, block_rows, budget_bytes=10_000)
    assert tri_c == tri_f
    ss = wn.make_seed_sets(n, np.random.SeedSequence(0), n_sets=10)
    same(wc.stats_chunked(src, dst, w, n, ss, block_rows=block_rows, budget_bytes=10_000),
         wn.stats(src, dst, w, n, ss))
    same(wc.stats_chunked(src, dst, w, n, ss, s4_only=True), wn.stats(src, dst, w, n, ss, s4_only=True))


LX_EDGES = LX / "processed" / "f1" / "lx" / "edges.parquet"


@pytest.mark.skipif(not LX_EDGES.exists(), reason="LX F1 edges absent")
def test_real_lx_graph_equal():
    src, dst, w, n, _ = wn.load_graph(LX, "f1", "lx")
    ss = wn.make_seed_sets(n, np.random.SeedSequence(0), n_sets=20)
    same(wc.stats_chunked(src, dst, w, n, ss, block_rows=500), wn.stats(src, dst, w, n, ss))


@pytest.mark.skipif(not LX_EDGES.exists(), reason="LX F1 edges absent")
def test_cli_rows_identical(tmp_path):
    from scripts import wiring_null as frozen
    from scripts import wiring_null_chunked as chunked
    outs = []
    for name, mod in (("f", frozen), ("c", chunked)):
        o = tmp_path / f"{name}.csv"
        args = ["run", "--region", "lx", "--split", "f1", "--n-perm", "3", "--end", "3",
                "--data-root", str(LX), "--out", str(o)]
        assert mod.main(args) == 0
        outs.append(pd.read_csv(o))
    f, c = outs
    s4 = "S4_spectral_radius"
    pd.testing.assert_frame_equal(f.drop(columns=s4), c.drop(columns=s4), check_exact=True)
    np.testing.assert_allclose(f[s4], c[s4], rtol=1e-9)
