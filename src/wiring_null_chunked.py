"""Memory-safe exact S2 for the wiring-null.

Implementation-only change: same statistic as src/wiring_null.stats, but the triangle count
tri = sum(B .* (B @ B)) is accumulated over row blocks, so B @ B is never materialised.
Triangle counts are integer sums of 0/1 products, so the result is exactly equal.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src import wiring_null as wn

BYTES_PER_PRODUCT_NNZ = 24  # float64 data + int32 index + temporaries (conservative)
DEFAULT_BUDGET_BYTES = 300 * 2**20


def _undirected(src, dst, n):
    a = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    return ((a + a.T) > 0).astype(np.float64).tocsr()


def _blocks(b, block_rows, budget_bytes):
    """Row ranges; each holds <= block_rows rows and <= budget/24 upper-bound product entries
    (a single row that alone exceeds the budget still forms its own block)."""
    deg = np.diff(b.indptr)
    flops = np.asarray(b @ deg.astype(np.float64)).ravel()  # bound on nnz of each product row
    cap = max(1, budget_bytes // BYTES_PER_PRODUCT_NNZ)
    start, n = 0, b.shape[0]
    cs = np.concatenate([[0.0], np.cumsum(flops)])
    while start < n:
        end = int(np.searchsorted(cs, cs[start] + cap, side="right")) - 1
        end = min(max(end, start + 1), n, start + (block_rows or n))
        yield start, end
        start = end


def _tri_and_wedges(b, block_rows=None, budget_bytes=DEFAULT_BUDGET_BYTES):
    deg = np.asarray(b.sum(1)).ravel()
    wedges = float((deg * (deg - 1)).sum())
    tri = 0
    for lo, hi in _blocks(b, block_rows, budget_bytes):
        r = b[lo:hi]
        tri += int(r.multiply(r @ b).sum())  # exact: integer-valued float64 sums
    return tri, wedges


def transitivity_chunked(src, dst, n, block_rows=None, budget_bytes=DEFAULT_BUDGET_BYTES):
    """Same value as S2 in wiring_null.stats. block_rows caps rows per block (None: budget only)."""
    tri, wedges = _tri_and_wedges(_undirected(src, dst, n), block_rows, budget_bytes)
    return float(tri) / wedges if wedges else 0.0


def stats_chunked(src, dst, w, n, seed_sets, s4_only=False, block_rows=None,
                  budget_bytes=DEFAULT_BUDGET_BYTES):
    """Same dict as wiring_null.stats; S2 chunked, S4 via the frozen spectral_radius.
    The frozen stats() is monolithic, so the S1/S3 lines below are verbatim copies;
    tests assert exact equality with wn.stats."""
    wn._check(src, dst, n)
    s4 = wn.spectral_radius(src, dst, w, n)
    if s4_only:
        return {"S4_spectral_radius": s4}
    keys = src * n + dst
    s1 = float(np.isin(dst * n + src, keys).mean())
    s2 = transitivity_chunked(src, dst, n, block_rows, budget_bytes)
    a = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    cur = np.zeros((n, len(seed_sets)), np.float32)
    for c, s in enumerate(seed_sets):
        cur[s, c] = 1
    at = a.T.tocsr().astype(np.float32)
    for _ in range(3):
        cur = np.minimum(1.0, cur + at @ cur)
    return {"S1_reciprocity": s1, "S2_transitivity": s2, "S3_spread3": float(cur.mean()),
            "S4_spectral_radius": s4}
