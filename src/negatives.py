"""Personalised, degree-corrected negative sampling for the gated metric.

Two pitfalls in the link-prediction literature motivate everything here, and
both papers are in ``literature/md/``:

* ``li2023_heart_linkpred_pitfalls`` -- a single globally shared set of random
  negatives makes the task trivially easy, because a random pair of nodes
  almost never shares a neighbour. Negatives must be *personalised*: each
  positive is ranked against corruptions of itself.
* ``implicit_degree_bias_linkpred`` -- positives are degree-biased by
  construction (a node with k edges appears k times in the edge list) while
  uniformly sampled negatives are not, so a pure-degree predictor reaches
  ~0.83 AUC-ROC on real graphs. Negatives must carry the *same* degree bias.

The sampler below does both at once: for every true edge it draws K
corruptions from the neurons innervating the held-out neuropil, with
probability proportional to ``degree ** alpha``.

The resulting sample is generated once, hashed, and committed. Every model --
all three baselines and every GNN variant -- is then scored against
byte-identical negatives. :meth:`NegativeSample.digest` is what makes that
checkable rather than aspirational.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import filterfalse

import numpy as np

__all__ = [
    "DEFAULT_SPEC",
    "SIDE_SOURCE",
    "SIDE_TARGET",
    "InsufficientPoolError",
    "NegativeSample",
    "NegativeSpec",
    "pack_edges",
    "sample_negatives",
]

#: Marker stored alongside each negative recording which endpoint was replaced.
SIDE_TARGET = 0  # (u -> v) became (u -> w): "who else could u talk to?"
SIDE_SOURCE = 1  # (u -> v) became (w -> v): "who else could talk to v?"


class InsufficientPoolError(RuntimeError):
    """K distinct, degree-weighted negatives cannot be built.

    Raised rather than silently returning a short row: MRR is a function of K,
    so a positive ranked against fewer negatives is not comparable to the
    others. It is raised for a pool smaller than ``K + 2`` and for a positive
    with too few positive-weight eligible partners; either aborts the sample.
    """


@dataclass(frozen=True)
class NegativeSpec:
    """Pinned sampling parameters. Any change invalidates the committed sample.

    Attributes
    ----------
    k
        Negatives per positive. Registered at 100 -- MRR depends on K,
        so this is not a free knob to tune after seeing results.
    alpha
        Degree-correction exponent. ``1.0`` in the degree-given setting (the
        gated one), ``0.0`` in the degree-free secondary setting, where it
        reduces to uniform sampling over the pool.
    master_seed
        Combined with the neuropil name to derive a per-neuropil generator, so
        the sample is reproducible and independent of evaluation order.
    """

    k: int = 100
    alpha: float = 1.0
    master_seed: int = 20260818

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")

    @property
    def n_target_side(self) -> int:
        """Corruptions replacing the target. Takes the odd one when K is odd."""
        return self.k - self.k // 2

    @property
    def n_source_side(self) -> int:
        """Corruptions replacing the source."""
        return self.k // 2

    def for_degree_free(self) -> NegativeSpec:
        """The matching spec for the degree-free secondary setting."""
        return replace(self, alpha=0.0)


@dataclass(frozen=True)
class NegativeSample:
    """K personalised negatives per positive, for one neuropil."""

    neuropil: str
    positives: np.ndarray  # (P, 2) int64, columns (source, target)
    negatives: np.ndarray  # (P, K, 2) int64
    sides: np.ndarray  # (P, K) uint8, SIDE_TARGET or SIDE_SOURCE
    spec: NegativeSpec

    @property
    def n_positives(self) -> int:
        return int(self.positives.shape[0])

    def digest(self) -> str:
        """Stable SHA-256 over the sample, for the identical-negatives rule.

        Byte-identical inputs give byte-identical digests across machines, so
        the fleet can assert that the 3090 and the 3070 scored the same
        negatives before their results are pooled.
        """
        h = hashlib.sha256()
        h.update(self.neuropil.encode("utf-8"))
        h.update(
            f"k={self.spec.k};alpha={self.spec.alpha!r};seed={self.spec.master_seed}".encode()
        )
        for arr in (self.positives, self.negatives, self.sides):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()


#: The pinned default. A module-level singleton rather than a call in the
#: signature: NegativeSpec is frozen, so sharing one instance is safe and it
#: gives the pre-registered configuration a single name to refer to.
DEFAULT_SPEC = NegativeSpec()


def pack_edges(edges: Iterable[tuple[int, int]]) -> set[int]:
    """Pack ``(source, target)`` pairs into a set of single ints for lookup.

    A Python ``set`` of tuples costs roughly three times the memory of a set of
    ints at connectome scale, and membership tests dominate the rejection loop.
    Node ids must fit in 32 bits after the caller has reindexed them densely,
    which ``src.graph`` does.
    """
    packed = set()
    for source, target in edges:
        if not (0 <= source < 2**32 and 0 <= target < 2**32):
            raise ValueError(
                f"node ids must be densely reindexed into [0, 2**32); got "
                f"({source}, {target})"
            )
        packed.add((int(source) << 32) | int(target))
    return packed


def _derive_rng(master_seed: int, neuropil: str) -> np.random.Generator:
    """Per-neuropil generator that does not depend on evaluation order.

    Uses blake2b rather than :func:`hash`, whose string hashing is randomised
    per process unless PYTHONHASHSEED is set.
    """
    name_hash = hashlib.blake2b(neuropil.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(
        np.random.SeedSequence([master_seed, int.from_bytes(name_hash, "big")])
    )


def _weights(degree: np.ndarray, alpha: float) -> np.ndarray:
    """Sampling weights ``degree ** alpha``, normalised.

    Neurons with zero degree in the neuropil get zero weight when ``alpha >
    0``, and that is deliberate: a true edge never has a zero-in-degree target,
    so admitting such nodes as negatives would reintroduce exactly the
    positive/negative degree mismatch this module exists to remove.
    """
    degree = np.asarray(degree, dtype=np.float64)
    if (degree < 0).any():
        raise ValueError("degrees must be non-negative")

    weights = np.ones_like(degree) if alpha == 0.0 else degree**alpha
    total = weights.sum()
    if total <= 0:
        raise InsufficientPoolError(
            "every candidate has zero sampling weight; check the degree vector"
        )
    return weights / total


class _Lookup:
    """``np.searchsorted(cumulative, u)`` for a block of uniforms ``u``.

    Bucket ``b`` holds the draws in ``[b / B, (b + 1) / B)``, both ends exact
    because ``B`` is a power of two. Every draw in a bucket that no cumulative
    boundary crosses resolves to the same index, so only draws in the other
    buckets need the binary search. ``B`` grows with the pool, between 2**16
    and 2**20, so that most buckets stay uncrossed.
    """

    def __init__(self, cumulative: np.ndarray) -> None:
        bits = (16 * cumulative.size - 1).bit_length()
        self._buckets = 1 << min(max(bits, 16), 20)
        bounds = np.searchsorted(
            cumulative, np.arange(self._buckets + 1) / self._buckets
        )
        self._cumulative = cumulative
        self._index = bounds[:-1]
        self._crossed = bounds[1:] != bounds[:-1]

    def __call__(self, u: np.ndarray) -> np.ndarray:
        bucket = (u * self._buckets).astype(np.intp)
        index = self._index[bucket]
        search = np.flatnonzero(self._crossed[bucket])
        index[search] = np.searchsorted(self._cumulative, u[search])
        return index


def _cumulative(weights: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(weights)
    # Guard against the last bin being unreachable through floating-point drift.
    cumulative[-1] = 1.0
    return cumulative


class _Candidates:
    """Both sides' candidate for each ``rng.random()`` draw, read ahead in blocks.

    ``Generator.random(n)`` returns the same doubles as ``n`` scalar calls, so
    reading ahead leaves the stream unchanged. Draws are taken with
    :meth:`peek` and :meth:`consume`; :meth:`sync` leaves the generator where
    one scalar call per consumed draw would have, before anything else draws
    from it.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        pool: np.ndarray,
        w_in: np.ndarray,
        w_out: np.ndarray,
        block: int = 1 << 16,
    ) -> None:
        self.rng = rng
        self._pool = pool
        self.weights = {True: w_in, False: w_out}
        self._lookups = {
            True: _Lookup(_cumulative(w_in)), False: _Lookup(_cumulative(w_out))
        }
        self._block = block
        self._state: dict | None = None  # generator state before the block
        self._drawn: dict[bool, list[int]] = {True: [], False: []}
        self._used = 0

    def peek(self, is_target_side: bool, n: int) -> list[int]:
        """The next ``n`` unconsumed draws, as candidates for one side."""
        if self._used + n > len(self._drawn[is_target_side]):
            self.sync()
            self._state = self.rng.bit_generator.state
            u = self.rng.random(max(n, self._block))
            self._drawn = {
                side: self._pool[lookup(u)].tolist()
                for side, lookup in self._lookups.items()
            }
        return self._drawn[is_target_side][self._used : self._used + n]

    def consume(self, n: int) -> None:
        self._used += n

    def sync(self) -> None:
        """Rewind the generator to the consumed position and drop the read-ahead."""
        if self._state is not None:
            self.rng.bit_generator.state = self._state
            self.rng.random(self._used)
        self._state = None
        self._drawn = {True: [], False: []}
        self._used = 0


class _Ineligible:
    """The candidates one side of one positive must reject.

    That is the fixed endpoint and every partner that would recreate a true
    edge. Node ids fit in 32 bits, so a node's partners are a slice of the
    packed keys, kept only for the positives' endpoints and sorted by that end.
    Each endpoint's set is built once per side.
    """

    def __init__(self, true_edges: set[int], positives: np.ndarray) -> None:
        try:
            keys = np.fromiter(true_edges, dtype=np.uint64, count=len(true_edges))
        except OverflowError:
            raise ValueError(
                "true_edges holds an int outside [0, 2**64); build it with pack_edges"
            ) from None
        swapped = (keys << np.uint64(32)) | (keys >> np.uint64(32))
        self._positives = positives.tolist()
        self._slices = {
            True: self._partner_slices(keys, positives[:, 0]),
            False: self._partner_slices(swapped, positives[:, 1]),
        }
        self._built: dict[tuple[int, bool], set[int]] = {}

    @staticmethod
    def _partner_slices(keys: np.ndarray, nodes: np.ndarray):
        """Partners (low halves) ordered by fixed end (high half), and each node's span."""
        nodes = nodes.astype(np.uint64)
        keys = keys[np.isin(keys >> np.uint64(32), nodes)]
        keys.sort()
        high = keys >> np.uint64(32)
        return (
            keys & np.uint64(0xFFFFFFFF),
            np.searchsorted(high, nodes, side="left").tolist(),
            np.searchsorted(high, nodes, side="right").tolist(),
        )

    def __call__(self, row: int, is_target_side: bool) -> set[int]:
        fixed_node = self._positives[row][0 if is_target_side else 1]
        ineligible = self._built.get((fixed_node, is_target_side))
        if ineligible is None:
            partners, starts, stops = self._slices[is_target_side]
            ineligible = set(partners[starts[row] : stops[row]].tolist())
            ineligible.add(fixed_node)
            self._built[fixed_node, is_target_side] = ineligible
        return ineligible


def _sample_side(
    candidates: _Candidates,
    ineligible: set[int],
    n_wanted: int,
    is_target_side: bool,
    pool: np.ndarray,
    fixed_node: int,
    neuropil: str,
) -> list[int]:
    """Draw ``n_wanted >= 1`` distinct eligible partners for one positive.

    Rejection sampling: one draw per candidate, accepting the first occurrence
    of each eligible one, for at most ``max_draws`` draws. A window of draws is
    screened at once and only the draws up to the ``n_wanted``-th acceptance
    are consumed, so the result and the generator's position match drawing one
    candidate at a time.
    """
    max_draws = 50 * n_wanted + 100
    window = min(max_draws, n_wanted + n_wanted // 4 + 16)
    while True:
        drawn = candidates.peek(is_target_side, window)
        # dict.fromkeys keeps each candidate's first occurrence, in draw order.
        accepted = list(dict.fromkeys(filterfalse(ineligible.__contains__, drawn)))
        if len(accepted) >= n_wanted:
            candidates.consume(drawn.index(accepted[n_wanted - 1]) + 1)
            return accepted[:n_wanted]
        if window == max_draws:
            break
        window = min(max_draws, 2 * window)

    # Rejection sampling stalled: the eligible weight is small. Enumerate the
    # eligible set and draw from it with the same weights, so zero-weight
    # nodes stay excluded.
    candidates.consume(max_draws)
    candidates.sync()
    keep = np.fromiter(
        (w not in ineligible for w in pool.tolist()), dtype=bool, count=pool.size
    )
    weights = candidates.weights[is_target_side][keep]
    n_positive = int(np.count_nonzero(weights > 0))
    if n_positive < n_wanted:
        side = "target" if is_target_side else "source"
        raise InsufficientPoolError(
            f"neuropil {neuropil!r}: node {fixed_node} has only {n_positive} "
            f"eligible partners with nonzero {side}-side weight but {n_wanted} "
            f"negatives are required, so this positive cannot be ranked against "
            f"K degree-weighted negatives."
        )
    return candidates.rng.choice(
        pool[keep], size=n_wanted, replace=False, p=weights / weights.sum()
    ).tolist()


def _draw_negatives(
    rng: np.random.Generator,
    positives: np.ndarray,
    pool: np.ndarray,
    w_in: np.ndarray,
    w_out: np.ndarray,
    n_target: np.ndarray,
    k: int,
    true_edges: set[int],
    neuropil: str,
    block: int = 1 << 16,
) -> tuple[np.ndarray, np.ndarray]:
    """The ``(P, K, 2)`` negatives and ``(P, K)`` sides, drawn from ``rng``.

    ``w_in`` and ``w_out`` are the normalised target-side and source-side
    weights aligned with ``pool``. ``block`` is how many draws to read ahead at
    a time; it never changes the result.
    """
    candidates = _Candidates(rng, pool, w_in, w_out, block)
    ineligible = _Ineligible(true_edges, positives)
    replacements: list[int] = []  # row-major (P, K)
    for row, (source, target) in enumerate(positives.tolist()):
        n_tgt = int(n_target[row])
        for is_target_side, n_wanted, fixed_node in (
            (True, n_tgt, source),
            (False, k - n_tgt, target),
        ):
            if n_wanted == 0:
                continue  # drawing one at a time, this side takes no draws
            replacements += _sample_side(
                candidates, ineligible(row, is_target_side), n_wanted,
                is_target_side, pool, fixed_node, neuropil,
            )
    candidates.sync()

    replaced = np.asarray(replacements, dtype=np.int64).reshape(len(positives), k)
    is_source_side = np.arange(k) >= np.asarray(n_target)[:, None]
    negatives = np.empty((len(positives), k, 2), dtype=np.int64)
    negatives[..., 0] = np.where(is_source_side, replaced, positives[:, :1])
    negatives[..., 1] = np.where(is_source_side, positives[:, 1:], replaced)
    sides = np.where(is_source_side, SIDE_SOURCE, SIDE_TARGET).astype(np.uint8)
    return negatives, sides


def _target_side_counts(
    positives: np.ndarray, masked_nodes: np.ndarray, *, spec: NegativeSpec
) -> np.ndarray:
    """Per-positive target-side counts that keep each positive's masked endpoint.

    A decoy that drops the masked endpoint is not a query about the masked
    neuron, and ``VisibleView.is_masked`` makes it trivially separable, so every
    negative keeps one. When exactly one endpoint is masked only the other is
    replaced; when both are, the K draws split as ``spec.n_target_side``.
    """
    flags = np.asarray(masked_nodes, dtype=bool)
    highest = int(positives.max())
    if flags.ndim != 1 or flags.size <= highest:
        raise ValueError(
            f"masked_nodes must be one flag per node covering every positive "
            f"endpoint; got shape {flags.shape} for endpoints up to {highest}"
        )
    source_masked = flags[positives[:, 0]]
    target_masked = flags[positives[:, 1]]
    if not (source_masked | target_masked).all():
        raise ValueError(
            f"{(~(source_masked | target_masked)).sum()} positive edge(s) touch "
            f"no masked node; masked_nodes does not describe these positives"
        )
    return np.where(
        source_masked & target_masked,
        spec.n_target_side,
        np.where(source_masked, spec.k, 0),
    ).astype(np.int64)


def sample_negatives(
    *,
    neuropil: str,
    positives: np.ndarray,
    pool: np.ndarray,
    in_degree: np.ndarray,
    out_degree: np.ndarray,
    true_edges: set[int],
    spec: NegativeSpec = DEFAULT_SPEC,
    masked_nodes: np.ndarray | None = None,
) -> NegativeSample:
    """Build K personalised, degree-corrected negatives per positive.

    Parameters
    ----------
    neuropil
        Held-out neuropil name; seeds this neuropil's generator.
    positives
        ``(P, 2)`` int array of true edges ``(source, target)`` inside the
        neuropil.
    pool
        ``(M,)`` int array of neurons innervating the neuropil -- the candidate
        set. Both endpoints of every negative come from here, which is what
        makes these "hard" negatives in the sense of the approved plan.
    in_degree, out_degree
        ``(M,)`` arrays aligned with ``pool``, giving each candidate's synapse
        count *within this neuropil*. Target-side corruptions are weighted by
        ``in_degree``, source-side by ``out_degree``. In the degree-given
        setting these are the same counts already handed to every model as a
        feature, so using them here leaks nothing new; in the degree-free
        setting ``alpha=0`` ignores them.
    true_edges
        Output of :func:`pack_edges` over *all* known edges among the pool, so
        a real connection is never handed to a model as a negative. Must
        include the held-out edges themselves.
    spec
        Pinned sampling parameters.
    masked_nodes
        ``(N,)`` boolean flag per dense node id, where ``N`` covers every
        positive endpoint. When given, each positive keeps its masked endpoint: a
        single-masked edge has only its visible partner replaced, a both-masked
        edge splits as ``spec.n_target_side``. ``None`` keeps the plain 50/50 split.

    Raises
    ------
    InsufficientPoolError
        If any positive cannot be given K distinct eligible negatives with
        nonzero weight.
    """
    positives = np.asarray(positives, dtype=np.int64)
    pool = np.asarray(pool, dtype=np.int64)
    in_degree = np.asarray(in_degree, dtype=np.float64)
    out_degree = np.asarray(out_degree, dtype=np.float64)

    if positives.ndim != 2 or positives.shape[1] != 2:
        raise ValueError(f"positives must have shape (P, 2), got {positives.shape}")
    if positives.shape[0] == 0:
        raise ValueError(f"neuropil {neuropil!r} has no positive edges to rank")
    if pool.ndim != 1:
        raise ValueError(f"pool must be 1-D, got shape {pool.shape}")
    if len(np.unique(pool)) != pool.size:
        raise ValueError("pool must not contain duplicate node ids")
    if pool.size and (pool.min() < 0 or pool.max() >= 2**32):
        raise ValueError(
            f"pool node ids must be densely reindexed into [0, 2**32), as "
            f"pack_edges requires; got [{pool.min()}, {pool.max()}]"
        )
    if in_degree.shape != pool.shape or out_degree.shape != pool.shape:
        raise ValueError(
            f"degree vectors must align with pool {pool.shape}; got "
            f"in_degree {in_degree.shape}, out_degree {out_degree.shape}"
        )
    if pool.size < spec.k + 2:
        raise InsufficientPoolError(
            f"neuropil {neuropil!r}: pool of {pool.size} neurons cannot supply "
            f"{spec.k} distinct negatives per positive"
        )

    missing = set(np.unique(positives).tolist()) - set(pool.tolist())
    if missing:
        raise ValueError(
            f"neuropil {neuropil!r}: {len(missing)} endpoint(s) of the positive "
            f"edges are absent from the candidate pool, e.g. {sorted(missing)[:5]}"
        )

    rng = _derive_rng(spec.master_seed, neuropil)
    w_in = _weights(in_degree, spec.alpha)
    w_out = _weights(out_degree, spec.alpha)

    n_target = (
        np.full(positives.shape[0], spec.n_target_side, dtype=np.int64)
        if masked_nodes is None
        else _target_side_counts(positives, masked_nodes, spec=spec)
    )
    negatives, sides = _draw_negatives(
        rng, positives, pool, w_in, w_out, n_target, spec.k, true_edges, neuropil
    )

    return NegativeSample(
        neuropil=neuropil,
        positives=positives,
        negatives=negatives,
        sides=sides,
        spec=spec,
    )
