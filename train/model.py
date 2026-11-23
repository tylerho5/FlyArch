"""M1: a directed GraphSAGE encoder and a two-head pair decoder.

The encoder aggregates in-edges and out-edges separately over the view's
*visible* edges and concatenates the two, so the output width is ``2*hidden``.
Each layer also carries the node's own transformed features, which is what
lets a masked neuron -- one with every incident edge hidden -- still reach its
own embedding. Because ``view.visible_edges`` excludes every edge touching a
masked endpoint, a masked node's features never reach an unmasked neighbour.

Edge message weights are ``log1p(visible_input_fraction)`` per visible edge,
as pinned in the Phase 1 design spec, section 7 (not published). The decoder is
an MLP over ``[h_u, h_v, pair_features]`` with an edge logit head and a
``log1p`` synapse-count head. In v1 the pair-feature block is the masked
endpoint's ``degree_given_total``, the one complete-graph value the registered
rules allow a model at scoring time.

No PyG import lives in this module's body, so the encoder runs on plain torch.
"""

from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.features import FeatureSchema, FeatureTable
from src.masks import VisibleView

__all__ = ["DirectedSAGE", "PairDecoder", "M1", "PreparedPairs"]

#: Opt-in ``M1`` switches with their off values. ``TrainConfig`` carries each
#: under the same name, and every place that rebuilds an ``M1`` from a config
#: passes them through by this table.
MODEL_KNOBS = {
    "type_pair_features": False,
    "pair_geometry": False,
    "sibling_pooling": False,
    "aggregation": "sum",
    "edge_weight": "fraction",
    "nt_edges": False,
    "type_pair_shares": False,
    "annotation_groups": False,
    "type_profile_rank": 0,
    "nt_pair_features": False,
}

#: Per-node ``extras.npz`` arrays each opt-in switch reads. A switch absent here
#: reads only the feature table and the schema.
EXTRAS_KEYS = {
    "annotation_groups": ("supertype", "hemilineage"),
}

#: Most pairs decoded per batch. F1-val scores ~1.4M F1/CX pairs in one call,
#: which overflows an 8 GB card as a single block.
_DECODE_CHUNK = 1 << 18


def _scatter_add(size: int, index: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(size, source.shape[1], dtype=source.dtype, device=source.device)
    return out.index_add_(0, index, source)


def _visible(view: VisibleView) -> VisibleView:
    """Refuse the evaluator's own view.

    ``MaskedView`` carries the full graph, ``hidden_edges``, ``hidden_weights``
    and ``both_masked`` -- the answer key. Handing it to the model would leave
    the section 7 experiment open to a leak no shape test would catch, so the
    model accepts only ``MaskedView.model_view()``.
    """
    if not isinstance(view, VisibleView):
        raise TypeError(
            f"M1 accepts the leak-free VisibleView, not {type(view).__name__}; "
            "pass view.model_view()"
        )
    return view


def _check_type_ids(ids: np.ndarray, vocabulary: int, label: str) -> None:
    if ids.size and (ids.min() < 0 or ids.max() >= vocabulary):
        raise ValueError(
            f"{label} ids run {int(ids.min())}..{int(ids.max())}, but the "
            f"schema's type embedding covers 0..{vocabulary - 1}; the table "
            f"was built with a different FeatureSchema"
        )


#: Encoder aggregations: plain weighted sum, the sum scaled by
#: ``1/sqrt((1+deg_source)(1+deg_target))`` per edge, or attention weights that
#: reweight neighbours while keeping the sum's scale.
AGGREGATIONS = ("sum", "degree_norm", "attention")

#: Edge message weights: ``log1p`` of the visible input fraction (pinned), or
#: ``0.1 * log1p(synapse count)``, which spans the same ~0.2..0.7 range.
EDGE_WEIGHTS = ("fraction", "count")


def _edge_softmax(scores: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of ``scores`` within each ``index`` group, times the group's size."""
    # Bounded scores keep exp finite without a per-group max.
    weights = torch.exp(5 * torch.tanh(scores / 5))
    total = torch.zeros(n, dtype=weights.dtype, device=weights.device).index_add_(
        0, index, weights
    )
    size = torch.zeros_like(total).index_add_(0, index, torch.ones_like(weights))
    return weights / total[index] * size[index]


class _DirectedSAGELayer(nn.Module):
    """One in/out-separated aggregation plus a self path."""

    def __init__(
        self, in_dim: int, hidden: int, aggregation: str = "sum", n_nt: int = 0
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.aggregation = aggregation
        self.self_linear = nn.Linear(in_dim, hidden)
        self.in_linear = nn.Linear(in_dim, hidden)
        self.out_linear = nn.Linear(in_dim, hidden)
        if aggregation == "attention":
            self.in_attention = nn.Linear(2 * hidden, 1)
            self.out_attention = nn.Linear(2 * hidden, 1)
        # Per presynaptic transmitter, a gate on each message channel;
        # ``2*sigmoid(0) = 1``, so a fresh layer starts as the ungated one.
        self.nt_gates = nn.Embedding(n_nt, 2 * hidden) if n_nt else None
        if self.nt_gates is not None:
            nn.init.zeros_(self.nt_gates.weight)

    def forward(
        self,
        x: torch.Tensor,
        edges: torch.Tensor,
        weights: torch.Tensor,
        edge_nt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = x.shape[0]
        self_features = torch.relu(self.self_linear(x))
        if edges.shape[0]:
            source, target = edges[:, 0], edges[:, 1]
            # Linear+ReLU are row-wise, so evaluating them once on the node
            # rows and gathering the endpoints is identical to evaluating them
            # on the edge rows; E is ~20x N at CX scale.
            h_in = torch.relu(self.in_linear(x))
            h_out = torch.relu(self.out_linear(x))
            in_weights = out_weights = weights
            if self.aggregation == "degree_norm":
                ones = torch.ones_like(weights)
                out_degree = torch.zeros(n, device=x.device).index_add_(0, source, ones)
                in_degree = torch.zeros(n, device=x.device).index_add_(0, target, ones)
                norm = torch.rsqrt((1 + out_degree[source]) * (1 + in_degree[target]))
                in_weights = out_weights = weights * norm
            elif self.aggregation == "attention":
                in_score = self.in_attention(
                    torch.cat([h_in[source], self_features[target]], dim=1)
                ).squeeze(-1)
                out_score = self.out_attention(
                    torch.cat([h_out[target], self_features[source]], dim=1)
                ).squeeze(-1)
                in_weights = weights * _edge_softmax(in_score, target, n)
                out_weights = weights * _edge_softmax(out_score, source, n)
            in_message = h_in[source] * in_weights[:, None]
            out_message = h_out[target] * out_weights[:, None]
            if self.nt_gates is not None:
                gate = 2 * torch.sigmoid(self.nt_gates(edge_nt))
                in_message = in_message * gate[:, : self.hidden]
                out_message = out_message * gate[:, self.hidden :]
            in_aggregate = _scatter_add(n, target, in_message)
            out_aggregate = _scatter_add(n, source, out_message)
        else:
            in_aggregate = x.new_zeros((n, self.hidden))
            out_aggregate = x.new_zeros((n, self.hidden))
        return torch.cat(
            [in_aggregate + self_features, out_aggregate + self_features], dim=1
        )


class DirectedSAGE(nn.Module):
    """``layers`` directed message-passing layers; output width is ``2*hidden``.

    ``dropout`` applies to each layer's output, after an optional LayerNorm.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
        layer_norm: bool = False,
        residual: bool = False,
        branch_norm: bool = False,
        aggregation: str = "sum",
        n_nt: int = 0,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.branch_norm = branch_norm
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        widths = [in_dim] + [2 * hidden] * (layers - 1)
        self.layers = nn.ModuleList(
            _DirectedSAGELayer(width, hidden, aggregation, n_nt) for width in widths
        )
        self.norms = (
            nn.ModuleList(nn.LayerNorm(2 * hidden) for _ in widths) if layer_norm else None
        )
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edges: torch.Tensor,
        weights: torch.Tensor,
        edge_nt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gates = () if edge_nt is None else (edge_nt,)
        for i, layer in enumerate(self.layers):
            out = layer(x, edges, weights, *gates)
            res = self.residual and i > 0
            if self.branch_norm and self.norms is not None:
                out = self.norms[i](out)  # norm the branch, then add (x + norm(f(x)))
                x = x + out if res else out
            else:
                x = x + out if res else out
                if self.norms is not None:
                    x = self.norms[i](x)
            x = self.dropout(x)
        return x


class PairDecoder(nn.Module):
    """MLP over ``[h_u, h_v, pair_features]`` with logit and count heads.

    ``hidden`` is the width of one node embedding (the encoder's ``2*hidden``).
    The first layer's endpoint blocks are evaluated once per node and gathered per pair.
    ``dropout`` applies after each ReLU in the trunk.
    """

    def __init__(
        self, hidden: int, pair_dim: int, dropout: float = 0.0, product_rank: int = 0
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        # product_rank > 0 adds a DistMult-style term: rank-r node projections
        # multiplied elementwise per pair, fed into the trunk's first layer.
        self.product_rank = product_rank
        if product_rank:
            self.prod_source = nn.Linear(hidden, product_rank, bias=False)
            self.prod_target = nn.Linear(hidden, product_rank, bias=False)
            self.prod_out = nn.Linear(product_rank, hidden, bias=False)
        width = 2 * hidden + pair_dim
        self.trunk = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.edge_head = nn.Linear(hidden, 1)
        self.count_head = nn.Linear(hidden, 1)

    def project(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each node's first-layer term as a pair's source and as its target."""
        weight = self.trunk[0].weight
        source = F.linear(nodes, weight[:, : self.hidden])
        target = F.linear(nodes, weight[:, self.hidden : 2 * self.hidden])
        if self.product_rank:  # extra columns carry the product factors
            source = torch.cat([source, self.prod_source(nodes)], dim=1)
            target = torch.cat([target, self.prod_target(nodes)], dim=1)
        return source, target

    def decode(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        ends: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both heads for the pairs ``ends``, from :meth:`project`'s node terms."""
        first = self.trunk[0]
        width = first.out_features
        pre = (
            source[ends[:, 0], :width]
            + target[ends[:, 1], :width]
            + F.linear(pair_features, first.weight[:, 2 * self.hidden :], first.bias)
        )
        if self.product_rank:
            pre = pre + self.prod_out(
                source[ends[:, 0], width:] * target[ends[:, 1], width:]
            )
        if self.dropout:
            features = pre
            for module in self.trunk[1:]:
                features = module(features)
                if isinstance(module, nn.ReLU):
                    features = F.dropout(features, self.dropout, self.training)
        else:
            features = self.trunk[1:](pre)
        return (
            self.edge_head(features).squeeze(-1),
            self.count_head(features).squeeze(-1),
        )


#: Width of the sibling pair-feature block (see :func:`_sibling_features`).
_SIBLING_COLUMNS = 3


def _lookup(keys: torch.Tensor, counts: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """``counts`` at each ``query`` found in the sorted ``keys``, else 0, as float."""
    if not keys.numel():
        return torch.zeros(query.shape, device=query.device)
    position = torch.searchsorted(keys, query).clamp_max(keys.numel() - 1)
    found = keys[position] == query
    return torch.where(found, counts[position], torch.zeros_like(counts[position])).float()


class _SiblingContext(NamedTuple):
    n_nodes: int
    type_ids: torch.Tensor
    masked: torch.Tensor
    keys_out: torch.Tensor
    counts_out: torch.Tensor
    keys_in: torch.Tensor
    counts_in: torch.Tensor
    siblings: torch.Tensor


def _sibling_context(
    n_nodes: int,
    edges: torch.Tensor,
    type_ids: torch.Tensor,
    masked: torch.Tensor,
) -> _SiblingContext:
    """Build the view-wide sibling lookup once for chunked inference."""
    n_types = int(type_ids.max().item()) + 1 if type_ids.numel() else 1
    edge_source, edge_target = edges[:, 0], edges[:, 1]
    keys_out, counts_out = torch.unique(
        type_ids[edge_source] * n_nodes + edge_target, return_counts=True
    )
    keys_in, counts_in = torch.unique(
        type_ids[edge_target] * n_nodes + edge_source, return_counts=True
    )
    connected_nodes = torch.zeros(n_nodes, dtype=torch.bool, device=type_ids.device)
    connected_nodes[edges.reshape(-1)] = True
    members = type_ids[connected_nodes & ~masked]
    siblings = torch.zeros(n_types, device=type_ids.device).index_add_(
        0, members, torch.ones(members.shape[0], device=type_ids.device)
    )
    return _SiblingContext(
        n_nodes, type_ids, masked, keys_out, counts_out, keys_in, counts_in, siblings
    )


def _sibling_features_for_pairs(
    index: torch.Tensor,
    context: _SiblingContext,
    anchor_dropout: torch.Tensor | None = None,
) -> torch.Tensor:
    n_nodes, type_ids, masked = context.n_nodes, context.type_ids, context.masked
    keys_out, counts_out = context.keys_out, context.counts_out
    keys_in, counts_in = context.keys_in, context.counts_in
    siblings = context.siblings
    source, target = index[:, 0], index[:, 1]
    source_masked = masked[source]
    anchor_type = torch.where(source_masked, type_ids[source], type_ids[target])
    flat = torch.where(
        source_masked, anchor_type * n_nodes + target, anchor_type * n_nodes + source
    )
    connected = torch.where(
        source_masked, _lookup(keys_out, counts_out, flat), _lookup(keys_in, counts_in, flat)
    )
    group = siblings[anchor_type]
    live = (source_masked | masked[target]) & (anchor_type > 0)
    if anchor_dropout is not None:
        live &= ~anchor_dropout[torch.where(source_masked, source, target)]
    connected = torch.where(live, connected, torch.zeros_like(connected))
    group = torch.where(live, group, torch.zeros_like(group))
    columns = [connected / group.clamp_min(1), torch.log1p(connected), torch.log1p(group)]
    return torch.stack(columns, dim=1)


def _sibling_features(
    prepared: "PreparedPairs",
    type_ids: torch.Tensor,
    anchor_dropout: torch.Tensor | None = None,
) -> torch.Tensor:
    """How the masked endpoint's visible same-type siblings connect to the candidate.

    For a pair whose masked endpoint ``u`` has type ``k`` and whose other end
    is ``v``: ``c`` = the number of type-``k`` neurons with a visible edge to
    ``v`` in the pair's direction, ``n`` = the number of unmasked type-``k``
    neurons with any visible edge. Columns are ``c / n``, ``log1p(c)``, ``log1p(n)``; all
    zero when ``u`` reads as OOV or neither end is masked. Built only from the
    visible edges, the mask flags, and the model's own undropped type ids, so
    training counts match evaluation; ``anchor_dropout`` (training's type
    dropout mask) zeros the rows whose ``u`` was dropped, as an OOV ``u`` is.
    """
    masked = prepared.is_masked
    if masked is None:
        raise ValueError("sibling features need the pairs' is_masked flags")
    context = _sibling_context(prepared.n_nodes, prepared.edges, type_ids, masked)
    return _sibling_features_for_pairs(prepared.index, context, anchor_dropout)


#: Width of the type-pair block (see :func:`_type_pair_columns`).
_TYPE_PAIR_COLUMNS = 4
#: Width of the geometry block (see :meth:`M1._geometry_columns_for`).
_GEOMETRY_COLUMNS = 7
#: Widths of the later opt-in blocks, in the order :meth:`M1._extra_pair_columns` appends them.
_SHARE_COLUMNS = 4
_ANNOTATION_COLUMNS = 2
_PROFILE_COLUMNS = 2
_NT_PAIR_COLUMNS = 2


class _GroupPairs(NamedTuple):
    """Visible-edge counts between groups of neurons (cell types, superclasses)."""

    n_groups: int
    ids: torch.Tensor
    members: torch.Tensor
    keys: torch.Tensor
    counts: torch.Tensor
    weighted: torch.Tensor
    #: Per group: visible edges (and their weight) leaving it and entering it.
    out_edges: torch.Tensor | None = None
    in_edges: torch.Tensor | None = None
    out_weight: torch.Tensor | None = None
    in_weight: torch.Tensor | None = None


def _group_pairs(
    edges: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, masked: torch.Tensor
) -> _GroupPairs:
    """Directed group-to-group edge counts over the visible edges; group 0 is unknown.

    ``members`` counts the unmasked neurons of each group with a visible edge,
    the neurons whose edges the counts can contain.
    """
    n_groups = int(ids.max().item()) + 1 if ids.numel() else 1
    flat = ids[edges[:, 0]] * n_groups + ids[edges[:, 1]]
    keys, inverse, counts = torch.unique(flat, return_inverse=True, return_counts=True)
    weighted = torch.zeros(len(keys), device=ids.device).index_add_(0, inverse, weights)
    connected = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    connected[edges.reshape(-1)] = True
    members = torch.bincount(ids[connected & ~masked], minlength=n_groups)
    source, target = ids[edges[:, 0]], ids[edges[:, 1]]
    zeros = torch.zeros(n_groups, device=ids.device)
    return _GroupPairs(
        n_groups, ids, members, keys, counts, weighted,
        out_edges=torch.bincount(source, minlength=n_groups).float(),
        in_edges=torch.bincount(target, minlength=n_groups).float(),
        out_weight=zeros.index_add(0, source, weights),
        in_weight=zeros.index_add(0, target, weights),
    )


def _group_pair_shares(
    index: torch.Tensor, table: _GroupPairs, live: torch.Tensor
) -> torch.Tensor:
    """Per pair: the share of the source group's visible output (edges, then weight) going
    to the target group, and the share of the target group's input coming from the source.

    Densities are 0/1 noise for groups of 2-3 neurons; a share of the group's whole
    output ranks the partner groups by how much of it each one takes.
    """
    source, target = table.ids[index[:, 0]], table.ids[index[:, 1]]
    live = live & (source > 0) & (target > 0)
    flat = source * table.n_groups + target
    count = _lookup(table.keys, table.counts, flat)
    weighted = _lookup(table.keys, table.weighted, flat)
    columns = torch.stack([
        count / table.out_edges[source].clamp_min(1),
        count / table.in_edges[target].clamp_min(1),
        weighted / table.out_weight[source].clamp_min(1e-6),
        weighted / table.in_weight[target].clamp_min(1e-6),
    ], dim=1)
    return torch.where(live[:, None], columns, torch.zeros_like(columns))


def _low_rank(matrix: torch.Tensor, rank: int, iterations: int = 2):
    """A deterministic randomized SVD: ``matrix ~ u @ diag(s) @ v.T`` at ``rank``.

    The sketch matrix comes from a fixed CPU generator, so the same view always
    yields the same factors and scoring stays reproducible.
    """
    generator = torch.Generator(device="cpu").manual_seed(0)
    width = min(rank + 8, *matrix.shape)
    sketch = torch.randn(matrix.shape[1], width, generator=generator).to(matrix.device)
    basis = torch.linalg.qr(matrix @ sketch).Q
    for _ in range(iterations):
        basis = torch.linalg.qr(matrix @ (matrix.T @ basis)).Q
    left, values, right = torch.linalg.svd(basis.T @ matrix, full_matrices=False)
    return (basis @ left)[:, :rank], values[:rank], right[:rank].T


class _TypeProfile(NamedTuple):
    sender: torch.Tensor  # per type: low-rank factor as a source
    receiver: torch.Tensor  # per type: low-rank factor as a target
    senders_into: torch.Tensor  # per node: mean sender factor of its visible inputs
    receivers_of: torch.Tensor  # per node: mean receiver factor of its visible outputs


def _type_profile(
    edges: torch.Tensor, type_ids: torch.Tensor, n_types: int, rank: int
) -> _TypeProfile:
    """Low-rank factors of the visible type x type matrix, and each node's partners' factors.

    A type pair with no visible edge still scores high when the two types connect
    the way other connected pairs do.
    """
    source, target = type_ids[edges[:, 0]], type_ids[edges[:, 1]]
    matrix = torch.zeros(n_types * n_types, device=type_ids.device).index_add_(
        0, source * n_types + target, torch.ones(len(source), device=type_ids.device)
    ).view(n_types, n_types)
    matrix[0], matrix[:, 0] = 0, 0  # unknown type
    left, values, right = _low_rank(torch.log1p(matrix), rank)
    sender, receiver = left * values.sqrt(), right * values.sqrt()
    n = type_ids.shape[0]

    def mean_into(index: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(n, factors.shape[1], device=factors.device).index_add_(0, index, factors)
        count = torch.bincount(index, minlength=n).clamp_min(1)
        return total / count[:, None]

    return _TypeProfile(
        sender, receiver,
        senders_into=mean_into(edges[:, 1], sender[source]),
        receivers_of=mean_into(edges[:, 0], receiver[target]),
    )


def _type_profile_columns(
    index: torch.Tensor, profile: _TypeProfile, type_ids: torch.Tensor,
    masked: torch.Tensor, live: torch.Tensor,
) -> torch.Tensor:
    """Per pair: the low-rank type x type score, and how well the visible end's partners
    match the masked end's type (cosine of factors)."""
    source, target = index[:, 0], index[:, 1]
    source_type, target_type = type_ids[source], type_ids[target]
    live = live & (source_type > 0) & (target_type > 0)
    score = (profile.sender[source_type] * profile.receiver[target_type]).sum(dim=1)
    match = torch.where(
        masked[source],
        F.cosine_similarity(profile.sender[source_type], profile.senders_into[target], dim=1),
        F.cosine_similarity(profile.receiver[target_type], profile.receivers_of[source], dim=1),
    )
    columns = torch.stack([score, match], dim=1)
    return torch.where(live[:, None], columns, torch.zeros_like(columns))


class _NTContext(NamedTuple):
    nt: torch.Tensor  # per node: transmitter class id
    has_nt: torch.Tensor
    node_inputs: torch.Tensor  # per node: visible input edges by presynaptic transmitter
    type_inputs: torch.Tensor  # per type: the same summed over its visible members


def _nt_context(
    x: torch.Tensor, nt_columns: list[int], edges: torch.Tensor,
    type_ids: torch.Tensor, masked: torch.Tensor, n_types: int,
) -> _NTContext:
    block = x[:, nt_columns]
    nt, has_nt = block.argmax(dim=1), block.sum(dim=1) > 0
    n, width = x.shape[0], len(nt_columns)
    node_inputs = torch.zeros(n * width, device=x.device).index_add_(
        0, edges[:, 1] * width + nt[edges[:, 0]], torch.ones(len(edges), device=x.device)
    ).view(n, width)
    donors = ~masked & (type_ids > 0)
    type_inputs = torch.zeros(n_types, width, device=x.device).index_add_(
        0, type_ids[donors], node_inputs[donors]
    )
    return _NTContext(nt, has_nt, node_inputs, type_inputs)


def _nt_pair_columns(
    index: torch.Tensor, context: _NTContext, type_ids: torch.Tensor,
    masked: torch.Tensor, live: torch.Tensor,
) -> torch.Tensor:
    """Per pair: how much of the receiving side's visible input carries the sender's transmitter.

    The receiver's own inputs when it is visible; its type-mates' when it is masked.
    Columns: that share, and ``log1p`` of the matching input count.
    """
    source, target = index[:, 0], index[:, 1]
    inputs = torch.where(
        masked[target][:, None], context.type_inputs[type_ids[target]], context.node_inputs[target]
    )
    matching = inputs.gather(1, context.nt[source][:, None]).squeeze(1)
    live = live & context.has_nt[source] & (~masked[target] | (type_ids[target] > 0))
    columns = torch.stack([matching / inputs.sum(dim=1).clamp_min(1), torch.log1p(matching)], dim=1)
    return torch.where(live[:, None], columns, torch.zeros_like(columns))


def _group_pair_density(
    index: torch.Tensor, table: _GroupPairs, live: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per pair: edge density and weighted density between the two groups, and log1p(edges)."""
    source, target = table.ids[index[:, 0]], table.ids[index[:, 1]]
    live = live & (source > 0) & (target > 0)
    flat = source * table.n_groups + target
    count = _lookup(table.keys, table.counts, flat)
    weighted = _lookup(table.keys, table.weighted, flat)
    possible = (table.members[source] * table.members[target]).float()
    possible = possible - torch.where(source == target, table.members[source], 0).float()
    zero = torch.zeros_like(count)
    count, weighted = torch.where(live, count, zero), torch.where(live, weighted, zero)
    possible = possible.clamp_min(1)
    return count / possible, weighted / possible, torch.log1p(count)


def _type_pair_columns(
    index: torch.Tensor,
    types: _GroupPairs,
    classes: _GroupPairs,
    masked: torch.Tensor,
    anchor_dropout: torch.Tensor | None = None,
) -> torch.Tensor:
    """How densely the pair's two cell types (and superclasses) connect in the visible graph.

    The sibling block asks whether the masked neuron's type-mates reach *this*
    candidate; this one asks whether they reach the candidate's *type*, which
    still fires when the mates connect to the candidate's homolog. Columns:
    type density, input-weighted type density, ``log1p`` type edges, and
    superclass density as the fallback for rare types. Zero unless one end is
    masked; a type-dropped masked end zeros them, as in the sibling block.
    """
    live = masked[index[:, 0]] | masked[index[:, 1]]
    if anchor_dropout is not None:
        anchor = torch.where(masked[index[:, 0]], index[:, 0], index[:, 1])
        live = live & ~anchor_dropout[anchor]
    density, weighted, log_edges = _group_pair_density(index, types, live)
    class_density, _, _ = _group_pair_density(index, classes, live)
    return torch.stack([density, weighted, log_edges, class_density], dim=1)


@dataclass(frozen=True, eq=False)
class PreparedPairs:
    """A view's encoder inputs and a pair block on the model's device, for repeated scoring."""

    n_nodes: int
    edges: torch.Tensor
    weights: torch.Tensor
    index: torch.Tensor
    pair_features: torch.Tensor
    row_width: int
    is_masked: torch.Tensor | None = None


class M1(nn.Module):
    """The region expert: encodes a masked view and scores candidate pairs.

    The ``cell_type`` column holds integer ids (1..T, 0 for OOV), so the
    encoder replaces it with a learned ``type_embedding`` of that id rather
    than reading it as a magnitude. ``sibling_features`` widens the decoder's
    first layer by the sibling columns, so a checkpoint loads only into an
    ``M1`` built with the same flag.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        hidden: int = 128,
        type_embedding: int = 32,
        *,
        layers: int = 2,
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
        sibling_features: bool = False,
        layer_norm: bool = False,
        residual: bool = False,
        branch_norm: bool = False,
        decoder_product_rank: int = 0,
        type_pair_features: bool = False,
        pair_geometry: bool = False,
        sibling_pooling: bool = False,
        aggregation: str = "sum",
        edge_weight: str = "fraction",
        nt_edges: bool = False,
        type_pair_shares: bool = False,
        annotation_groups: bool = False,
        type_profile_rank: int = 0,
        nt_pair_features: bool = False,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.sibling_features = sibling_features
        self.type_pair_features = type_pair_features
        self.pair_geometry = pair_geometry
        self.sibling_pooling = sibling_pooling
        self.nt_edges = nt_edges
        self.type_pair_shares = type_pair_shares
        self.annotation_groups = annotation_groups
        self.type_profile_rank = type_profile_rank
        self.nt_pair_features = nt_pair_features
        if edge_weight not in EDGE_WEIGHTS:
            raise ValueError(f"edge_weight must be one of {EDGE_WEIGHTS}, got {edge_weight!r}")
        self.edge_weight = edge_weight
        self.hidden = hidden
        self.in_dim = schema.n_features
        names = tuple(map(str, schema.column_names))
        if "cell_type" not in names:
            raise ValueError("the schema has no cell_type column to embed")
        self.type_column = names.index("cell_type")

        def columns(match) -> list[int]:
            """Positions in the encoder input, which drops the cell_type column."""
            return [i - (i > self.type_column) for i, name in enumerate(names) if match(name)]

        self.centroid_columns = columns(lambda n: n.startswith("centroid_"))
        self.shape_columns = columns(lambda n: n.startswith(("centroid_", "extent_")))
        self.side_columns = columns(lambda n: n.startswith("side_") and n != "side_na")
        self.innervation_columns = columns(lambda n: n.startswith("innervation_"))
        self.superclass_columns = columns(
            lambda n: n.startswith("superclass_") and n != "superclass_OOV"
        )
        self.nt_columns = columns(lambda n: n.startswith("nt_"))
        self.type_embedding = nn.Embedding(len(schema.type_vocab) + 1, type_embedding)
        # Unit-norm rows, on the scale of the one-hot float features.
        nn.init.normal_(self.type_embedding.weight, std=type_embedding**-0.5)
        # A buffer, so a checkpoint carries it to F2 scoring.
        self.register_buffer(
            "trained_types",
            torch.ones(self.type_embedding.num_embeddings, dtype=torch.bool),
        )
        self.encoder = DirectedSAGE(
            self.in_dim - 1 + type_embedding,
            hidden=hidden,
            layers=layers,
            dropout=encoder_dropout,
            layer_norm=layer_norm,
            residual=residual,
            branch_norm=branch_norm,
            aggregation=aggregation,
            n_nt=len(self.nt_columns) if nt_edges else 0,
        )
        self.decoder = PairDecoder(
            2 * hidden,
            pair_dim=1
            + (_SIBLING_COLUMNS if sibling_features else 0)
            + (_TYPE_PAIR_COLUMNS if type_pair_features else 0)
            + (_GEOMETRY_COLUMNS if pair_geometry else 0)
            + (_SHARE_COLUMNS if type_pair_shares else 0)
            + (_ANNOTATION_COLUMNS if annotation_groups else 0)
            + (_PROFILE_COLUMNS if type_profile_rank else 0)
            + (_NT_PAIR_COLUMNS if nt_pair_features else 0),
            dropout=decoder_dropout,
            product_rank=decoder_product_rank,
        )
        if sibling_pooling:
            # Zero init: a fresh model scores exactly as one without pooling.
            self.pool = nn.Linear(2 * hidden, 2 * hidden)
            nn.init.zeros_(self.pool.weight)
            nn.init.zeros_(self.pool.bias)
        self._feature_cache: tuple | None = None
        #: The last table's ``extras`` on the model's device, set by :meth:`_node_features`,
        #: which every scoring path calls before it builds pair features.
        self._extras: dict[str, torch.Tensor] | None = None
        self._extras_source: tuple | None = None

    @property
    def _device(self) -> torch.device:
        """The device the parameters are on, resolved per call.

        Every tensor ``encode``/``score`` builds has to land here: the numpy
        masks, edge indices and pair features are device-agnostic, so nothing
        else would catch a CPU tensor in a CUDA forward pass but the runtime
        mismatch.
        """
        return next(self.encoder.parameters()).device

    def __getstate__(self) -> dict:
        state = dict(super().__getstate__())
        state["_feature_cache"] = None
        state["_extras"] = state["_extras_source"] = None
        return state

    def _static_features(self, table: FeatureTable) -> tuple[torch.Tensor, torch.Tensor]:
        """The table's float columns and type ids on the model's device.

        Checked and converted once per table and device, since every forward
        pass of a fit reuses the same table. The cache holds a weak reference
        and a digest of the values, so it neither keeps the table alive nor
        serves tensors from before an in-place edit.
        """
        device = self._device
        digest = hashlib.blake2b(
            np.ascontiguousarray(table.values).data, digest_size=16
        ).digest()
        cached = self._feature_cache
        if (
            cached is not None
            and cached[0]() is table
            and cached[1] == device
            and cached[2] == digest
        ):
            return cached[3], cached[4]
        # Drop the previous matrix before building this one, so a caller that rebuilds the
        # features per step (an innervation ablation) holds one matrix rather than two.
        self._feature_cache = None
        values = np.asarray(table.values, dtype=np.float32)
        if values.shape[1] != self.in_dim:
            raise ValueError(
                f"feature table has {values.shape[1]} columns, but the schema "
                f"declares {self.in_dim}"
            )
        expected = tuple(map(str, self.schema.column_names))
        names = tuple(map(str, table.columns))
        if names != expected:
            index = next(
                position
                for position, (left, right) in enumerate(zip(names, expected))
                if left != right
            )
            raise ValueError(
                f"feature table columns do not match the schema order: "
                f"position {index} is {names[index]!r}, expected {expected[index]!r}"
            )
        rest = torch.as_tensor(
            np.delete(values, self.type_column, axis=1),
            dtype=torch.float32,
            device=device,
        )
        ids = values[:, self.type_column]
        _check_type_ids(ids, self.type_embedding.num_embeddings, "cell_type")
        type_ids = torch.as_tensor(ids, dtype=torch.long, device=device)
        self._feature_cache = (weakref.ref(table), device, digest, rest, type_ids)
        return rest, type_ids

    def set_trained_types(self, type_ids: np.ndarray) -> None:
        """Keep ``type_ids`` (and OOV) as their own rows; every other id reads as OOV."""
        ids = np.asarray(type_ids, dtype=np.int64)
        _check_type_ids(ids, self.trained_types.numel(), "trained type")
        ids = torch.as_tensor(ids, device=self.trained_types.device)
        trained = torch.zeros_like(self.trained_types)
        trained[ids] = True
        trained[0] = True
        self.trained_types.copy_(trained)

    def _node_features(
        self, table: FeatureTable, type_dropout: np.ndarray | None = None
    ) -> torch.Tensor:
        rest, _ = self._static_features(table)
        type_ids = self.type_ids(table, type_dropout)
        self._load_extras(table)
        return torch.cat([rest, self.type_embedding(type_ids)], dim=1)

    def _load_extras(self, table: FeatureTable) -> None:
        """Move ``table.extras`` to the model's device once per table and device."""
        source = self._extras_source
        if source is not None and source[0]() is table and source[1] == self._device:
            return
        self._extras = None if table.extras is None else {
            name: torch.as_tensor(np.asarray(array), device=self._device)
            for name, array in table.extras.items()
        }
        self._extras_source = (weakref.ref(table), self._device)

    def type_ids(
        self, table: FeatureTable, type_dropout: np.ndarray | None = None
    ) -> torch.Tensor:
        """The type id each node's embedding reads: untrained and dropped types as OOV (0)."""
        _, type_ids = self._static_features(table)
        type_ids = type_ids.masked_fill(~self.trained_types[type_ids], 0)
        if type_dropout is not None:
            drop = torch.as_tensor(
                np.asarray(type_dropout, dtype=bool), device=type_ids.device
            )
            if drop.shape[0] != type_ids.shape[0]:
                raise ValueError(
                    f"type dropout mask has {drop.shape[0]} entries, but the table "
                    f"has {type_ids.shape[0]} rows"
                )
            type_ids = type_ids.masked_fill(drop, 0)
        return type_ids

    def encode(
        self,
        view: VisibleView,
        table: FeatureTable,
        *,
        type_dropout: np.ndarray | None = None,
    ) -> torch.Tensor:
        view = _visible(view)
        x = self._node_features(table, type_dropout=type_dropout)
        return self._encode(x, view.n_nodes, *self._edge_inputs(view))

    def _encode(
        self, x: torch.Tensor, n_nodes: int, edges: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        if x.shape[0] != n_nodes:
            raise ValueError(
                f"feature table has {x.shape[0]} rows, but the view has "
                f"{n_nodes} nodes; the table is not in this graph's node order"
            )
        edge_nt = None
        if self.nt_edges and edges.shape[0]:
            # The synapse's transmitter is its presynaptic neuron's.
            edge_nt = x[:, self.nt_columns].argmax(dim=1)[edges[:, 0]]
        return self.encoder(x, edges, weights, edge_nt)

    def _pool_siblings(
        self,
        hidden: torch.Tensor,
        type_ids: torch.Tensor,
        masked: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        """Add to each masked neuron the mean embedding of its visible same-type neurons.

        A masked neuron has no edges, so its own embedding sees only its
        features; its type-mates' embeddings carry their neighbourhoods. Donors
        are unmasked neurons with a visible edge, so no hidden edge reaches it.
        """
        n_types = self.type_embedding.num_embeddings
        connected = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        connected[edges.reshape(-1)] = True
        donors = connected & ~masked & (type_ids > 0)
        sums = torch.zeros(n_types, hidden.shape[1], device=hidden.device).index_add_(
            0, type_ids[donors], hidden[donors]
        )
        counts = torch.bincount(type_ids[donors], minlength=n_types).clamp_min(1)
        pooled = self.pool((sums / counts[:, None])[type_ids])
        receive = masked & (type_ids > 0)
        return hidden + torch.where(receive[:, None], pooled, torch.zeros_like(pooled))

    @property
    def _pair_context_needs_types(self) -> bool:
        return bool(
            self.type_pair_features or self.type_pair_shares or self.type_profile_rank
            or self.nt_pair_features
        )

    def _pair_context(
        self,
        x: torch.Tensor,
        edges: torch.Tensor,
        weights: torch.Tensor,
        masked: torch.Tensor,
        type_ids: torch.Tensor | None,
    ) -> dict | None:
        """The view-wide tables the opt-in pair blocks read, built once per scoring call."""
        context: dict = {}
        if self._pair_context_needs_types and type_ids is None:
            raise ValueError("type-pair, share, profile and NT pair features need the nodes' type ids")
        if self.type_pair_features or self.type_pair_shares:
            context["types"] = _group_pairs(edges, weights, type_ids, masked)
        if self.type_pair_features:
            block = x[:, self.superclass_columns]
            classes = torch.where(block.sum(dim=1) > 0, block.argmax(dim=1) + 1, 0)
            context["classes"] = _group_pairs(edges, weights, classes, masked)
        if self.annotation_groups:
            if self._extras is None:
                raise ValueError(
                    "annotation-group features need the table's extras; build "
                    "extras.npz with scripts/build_node_extras.py"
                )
        if self.annotation_groups:
            for name in EXTRAS_KEYS["annotation_groups"]:
                context[name] = _group_pairs(edges, weights, self._extras[name].long(), masked)
        n_types = self.type_embedding.num_embeddings
        if self.type_profile_rank:
            context["profile"] = _type_profile(edges, type_ids, n_types, self.type_profile_rank)
        if self.nt_pair_features:
            context["nt"] = _nt_context(x, self.nt_columns, edges, type_ids, masked, n_types)
        context["type_ids"] = type_ids
        return context

    def _geometry_columns_for(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Per pair: |dx|, |dy|, |dz|, centroid distance, both-located, same side, innervation cosine.

        Unlocated neurons sit at the z-scored origin with zero extent rows, so
        ``located`` lets the decoder discount their distances.
        """
        source, target = index[:, 0], index[:, 1]
        delta = (x[source][:, self.centroid_columns] - x[target][:, self.centroid_columns]).abs()
        shape = x[:, self.shape_columns]
        located = (shape.abs().sum(dim=1) > 0).float()
        sides = x[:, self.side_columns]
        has_side = sides.sum(dim=1) > 0
        same_side = (
            has_side[source] & has_side[target]
            & (sides.argmax(dim=1)[source] == sides.argmax(dim=1)[target])
        ).float()
        inn = x[:, self.innervation_columns]
        cosine = F.cosine_similarity(inn[source], inn[target], dim=1, eps=1e-8)
        return torch.cat([
            delta,
            torch.linalg.vector_norm(delta, dim=1, keepdim=True),
            (located[source] * located[target])[:, None],
            same_side[:, None],
            cosine[:, None],
        ], dim=1)

    def _extra_pair_columns(
        self,
        x: torch.Tensor,
        index: torch.Tensor,
        masked: torch.Tensor,
        context: dict | None,
        anchor_dropout: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """The opt-in pair blocks for one decoder chunk.

        Order: type-pair, geometry, shares, annotation groups, type profile, NT pair,
        sites; each present only when its switch is on. Type-derived blocks are zero
        unless one end is masked, and zero when training's type dropout hid the
        masked end's type, as the sibling block is.
        """
        extra = []
        live = masked[index[:, 0]] | masked[index[:, 1]]
        typed_live = live
        if anchor_dropout is not None:
            anchor = torch.where(masked[index[:, 0]], index[:, 0], index[:, 1])
            typed_live = live & ~anchor_dropout[anchor]
        if self.type_pair_features:
            extra.append(_type_pair_columns(
                index, context["types"], context["classes"], masked, anchor_dropout
            ))
        if self.pair_geometry:
            extra.append(self._geometry_columns_for(x, index))
        if self.type_pair_shares:
            extra.append(_group_pair_shares(index, context["types"], typed_live))
        if self.annotation_groups:
            extra.append(torch.stack([
                _group_pair_density(index, context[name], live)[0]
                for name in ("supertype", "hemilineage")
            ], dim=1))
        if self.type_profile_rank:
            extra.append(_type_profile_columns(
                index, context["profile"], context["type_ids"], masked, typed_live
            ))
        if self.nt_pair_features:
            extra.append(_nt_pair_columns(
                index, context["nt"], context["type_ids"], masked, typed_live
            ))
        return extra

    def _edge_inputs(self, view: VisibleView) -> tuple[torch.Tensor, torch.Tensor]:
        edges = torch.as_tensor(
            view.visible_edges, dtype=torch.long, device=self._device
        )
        if self.edge_weight == "count":
            counts = np.asarray(view.visible_weights, dtype=np.float32)
            return edges, 0.1 * torch.log1p(
                torch.as_tensor(counts, dtype=torch.float32, device=self._device)
            )
        fractions = np.asarray(view.visible_input_fraction(), dtype=np.float32)
        weights = torch.log1p(
            torch.as_tensor(fractions, dtype=torch.float32, device=self._device)
        )
        return edges, weights

    def _pair_features(self, view: VisibleView, pairs: np.ndarray) -> torch.Tensor:
        degree = np.asarray(view.degree_given_total(), dtype=np.float32)
        masked = np.asarray(view.is_masked, dtype=bool)
        return self._pair_features_from_context(pairs, degree, masked)

    def _pair_features_from_context(
        self, pairs: np.ndarray, degree: np.ndarray, masked: np.ndarray
    ) -> torch.Tensor:
        source, target = pairs[:, 0], pairs[:, 1]
        column = np.where(
            masked[source], degree[source], np.where(masked[target], degree[target], 0.0)
        )
        # The raw total runs to ~23k on real CX and dominates the decoder's
        # first layer ~200x over the encoder half; log1p matches the scale the
        # count head already regresses on without changing the ordering.
        column = np.log1p(column)
        return torch.as_tensor(
            column[:, None], dtype=torch.float32, device=self._device
        )

    def prepare(
        self, view: VisibleView, pairs: np.ndarray, *, row_width: int = 1
    ) -> PreparedPairs:
        """``view``'s encoder inputs and ``pairs`` on the model's device.

        ``row_width`` is the length of one ranking row (``K + 1``) in a
        row-major ``pairs`` block; a decoder batch never splits a row.
        """
        view = _visible(view)
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        edges, weights = self._edge_inputs(view)
        return PreparedPairs(
            n_nodes=view.n_nodes,
            edges=edges,
            weights=weights,
            index=torch.as_tensor(pairs, dtype=torch.long, device=self._device),
            pair_features=self._pair_features(view, pairs),
            row_width=row_width,
            is_masked=torch.tensor(
                np.asarray(view.is_masked, dtype=bool), device=self._device
            ),
        )

    def score(
        self,
        view: VisibleView,
        table: FeatureTable,
        pairs: np.ndarray,
        *,
        type_dropout: np.ndarray | None = None,
        row_width: int = 1,
        feature_noise: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score and count heads for ``pairs``; ``row_width`` as in :meth:`prepare`.

        ``feature_noise`` adds Gaussian noise of that standard deviation to the
        z-scored centroid and extent columns of every located neuron, so a
        neuron cannot be recognised by its exact coordinates (training only).
        """
        view = _visible(view)
        x = self._node_features(table, type_dropout=type_dropout)
        if feature_noise:
            shape = x[:, self.shape_columns]
            located = (shape.abs().sum(dim=1, keepdim=True) > 0).to(x.dtype)
            x = x.clone()
            x[:, self.shape_columns] = shape + feature_noise * torch.randn_like(shape) * located
        anchor_dropout = (
            torch.as_tensor(np.asarray(type_dropout, dtype=bool), device=self._device)
            if type_dropout is not None
            else None
        )
        return self.score_prepared(
            x,
            self.prepare(view, pairs, row_width=row_width),
            self.type_ids(table),
            anchor_dropout,
        )

    @torch.no_grad()
    def score_inference(
        self,
        view: VisibleView,
        table: FeatureTable,
        pairs: np.ndarray,
        *,
        row_width: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score complete ranking rows in bounded chunks and return CPU outputs.

        Runs in eval mode whatever mode the model was in, and restores it. The
        graph is encoded and projected once. Pair indices and features are
        materialized only for one decoder chunk at a time; returned logits and
        count estimates are accumulated on CPU to bound device output storage.
        """
        if (
            not isinstance(row_width, (int, np.integer))
            or isinstance(row_width, bool)
            or row_width <= 0
        ):
            raise ValueError("row_width must be a positive integer")
        view = _visible(view)
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"pairs must have shape (n, 2); got {pairs.shape}")
        if len(pairs) % row_width:
            raise ValueError(
                f"pair count {len(pairs)} is not a multiple of ranking row_width {row_width}"
            )

        was_training = self.training
        self.eval()
        try:
            return self._score_inference_eval(view, table, pairs, row_width=row_width)
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def _score_inference_eval(
        self,
        view: VisibleView,
        table: FeatureTable,
        pairs: np.ndarray,
        *,
        row_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._node_features(table)
        edges, weights = self._edge_inputs(view)
        hidden = self._encode(x, view.n_nodes, edges, weights)
        masked_np = np.array(view.is_masked, dtype=bool, copy=True)
        masked = torch.as_tensor(masked_np, dtype=torch.bool, device=self._device)
        uses_types = self.sibling_features or self.sibling_pooling or self._pair_context_needs_types
        type_ids = self.type_ids(table) if uses_types else None
        if self.sibling_pooling:
            hidden = self._pool_siblings(hidden, type_ids, masked, edges)
        source, target = self.decoder.project(hidden)
        degree = np.asarray(view.degree_given_total(), dtype=np.float32)
        sibling_context = (
            _sibling_context(view.n_nodes, edges, type_ids, masked)
            if self.sibling_features
            else None
        )
        pair_context = self._pair_context(x, edges, weights, masked, type_ids)

        step = max(int(row_width), _DECODE_CHUNK - _DECODE_CHUNK % int(row_width))
        logits_cpu: list[torch.Tensor] = []
        counts_cpu: list[torch.Tensor] = []
        for start in range(0, len(pairs), step):
            chunk = pairs[start : start + step]
            index = torch.as_tensor(chunk, dtype=torch.long, device=self._device)
            pair_features = self._pair_features_from_context(chunk, degree, masked_np)
            if sibling_context is not None:
                pair_features = torch.cat(
                    [pair_features, _sibling_features_for_pairs(index, sibling_context)], dim=1
                )
            extra = self._extra_pair_columns(x, index, masked, pair_context)
            if extra:
                pair_features = torch.cat([pair_features, *extra], dim=1)
            logits, counts = self.decoder.decode(source, target, index, pair_features)
            logits_cpu.append(logits.cpu())
            counts_cpu.append(counts.cpu())
            del chunk, index, pair_features, logits, counts
        if not logits_cpu:
            empty = torch.empty((0,), dtype=x.dtype, device="cpu")
            return empty, empty.clone()
        return torch.cat(logits_cpu), torch.cat(counts_cpu)

    def score_prepared(
        self,
        x: torch.Tensor,
        prepared: PreparedPairs,
        type_ids: torch.Tensor | None = None,
        anchor_dropout: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score and count heads from node features ``x`` and prebuilt inputs.

        ``type_ids`` (undropped, from :meth:`type_ids`) is required with sibling
        features; ``anchor_dropout`` is as in :func:`_sibling_features`.
        """
        hidden = self._encode(x, prepared.n_nodes, prepared.edges, prepared.weights)
        uses_types = self.sibling_features or self.sibling_pooling or self._pair_context_needs_types
        if uses_types:
            if type_ids is None:
                raise ValueError("sibling and type-pair features need the nodes' type ids")
            if prepared.is_masked is None:
                raise ValueError("sibling and type-pair features need the pairs' is_masked flags")
        if self.sibling_pooling:
            pool_ids = (
                type_ids.masked_fill(anchor_dropout, 0) if anchor_dropout is not None else type_ids
            )
            hidden = self._pool_siblings(hidden, pool_ids, prepared.is_masked, prepared.edges)
        source, target = self.decoder.project(hidden)
        index, pair_features = prepared.index, prepared.pair_features
        if self.sibling_features:
            pair_features = torch.cat(
                [pair_features, _sibling_features(prepared, type_ids, anchor_dropout)], dim=1
            )
        pair_context = self._pair_context(
            x, prepared.edges, prepared.weights, prepared.is_masked, type_ids
        )

        def decode(rows: slice) -> tuple[torch.Tensor, torch.Tensor]:
            features = pair_features[rows]
            extra = self._extra_pair_columns(
                x, index[rows], prepared.is_masked, pair_context, anchor_dropout
            )
            if extra:
                features = torch.cat([features, *extra], dim=1)
            return self.decoder.decode(source, target, index[rows], features)

        row_width = prepared.row_width
        step = max(row_width, _DECODE_CHUNK - _DECODE_CHUNK % row_width)
        logits, counts = zip(*(
            decode(slice(start, start + step))
            for start in range(0, max(len(index), 1), step)
        ))
        return torch.cat(logits), torch.cat(counts)
