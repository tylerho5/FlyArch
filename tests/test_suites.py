"""Frozen, digest-pinned mask suites: a suite is graph-free JSON that resolves
against a graph and is caught if that graph ever changes underneath it."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from src.masks import apply_mask, random_mask, spatial_mask, typed_mask
from src.negatives import SIDE_SOURCE, SIDE_TARGET, pack_edges
from src.suites import SUITE_SIZES, Suite, build_suite, f1_train_types, suite_negatives, view_negatives

_TAGS = ("R01", "R05", "R10", "T", "S05")
_FRACTIONS = {"R01": 0.01, "R05": 0.05, "R10": 0.10}
_OFFSETS = {"f1_val": 0, "f2_dev": 1000, "f2_confirm": 2000}


def test_suite_sizes():
    assert SUITE_SIZES == {"f1_val": 5, "f2_dev": 15, "f2_confirm": 20, "f2_side": 2, "f3_eval": 15}


def test_dev_and_confirm_suites_have_disjoint_seeds_and_ids(cx_graph, cx_types, cx_centroids):
    dev = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    confirm = build_suite(cx_graph, name="f2_confirm", types=cx_types, centroids=cx_centroids)
    assert {i.seed for i in dev.instances}.isdisjoint({i.seed for i in confirm.instances})
    assert {i.instance_id for i in dev.instances}.isdisjoint(
        {i.instance_id for i in confirm.instances}
    )


def test_confirm_typed_masks_are_disjoint_from_dev_and_distinct(cx_graph, cx_types, cx_centroids):
    dev = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    confirm = build_suite(cx_graph, name="f2_confirm", types=cx_types, centroids=cx_centroids)
    dev_types = {i.params["cell_type"] for i in dev.instances if i.regime == "T"}
    confirm_types = [i.params["cell_type"] for i in confirm.instances if i.regime == "T"]
    assert dev_types.isdisjoint(confirm_types)  # no confirm instance duplicates a dev one
    assert len(confirm_types) == len(set(confirm_types))  # confirm's own T masks are distinct


def test_draw_types_limit_typed_draws_and_leave_other_regimes(cx_graph, cx_types, cx_centroids):
    # A type absent from the F1 pool is never drawn; R and S instances do not move.
    plain_dev = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    dropped = next(i.params["cell_type"] for i in plain_dev.instances if i.regime == "T")
    drawable = set(cx_types.dropna().unique()) - {dropped}
    for name in ("f2_dev", "f2_confirm"):
        plain = build_suite(cx_graph, name=name, types=cx_types, centroids=cx_centroids)
        pooled = build_suite(cx_graph, name=name, types=cx_types, centroids=cx_centroids, draw_types=drawable)
        for a, b in zip(plain.instances, pooled.instances):
            if b.regime == "T":
                assert b.params["cell_type"] in drawable
            else:
                assert a.to_dict() == b.to_dict()
    dev = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids, draw_types=drawable)
    confirm = build_suite(cx_graph, name="f2_confirm", types=cx_types, centroids=cx_centroids, draw_types=drawable)
    dev_types = {i.params["cell_type"] for i in dev.instances if i.regime == "T"}
    assert dev_types.isdisjoint(i.params["cell_type"] for i in confirm.instances if i.regime == "T")


@pytest.mark.parametrize(
    ("name", "graph_fixture"),
    [("f1_val", "cx_graph_f1"), ("f2_dev", "cx_graph"), ("f2_confirm", "cx_graph")],
)
def test_every_suite_spans_all_three_regimes(
    name, graph_fixture, request, cx_types, cx_centroids
):
    graph = request.getfixturevalue(graph_fixture)
    suite = build_suite(graph, name=name, types=cx_types, centroids=cx_centroids)
    assert {i.regime for i in suite.instances} == {"R", "T", "S"}


def test_write_read_round_trip_preserves_digest_and_order(tmp_path, cx_graph, cx_types, cx_centroids):
    suite = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    path = tmp_path / "suite.json"
    suite.write(path)
    restored = Suite.read(path)
    assert restored.digest() == suite.digest()
    assert [i.instance_id for i in restored.instances] == [
        i.instance_id for i in suite.instances
    ]


def test_validate_passes_fresh_and_raises_on_graph_digest_mismatch(
    cx_graph_f1, cx_types, cx_centroids
):
    suite = build_suite(cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids)
    suite.validate(cx_graph_f1)  # no raise

    corrupted = dataclasses.replace(suite, graph_digest="deadbeef")
    with pytest.raises(ValueError, match="graph digest"):
        corrupted.validate(cx_graph_f1)


def test_build_suite_rejects_a_name_from_the_other_dataset(
    cx_graph, cx_graph_f1, cx_types, cx_centroids
):
    with pytest.raises(ValueError, match="registered to dataset 'f1'"):
        build_suite(cx_graph, name="f1_val", types=cx_types, centroids=cx_centroids)
    with pytest.raises(ValueError, match="registered to dataset 'f2'"):
        build_suite(cx_graph_f1, name="f2_dev", types=cx_types, centroids=cx_centroids)


def test_validate_rejects_a_suite_whose_name_and_dataset_disagree(
    cx_graph, cx_types, cx_centroids
):
    suite = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    with pytest.raises(ValueError, match="registered to dataset"):
        dataclasses.replace(suite, name="f1_val").validate(cx_graph)
    with pytest.raises(ValueError, match="unknown suite name"):
        dataclasses.replace(suite, name="f9_nope").validate(cx_graph)


def test_validate_rejects_a_suite_from_another_dataset(
    cx_graph, cx_graph_f1, cx_types, cx_centroids
):
    # An f1_val suite is internally consistent, but is not an F2 graph's suite.
    suite = build_suite(cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids)
    with pytest.raises(ValueError, match="does not match graph dataset"):
        suite.validate(cx_graph)


def test_resolve_reproduces_generators_run_directly_at_the_same_seeds(
    cx_graph, cx_types, cx_centroids
):
    suite = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    resolved = suite.resolve(cx_graph)
    offset = _OFFSETS["f2_dev"]

    for i, mi in enumerate(resolved):
        seed = offset + i
        tag = _TAGS[i % 5]
        if tag == "T":
            expected = typed_mask(cx_graph, cx_types, seed=seed)
        elif tag == "S05":
            expected = spatial_mask(cx_graph, cx_centroids, seed=seed, fraction=0.05)
        else:
            expected = random_mask(cx_graph, fraction=_FRACTIONS[tag], seed=seed)
        assert mi.instance_id == expected.instance_id
        assert np.array_equal(mi.masked, expected.masked)


def test_read_resolve_matches_in_memory_resolve(tmp_path, cx_graph, cx_types, cx_centroids):
    suite = build_suite(cx_graph, name="f2_confirm", types=cx_types, centroids=cx_centroids)
    path = tmp_path / "suite.json"
    suite.write(path)

    restored = Suite.read(path).resolve(cx_graph)
    original = suite.resolve(cx_graph)
    assert len(restored) == len(original)
    for a, b in zip(restored, original):
        assert a.instance_id == b.instance_id
        assert np.array_equal(a.masked, b.masked)


def test_suite_negatives_shapes_and_stable_digest(cx_graph, cx_types, cx_centroids):
    suite = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    first = suite_negatives(cx_graph, suite)
    second = suite_negatives(cx_graph, suite)

    assert set(first) == {i.instance_id for i in suite.instances}
    for instance_id, sample in first.items():
        assert sample.spec.k == 100
        assert sample.negatives.shape[1] == 100
        assert sample.digest() == second[instance_id].digest()


def test_every_negative_keeps_a_masked_endpoint(cx_graph, cx_types, cx_centroids):
    """A decoy that drops the masked neuron is not a query about it.

    ``VisibleView.is_masked`` is handed to every model, so a decoy with no
    masked endpoint is trivially separable from the positive it is ranked
    against.
    """
    suite = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    negatives = suite_negatives(cx_graph, suite)

    for instance in suite.resolve(cx_graph):
        view = apply_mask(cx_graph, instance)
        sample = negatives[instance.instance_id]
        keeps = view.is_masked[sample.negatives[..., 0]] | view.is_masked[
            sample.negatives[..., 1]
        ]
        assert keeps.all()

        source_masked = view.is_masked[view.hidden_edges[:, 0]]
        target_masked = view.is_masked[view.hidden_edges[:, 1]]
        assert (sample.sides[source_masked & ~target_masked] == SIDE_TARGET).all()
        assert (sample.sides[target_masked & ~source_masked] == SIDE_SOURCE).all()


def test_precomputed_graph_inputs_preserve_negative_bytes(cx_graph, cx_types, cx_centroids):
    suite = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    instance = suite.resolve(cx_graph)[0]
    view = apply_mask(cx_graph, instance)
    plain = view_negatives(view)
    prepared = view_negatives(
        view, pool=np.arange(cx_graph.n_nodes, dtype=np.int64),
        true_edges=pack_edges(cx_graph.edges),
    )
    assert np.array_equal(plain.positives, prepared.positives)
    assert np.array_equal(plain.negatives, prepared.negatives)
    assert np.array_equal(plain.sides, prepared.sides)
    assert plain.digest() == prepared.digest()
    assert suite_negatives(cx_graph, suite)[instance.instance_id].digest() == plain.digest()


def test_sample_for_reproduces_suite_negatives_for_one_instance(
    cx_graph, cx_types, cx_centroids, sample_for
):
    """The conftest fixture and suite_negatives must build identical negatives."""
    suite = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    negatives = suite_negatives(cx_graph, suite)
    instance = suite.resolve(cx_graph)[0]
    assert sample_for(apply_mask(cx_graph, instance)).digest() == negatives[
        instance.instance_id
    ].digest()


def test_build_suite_requires_types_for_t_instance(cx_graph, cx_centroids):
    with pytest.raises(ValueError):
        build_suite(cx_graph, name="f2_dev", types=None, centroids=cx_centroids)


def test_build_suite_requires_centroids_for_s_instance(cx_graph, cx_types):
    with pytest.raises(ValueError):
        build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=None)


def test_build_suite_rejects_non_default_master_seed(cx_graph, cx_types, cx_centroids):
    with pytest.raises(ValueError, match="master_seed"):
        build_suite(
            cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids, master_seed=1
        )


def _f1_pool_args(tmp_path, cx_graph_f1, cx_types, cx_centroids, types_path):
    """The ``--f1-*`` flags for a toy region's F1 build, and its F1-val suite."""
    cx_graph_f1.write(tmp_path / "f1_graph")
    val = build_suite(cx_graph_f1, name="f1_val", types=cx_types, centroids=cx_centroids)
    val.write(tmp_path / "f1_val.json")
    args = ["--f1-graph", str(tmp_path / "f1_graph"), "--f1-types", str(types_path),
            "--f1-val", str(tmp_path / "f1_val.json")]
    return args, val


def test_build_suites_cli_round_trip(tmp_path, cx_graph, cx_graph_f1, cx_types, cx_centroids):
    from scripts.build_suites import main

    graph_dir = tmp_path / "graph"
    cx_graph.write(graph_dir)

    types_path = tmp_path / "types.parquet"
    cx_types.rename_axis("root_id").reset_index(name="cell_type").to_parquet(
        types_path, index=False
    )

    centroids_path = tmp_path / "centroids.parquet"
    centroids_df = pd.DataFrame(cx_centroids, columns=["x", "y", "z"])
    centroids_df.insert(0, "root_id", cx_graph.node_ids)
    centroids_df.to_parquet(centroids_path, index=False)

    out = tmp_path / "out"
    base = ["--graph", str(graph_dir), "--name", "f2_dev", "--types", str(types_path),
            "--centroids", str(centroids_path)]
    with pytest.raises(SystemExit):  # an F2 dev/confirm suite needs the region's F1 draw pool
        main([*base, "--out", str(tmp_path / "no-pool")])
    pool, _ = _f1_pool_args(tmp_path, cx_graph_f1, cx_types, cx_centroids, types_path)
    assert main([*base, *pool, "--out", str(out)]) == 0

    suite_path = out / "f2_dev.json"
    negatives_path = out / "f2_dev.negatives.json"
    assert suite_path.exists()
    assert negatives_path.exists()

    suite = Suite.read(suite_path)
    suite.validate(cx_graph)  # no raise

    negatives_doc = json.loads(negatives_path.read_text(encoding="utf-8"))
    expected = {
        iid: sample.digest() for iid, sample in suite_negatives(cx_graph, suite).items()
    }
    assert negatives_doc["digests"] == expected
    assert negatives_doc["name"] == "f2_dev"

    # A region with an exclusion record beside its graph cannot build an unexcluded suite.
    (graph_dir / "excluded_types.json").write_text('{"types": []}', encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--graph", str(graph_dir), "--name", "f2_dev", "--types", str(types_path),
              "--centroids", str(centroids_path), *pool, "--out", str(tmp_path / "refused")])


def test_cli_centroids_keyed_by_root_id_not_row_order(
    tmp_path, cx_graph, cx_graph_f1, cx_types, cx_centroids
):
    # A centroids file whose rows are not in node order must still yield the same
    # suite: load_centroids keys on root_id, so a mis-ordered file cannot silently
    # misalign centroids and produce wrong-but-valid spatial masks.
    from scripts.build_suites import main

    graph_dir = tmp_path / "graph"
    cx_graph.write(graph_dir)

    types_path = tmp_path / "types.parquet"
    cx_types.rename_axis("root_id").reset_index(name="cell_type").to_parquet(
        types_path, index=False
    )

    order = np.arange(cx_graph.n_nodes)[::-1]
    centroids_path = tmp_path / "centroids.parquet"
    shuffled = pd.DataFrame(cx_centroids[order], columns=["x", "y", "z"])
    shuffled.insert(0, "root_id", cx_graph.node_ids[order])
    shuffled.to_parquet(centroids_path, index=False)

    pool, val = _f1_pool_args(tmp_path, cx_graph_f1, cx_types, cx_centroids, types_path)
    out = tmp_path / "out"
    assert main([
        "--graph", str(graph_dir),
        "--name", "f2_dev",
        "--types", str(types_path),
        "--centroids", str(centroids_path),
        *pool,
        "--out", str(out),
    ]) == 0

    from_shuffled = Suite.read(out / "f2_dev.json")
    reference = build_suite(
        cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids,
        draw_types=f1_train_types(cx_graph_f1, cx_types, val),
    )
    assert from_shuffled.digest() == reference.digest()


def test_load_centroids_rejects_duplicate_root_ids(tmp_path, cx_graph, cx_centroids):
    from scripts.build_suites import load_centroids

    df = pd.DataFrame(cx_centroids, columns=["x", "y", "z"])
    df.insert(0, "root_id", cx_graph.node_ids)
    df.loc[1, "root_id"] = df.loc[0, "root_id"]  # collide two rows on one root id
    path = tmp_path / "dup.parquet"
    df.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="duplicate root ids"):
        load_centroids(path, cx_graph)


def _lamina_like(cx_types):
    return (cx_types.value_counts().index[0],)


def test_unexcluded_suites_keep_their_serialization_and_pool_free_draws(
    cx_graph, cx_types, cx_centroids
):
    plain = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids)
    assert "mask_exclude_types" not in plain.to_json()
    # An excluded type the graph does not carry makes the pool the whole graph.
    absent = build_suite(
        cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids,
        mask_exclude_types=("no-such-type",),
    )
    assert absent.to_json()["instances"] == plain.to_json()["instances"]


def test_mask_exclude_types_are_never_masked_and_sizes_count_the_eligible(
    tmp_path, cx_graph, cx_types, cx_centroids
):
    ex = _lamina_like(cx_types)
    banned = set(cx_types.index[cx_types.isin(ex)]) & set(cx_graph.node_ids.tolist())
    eligible = cx_graph.n_nodes - len(banned)
    dev = build_suite(cx_graph, name="f2_dev", types=cx_types, centroids=cx_centroids, mask_exclude_types=ex)
    confirm = build_suite(
        cx_graph, name="f2_confirm", types=cx_types, centroids=cx_centroids, mask_exclude_types=ex
    )
    for suite in (dev, confirm):
        assert suite.mask_exclude_types == tuple(sorted(ex))
        for inst in suite.instances:
            assert banned.isdisjoint(inst.masked_root_ids.tolist())
            if inst.regime == "R":
                assert inst.masked_root_ids.size == round(inst.params["fraction"] * eligible)
            if inst.regime == "T":
                assert inst.params["cell_type"] not in ex
            if inst.regime == "S":
                assert inst.masked_root_ids.size >= round(0.05 * eligible)
    dev_types = {i.params["cell_type"] for i in dev.instances if i.regime == "T"}
    assert dev_types.isdisjoint(i.params["cell_type"] for i in confirm.instances if i.regime == "T")

    path = tmp_path / "suite.json"
    dev.write(path)
    restored = Suite.read(path)
    restored.validate(cx_graph)
    assert restored.digest() == dev.digest() and restored.mask_exclude_types == dev.mask_exclude_types


def test_mask_exclude_types_need_types(cx_graph):
    with pytest.raises(ValueError, match="types is required"):
        build_suite(cx_graph, name="f2_dev", mask_exclude_types=("L1",))
