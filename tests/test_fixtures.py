"""Shape and contract checks for the shared synthetic CX fixtures in conftest.py."""

from src.masks import apply_mask, random_mask


def test_cx_graph_has_enough_nodes_for_negative_sampling(cx_graph):
    assert cx_graph.n_nodes >= 102


def test_cx_annotations_has_exactly_the_required_columns_and_a_non_trivial_type(cx_annotations):
    assert set(cx_annotations.columns) == {
        "root_id", "super_class", "class", "cell_type", "side", "flow", "nt_type",
    }
    counts = cx_annotations["cell_type"].value_counts()
    assert counts.max() >= 2


def test_cx_types_is_aligned_to_graph_node_order(cx_graph, cx_types):
    assert len(cx_types) == cx_graph.n_nodes
    assert list(cx_types.index) == list(cx_graph.node_ids)


def test_cx_centroids_has_one_row_per_node(cx_graph, cx_centroids):
    assert cx_centroids.shape == (cx_graph.n_nodes, 3)


def test_cx_features_and_schema_agree_on_feature_count(cx_graph, cx_schema, cx_features):
    assert cx_features.values.shape[0] == cx_graph.n_nodes
    assert cx_schema.n_features == cx_features.values.shape[1]


def test_sample_for_uses_visible_degrees_and_yields_k_100_negatives(cx_graph, sample_for):
    instance = random_mask(cx_graph, fraction=0.05, seed=1)
    view = apply_mask(cx_graph, instance)
    sample = sample_for(view)
    assert sample.spec.k == 100
    assert sample.negatives.shape[1] == 100
