from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from iltm.tree_embedding import TreeEmbedding


def test_sparse_top_feature_encoding_matches_dense_reference(monkeypatch):
    n_rows = 120
    n_original_features = 6145
    n_trees = 60
    X_train = pd.DataFrame(
        np.zeros((n_rows, n_original_features), dtype=np.float32)
    )
    X_test = X_train.iloc[:7].copy()
    train_leaves = np.column_stack(
        [
            (np.arange(n_rows) * (column + 1)) % (70 + column)
            for column in range(n_trees)
        ]
    )
    test_leaves = train_leaves[:7].copy()
    test_leaves[0, 0] = 10_000

    tree = TreeEmbedding(
        tree_model="XGBoost_hist",
        cat_features=[],
        task_type="regression",
        onehot_max_features=True,
    )

    def fake_fit_model(X, y, eval_set=None):
        tree.model = object()

    monkeypatch.setattr(tree, "_fit_model", fake_fit_model)
    monkeypatch.setattr(
        tree,
        "_get_embeddings",
        lambda X: train_leaves if len(X) == n_rows else test_leaves,
    )

    tree.fit_tree(
        X_train,
        np.arange(n_rows),
        concat_with_orig_features=True,
    )
    actual = tree.transform(X_test)

    reference_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        dtype=np.int8,
    ).fit(train_leaves)
    dense_train = reference_encoder.transform(train_leaves)
    feature_counts = dense_train.sum(axis=0)
    top_features = np.argsort(feature_counts)[::-1][
        : 8192 - tree.n_orig_features_to_keep_
    ]
    expected = reference_encoder.transform(test_leaves)[:, top_features]

    assert tree.onehot_encoder.sparse_output is True
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.int8
    np.testing.assert_array_equal(actual, expected)


def test_full_onehot_encoding_remains_dense(monkeypatch):
    leaves = np.array([[0, 2], [1, 2], [0, 3]])
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    tree = TreeEmbedding(
        tree_model="XGBoost_hist",
        cat_features=[],
        task_type="regression",
        onehot_max_features=False,
    )

    def fake_fit_model(X, y, eval_set=None):
        tree.model = object()

    monkeypatch.setattr(tree, "_fit_model", fake_fit_model)
    monkeypatch.setattr(tree, "_get_embeddings", lambda X: leaves)

    tree.fit_tree(X, np.arange(len(X)))
    actual = tree.transform(X)

    assert tree.onehot_encoder.sparse_output is False
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(
        actual,
        np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.int8),
    )


def test_top_feature_encoding_stays_dense_when_pruning_is_unnecessary(
    monkeypatch,
):
    leaves = np.array([[0, 2], [1, 2], [0, 3]])
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    tree = TreeEmbedding(
        tree_model="XGBoost_hist",
        cat_features=[],
        task_type="regression",
        onehot_max_features=True,
    )

    def fake_fit_model(X, y, eval_set=None):
        tree.model = object()

    monkeypatch.setattr(tree, "_fit_model", fake_fit_model)
    monkeypatch.setattr(tree, "_get_embeddings", lambda X: leaves)

    tree.fit_tree(X, np.arange(len(X)))
    actual = tree.transform(X)

    assert tree.onehot_encoder.sparse_output is False
    assert isinstance(actual, np.ndarray)
    expected_full = np.array(
        [[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]],
        dtype=np.int8,
    )
    expected_order = np.argsort(expected_full.sum(axis=0))[::-1]
    np.testing.assert_array_equal(actual, expected_full[:, expected_order])


@pytest.mark.parametrize("dtype", [np.float32, np.uint32, np.int64])
@pytest.mark.parametrize("capped,sparse", [(False, False), (True, False), (True, True)])
def test_leaf_lookup_matches_encoder_across_batches(monkeypatch, dtype, capped, sparse):
    rng = np.random.default_rng(42)
    leaves = (rng.integers(0, 8, size=(40, 70)) * 2).astype(dtype)
    query = np.tile(leaves, (50, 1))
    query[::3, 0] = 5  # An unseen leaf between two known IDs.
    query[::4, -1] = 22
    if dtype != np.uint32:
        query[::5, 1] = -1
    if dtype == np.float32:
        query[::6, 2] = 2.5
    monkeypatch.setattr("iltm.tree_embedding._DENSE_ONEHOT_WORKING_SET_BYTES", 12_000 if sparse else 64 * 1024**2)
    X = pd.DataFrame({"x": np.arange(len(leaves))})
    tree = TreeEmbedding("GB", [], "regression", onehot_max_features=capped)
    monkeypatch.setattr(tree, "_fit_model", lambda *args: None)
    monkeypatch.setattr(tree, "_get_embeddings", lambda X: leaves)
    tree.fit_tree(X, X.x.to_numpy())
    assert tree._leaf_lookup is not None
    assert tree.onehot_encoder.sparse_output == sparse
    expected = tree.onehot_encoder.transform(query)
    if capped:
        expected = expected[:, tree.onehot_top_features_idx_]
    if sparse:
        expected = expected.toarray()

    def unexpected_transform(*args):
        pytest.fail("Compact leaf IDs should use the lookup")

    monkeypatch.setattr(tree.onehot_encoder, "transform", unexpected_transform)
    actual = tree._encode_embeddings(query)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert actual.strides == expected.strides


def test_leaf_lookup_refit_and_encoder_fallback(monkeypatch):
    tree = TreeEmbedding("GB", [], "regression")
    X = pd.DataFrame({"x": [1, 2, 3]})
    leaves = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.float32)
    monkeypatch.setattr(tree, "_fit_model", lambda *args: None)
    monkeypatch.setattr(tree, "_get_embeddings", lambda X: leaves)
    tree.fit_tree(X, X.x.to_numpy())
    assert tree._leaf_lookup is not None
    query = np.array([[np.nan, 1]], dtype=leaves.dtype)
    np.testing.assert_array_equal(tree._encode_embeddings(query), tree.onehot_encoder.transform(query))
    with pytest.raises(ValueError):
        tree._encode_embeddings(np.array([[np.inf, 1]], dtype=leaves.dtype))
    for query in (leaves[:, :1], leaves[:0]):
        with pytest.raises(ValueError):
            tree._encode_embeddings(query)

    leaves[0, 0] = 1_000_000
    tree.fit_tree(X, X.x.to_numpy())
    assert tree._leaf_lookup is None
    np.testing.assert_array_equal(tree._encode_embeddings(leaves), tree.onehot_encoder.transform(leaves))


def test_leaf_lookup_serialization_and_older_tree():
    tree = TreeEmbedding("GB", [], "regression", n_estimators=3, select_best_model=False, device="cpu")
    X = pd.DataFrame({"x": np.arange(30, dtype=float)})
    tree.fit_tree(X, X.x.to_numpy())
    leaves = tree.transform(X, onehot_encode=False)
    expected = tree.onehot_encoder.transform(leaves)
    assert tree._leaf_lookup is not None
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(tree)).transform(X), expected)
    del tree._leaf_lookup
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(tree)).transform(X), expected)
