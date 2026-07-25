from __future__ import annotations

import numpy as np
import pandas as pd
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
