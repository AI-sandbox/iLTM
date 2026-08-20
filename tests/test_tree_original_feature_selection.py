from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from iltm import iLTMRegressor
from iltm.tree_embedding import TreeEmbedding


N_FEATURES = 6152
ORIGINAL_BUDGET = 6144


def _mock_tree(monkeypatch, *, task_type="classification", leaves=None):
    tree = TreeEmbedding(
        tree_model="XGBoost_hist",
        cat_features=[],
        task_type=task_type,
        onehot_max_features=True,
    )

    def fake_fit_model(self, X, y, eval_set=None):
        self.model = {"fitted": True}

    def fake_get_embeddings(self, X):
        n_rows = len(X)
        if leaves is not None:
            return leaves[:n_rows]
        rows = np.arange(n_rows)
        return np.column_stack([rows % 2, (rows // 2) % 2])

    monkeypatch.setattr(TreeEmbedding, "_fit_model", fake_fit_model)
    monkeypatch.setattr(TreeEmbedding, "_get_embeddings", fake_get_embeddings)
    return tree


def _classification_data(*, multiclass=False):
    rng = np.random.default_rng(7)
    n_rows = 60
    y = np.arange(n_rows) % (3 if multiclass else 2)
    X = rng.normal(size=(n_rows, N_FEATURES)).astype(np.float32)
    X[:, 0] = 1.0
    X[:, 1] = np.nan
    X[:, 2] = np.inf
    X[:, -2] = y
    X[:, -1] = y == 1
    return X, y


@pytest.mark.parametrize("as_frame", [False, True])
def test_late_informative_original_features_are_selected(monkeypatch, as_frame):
    X, y = _classification_data()
    if as_frame:
        X = pd.DataFrame(X, columns=[f"feature_{idx}" for idx in range(N_FEATURES)])
    tree = _mock_tree(monkeypatch)
    tree.fit_tree(X, y, concat_with_orig_features=True)

    selected_indices = tree.orig_feature_indices_to_keep_
    assert tree.n_orig_features_to_keep_ == ORIGINAL_BUDGET
    assert selected_indices.shape == (ORIGINAL_BUDGET,)
    assert N_FEATURES - 2 in selected_indices
    assert N_FEATURES - 1 in selected_indices
    assert 0 not in selected_indices
    assert 1 not in selected_indices
    assert 2 not in selected_indices

    selected = iLTMRegressor._select_original_features_for_tree(tree, X)
    assert selected.shape == (len(X), ORIGINAL_BUDGET)
    if isinstance(selected, pd.DataFrame):
        np.testing.assert_array_equal(selected.iloc[:, -2:].to_numpy(), X.iloc[:, -2:].to_numpy())
    else:
        selected_positions = {
            feature_idx: position
            for position, feature_idx in enumerate(selected_indices)
        }
        np.testing.assert_array_equal(
            selected[:, selected_positions[N_FEATURES - 2]],
            X[:, N_FEATURES - 2],
        )


def test_selection_is_invariant_to_column_order(monkeypatch):
    X, y = _classification_data()
    permutation = np.random.default_rng(11).permutation(N_FEATURES)
    first = _mock_tree(monkeypatch)
    second = _mock_tree(monkeypatch)
    first.fit_tree(X, y)
    second.fit_tree(X[:, permutation], y)

    selected_first = set(first.orig_feature_indices_to_keep_)
    selected_second = set(permutation[second.orig_feature_indices_to_keep_])
    assert selected_first == selected_second


def test_multiclass_selection_is_invariant_to_label_values(monkeypatch):
    X, y = _classification_data(multiclass=True)
    relabeled = np.array([17, -4, 93])[y]
    first = _mock_tree(monkeypatch)
    second = _mock_tree(monkeypatch)
    first.fit_tree(X, y)
    second.fit_tree(X, relabeled)

    np.testing.assert_array_equal(
        first.orig_feature_indices_to_keep_,
        second.orig_feature_indices_to_keep_,
    )


def test_regression_selection_is_invariant_to_target_sign(monkeypatch):
    rng = np.random.default_rng(19)
    X = rng.normal(size=(64, N_FEATURES)).astype(np.float32)
    y = rng.normal(size=len(X))
    X[:, -2] = y
    X[:, -1] = -y
    first = _mock_tree(monkeypatch, task_type="regression")
    second = _mock_tree(monkeypatch, task_type="regression")
    first.fit_tree(X, y)
    second.fit_tree(X, -y)

    np.testing.assert_array_equal(
        first.orig_feature_indices_to_keep_,
        second.orig_feature_indices_to_keep_,
    )
    assert N_FEATURES - 2 in first.orig_feature_indices_to_keep_
    assert N_FEATURES - 1 in first.orig_feature_indices_to_keep_


def test_selected_original_features_survive_pickle(monkeypatch):
    X, y = _classification_data()
    X = pd.DataFrame(X, columns=[f"feature_{idx}" for idx in range(N_FEATURES)])
    tree = _mock_tree(monkeypatch)
    tree.fit_tree(X, y)

    expected = iLTMRegressor._select_original_features_for_tree(tree, X)
    restored = pickle.loads(pickle.dumps(tree))
    actual = iLTMRegressor._select_original_features_for_tree(restored, X)
    pd.testing.assert_frame_equal(actual, expected)


def test_legacy_tree_pickle_keeps_first_column_semantics(monkeypatch):
    X, y = _classification_data()
    tree = _mock_tree(monkeypatch)
    tree.fit_tree(X, y)
    del tree.orig_feature_indices_to_keep_
    restored = pickle.loads(pickle.dumps(tree))

    actual = iLTMRegressor._select_original_features_for_tree(restored, X)
    np.testing.assert_array_equal(actual, X[:, :ORIGINAL_BUDGET])


def test_refit_clears_supervised_original_feature_selection(monkeypatch):
    X, y = _classification_data()
    tree = _mock_tree(monkeypatch)
    tree.fit_tree(X, y)
    assert tree.orig_feature_indices_to_keep_ is not None

    X_small = X[:, :32]
    tree.fit_tree(X_small, y)
    assert tree.orig_feature_indices_to_keep_ is None
    assert tree.n_orig_features_to_keep_ == X_small.shape[1]
    np.testing.assert_array_equal(
        iLTMRegressor._select_original_features_for_tree(tree, X_small),
        X_small,
    )


def test_leaf_embedding_budget_remains_2048(monkeypatch):
    X, y = _classification_data()
    rows = np.arange(len(X))
    leaves = np.column_stack([(rows + idx) % len(X) for idx in range(40)])
    tree = _mock_tree(monkeypatch, leaves=leaves)
    tree.fit_tree(X, y)

    assert tree.n_orig_features_to_keep_ == ORIGINAL_BUDGET
    assert tree.onehot_top_features_idx_.shape == (8192 - ORIGINAL_BUDGET,)
    assert tree.transform(X).shape[1] == 8192 - ORIGINAL_BUDGET
