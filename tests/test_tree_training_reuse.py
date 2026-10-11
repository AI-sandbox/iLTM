from __future__ import annotations

from types import MethodType

import numpy as np
import pandas as pd
import pytest
from iltm.tree_embedding import TreeEmbedding


def make_tree(cls, kind, capped=False):
    return cls(
        tree_model=kind,
        cat_features=[1],
        task_type="classification",
        seed=17,
        n_estimators=4,
        max_depth=2,
        device="cpu",
        select_best_model=False,
        onehot_max_features=capped,
    )


@pytest.mark.parametrize("kind", ["GB", "RT", "CatBoost", "XGBoost_hist", "XGBoost_approx"])
@pytest.mark.parametrize("capped", [False, True])
def test_fit_transform_matches_public_transform_and_fitted_state(kind, capped):
    rng = np.random.RandomState(13)
    X = pd.DataFrame({"num": rng.normal(size=96), "cat": ["red", "green", "blue", None] * 24})
    y = (X["num"].to_numpy() > 0).astype(int)
    reference = make_tree(TreeEmbedding, kind, capped)
    optimized = make_tree(TreeEmbedding, kind, capped)
    assert reference.fit_tree(X, y) is None
    expected = reference.transform(X)
    actual = optimized._fit_transform_training(X, y)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, optimized.transform(X))
    query = X.iloc[:11].copy()
    query.loc[query.index[0], "cat"] = "unknown"
    np.testing.assert_array_equal(optimized.transform(query), reference.transform(query))
    np.testing.assert_array_equal(optimized.onehot_top_features_idx_, reference.onehot_top_features_idx_)
    assert optimized.__dict__.keys() == reference.__dict__.keys()
    if kind != "RT":
        assert optimized.onehot_encoder.sparse_output == reference.onehot_encoder.sparse_output
        for actual_categories, expected_categories in zip(
            optimized.onehot_encoder.categories_, reference.onehot_encoder.categories_, strict=True
        ):
            np.testing.assert_array_equal(actual_categories, expected_categories)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float32, np.float64])
@pytest.mark.parametrize("concat", [False, True])
def test_direct_frequencies_match_dense_and_sparse_encoder(monkeypatch, dtype, concat):
    rng = np.random.RandomState(23)
    leaves = rng.randint(0, 37, size=(400, 260)).astype(dtype)
    X = pd.DataFrame({"num": np.arange(len(leaves)), "cat": ["same"] * len(leaves)})
    y = np.arange(len(X)) % 2
    trees = [make_tree(cls, "GB", True) for cls in [TreeEmbedding, TreeEmbedding]]
    for tree in trees:
        monkeypatch.setattr(tree, "_fit_model", lambda *args: None)
        monkeypatch.setattr(tree, "_get_embeddings", lambda *args: leaves)
        tree.fit_tree(X, y, concat_with_orig_features=concat)
    reference, optimized = trees
    reference_onehot = reference.onehot_encoder.transform(leaves)
    counts = np.asarray(reference_onehot.sum(axis=0)).ravel()
    limit = 8192 - X.shape[1] if concat else 8192
    expected_indices = np.argsort(counts)[::-1][:limit]
    np.testing.assert_array_equal(optimized.onehot_top_features_idx_, expected_indices)
    assert optimized.onehot_encoder.sparse_output == reference.onehot_encoder.sparse_output
    np.testing.assert_array_equal(optimized._encode_embeddings(leaves), reference_onehot[:, expected_indices].toarray())


def test_reuse_computes_training_leaves_once(monkeypatch):
    X = pd.DataFrame({"num": [0, 1, 2], "cat": ["a", "b", "a"]})
    tree = make_tree(TreeEmbedding, "GB", True)
    calls = []
    monkeypatch.setattr(tree, "_fit_model", lambda *args: None)

    def embeddings(encoded):
        calls.append(encoded)
        return np.array([[0, 1], [1, 2], [0, 2]])

    monkeypatch.setattr(tree, "_get_embeddings", embeddings)
    transformed = tree._fit_transform_training(X, np.array([0, 1, 0]))
    assert len(calls) == 1
    assert transformed.shape == (3, 4)


@pytest.mark.parametrize("kind", ["GB", "CatBoost", "XGBoost_hist"])
@pytest.mark.parametrize("task", ["regression", "multiclass"])
@pytest.mark.parametrize("external_validation", [False, True])
def test_training_reuse_with_validation_and_other_targets(kind, task, external_validation):
    rng = np.random.RandomState(37)
    X = pd.DataFrame({"num": rng.normal(size=120), "cat": ["a", "b", "c", None] * 30})
    y = X["num"].to_numpy() * 2 if task == "regression" else np.arange(len(X)) % 3
    params = dict(
        tree_model=kind,
        cat_features=[1],
        task_type="regression" if task == "regression" else "classification",
        seed=19,
        n_estimators=12,
        max_depth=2,
        device="cpu",
        select_best_model=True,
        onehot_max_features=True,
    )
    reference = TreeEmbedding(**params)
    optimized = TreeEmbedding(**params)
    X_train, y_train = X.iloc[:96], y[:96]
    evaluation = (X.iloc[96:], y[96:]) if external_validation else None
    reference.fit_tree(X_train, y_train, eval_set=evaluation)
    expected = reference.transform(X_train)
    actual = optimized._fit_transform_training(X_train, y_train, eval_set=evaluation)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(optimized.transform(X.iloc[96:]), reference.transform(X.iloc[96:]))
    np.testing.assert_array_equal(optimized.onehot_top_features_idx_, reference.onehot_top_features_idx_)


@pytest.mark.parametrize("method", ["fit_tree", "transform"])
@pytest.mark.parametrize("override", ["subclass", "instance", "class"])
def test_training_reuse_respects_public_method_overrides(method, override, monkeypatch):
    original_fit = TreeEmbedding.fit_tree
    original_transform = TreeEmbedding.transform

    def fit_tree(self, X, y, eval_set=None, concat_with_orig_features=True):
        self.public_fit_called = True
        return original_fit(self, X, y, eval_set, concat_with_orig_features)

    def transform(self, X):
        return original_transform(self, X) + 100

    replacement = fit_tree if method == "fit_tree" else transform
    cls = TreeEmbedding
    if override == "subclass":
        cls = type("CustomTree", (TreeEmbedding,), {method: replacement})
    elif override == "class":
        monkeypatch.setattr(TreeEmbedding, method, replacement)
    trees = [
        cls(
            tree_model="GB",
            cat_features=[],
            task_type="classification",
            device="cpu",
            select_best_model=False,
            n_estimators=2,
            max_depth=1,
            seed=19,
        )
        for _ in range(2)
    ]
    if override == "instance":
        for tree in trees:
            setattr(tree, method, MethodType(replacement, tree))
    X = pd.DataFrame({"x": np.arange(20)})
    y = np.arange(20) % 2
    reference, optimized = trees
    assert reference.fit_tree(X, y) is None
    expected = reference.transform(X)
    actual = optimized._fit_transform_training(X, y)
    np.testing.assert_array_equal(actual, expected)
    if method == "fit_tree":
        assert optimized.public_fit_called is reference.public_fit_called is True
