from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostError, Pool

from iltm.tree_embedding import TreeEmbedding
from iltm.utils import preprocess_cat_features_for_catboost


def full_leaves(tree, X):
    frame = preprocess_cat_features_for_catboost(pd.DataFrame(X, copy=True), tree.cat_features)
    pool = Pool(frame, cat_features=tree.cat_features, thread_count=2)
    best = tree.model.get_best_iteration() if tree.select_best_model else None
    return tree.model.calc_leaf_indexes(pool, ntree_end=0 if best is None else best + 1, thread_count=2)


@pytest.mark.parametrize("categorical", [False, True])
def test_reduced_pool_matches_full_pool(categorical):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(128, 320)))
    cats = [10, 299] if categorical else []
    task = "classification" if categorical else "regression"
    if categorical:
        X[10] = 0
        X[299] = pd.Categorical(np.tile(["a", "b", "c", "d"], 32))
        y = X[299].isin(["a", "b"]).to_numpy().astype(int)
        X.columns = [f"f{i}" for i in range(X.shape[1])]
    else:
        y = X[298].to_numpy(copy=True)
    tree = TreeEmbedding("CatBoost", cats, task, device="cpu", n_jobs=2, n_estimators=8, max_depth=2)
    train = X.iloc[:96] if categorical else X.iloc[:96].to_numpy()
    encoded = tree._fit_transform_training(train, y[:96], eval_set=(X.iloc[96:], y[96:]))
    np.testing.assert_array_equal(encoded, tree._encode_embeddings(full_leaves(tree, train)))
    tree = pickle.loads(pickle.dumps(tree))
    query = X.iloc[96:].copy()
    if categorical:
        col = query.columns[299]
        query[col] = query[col].cat.add_categories("unseen")
        query.iloc[[0, 1], 299] = ["unseen", np.nan]
    else:
        index, borders = next((i, borders) for i, borders in tree.model.get_borders().items() if borders)
        threshold = np.float32(borders[0])
        query.iloc[:8, index] = [
            np.nan, -np.inf, np.inf, -0.0, 0.0, np.nextafter(threshold, np.float32(-np.inf)),
            threshold, np.nextafter(threshold, np.float32(np.inf)),
        ]
    original = query.copy()
    pool = tree._catboost_leaf_pool(query)
    assert 0 < pool.num_col() <= tree.n_estimators * tree.max_depth + len(cats)
    if categorical:
        assert pool.get_cat_feature_indices() == [pool.get_feature_names().index(f"f{i}") for i in cats]
    for batch in (query.iloc[:1], query, query.iloc[::-1]):
        leaves = full_leaves(tree, batch)
        data = batch if categorical else batch.to_numpy()
        np.testing.assert_array_equal(tree.transform(data, onehot_encode=False), leaves)
        np.testing.assert_array_equal(tree.transform(data), tree._encode_embeddings(leaves))
    pd.testing.assert_frame_equal(query, original)


def test_refit_schema_and_input_validation():
    rng = np.random.default_rng(23)
    X = pd.DataFrame(rng.normal(size=(48, 260)), columns=[f"f{i}" for i in range(260)])
    tree = TreeEmbedding("CatBoost", [], "regression", device="cpu", n_jobs=2, n_estimators=3,
                         max_depth=1, select_best_model=False)
    y = X["f230"].to_numpy()
    tree.fit_tree(X, y)
    assert tree._catboost_leaf_layout is not None
    unused = np.flatnonzero(~tree._catboost_leaf_layout[1])[0]
    bad = X.copy()
    bad[bad.columns[unused]] = "invalid number"
    with pytest.raises(CatBoostError):
        tree.transform(bad)
    reordered = X.iloc[:, ::-1]
    assert tree._catboost_leaf_pool(reordered).num_col() == X.shape[1]
    np.testing.assert_array_equal(tree.transform(reordered, False), full_leaves(tree, reordered))
    del tree._catboost_leaf_layout
    np.testing.assert_array_equal(tree.transform(X, False), full_leaves(tree, X))
    tree.fit_tree(X, y)
    subset = X.iloc[:, :4]
    tree.fit_tree(subset, y)
    assert tree._catboost_leaf_layout is None
    np.testing.assert_array_equal(tree.transform(subset, False), full_leaves(tree, subset))
