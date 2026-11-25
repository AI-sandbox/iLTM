from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import xgboost as xgb

from iltm.tree_embedding import TreeEmbedding


@pytest.fixture(scope="module", params=["cpu", "cuda:0"])
def fitted(request):
    device = request.param
    if device != "cpu" and not (torch.cuda.is_available() and xgb.build_info().get("USE_CUDA")):
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(23)
    X = pd.DataFrame(rng.normal(size=(161, 320)), columns=[f"column_{i}" for i in range(320)])
    target = X.iloc[:, 173].to_numpy(copy=True)
    X.iloc[::5, 173] = np.nan
    target[::5] = -3
    tree = TreeEmbedding("XGBoost_hist", [], "regression", device=device, n_jobs=2,
                         n_estimators=12, max_depth=2, lr=0.4)
    valid = X.iloc[::4]
    tree.fit_tree(X, target, eval_set=(valid, 0.4 * target[::4]))
    assert tree._sparse_leaf_layout is not None
    assert tree.model.best_iteration + 1 < tree.model.num_boosted_rounds()
    return tree, X


def dense_leaves(tree, X):
    # Use a CPU copy as an independent reference even when the fitted booster is on CUDA.
    booster = tree.model.copy()
    booster.set_param({"device": "cpu", "nthread": 2})
    end = tree.model.best_iteration + 1 if tree.select_best_model else 0
    matrix = tree._make_xgboost_dmatrix(tree._handle_inf_and_value_too_large(X))
    return booster.predict(matrix, pred_leaf=True, iteration_range=(0, end)).reshape(len(X), -1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sparse_leaf_equivalence(fitted, dtype, monkeypatch):
    tree, X = fitted
    X = X.astype(dtype)
    split = json.loads(tree.model.get_dump(dump_format="json")[0])
    threshold = np.float32(split["split_condition"])
    X.loc[:8, split["split"]] = [
        threshold, np.nextafter(threshold, np.float32(-np.inf)), np.nextafter(threshold, np.float32(np.inf)),
        0, -0., np.inf, -np.inf, 1e30, -1e30,
    ]
    X.iloc[9, :] = np.nan
    X.iloc[10, :] = 0
    for batch in [X.iloc[:n] for n in (1, 17, 127, 128, 129, 161)] + [X.iloc[[9]], X.iloc[[10]]]:
        reference = dense_leaves(tree, batch)
        with monkeypatch.context() as patch:
            def unexpected(*args):
                pytest.fail("dense leaf matrix constructed")
            patch.setattr(tree, "_make_xgboost_dmatrix", unexpected)
            np.testing.assert_array_equal(tree.transform(batch, onehot_encode=False), reference)
            expected = tree._encode_embeddings(reference)
            np.testing.assert_array_equal(tree.transform(batch), expected)


def test_sparse_leaf_fallbacks(fitted):
    fitted_tree, X = fitted
    tree = pickle.loads(pickle.dumps(fitted_tree))
    with pytest.raises(ValueError, match="feature_names mismatch"):
        tree.transform(X.iloc[:, ::-1])
    bad = X.assign(**{X.columns[-1]: "invalid"})
    with pytest.raises(ValueError):
        tree.transform(bad)
    tree.select_best_model = False
    np.testing.assert_array_equal(tree.transform(X, onehot_encode=False), dense_leaves(tree, X))
    del tree._sparse_leaf_layout
    np.testing.assert_array_equal(tree.transform(X, onehot_encode=False), dense_leaves(tree, X))


def test_refit_replaces_sparse_layout_and_dense_cutoff(fitted):
    fitted_tree, X = fitted
    tree = pickle.loads(pickle.dumps(fitted_tree))
    tree.select_best_model = False
    tree.fit_tree(X, X.iloc[:, 28].to_numpy())
    assert 28 in tree._sparse_leaf_layout[2]
    np.testing.assert_array_equal(tree.transform(X, onehot_encode=False), dense_leaves(tree, X))
    tree.fit_tree(X.iloc[:, :5], X.iloc[:, 0].to_numpy())
    assert tree._sparse_leaf_layout is None

def test_wide_tree_with_many_split_columns_uses_dense_path(fitted, monkeypatch):
    fitted_tree, X = fitted
    tree = pickle.loads(pickle.dumps(fitted_tree))
    tree.select_best_model = False
    monkeypatch.setattr(tree.model, "get_score", lambda **kwargs: dict.fromkeys(X.columns[:80], 1))
    assert tree._xgboost_sparse_leaf_layout(X) is None
