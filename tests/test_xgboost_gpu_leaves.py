from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import xgboost as xgb
from iltm.tree_embedding import TreeEmbedding


def _features(n_rows, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_rows, 18)).astype(np.float32)
    values[rng.random(values.shape) < 0.3] = np.nan
    frame = pd.DataFrame(values, columns=[f"feature_{i}" for i in range(values.shape[1])])
    frame["category"] = pd.Categorical(rng.choice(["alpha", "beta", "gamma"], size=n_rows))
    frame.index = np.arange(n_rows)[::-1] + 100
    return frame


@pytest.fixture(scope="module", params=["XGBoost_hist", "XGBoost_approx"])
def fitted_gpu_tree(request):
    if not torch.cuda.is_available() or not xgb.build_info().get("USE_CUDA", False):
        pytest.skip("Requires XGBoost with CUDA and an available GPU")
    X_train = _features(257, seed=0)
    y_train = X_train["feature_0"].fillna(-3).to_numpy() + X_train["feature_1"].fillna(2).to_numpy()
    tree = TreeEmbedding(
        tree_model=request.param,
        cat_features=[18],
        task_type="regression",
        device="cuda:0",
        n_estimators=12,
        max_depth=3,
        min_samples_leaf=1,
        select_best_model=False,
    )
    tree.fit_tree(X_train, y_train)
    cpu_model = tree.model.copy()
    cpu_model.set_param({"device": "cpu", "nthread": 2})
    return tree, cpu_model, X_train


def _cpu_leaves(tree, cpu_model, frame):
    encoded = tree._handle_categorical_features(frame, fit=False)
    encoded = tree._handle_inf_and_value_too_large(encoded)
    return cpu_model.predict(xgb.DMatrix(encoded, enable_categorical=True), pred_leaf=True)


def test_gpu_training_leaf_categories_match_cpu(fitted_gpu_tree):
    tree, cpu_model, X_train = fitted_gpu_tree
    expected = _cpu_leaves(tree, cpu_model, X_train)
    for column, categories in enumerate(tree.onehot_encoder.categories_):
        np.testing.assert_array_equal(categories, np.unique(expected[:, column]))


@pytest.mark.parametrize("n_rows", [1, 17, 127, 128, 129, 10000])
def test_gpu_leaf_predictions_match_cpu(fitted_gpu_tree, n_rows):
    tree, cpu_model, _ = fitted_gpu_tree
    frame = _features(n_rows, seed=1)
    original = frame.copy(deep=True)
    expected = _cpu_leaves(tree, cpu_model, frame)
    for _ in range(2):
        actual = tree.transform(frame, onehot_encode=False)
        np.testing.assert_array_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, original)
