from __future__ import annotations

import weakref

import numpy as np
import pandas as pd
import pytest
import torch
import xgboost as xgb

from iltm import iLTMClassifier, iLTMRegressor
from iltm.iltm_model import iLTM
from iltm.tree_embedding import TreeEmbedding


@pytest.mark.parametrize(
    "split,external,select_best,mixed,regression,expected",
    [
        ("all", True, True, False, False, 2),
        ("all", True, True, True, True, 2),
        ("half", True, True, False, False, 6),
        ("all", False, True, False, False, 6),
        ("all", False, False, False, True, 1),
    ],
)
def test_matrix_reuse_within_ensemble_fit(tmp_path, monkeypatch, split, external, select_best, mixed, regression, expected):
    architecture = dict(n_dims=4, hn_n_layers=3, hn_hidden_size=8, rf_size=12, n_classes_limit=5)
    xgb_path, cb_path = [str(tmp_path / name) for name in ("xgbrconcat.pth", "cbrconcat.pth")]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(19)
        weights = iLTM(**architecture).state_dict()
        torch.save(weights, xgb_path)
        torch.save(weights, cb_path)

    cls = iLTMRegressor if regression else iLTMClassifier
    monkeypatch.setattr(cls, "_model_cache", {})
    model = cls(
        checkpoint=xgb_path, checkpoint_mix=(xgb_path, cb_path) if mixed else None,
        device="cpu", n_jobs=1, n_ensemble=3, tree_n_estimators=6, tree_max_depth=2,
        tree_data_split=split, tree_select_best_model=select_best, tree_use_external_eval_set=external,
        tree_subsample=0.7, tree_feature_fraction=0.8, finetuning=False, retrieval_alpha_adaptive=False,
        auto_stop_on_low_cpu_memory=False, **architecture,
    )
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
    X.loc[::7, "c"] = np.nan
    X["category"] = ["a", "b", None, "c"] * 20 + ["unseen"] * 20
    target = X["a"].to_numpy() + 0.5 * X["b"].to_numpy()
    y = target if regression else (target > 0).astype(int)
    train, valid = X.iloc[:80], X.iloc[80:]
    matrices = []
    make_matrix = TreeEmbedding._make_xgboost_dmatrix

    def record(self, frame, labels=None):
        matrix = make_matrix(self, frame, labels)
        if labels is not None:
            matrices.append(weakref.ref(matrix))
        return matrix

    monkeypatch.setattr(TreeEmbedding, "_make_xgboost_dmatrix", record)

    def fit():
        model.fit(train, y[:80], eval_set=(valid, y[80:]))
        trees = [t for t in model.tr_ if t.tree_model.startswith("XGBoost")]
        result = [(t.seed, t.model.get_dump(), getattr(t.model, "best_iteration", None)) for t in trees]
        preds = model.predict(valid) if regression else model.predict_proba(valid)
        return result, preds

    actual, predictions = fit()
    assert len(matrices) == expected
    assert all(ref() is None for ref in matrices)
    assert len({seed for seed, _, _ in actual}) == len(actual)

    original_fit = TreeEmbedding._fit_model

    def fresh(self, X, y, eval_set=None, **kwargs):
        return original_fit(self, X, y, eval_set)

    monkeypatch.setattr(TreeEmbedding, "_fit_model", fresh)
    reference, reference_predictions = fit()
    assert actual == reference
    np.testing.assert_array_equal(predictions, reference_predictions)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["XGBoost_hist", "XGBoost_approx"])
def test_reused_matrix_matches_fresh_for_each_seed(device, kind):
    if device == "cuda" and not (torch.cuda.is_available() and xgb.build_info().get("USE_CUDA")):
        pytest.skip("CUDA required")
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(180, 300)).astype(np.float32))
    X.iloc[0, :3] = [np.inf, -np.inf, np.nan]
    y = rng.normal(size=len(X))
    cache = {}
    for seed in (12, 13):
        options = dict(tree_model=kind, cat_features=[], task_type="regression", device=device,
                       n_jobs=1, n_estimators=6, max_depth=2, subsample=0.7, seed=seed)
        cached, fresh = TreeEmbedding(**options), TreeEmbedding(**options)
        valid = X.iloc[138:], y[138:]
        leaves = cached._fit_transform_training(X.iloc[:138], y[:138], eval_set=valid, xgb_fit_cache=cache)
        reference = fresh._fit_transform_training(X.iloc[:138], y[:138], eval_set=valid)
        assert cached.model.get_dump() == fresh.model.get_dump()
        assert cached.model.best_iteration == fresh.model.best_iteration
        np.testing.assert_array_equal(leaves, reference)
        np.testing.assert_array_equal(cached.transform(valid[0]), fresh.transform(valid[0]))
        if kind == "XGBoost_approx":
            assert not cache


def test_out_of_memory_retry_discards_shared_matrices(monkeypatch):
    import iltm.tree_embedding as tree_module

    X = pd.DataFrame({"x": np.arange(32, dtype=np.float32)})
    y = np.arange(len(X)) % 2
    options = dict(tree_model="XGBoost_hist", cat_features=[], task_type="classification",
                   device="cpu", n_jobs=1, n_estimators=3, select_best_model=False)
    cache = {}
    monkeypatch.setattr(tree_module, "get_gpu_memory_info", lambda: None)
    TreeEmbedding(**options)._fit_transform_training(X, y, xgb_fit_cache=cache)
    assert cache
    train = xgb.train
    attempts = []

    def fail_once(**kwargs):
        attempts.append(kwargs["params"].get("max_bin"))
        if len(attempts) == 1:
            raise RuntimeError("CUDA out of memory")
        return train(**kwargs)

    monkeypatch.setattr(xgb, "train", fail_once)
    TreeEmbedding(**options)._fit_transform_training(X, y, xgb_fit_cache=cache)
    assert attempts == [None, 256]
    assert not cache
