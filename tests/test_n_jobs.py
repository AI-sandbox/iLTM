import numpy as np
import pandas as pd
import pytest
import torch
from catboost import CatBoost
from sklearn.base import clone

from iltm import iLTMClassifier, iLTMRegressor
from iltm.tree_embedding import TreeEmbedding


@pytest.mark.parametrize("cls", [iLTMClassifier, iLTMRegressor])
def test_cpu_thread_count_is_scoped_to_fit_and_predict(monkeypatch, cls):
    caller_threads = torch.get_num_threads()
    budget = 2 if caller_threads == 1 else 1
    estimator = clone(cls(checkpoint=None, device="cpu", n_jobs=budget, retrieval_alpha_adaptive=False))
    X = np.zeros((2, 1), dtype=np.float32)
    y = np.array([0, 1])
    assert estimator.n_jobs == budget
    assert estimator._create_tree_embedding(tree_model="CatBoost", seed=1).n_jobs == budget
    assert torch.get_num_threads() == caller_threads

    def fit_common(*args, **kwargs):
        assert torch.get_num_threads() == budget
        estimator.normalize_predictions_ = False
        estimator.clip_predictions_ = False
        return estimator

    monkeypatch.setattr(estimator, "_fit_common", fit_common)
    assert estimator.fit(X, y) is estimator
    assert torch.get_num_threads() == caller_threads

    def fail_prediction(*args, **kwargs):
        assert torch.get_num_threads() == budget
        raise ValueError("prediction failed")

    monkeypatch.setattr(estimator, "_predict_ensemble", fail_prediction)
    predict = estimator.predict_proba if cls is iLTMClassifier else estimator.predict
    with pytest.raises(ValueError, match="prediction failed"):
        predict(X)
    assert torch.get_num_threads() == caller_threads


def test_cuda_fit_respects_cpu_thread_count_during_retrieval_scoring(monkeypatch):
    caller_threads = torch.get_num_threads()
    budget = 2 if caller_threads == 1 else 1
    estimator = iLTMRegressor(checkpoint=None, device="cuda:0", n_jobs=budget, retrieval_alpha_adaptive=True)
    X = np.zeros((2, 1), dtype=np.float32)
    y = np.array([0.0, 1.0])
    observed_threads = []
    dot = torch.dot

    def fit_common(*args, **kwargs):
        estimator.normalize_predictions_ = False
        estimator.clip_predictions_ = False
        return estimator

    def score_dot(a, b):
        assert a.device.type == "cpu"
        observed_threads.append(torch.get_num_threads())
        return dot(a, b)

    monkeypatch.setattr(estimator, "_fit_common", fit_common)
    monkeypatch.setattr(estimator, "_predict_ensemble", lambda *args, **kwargs: (torch.zeros(1, 2), torch.ones(1, 2)))
    monkeypatch.setattr(torch, "dot", score_dot)
    estimator.fit(X, y, eval_set=(X, y))
    assert observed_threads and all(threads == budget for threads in observed_threads)
    assert torch.get_num_threads() == caller_threads


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_catboost_cpu_fallback_limits_automatic_feature_importance(monkeypatch, task):
    tree = TreeEmbedding(
        "CatBoost", [0], task, device="cuda:0", n_jobs=2, n_estimators=3, select_best_model=False,
    )
    X = pd.DataFrame({"category": ["a", "b"] * 16, "x": np.arange(32)})
    y = np.arange(len(X)) % 2
    get_importance = CatBoost.get_feature_importance
    observed_threads = []

    def record_importance(model, *args, thread_count=-1, **kwargs):
        observed_threads.append(thread_count)
        return get_importance(model, *args, thread_count=thread_count, **kwargs)

    monkeypatch.setattr(CatBoost, "get_feature_importance", record_importance)
    tree.fit_tree(X, y)
    assert tree.model.get_param("task_type") == "CPU"
    assert observed_threads == [2]
    assert "get_feature_importance" not in vars(tree.model)
    assert tree.model.feature_importances_.shape == (X.shape[1],)
