import numpy as np
import pandas as pd
import pytest

import iltm.tree_embedding as tree_embedding
from iltm.tree_embedding import TreeEmbedding


def _tree(model, task):
    return TreeEmbedding(
        tree_model=model,
        cat_features=[],
        task_type=task,
        device="cuda:0",
        select_best_model=False,
        n_estimators=200,
        max_depth=6,
        max_leaves=31,
    )


def _data(task):
    X = pd.DataFrame({"feature": [0.0, 1.0]})
    if task == "classification":
        return X, np.array([0, 1])
    return X, np.array([0.0, 1.0])


def test_xgboost_oom_backoff_reaches_cpu(monkeypatch):
    calls = []
    fitted_model = object()

    def train(*, params, **kwargs):
        calls.append(dict(params))
        if str(params["device"]).startswith("cuda"):
            raise RuntimeError("CUDA out of memory")
        return fitted_model

    monkeypatch.setattr(tree_embedding.xgb, "DMatrix", lambda *args, **kwargs: object())
    monkeypatch.setattr(tree_embedding.xgb, "train", train)
    monkeypatch.setattr(tree_embedding, "get_gpu_memory_info", lambda *args, **kwargs: None)

    tree = _tree("XGBoost_hist", "regression")
    tree._fit_model(*_data("regression"))

    assert tree.model is fitted_model
    assert calls[-1]["device"] == "cpu"
    assert calls[-1]["max_leaves"] == 31


def test_xgboost_oom_backoff_reraises_after_cpu_failure(monkeypatch):
    calls = []

    def train(*, params, **kwargs):
        calls.append(dict(params))
        raise RuntimeError("CUDA out of memory: xgboost retry test")

    monkeypatch.setattr(tree_embedding.xgb, "DMatrix", lambda *args, **kwargs: object())
    monkeypatch.setattr(tree_embedding.xgb, "train", train)
    monkeypatch.setattr(tree_embedding, "get_gpu_memory_info", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="xgboost retry test"):
        _tree("XGBoost_hist", "regression")._fit_model(*_data("regression"))

    assert calls[-1]["device"] == "cpu"


def test_catboost_oom_backoff_reaches_cpu(monkeypatch):
    calls = []

    class CatBoost:
        def __init__(self, **params):
            self.params = dict(params)
            calls.append(self.params)

        def fit(self, *args, **kwargs):
            if self.params["task_type"] == "GPU":
                raise RuntimeError("CUDA out of memory")
            return self

    monkeypatch.setattr(tree_embedding, "CatBoostClassifier", CatBoost)
    monkeypatch.setattr(tree_embedding, "Pool", lambda *args, **kwargs: object())
    monkeypatch.setattr(tree_embedding, "get_gpu_memory_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(tree_embedding, "pick_gpu_ram_part", lambda *args, **kwargs: 0.8)

    tree = _tree("CatBoost", "classification")
    tree._fit_model(*_data("classification"))

    assert tree.model.params["task_type"] == "CPU"
    assert calls[-1]["task_type"] == "CPU"


def test_catboost_oom_backoff_reraises_after_cpu_failure(monkeypatch):
    calls = []

    class CatBoost:
        def __init__(self, **params):
            self.params = dict(params)
            calls.append(self.params)

        def fit(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory: catboost retry test")

    monkeypatch.setattr(tree_embedding, "CatBoostClassifier", CatBoost)
    monkeypatch.setattr(tree_embedding, "Pool", lambda *args, **kwargs: object())
    monkeypatch.setattr(tree_embedding, "get_gpu_memory_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(tree_embedding, "pick_gpu_ram_part", lambda *args, **kwargs: 0.8)

    with pytest.raises(RuntimeError, match="catboost retry test"):
        _tree("CatBoost", "classification")._fit_model(*_data("classification"))

    assert calls[-1]["task_type"] == "CPU"
