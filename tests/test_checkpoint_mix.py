from types import SimpleNamespace

import numpy as np
import torch
from sklearn.base import clone

import iltm.inference_interface as inference_interface
from iltm.inference_interface import iLTMClassifier


def _checkpoint_config(name):
    configs = {
        "base": {
            "checkpoint": "/base.pth",
            "preprocessing": "realmlp_td_s_v0",
            "tree_embedding": True,
            "tree_model": "XGBoost_hist",
            "concat_tree_with_orig_features": True,
        },
        "cb": {
            "checkpoint": "/cb.pth",
            "preprocessing": "realmlp_td_s_v0",
            "tree_embedding": True,
            "tree_model": "CatBoost",
            "concat_tree_with_orig_features": True,
        },
        "xgb": {
            "checkpoint": "/xgb.pth",
            "preprocessing": "realmlp_td_s_v0",
            "tree_embedding": True,
            "tree_model": "XGBoost_hist",
            "concat_tree_with_orig_features": True,
        },
        "rtr": {
            "checkpoint": "/rtr.pth",
            "preprocessing": "realmlp_td_s_v0",
            "tree_embedding": False,
            "concat_tree_with_orig_features": False,
            "do_retrieval": True,
        },
    }
    return configs.get(name, {"checkpoint": name})


def test_checkpoint_mix_is_cloneable(monkeypatch):
    monkeypatch.setattr(
        inference_interface,
        "resolve_model_checkpoint",
        _checkpoint_config,
    )
    estimator = iLTMClassifier(
        checkpoint="base",
        checkpoint_mix=("cb", "xgb", "rtr"),
        device="cpu",
    )

    cloned = clone(estimator)

    assert cloned.checkpoint_mix == ("cb", "xgb", "rtr")
    assert [config["checkpoint"] for config in cloned._checkpoint_mix_configs] == [
        "/cb.pth",
        "/xgb.pth",
        "/rtr.pth",
    ]


def test_checkpoint_mix_fits_each_predictor_with_its_checkpoint(monkeypatch):
    monkeypatch.setattr(
        inference_interface,
        "resolve_model_checkpoint",
        _checkpoint_config,
    )

    class FakeTree:
        def __init__(self, **kwargs):
            self.tree_model = kwargs["tree_model"]

        def fit_tree(self, X, y, **kwargs):
            return self

        def transform(self, X):
            return np.ones((len(X), 1), dtype=np.float32)

    monkeypatch.setattr(inference_interface, "TreeEmbedding", FakeTree)
    estimator = iLTMClassifier(
        checkpoint="base",
        checkpoint_mix=("cb", "xgb", "rtr"),
        n_ensemble=3,
        device="cpu",
        retrieval_alpha_adaptive=False,
    )

    initialized = []

    def initialize_model():
        initialized.append(estimator.model_path)
        return SimpleNamespace(
            pca_sampling="zeropad",
            n_dims=512,
            clip_data_value=1_000_000,
        )

    estimator._initialize_model = initialize_model
    estimator._release_training_model = lambda: setattr(estimator, "_model", None)
    estimator._auto_tune_for_memory = lambda: None
    estimator._should_stop_for_cpu_memory_before_predictor = lambda index: (False, None)
    estimator._preprocess_fitting_data = lambda X, y, **kwargs: (
        np.asarray(X, dtype=np.float32),
        np.asarray(y),
        {},
    )
    estimator._generate_predictor = lambda *args, **kwargs: {
        "timed_out": False,
        "retrieval_parameters": {},
    }

    X = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    estimator._fit_common(X, y, n_outputs=2)

    assert initialized == ["/cb.pth", "/xgb.pth", "/rtr.pth"]
    assert [config["checkpoint"] for config in estimator.predictor_configs_] == initialized
    assert [tree.tree_model if tree is not None else None for tree in estimator.tr_] == [
        "CatBoost",
        "XGBoost_hist",
        None,
    ]
    assert len(estimator.predictors_) == len(estimator.preprocessors_) == 3


def test_checkpoint_mix_uses_predictor_specific_inference_path(monkeypatch):
    monkeypatch.setattr(
        inference_interface,
        "resolve_model_checkpoint",
        _checkpoint_config,
    )

    class FakeTree:
        def transform(self, X):
            return np.asarray(X) + 10

    estimator = iLTMClassifier(
        checkpoint="base",
        checkpoint_mix=("cb", "rtr"),
        n_ensemble=2,
        device="cpu",
        retrieval_alpha_adaptive=False,
    )
    estimator._fit_succeeded = True
    estimator.predictors_ = [{"value": 1.0}, {"value": 3.0}]
    estimator.predictor_configs_ = estimator._checkpoint_mix_configs.copy()
    estimator.preprocessors_ = [{"id": "tree"}, {"id": "plain"}]
    estimator.tr_ = [FakeTree(), None]
    estimator._move_predictor_to_device = lambda predictor, device=None: predictor
    estimator._move_predictor_to_cpu = lambda predictor: predictor

    seen_shapes = []

    def preprocess(X, preprocessing):
        seen_shapes.append(np.asarray(X).shape)
        return torch.as_tensor(np.asarray(X), dtype=torch.float32)

    estimator._preprocess_test_data = preprocess
    estimator._forward_pass_predictor = lambda predictor, X, **kwargs: torch.full(
        (len(X), 2), predictor["value"]
    )

    outputs = estimator._predict_ensemble(
        np.arange(6, dtype=np.float32).reshape(3, 2),
        n_outputs=2,
    )

    assert seen_shapes == [(3, 4), (3, 2)]
    torch.testing.assert_close(outputs, torch.full((3, 2), 2.0))
