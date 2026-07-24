from __future__ import annotations

import pytest
import torch

import iltm.inference_interface as inference_interface
from iltm import iLTMClassifier


class _FakeHypernetwork:
    def __call__(self, X, y, n_outputs):
        tensors = X.values() if isinstance(X, dict) else [X]
        assert all(value.device.type == "meta" for value in tensors)
        assert y.device.type == "meta"
        return None, None, [], None


def _estimator(**kwargs):
    estimator = iLTMClassifier(
        checkpoint=None,
        device="meta",
        n_ensemble=1,
        batch_size=2,
        n_dims=2,
        preprocessing="none",
        corr_select_k=0,
        finetuning=True,
        adaptive_memory=False,
        **kwargs,
    )
    estimator._model = _FakeHypernetwork()
    return estimator


def _capture_validation(monkeypatch):
    captured = {}

    def fake_fine_tune(*args, **kwargs):
        captured["X_val"] = kwargs["X_val"]
        captured["y_val"] = kwargs["y_val"]
        return {
            "rf": None,
            "pca": None,
            "main_network": [],
            "norm": None,
            "timed_out": False,
        }

    monkeypatch.setattr(
        inference_interface,
        "fine_tune_main_network",
        fake_fine_tune,
    )
    return captured


@pytest.mark.parametrize("dict_input", [False, True])
def test_external_validation_remains_on_cpu(monkeypatch, dict_input):
    estimator = _estimator()
    captured = _capture_validation(monkeypatch)

    if dict_input:
        X = {
            "x_num": torch.ones((4, 1)),
            "x_cat": torch.zeros((4, 1)),
        }
        X_val = {
            "x_num": torch.full((2, 1), 2.0),
            "x_cat": torch.full((2, 1), 3.0),
        }
    else:
        X = torch.ones((4, 2))
        X_val = torch.full((2, 2), 2.0)

    estimator._generate_predictor(
        X,
        torch.tensor([0, 1, 0, 1]),
        n_outputs=2,
        X_val=X_val,
        y_val=torch.tensor([0, 1]),
    )

    validation = captured["X_val"]
    tensors = validation.values() if isinstance(validation, dict) else [validation]
    assert all(value.device.type == "cpu" for value in tensors)
    assert captured["y_val"].device.type == "cpu"


def test_cpu_validation_preserves_feature_bagging(monkeypatch):
    estimator = _estimator(
        feature_bagging=True,
        feature_bagging_size=1,
    )
    captured = _capture_validation(monkeypatch)
    feature_indices = torch.tensor([1])

    def sample_data(X, y, pca_sampling):
        return X[:2, feature_indices], y[:2], feature_indices

    monkeypatch.setattr(estimator, "_sample_data", sample_data)
    X = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    X_val = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    estimator._generate_predictor(
        X,
        torch.tensor([0, 1, 0, 1]),
        n_outputs=2,
        X_val=X_val,
        y_val=torch.tensor([0, 1]),
    )

    assert captured["X_val"].device.type == "cpu"
    torch.testing.assert_close(
        captured["X_val"],
        X_val[:, feature_indices],
    )
