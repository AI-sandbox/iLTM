from __future__ import annotations

import types

import numpy as np
import pytest
import torch

import iltm.inference_interface as inference_interface
from iltm import iLTMClassifier, iLTMRegressor


class _NumericTree:
    n_orig_features_to_keep_ = None

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


def _minimal_predictor():
    return {
        "feature_bagging_idxs": None,
        "rf": None,
        "pca": None,
        "norm": None,
        "main_network": [],
        "X_ctxt_superset": None,
        "y_ctxt_superset": None,
        "retrieval_parameters": {
            "do_retrieval": False,
            "retrieval_alpha": 0.0,
            "retrieval_temperature": 1.0,
            "retrieval_distance": "euclidean",
        },
        "timed_out": False,
    }


def _configure_chunked(estimator):
    estimator.inference_chunk_rows = 3
    estimator.tree_embedding = True
    estimator.tree_for_each_predictor = True
    estimator.concat_tree_with_orig_features = False
    estimator.tr_ = [_NumericTree()]
    estimator.preprocessors_ = [{"corr_selected_indices": None}]
    estimator.predictors_ = [_minimal_predictor()]
    estimator._model = types.SimpleNamespace(
        pca_sampling="zeropad",
        n_dims=1,
        clip_data_value=100.0,
    )
    return estimator


def _install_transfer_probe(monkeypatch, estimator, *, fail_forward=False):
    moves = []

    def move_to_device(self, predictor, device=None):
        moves.append("device")
        predictor["probe_device"] = "device"
        return predictor

    def move_to_cpu(self, predictor):
        moves.append("cpu")
        predictor["probe_device"] = "cpu"
        return predictor

    def forward(X, n_outputs, *args, **kwargs):
        if fail_forward:
            raise ValueError("intentional forward failure")
        values = X[:, 0]
        if n_outputs == 1:
            return values
        return torch.stack((values, -values), dim=1)

    estimator._move_predictor_to_device = types.MethodType(
        move_to_device,
        estimator,
    )
    estimator._move_predictor_to_cpu = types.MethodType(
        move_to_cpu,
        estimator,
    )
    monkeypatch.setattr(inference_interface, "full_main_forward", forward)
    return moves


@pytest.mark.parametrize("task", ["regression", "soft", "hard"])
def test_chunked_prediction_uses_one_device_lifetime(monkeypatch, task):
    if task == "regression":
        estimator = iLTMRegressor(
            checkpoint=None,
            device="cpu",
            preprocessing="none",
            corr_select_k=0,
            adaptive_memory=False,
        )
    else:
        estimator = iLTMClassifier(
            checkpoint=None,
            device="cpu",
            preprocessing="none",
            corr_select_k=0,
            adaptive_memory=False,
            voting=task,
        )
        estimator.n_outputs_ = 2
        estimator.classes_ = np.array([10, 20])

    _configure_chunked(estimator)
    moves = _install_transfer_probe(monkeypatch, estimator)
    X = np.arange(7, dtype=np.float32).reshape(-1, 1)

    if task == "regression":
        actual = estimator._predict_ensemble(X, n_outputs=1)
        torch.testing.assert_close(actual, torch.from_numpy(X[:, 0]))
    elif task == "soft":
        actual = estimator.predict_proba(X)
        assert actual.shape == (7, 2)
    else:
        actual = estimator.predict(X)
        np.testing.assert_array_equal(actual, np.full(7, 10))

    assert moves == ["device", "cpu"]
    assert estimator.predictors_[0]["probe_device"] == "cpu"


@pytest.mark.parametrize("chunked", [False, True])
def test_prediction_failure_restores_predictor_to_cpu(monkeypatch, chunked):
    estimator = iLTMRegressor(
        checkpoint=None,
        device="cpu",
        preprocessing="none",
        corr_select_k=0,
        adaptive_memory=False,
    )
    if chunked:
        _configure_chunked(estimator)
    else:
        estimator.tree_embedding = False
        estimator.predictors_ = [_minimal_predictor()]
        estimator.preprocessors_ = [{"corr_selected_indices": None}]
        estimator._model = types.SimpleNamespace(
            pca_sampling="zeropad",
            n_dims=1,
            clip_data_value=100.0,
        )
    moves = _install_transfer_probe(
        monkeypatch,
        estimator,
        fail_forward=True,
    )

    with pytest.raises(ValueError, match="intentional forward failure"):
        estimator._predict_ensemble(
            np.arange(7, dtype=np.float32).reshape(-1, 1),
            n_outputs=1,
        )

    assert moves == ["device", "cpu"]
    assert estimator.predictors_[0]["probe_device"] == "cpu"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("storage_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_inference_restores_original_rf_weight(device, storage_dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    estimator = iLTMRegressor(checkpoint=None, device=device, inference_storage_dtype=storage_dtype)
    layer = torch.nn.Linear(3, 4, bias=False).to(storage_dtype)
    rf = torch.nn.Sequential(layer, torch.nn.ReLU())
    predictor = {"rf": rf}
    original = layer.weight
    stored_pointer = original.data_ptr()
    x = torch.tensor([[1., -2., 3.]], device=device)
    expected = torch.nn.functional.linear(x, original.detach().to(x)).relu()

    for fail in (False, True):
        def run():
            with estimator._predictor_on_device(predictor):
                assert layer.weight.dtype == torch.float32
                assert layer.weight.device == x.device
                torch.testing.assert_close(rf(x), expected, rtol=0, atol=0)
                if fail:
                    raise RuntimeError("interrupted prediction")

        if fail:
            with pytest.raises(RuntimeError, match="interrupted prediction"):
                run()
        else:
            run()
        assert layer.weight is original
        assert layer.weight.data_ptr() == stored_pointer
        assert layer.weight.dtype == storage_dtype


def test_failed_device_transfer_restores_original_rf_weight(monkeypatch):
    estimator = iLTMRegressor(checkpoint=None, device="cpu", inference_storage_dtype="float16")
    layer = torch.nn.Linear(3, 4, bias=False).half()
    predictor = {"rf": torch.nn.Sequential(layer)}
    original = layer.weight
    move = estimator._move_predictor_to_device

    def fail_after_move(predictor, device=None):
        move(predictor, device)
        assert layer.weight.dtype == torch.float32
        raise RuntimeError("interrupted transfer")

    monkeypatch.setattr(estimator, "_move_predictor_to_device", fail_after_move)
    with pytest.raises(RuntimeError, match="interrupted transfer"):
        with estimator._predictor_on_device(predictor):
            pytest.fail("transfer should have failed before prediction")
    assert layer.weight is original
    assert layer.weight.dtype == torch.float16
