from __future__ import annotations

import types

import pytest
import torch

import iltm.inference_interface as inference_interface
import iltm.utils as iltm_utils
from iltm import iLTMRegressor


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


def test_oom_retry_discards_outputs_from_failed_attempt(monkeypatch):
    estimator = iLTMRegressor(
        checkpoint=None,
        device="cpu",
        batch_size=256,
        preprocessing="none",
        corr_select_k=0,
        adaptive_memory=False,
    )
    estimator._model = types.SimpleNamespace(
        pca_sampling="zeropad",
        n_dims=1,
        clip_data_value=100.0,
    )
    predictor = _minimal_predictor()
    moves = []
    forward_calls = []
    cache_clears = []

    def move_to_device(self, predictor, device=None):
        moves.append("device")
        return predictor

    def move_to_cpu(self, predictor):
        moves.append("cpu")
        return predictor

    def fail_after_one_batch(X, n_outputs, batch_size, *args, **kwargs):
        forward_calls.append((len(X), batch_size))
        if len(forward_calls) == 2:
            raise RuntimeError("CUDA out of memory: retry test")
        return X[:, 0]

    estimator._move_predictor_to_device = types.MethodType(
        move_to_device,
        estimator,
    )
    estimator._move_predictor_to_cpu = types.MethodType(
        move_to_cpu,
        estimator,
    )
    monkeypatch.setattr(
        inference_interface,
        "full_main_forward",
        fail_after_one_batch,
    )
    monkeypatch.setattr(
        iltm_utils,
        "clear_cuda_cache",
        lambda: cache_clears.append(None),
    )
    X = torch.arange(300, dtype=torch.float32).reshape(-1, 1)

    actual = estimator._forward_pass_predictor(
        predictor,
        X,
        n_outputs=1,
    )

    torch.testing.assert_close(actual, X[:, 0])
    assert actual.shape == (300,)
    assert forward_calls == [
        (256, 256),
        (44, 256),
        (128, 128),
        (128, 128),
        (44, 128),
    ]
    assert cache_clears == [None]
    assert moves == ["device", "cpu"]


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory",
        "Tried to allocate 2.00 GiB",
    ],
)
def test_cuda_oom_signatures_are_recognized(message):
    assert iltm_utils.is_cuda_oom(RuntimeError(message))


def test_unrelated_cuda_error_is_not_treated_as_oom():
    error = RuntimeError("CUDA error: invalid device ordinal")

    assert not iltm_utils.is_cuda_oom(error)
