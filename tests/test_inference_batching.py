from __future__ import annotations

import iltm.inference_interface as interface
import numpy as np
import pytest
import torch
from iltm import iLTMClassifier, iLTMRegressor


class _IdentityPCA:
    def transform(self, values):
        return values


class _NumericTree:
    def transform(self, values):
        return np.asarray(values, dtype=np.float32)


def _make_estimator(task, *, distance="cosine", bagging=False, dict_context=False):
    estimator_class = iLTMRegressor if task == "regression" else iLTMClassifier
    estimator = estimator_class(
        checkpoint=None,
        device="cpu",
        n_ensemble=2,
        batch_size=4,
        preprocessing="none",
        corr_select_k=0,
        tree_embedding=True,
        tree_for_each_predictor=True,
        concat_tree_with_orig_features=False,
        adaptive_memory=False,
        feature_bagging=bagging,
    )
    estimator.inference_chunk_rows = 3
    estimator._fit_succeeded = True
    n_outputs = 1 if task == "regression" else 2
    estimator.n_outputs_ = n_outputs
    estimator.classes_ = np.array([10, 20])
    estimator.voting = "hard" if task == "hard" else "soft"
    estimator._inference_model_config_ = {
        "pca_sampling": "zeropad",
        "n_dims": 2 if bagging else 4,
        "clip_data_value": 100.0,
    }
    estimator.preprocessors_ = [{}, {}]
    estimator.tr_ = [_NumericTree(), _NumericTree()]
    estimator.predictors_ = []
    for index in range(2):
        context = torch.arange(28, dtype=torch.float32).reshape(7, 4) / 11 - 1
        context = context + index
        width = 2 if bagging else 4
        hidden = torch.nn.Linear(width, width)
        output = torch.nn.Linear(width, n_outputs)
        with torch.no_grad():
            hidden.weight.copy_(torch.eye(width))
            hidden.bias.fill_(0.1)
            output.weight.copy_(torch.arange(width * n_outputs, dtype=torch.float32).reshape(n_outputs, width) / 7)
            output.bias.fill_(-0.3)
        for layer in (hidden, output):
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
        targets = torch.linspace(-1, 1, len(context)) if n_outputs == 1 else torch.arange(len(context)) % n_outputs
        estimator.predictors_.append(
            {
                "feature_bagging_idxs": [0, 2] if bagging else None,
                "rf": torch.nn.Identity(),
                "pca": _IdentityPCA(),
                "norm": None,
                "main_network": torch.nn.ModuleList([hidden, output]),
                "X_ctxt_superset": ({"x_num": context[:, :2], "x_cat": context[:, 2:]} if dict_context else context),
                "y_ctxt_superset": targets,
                "retrieval_parameters": {
                    "do_retrieval": True,
                    "retrieval_alpha": 0.75,
                    "retrieval_temperature": 1.05,
                    "retrieval_distance": distance,
                },
            }
        )
    moves = []
    estimator._move_predictor_to_device = lambda predictor, device=None: moves.append("device") or predictor
    estimator._move_predictor_to_cpu = lambda predictor: moves.append("cpu") or predictor

    def preprocess(values, preprocessor):
        values = torch.as_tensor(values, dtype=torch.float32)
        return {"x_num": values[:, :2], "x_cat": values[:, 2:]} if dict_context else values

    estimator._preprocess_test_data = preprocess
    return estimator, moves


@pytest.mark.parametrize("task", ["soft", "hard", "regression", "components"])
@pytest.mark.parametrize("distance", ["cosine", "euclidean"])
@pytest.mark.parametrize(("dict_context", "bagging"), [(False, False), (True, True)])
def test_reuses_context_per_predictor_and_prediction(monkeypatch, task, distance, dict_context, bagging):
    estimator, moves = _make_estimator(task, distance=distance, dict_context=dict_context, bagging=bagging)
    X = np.arange(28, dtype=np.float32).reshape(7, 4) / 13 - 0.5
    original_forward = estimator._forward_pass_predictor

    def predict():
        if task == "hard":
            return estimator.predict(X)
        return estimator._predict_ensemble(
            X,
            n_outputs=estimator.n_outputs_,
            softmax_per_predictor=task == "soft",
            return_retrieval_components=task == "components",
        )

    def uncached_forward(*args, **kwargs):
        kwargs.pop("retrieval_context_cache", None)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(estimator, "_forward_pass_predictor", uncached_forward)
    expected = predict()
    monkeypatch.setattr(estimator, "_forward_pass_predictor", original_forward)
    prepare = interface.prepare_retrieval_context
    prepared = []

    def prepare_spy(*args, **kwargs):
        result = prepare(*args, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr(interface, "prepare_retrieval_context", prepare_spy)
    moves.clear()
    actual = predict()
    assert len(prepared) == 2
    assert moves == ["device", "cpu", "device", "cpu"]
    if task == "hard":
        np.testing.assert_array_equal(actual, expected)
    elif task == "components":
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for predictor in estimator.predictors_:
        predictor["retrieval_parameters"]["retrieval_alpha"] = 0.5
        predictor["y_ctxt_superset"] = predictor["y_ctxt_superset"].flip(0)
        assert not any("cache" in key for key in predictor)
    predict()
    assert len(prepared) == 4
    assert not any("cache" in key for key in estimator.__dict__)


@pytest.mark.parametrize("task", ["soft", "hard"])
def test_failure_clears_transient_context_and_restores_cpu(monkeypatch, task):
    estimator, moves = _make_estimator(task)
    forward = estimator._forward_pass_predictor
    caches = []

    def failing_forward(*args, **kwargs):
        caches.append(kwargs["retrieval_context_cache"])
        if len(caches) == 2:
            assert caches[-1]["context"] is caches[0]["context"]
            raise ValueError("test prediction failure")
        return forward(*args, **kwargs)

    monkeypatch.setattr(estimator, "_forward_pass_predictor", failing_forward)
    X = np.arange(28, dtype=np.float32).reshape(7, 4)

    def predict():
        if task == "hard":
            return estimator.predict(X)
        return estimator._predict_ensemble(X, n_outputs=2)

    with pytest.raises(ValueError, match="test prediction failure"):
        predict()
    assert moves == ["device", "cpu"]
    assert caches
    assert all(cache == {} for cache in caches)


def test_contiguous_slices_preserve_rows_and_batch_boundaries(monkeypatch):
    estimator, _ = _make_estimator("regression")
    predictor = estimator.predictors_[0]
    predictor["retrieval_parameters"]["do_retrieval"] = False
    X = torch.arange(40, dtype=torch.float32).reshape(4, 10).T
    assert not X.is_contiguous()
    seen = []

    def forward(values, n_outputs, batch_size, *args, **kwargs):
        assert values.is_contiguous()
        seen.append(values.clone())
        return values[:, 0]

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("Prediction should slice the existing tensor")

    monkeypatch.setattr(interface, "full_main_forward", forward)
    monkeypatch.setattr(torch.utils.data, "DataLoader", forbidden_loader)
    actual = estimator._forward_pass_predictor(predictor, X, n_outputs=1)
    assert [len(batch) for batch in seen] == [4, 4, 2]
    torch.testing.assert_close(torch.cat(seen), X, rtol=0, atol=0)
    torch.testing.assert_close(actual, X[:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("retry", [False, True])
def test_sliced_batches_preserve_dataloader_rng_state(monkeypatch, retry):
    estimator, _ = _make_estimator("regression")
    estimator.batch_size = 256
    predictor = estimator.predictors_[0]
    predictor["retrieval_parameters"]["do_retrieval"] = False
    X = torch.arange(1200, dtype=torch.float32).reshape(300, 4)
    calls = []

    def forward(values, n_outputs, batch_size, *args, **kwargs):
        calls.append(len(values))
        if retry and len(calls) == 2:
            raise RuntimeError("CUDA out of memory: test")
        return values[:, 0]

    monkeypatch.setattr(interface, "full_main_forward", forward)
    monkeypatch.setattr(interface, "clear_cuda_cache", lambda: None)
    start_state = torch.get_rng_state()
    dataset = torch.utils.data.TensorDataset(X)
    for batch_size in [256, 128] if retry else [256]:
        list(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False))
    expected_state = torch.get_rng_state()
    torch.set_rng_state(start_state)
    actual = estimator._forward_pass_predictor(predictor, X, n_outputs=1)
    torch.testing.assert_close(torch.get_rng_state(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(actual, X[:, 0], rtol=0, atol=0)
    assert calls == ([256, 44, 128, 128, 44] if retry else [256, 44])


def test_rf_dtype_transfers_release_old_storage_without_full_gc(monkeypatch):
    estimator = iLTMClassifier(checkpoint=None, device="cpu", inference_storage_dtype="float16")
    predictor = {"rf": torch.nn.Sequential(torch.nn.Linear(4, 8, bias=False).half())}

    def forbidden_collect():
        raise AssertionError("Predictor transfer should not run a full garbage collection")

    monkeypatch.setattr(interface.gc, "collect", forbidden_collect)
    half_storage = predictor["rf"][0].weight.untyped_storage()._weak_ref()
    float_storage = None
    try:
        estimator._move_predictor_to_device(predictor)
        assert predictor["rf"][0].weight.dtype == torch.float32
        assert torch.UntypedStorage._expired(half_storage)
        float_storage = predictor["rf"][0].weight.untyped_storage()._weak_ref()
        estimator._move_predictor_to_cpu(predictor)
        assert predictor["rf"][0].weight.dtype == torch.float16
        assert torch.UntypedStorage._expired(float_storage)
    finally:
        torch.UntypedStorage._free_weak_ref(half_storage)
        if float_storage is not None:
            torch.UntypedStorage._free_weak_ref(float_storage)
