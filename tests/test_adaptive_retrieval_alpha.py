from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from iltm import iLTMClassifier, iLTMRegressor, inference_interface


def _predictors(count):
    return [
        {
            "retrieval_parameters": {
                "retrieval_alpha": 0.75,
            }
        }
        for _ in range(count)
    ]


def test_adaptive_retrieval_alpha_is_default():
    model = iLTMRegressor(checkpoint=None, device="cpu")

    assert model.retrieval_alpha == pytest.approx(0.75)
    assert model.retrieval_alpha_finetuning is False
    assert model.retrieval_alpha_adaptive is True


@pytest.mark.parametrize("weight", [1.0, 1e-8])
def test_adaptive_retrieval_alpha_regression(monkeypatch, weight):
    model = iLTMRegressor(
        checkpoint=None,
        device="cpu",
        retrieval_alpha_adaptive=True,
        clip_predictions=False,
    )
    model.clip_predictions_ = False
    model.predictors_ = _predictors(2)
    context = torch.ones((4, 1))
    for predictor in model.predictors_:
        predictor["X_ctxt_superset"] = context
    main = torch.zeros((2, 4))
    retrieval = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        ]
    )
    monkeypatch.setattr(
        model,
        "_predict_ensemble",
        lambda *args, **kwargs: (main, retrieval),
    )

    model._fit_adaptive_retrieval_alpha(
        np.zeros((4, 1)),
        weight * np.arange(4, dtype=np.float64),
        n_outputs=1,
    )

    assert model.retrieval_alpha_ == pytest.approx(weight)
    assert all(
        predictor["retrieval_parameters"]["retrieval_alpha"] == pytest.approx(weight)
        and predictor["X_ctxt_superset"] is context
        for predictor in model.predictors_
    )


def test_adaptive_retrieval_alpha_classification(monkeypatch):
    model = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        retrieval_alpha_adaptive=True,
        finetuning_classification_val_metric="logloss",
    )
    model.predictors_ = _predictors(1)
    main = torch.zeros((1, 4, 2))
    retrieval = torch.tensor(
        [
            [
                [5.0, -5.0],
                [-5.0, 5.0],
                [5.0, -5.0],
                [-5.0, 5.0],
            ]
        ]
    )
    monkeypatch.setattr(
        model,
        "_predict_ensemble",
        lambda *args, **kwargs: (main, retrieval),
    )

    model._fit_adaptive_retrieval_alpha(
        np.zeros((4, 1)),
        np.array([0, 1, 0, 1]),
        n_outputs=2,
    )

    assert model.retrieval_alpha_ == pytest.approx(1.0)


def test_adaptive_and_finetuned_alpha_are_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot both be enabled"):
        iLTMRegressor(
            checkpoint=None,
            device="cpu",
            retrieval_alpha_adaptive=True,
            retrieval_alpha_finetuning=True,
        )


def test_retrieval_components_are_collected_once_per_predictor(monkeypatch):
    model = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        tree_embedding=False,
    )
    model._fit_succeeded = True
    model.predictors_ = [{"index": 0}, {"index": 1}]
    monkeypatch.setattr(
        model,
        "_preprocess_for_predict_once",
        lambda X: torch.as_tensor(X, dtype=torch.float32),
    )
    calls = []

    def forward(predictor, X, n_outputs, **kwargs):
        calls.append(predictor["index"])
        offset = float(predictor["index"])
        return X[:, :n_outputs] + offset, X[:, :n_outputs] + offset + 10.0

    monkeypatch.setattr(model, "_forward_pass_predictor", forward)
    main, retrieval = model._predict_ensemble(
        np.ones((3, 2), dtype=np.float32),
        n_outputs=2,
        return_retrieval_components=True,
    )

    assert calls == [0, 1]
    assert main.shape == (2, 3, 2)
    assert retrieval.shape == (2, 3, 2)
    assert torch.allclose(retrieval - main, torch.full_like(main, 10.0))


@pytest.mark.parametrize("model_cls", [iLTMRegressor, iLTMClassifier])
@pytest.mark.parametrize(
    "mode", ["reuse", "subsampled", "chunked", "over_budget", "partial"]
)
def test_fit_reuses_validation_inputs_only_when_equivalent(
    monkeypatch, model_cls, mode
):
    model = model_cls(
        checkpoint=None,
        device="cpu",
        n_jobs=1,
        n_ensemble=2,
        tree_embedding=True,
        tree_for_each_predictor=True,
        tree_model="XGBoost_hist",
        tree_n_estimators=3,
        preprocessing="none",
        val_max_samples=4 if mode == "subsampled" else None,
    )
    model.checkpoint = "unused"
    monkeypatch.setattr(
        model,
        "_initialize_model",
        lambda: SimpleNamespace(pca_sampling="zeropad", n_dims=1, clip_data_value=1.0),
    )
    if mode == "chunked":
        model.inference_chunk_rows = 2
    if mode == "over_budget":
        monkeypatch.setattr(inference_interface, "_ALPHA_VALIDATION_CACHE_BYTES", 1)

    def generate(*args, **kwargs):
        predictor = _predictors(1)[0]
        predictor["index"] = len(model.predictors_)
        predictor["timed_out"] = mode == "partial" and predictor["index"] == 1
        return predictor

    def forward(predictor, X, n_outputs, **kwargs):
        values = X[:, :n_outputs] + predictor["index"]
        return values, -values

    monkeypatch.setattr(model, "_generate_predictor", generate)
    monkeypatch.setattr(model, "_forward_pass_predictor", forward)
    preprocess = model._preprocess_test_data
    preprocess_calls = []
    ensemble = model._predict_ensemble
    alpha_passes = []

    def record_preprocessing(X, preprocessor):
        preprocess_calls.append(None)
        return preprocess(X, preprocessor)

    def record_alpha_pass(X, **kwargs):
        before = len(preprocess_calls)
        components = ensemble(X, **kwargs)
        alpha_passes.append((len(preprocess_calls) - before, components))
        return components

    monkeypatch.setattr(model, "_preprocess_test_data", record_preprocessing)
    monkeypatch.setattr(model, "_predict_ensemble", record_alpha_pass)
    rng = np.random.default_rng(42)
    data = rng.normal(size=(26, 3)).astype(np.float32)
    y = np.tile([0, 1], 13)
    model.fit(
        data[:18], y[:18], eval_set=(data[18:], y[18:]), return_partial_on_timeout=False
    )
    assert len(model.predictors_) == (1 if mode == "partial" else 2)
    alpha = model.retrieval_alpha_

    y_val = y[18:]
    if model_cls is iLTMRegressor:
        y_val = ((y_val.astype(np.float32) - model._y_mean) / model._y_std).astype(
            np.float32
        )
    model._fit_adaptive_retrieval_alpha(data[18:], y_val, n_outputs=model.n_outputs_)
    assert model.retrieval_alpha_ == alpha
    (fit_calls, fit_components), (raw_calls, raw_components) = alpha_passes
    assert raw_calls > 0
    assert fit_calls == (0 if mode in ("reuse", "partial") else raw_calls)
    for fit_component, raw_component in zip(
        fit_components, raw_components, strict=True
    ):
        torch.testing.assert_close(fit_component, raw_component, rtol=0, atol=0)
