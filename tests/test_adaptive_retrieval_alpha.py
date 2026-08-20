import numpy as np
import pytest
import torch

from iltm import iLTMClassifier, iLTMRegressor


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


def test_adaptive_retrieval_alpha_regression(monkeypatch):
    model = iLTMRegressor(
        checkpoint=None,
        device="cpu",
        retrieval_alpha_adaptive=True,
        clip_predictions=False,
    )
    model.clip_predictions_ = False
    model.predictors_ = _predictors(2)
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
        np.arange(4, dtype=np.float32),
        n_outputs=1,
    )

    assert model.retrieval_alpha_ == pytest.approx(1.0)
    assert all(
        predictor["retrieval_parameters"]["retrieval_alpha"] == pytest.approx(1.0)
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
