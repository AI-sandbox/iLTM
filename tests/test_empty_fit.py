import numpy as np
import pytest
import torch

from iltm import iLTMClassifier, iLTMRegressor


class _TrainingModel(torch.nn.Module):
    pca_sampling = "zeropad"
    n_dims = 2
    clip_data_value = 1.0


def _classifier(monkeypatch, *, n_ensemble=1):
    classifier = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        n_ensemble=n_ensemble,
        n_dims=2,
        preprocessing="none",
        corr_select_k=0,
        finetuning=False,
        adaptive_memory=False,
    )
    classifier.checkpoint = "/unused/local-checkpoint.pth"
    classifier.model_path = classifier.checkpoint
    monkeypatch.setattr(classifier, "_initialize_model", _TrainingModel)
    return classifier


def test_fit_raises_timeout_if_budget_expires_before_first_predictor(monkeypatch):
    classifier = _classifier(monkeypatch)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    y = np.array([0, 1])

    with pytest.raises(TimeoutError, match="before the first predictor"):
        classifier.fit(X, y, fit_max_time=1.0)

    assert classifier.predictors_ == []
    assert classifier._model is None


def test_fit_rejects_empty_ensemble_without_timeout(monkeypatch):
    classifier = _classifier(monkeypatch, n_ensemble=0)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    y = np.array([0, 1])

    with pytest.raises(RuntimeError, match="without generating any predictors"):
        classifier.fit(X, y)

    assert classifier.predictors_ == []
    assert classifier._model is None


@pytest.mark.parametrize(
    ("estimator_class", "y"),
    [
        (iLTMClassifier, np.array([0, 1])),
        (iLTMRegressor, np.array([0.0, 1.0])),
    ],
)
def test_zero_fit_budget_times_out_immediately(estimator_class, y):
    estimator = estimator_class(checkpoint=None, device="cpu")
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    with pytest.raises(TimeoutError, match="before fitting could start"):
        estimator.fit(X, y, fit_max_time=0)

    assert estimator.predictors_ == []
    assert estimator._model is None


@pytest.mark.parametrize("fit_max_time", [-1.0, np.inf, np.nan])
@pytest.mark.parametrize(
    ("estimator_class", "y"),
    [
        (iLTMClassifier, np.array([0, 1])),
        (iLTMRegressor, np.array([0.0, 1.0])),
    ],
)
def test_invalid_fit_budget_is_rejected(estimator_class, y, fit_max_time):
    estimator = estimator_class(checkpoint=None, device="cpu")
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="finite and non-negative"):
        estimator.fit(X, y, fit_max_time=fit_max_time)
