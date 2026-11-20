from types import SimpleNamespace

import numpy as np
import pytest
import torch

from iltm import iLTMClassifier, iLTMRegressor
from iltm import inference_interface, utils


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
    monkeypatch.setattr(classifier, "_initialize_model", lambda: pytest.fail("loaded checkpoint with no fit budget"))
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    y = np.array([0, 1])

    with pytest.raises(TimeoutError, match="before the first predictor"):
        classifier.fit(X, y, fit_max_time=1.0)

    assert classifier.predictors_ == []
    assert classifier._model is None


@pytest.fixture
def fit_clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    clock.time = lambda: clock.now
    monkeypatch.setattr(inference_interface, "time", clock)
    monkeypatch.setattr(utils, "time", clock)
    return clock


@pytest.mark.parametrize("tree_for_each_predictor", [False, True])
def test_deadline_checked_before_tree_fit(monkeypatch, fit_clock, tree_for_each_predictor):
    model = _classifier(monkeypatch)
    model.set_params(tree_embedding=True, tree_for_each_predictor=tree_for_each_predictor)

    def split_at_deadline(X, y, **kwargs):
        fit_clock.now = 101.0
        return X, y, X, y

    tree = SimpleNamespace(fit_tree=lambda *a, **k: pytest.fail("started tree fit after the deadline"))
    monkeypatch.setattr(model, "_split_data_tree_embedding", split_at_deadline)
    monkeypatch.setattr(model, "_create_tree_embedding", lambda **kwargs: tree)

    X = np.eye(2, dtype=np.float32)
    with pytest.raises(TimeoutError):
        model.fit(X, np.array([0, 1]), fit_max_time=11.0)
    assert not model.__sklearn_is_fitted__()


def test_tree_overrun_keeps_completed_predictors_and_skips_alpha_scoring(monkeypatch, fit_clock):
    model = _classifier(monkeypatch, n_ensemble=2)
    model.set_params(tree_embedding=True, tree_for_each_predictor=True, tree_data_split="all")

    def fit_slow_tree(*args, **kwargs):
        fit_clock.now = 102.0

    fast_tree = SimpleNamespace(fit_tree=lambda *a, **k: None, transform=lambda X: X[:, :1])
    slow_tree = SimpleNamespace(
        fit_tree=fit_slow_tree,
        transform=lambda X: pytest.fail("transformed data after tree fit exhausted the budget"),
    )
    trees = iter([fast_tree, slow_tree])
    predictor = {"timed_out": False}
    monkeypatch.setattr(model, "_create_tree_embedding", lambda **kwargs: next(trees))
    monkeypatch.setattr(model, "_generate_predictor", lambda *args, **kwargs: predictor)
    monkeypatch.setattr(model, "_predict_ensemble", lambda *args, **kwargs: pytest.fail("started alpha scoring"))

    X = np.eye(2, dtype=np.float32)
    y = np.array([0, 1])
    assert model.fit(X, y, eval_set=(X, y), fit_max_time=11.0) is model
    assert model.predictors_ == [predictor]
    assert model.tr_ == [fast_tree]
    assert model.retrieval_alpha_ == model.retrieval_alpha


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
