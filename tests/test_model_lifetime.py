import pickle

import numpy as np
import torch

from iltm import iLTMClassifier


class IdentityPCA(torch.nn.Module):
    def transform(self, X):
        return X


class RecordingTrainingModel(torch.nn.Module):
    pca_sampling = "zeropad"
    n_dims = 2
    clip_data_value = 17.0

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.moves = []

    def to(self, device, *args, **kwargs):
        # Record logical placement without requiring a CUDA device in this unit
        # test. The real iLTM module uses nn.Module.to unchanged.
        self.moves.append(str(torch.device(device)))
        return self


def _predictor():
    hidden = torch.nn.Linear(2, 2)
    output = torch.nn.Linear(2, 2)
    with torch.no_grad():
        hidden.weight.copy_(torch.tensor([[0.5, -0.25], [0.75, 0.125]]))
        hidden.bias.copy_(torch.tensor([0.1, -0.2]))
        output.weight.copy_(torch.tensor([[0.25, 0.5], [-0.75, 0.125]]))
        output.bias.copy_(torch.tensor([0.3, -0.4]))
    for parameter in (*hidden.parameters(), *output.parameters()):
        parameter.requires_grad_(False)

    return {
        "feature_bagging_idxs": None,
        "rf": torch.nn.Identity(),
        "pca": IdentityPCA(),
        "norm": None,
        "main_network": torch.nn.ModuleList([hidden, output]),
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


def _cache_key(estimator):
    arch_params = (
        estimator.n_dims,
        estimator.hn_n_layers,
        estimator.hn_hidden_size,
        estimator.clip_data_value,
        estimator.rf_size,
        estimator.n_classes_limit,
        estimator.dim_exp_type,
        estimator.bottleneck_size,
        estimator.main_n_layers,
        estimator.pca_fit,
        estimator.pca_svd_driver,
        estimator.hyper_dropout,
        estimator.pca_sampling,
    )
    return (estimator.model_path, str(estimator.device)) + arch_params


def test_fit_releases_training_model_and_preserves_predictions(monkeypatch):
    classifier = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        n_ensemble=1,
        n_dims=2,
        preprocessing="none",
        corr_select_k=0,
        finetuning=False,
        adaptive_memory=False,
    )
    classifier.checkpoint = "/unused/local-checkpoint.pth"
    classifier.model_path = classifier.checkpoint
    monkeypatch.setattr(type(classifier), "_model_cache", {})

    training_model = RecordingTrainingModel()
    cache_key = ("unit-test-cache-entry",)

    def initialize_model():
        classifier._model_cache[cache_key] = training_model
        return training_model

    monkeypatch.setattr(classifier, "_initialize_model", initialize_model)
    monkeypatch.setattr(classifier, "_generate_predictor", lambda *args, **kwargs: _predictor())

    X_train = np.array(
        [[-1.0, 0.5], [0.0, 1.0], [1.0, -0.5], [2.0, 0.25]],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.25, -0.75], [1.5, 0.5]], dtype=np.float32)
    predictions_before_release = []
    release_training_model = classifier._release_training_model

    def record_then_release():
        predictions_before_release.append(classifier.predict_proba(X_test))
        release_training_model()

    monkeypatch.setattr(classifier, "_release_training_model", record_then_release)
    classifier.fit(X_train, y_train)
    predictions_after_release = classifier.predict_proba(X_test)

    assert classifier._model is None
    assert classifier._model_cache[cache_key] is training_model
    assert training_model.moves[-1] == "cpu"
    assert classifier._inference_model_config_ == {
        "pca_sampling": "zeropad",
        "n_dims": 2,
        "clip_data_value": 17.0,
    }
    np.testing.assert_array_equal(
        predictions_after_release,
        predictions_before_release[0],
    )


def test_cached_training_model_moves_back_to_requested_device(monkeypatch):
    classifier = iLTMClassifier(
        checkpoint=None,
        device="cuda:7",
        n_dims=2,
        adaptive_memory=False,
    )
    classifier.model_path = "/unused/local-checkpoint.pth"
    training_model = RecordingTrainingModel()
    monkeypatch.setattr(
        type(classifier),
        "_model_cache",
        {_cache_key(classifier): training_model},
    )
    classifier._model = training_model
    classifier._inference_model_config_ = {
        "pca_sampling": "zeropad",
        "n_dims": 2,
        "clip_data_value": 17.0,
    }

    classifier._release_training_model()

    assert classifier._model is None
    assert training_model.moves == ["cpu"]

    restored = classifier._initialize_model()

    assert restored is training_model
    assert training_model.moves == ["cpu", "cuda:7"]
    assert not training_model.training


def test_serialization_excludes_training_model_and_preserves_predictions():
    classifier = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        preprocessing="none",
        n_dims=2,
        corr_select_k=0,
    )
    classifier.classes_ = np.array([0, 1])
    classifier.n_classes_ = 2
    classifier.n_outputs_ = 2
    classifier.preprocessors_ = [{"corr_selected_indices": None}]
    classifier.predictors_ = [_predictor()]
    classifier._model = torch.nn.Linear(1024, 1024, bias=False)
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    predictions_before = classifier.predict_proba(X)

    serialized = pickle.dumps(classifier)
    restored = pickle.loads(serialized)
    predictions_after = restored.predict_proba(X)

    assert classifier._model is not None
    assert restored._model is None
    assert len(serialized) < 1_000_000
    np.testing.assert_allclose(predictions_after, predictions_before)
