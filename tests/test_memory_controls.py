import logging

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification

from iltm import iLTMClassifier


def test_default_inference_storage_dtype_compacts_rf_after_fit_and_predict():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=7)

    classifier = iLTMClassifier(
        checkpoint="xgbrconcat",
        device="cpu",
        n_ensemble=1,
        batch_size=32,
        finetuning=False,
        rf_size=128,
    )

    classifier.fit(X, y)

    assert classifier.predictors_[0]["rf"][0].weight.dtype == torch.float16
    probabilities = classifier.predict_proba(X[:5])
    assert classifier.predictors_[0]["rf"][0].weight.dtype == torch.float16
    assert probabilities.shape == (5, 2)
    assert np.isfinite(probabilities).all()


def test_inference_storage_dtype_float32_keeps_rf_float32():
    classifier = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        inference_storage_dtype="float32",
    )
    predictor = {
        "rf": torch.nn.Sequential(torch.nn.Linear(8, 16, bias=False)),
        "pca": torch.nn.Identity(),
        "norm": None,
        "main_network": torch.nn.ModuleList([torch.nn.Linear(16, 2)]),
    }

    classifier._move_predictor_to_cpu(predictor)

    assert predictor["rf"][0].weight.dtype == torch.float32


def test_inference_storage_dtype_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported inference_storage_dtype"):
        iLTMClassifier(checkpoint=None, device="cpu", inference_storage_dtype="int8")


def test_cpu_memory_guard_triggers_before_next_predictor(monkeypatch):
    classifier = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        n_ensemble=4,
        auto_stop_on_low_cpu_memory=True,
        cpu_memory_safety_margin_gb=0.0,
    )
    classifier.predictors_ = [
        {"rf": torch.nn.Sequential(torch.nn.Linear(8, 128, bias=False))},
        {"rf": torch.nn.Sequential(torch.nn.Linear(8, 128, bias=False))},
    ]
    current_model_bytes = classifier._estimate_predictors_storage_bytes()
    avg_predictor_bytes = current_model_bytes / len(classifier.predictors_)

    monkeypatch.setattr(classifier, "_get_effective_cpu_memory_limit_bytes", lambda: int(64 * 1024**2))
    monkeypatch.setattr(
        classifier,
        "_get_cpu_memory_usage_bytes",
        lambda limit_bytes=None: int(64 * 1024**2 - avg_predictor_bytes),
    )

    should_stop, details = classifier._should_stop_for_cpu_memory_before_predictor(next_predictor_index=3)

    assert should_stop
    assert details is not None
    assert details["predictors_fit"] == 2
    assert details["next_predictor_index"] == 3
    assert details["projected_peak_gb"] > details["limit_gb"]


def test_cpu_memory_guard_allows_when_no_limit(monkeypatch):
    classifier = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        n_ensemble=4,
        auto_stop_on_low_cpu_memory=True,
    )
    classifier.predictors_ = [{"rf": torch.nn.Sequential(torch.nn.Linear(8, 128, bias=False))}]

    monkeypatch.setattr(classifier, "_get_effective_cpu_memory_limit_bytes", lambda: None)

    should_stop, details = classifier._should_stop_for_cpu_memory_before_predictor(next_predictor_index=2)

    assert not should_stop
    assert details is None


def test_cpu_memory_guard_reads_autogluon_memory_limit(monkeypatch):
    classifier = iLTMClassifier(checkpoint=None, device="cpu", n_ensemble=1)

    monkeypatch.setenv("AG_MEMORY_LIMIT_IN_GB", "12.5")
    monkeypatch.setattr(classifier, "_get_cgroup_memory_limit_bytes", lambda: None)
    monkeypatch.setattr(classifier, "_get_slurm_memory_limit_bytes", lambda: None)

    assert classifier._get_effective_cpu_memory_limit_bytes() == int(12.5 * 1024**3)


def test_cpu_memory_guard_explicit_limit_without_cgroup(monkeypatch):
    classifier = iLTMClassifier(checkpoint=None, device="cpu", n_ensemble=1, cpu_memory_limit_gb=4)

    monkeypatch.delenv("AG_MEMORY_LIMIT_IN_GB", raising=False)
    monkeypatch.setattr(classifier, "_get_cgroup_memory_limit_bytes", lambda: None)
    monkeypatch.setattr(classifier, "_get_slurm_memory_limit_bytes", lambda: None)
    monkeypatch.setattr(classifier, "_get_process_rss_bytes", lambda: 123)
    monkeypatch.setattr(classifier, "_get_cgroup_memory_usage_bytes", lambda: 456)

    limit_bytes = classifier._get_effective_cpu_memory_limit_bytes()

    assert limit_bytes == 4 * 1024**3
    assert classifier._get_cpu_memory_usage_bytes(limit_bytes=limit_bytes) == 123


def test_cpu_memory_guard_explicit_limit_with_larger_cgroup(monkeypatch):
    classifier = iLTMClassifier(checkpoint=None, device="cpu", n_ensemble=1, cpu_memory_limit_gb=4)

    monkeypatch.delenv("AG_MEMORY_LIMIT_IN_GB", raising=False)
    monkeypatch.setattr(classifier, "_get_cgroup_memory_limit_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(classifier, "_get_slurm_memory_limit_bytes", lambda: None)
    monkeypatch.setattr(classifier, "_get_process_rss_bytes", lambda: 123)
    monkeypatch.setattr(classifier, "_get_cgroup_memory_usage_bytes", lambda: 456)

    limit_bytes = classifier._get_effective_cpu_memory_limit_bytes()

    assert limit_bytes == 4 * 1024**3
    assert classifier._get_cpu_memory_usage_bytes(limit_bytes=limit_bytes) == 123


def test_cpu_memory_guard_explicit_limit_with_smaller_cgroup(monkeypatch):
    classifier = iLTMClassifier(checkpoint=None, device="cpu", n_ensemble=1, cpu_memory_limit_gb=64)

    monkeypatch.delenv("AG_MEMORY_LIMIT_IN_GB", raising=False)
    monkeypatch.setattr(classifier, "_get_cgroup_memory_limit_bytes", lambda: 4 * 1024**3)
    monkeypatch.setattr(classifier, "_get_slurm_memory_limit_bytes", lambda: None)
    monkeypatch.setattr(classifier, "_get_process_rss_bytes", lambda: 123)
    monkeypatch.setattr(classifier, "_get_cgroup_memory_usage_bytes", lambda: 456)

    limit_bytes = classifier._get_effective_cpu_memory_limit_bytes()

    assert limit_bytes == 4 * 1024**3
    assert classifier._get_cpu_memory_usage_bytes(limit_bytes=limit_bytes) == 456


def test_cpu_memory_guard_stops_fit_loop():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=11)

    classifier = iLTMClassifier(
        checkpoint="xgbrconcat",
        device="cpu",
        n_ensemble=3,
        batch_size=32,
        finetuning=False,
        rf_size=128,
        auto_stop_on_low_cpu_memory=True,
        cpu_memory_limit_gb=0.001,
        cpu_memory_safety_margin_gb=0.0,
        logging_level=logging.WARNING,
    )

    classifier.fit(X, y)

    assert len(classifier.predictors_) == 1
    probabilities = classifier.predict_proba(X[:5])
    assert probabilities.shape == (5, 2)
    assert np.isfinite(probabilities).all()
