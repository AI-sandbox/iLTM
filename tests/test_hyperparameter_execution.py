"""Verify hyperparameter configurations route active settings to each predictor."""

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.base import clone

import iltm.inference_interface as interface
import iltm.model_checkpoints as checkpoints
from iltm import (
    AVAILABLE_CHECKPOINTS, get_hyperparameter_configs,
    iLTMClassifier, iLTMRegressor, sample_hyperparameters,
)
from iltm.iltm_model import iLTM


@pytest.fixture
def local_checkpoint_registry(monkeypatch, tmp_path):
    # Exercise the real registry while avoiding downloads of pretrained weights.
    # Existing paths also exercise the normal cloning path through the resolver.
    for checkpoint in AVAILABLE_CHECKPOINTS:
        (tmp_path / f"{checkpoint}.pth").touch()
    monkeypatch.setattr(checkpoints, "_get_default_ckpt_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        checkpoints, "_ensure_checkpoint",
        lambda repo_id, filename, ckpt_dir: str(tmp_path / filename),
    )


@pytest.mark.parametrize("estimator_class", [iLTMRegressor, iLTMClassifier])
@pytest.mark.parametrize("n_ensemble", [8, 12])
def test_named_mix_preserves_portfolio_and_clones(
    local_checkpoint_registry, estimator_class, n_ensemble,
):
    config = sample_hyperparameters(np.random.default_rng(17))
    config.update(device="cpu", checkpoint="rnobn", checkpoint_mix="default", n_ensemble=n_ensemble)
    estimator = estimator_class(**config)
    expected = next(
        item["checkpoint_mix"] for item in get_hyperparameter_configs()
        if item["n_ensemble"] == n_ensemble and "checkpoint_mix" in item
    )
    cloned = clone(estimator)
    assert cloned.get_params() == estimator.get_params()
    assert cloned.checkpoint_mix == "default"
    assert Path(cloned.checkpoint).stem == expected[0]
    actual = [Path(item["checkpoint"]).stem for item in cloned._checkpoint_mix_configs]
    assert actual[:n_ensemble] == expected


@pytest.mark.parametrize("mix,error", [
    ("unknown", ValueError), ([], ValueError), ((), ValueError), (True, TypeError),
])
def test_invalid_checkpoint_mix_is_rejected(mix, error):
    with pytest.raises(error, match="checkpoint_mix"):
        iLTMRegressor(checkpoint=None, checkpoint_mix=mix, device="cpu")


@pytest.mark.parametrize("estimator_class", [iLTMRegressor, iLTMClassifier])
@pytest.mark.parametrize("checkpoint", AVAILABLE_CHECKPOINTS)
def test_sampled_config_preserves_effective_feature_selection(
    local_checkpoint_registry, estimator_class, checkpoint,
):
    config = sample_hyperparameters(np.random.default_rng(17), [checkpoint])
    config.update(device="cpu", corr_select_k=10, corr_select_k_non_tree=5)
    estimator = estimator_class(**config)
    assert clone(estimator).get_params() == estimator.get_params()
    # Compare against the corresponding scalar feature-selection setting.
    reference = estimator_class(
        checkpoint=checkpoint, device="cpu",
        corr_select_k=10 if estimator.tree_embedding else 5,
    )
    estimator.set_params(preprocessing="none")
    reference.set_params(preprocessing="none")
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 20)).astype(np.float32)
    y = np.arange(30) % 2 if estimator_class is iLTMClassifier else rng.normal(size=30)
    actual = estimator._preprocess_fitting_data(
        X, y, is_classification=estimator_class is iLTMClassifier,
    )
    expected = reference._preprocess_fitting_data(
        X, y, is_classification=estimator_class is iLTMClassifier,
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(
        actual[2]["corr_selected_indices"], expected[2]["corr_selected_indices"],
    )


@pytest.mark.parametrize("tree_model", ["XGBoost_hist", "XGBoost_approx", "CatBoost", "GB", "RT"])
def test_xgboost_overrides_do_not_leak_to_other_tree_families(tree_model):
    estimator = iLTMRegressor(
        checkpoint=None, device="cpu", tree_model="CatBoost",
        xgb_min_samples_leaf=2, xgb_subsample=0.8, xgb_feature_fraction=0.7,
    )
    # Use the predictor's family, rather than the primary checkpoint's family.
    tree = estimator._create_tree_embedding(tree_model=tree_model, seed=3)
    actual = (tree.min_samples_leaf, tree.subsample, tree.feature_fraction)
    if tree_model.startswith("XGBoost"):
        assert actual == (2, 0.8, 0.7)
    else:
        assert actual == (12, 0.63, 0.99)


@pytest.mark.parametrize("estimator_class", [iLTMRegressor, iLTMClassifier])
@pytest.mark.parametrize("checkpoint", ["xgbrconcat", "cbrconcat"])
def test_sampled_tree_config_produces_identical_embeddings(
    local_checkpoint_registry, estimator_class, checkpoint,
):
    config = sample_hyperparameters(np.random.default_rng(41), [checkpoint])
    config.update(device="cpu", tree_n_estimators=4, tree_max_depth=2, tree_select_best_model=False)
    sampled = estimator_class(**config)
    # Generic settings should match the equivalent optional overrides. Inactive
    # settings should not affect the embedding produced by either tree family.
    legacy = dict(config)
    if checkpoint == "xgbrconcat":
        for suffix in ("min_samples_leaf", "subsample", "feature_fraction"):
            legacy[f"xgb_{suffix}"] = legacy.pop(f"tree_{suffix}")
        legacy.pop("tree_bagging_temperature")
    else:
        legacy.pop("tree_gamma")
    reference = estimator_class(**legacy)
    rng = np.random.default_rng(12)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    y = np.arange(40) % 2 if estimator_class is iLTMClassifier else X[:, 0] + X[:, 1]
    results = []
    for estimator in (sampled, reference):
        tree = estimator._create_tree_embedding(tree_model=estimator.tree_model, seed=19)
        tree.fit_tree(X, y)
        results.append(tree.transform(X))
    np.testing.assert_array_equal(*results)


def test_overrides_follow_set_params_and_preserve_generic_parameters():
    estimator = iLTMRegressor(
        checkpoint=None, device="cpu", tree_min_samples_leaf=7, tree_subsample=0.9,
        tree_feature_fraction=0.85, corr_select_k=9,
    )
    estimator.set_params(xgb_min_samples_leaf=2, xgb_subsample=0.6, xgb_feature_fraction=0.7)
    copied = clone(estimator)
    assert copied.get_params() == estimator.get_params()
    xgb = copied._create_tree_embedding(tree_model="XGBoost_hist", seed=1)
    catboost = copied._create_tree_embedding(tree_model="CatBoost", seed=1)
    assert (xgb.min_samples_leaf, xgb.subsample, xgb.feature_fraction) == (2, 0.6, 0.7)
    assert (catboost.min_samples_leaf, catboost.subsample, catboost.feature_fraction) == (7, 0.9, 0.85)
    copied.set_params(xgb_min_samples_leaf=None, xgb_subsample=None, xgb_feature_fraction=None)
    tree = copied._create_tree_embedding(tree_model="XGBoost_hist", seed=1)
    assert (tree.min_samples_leaf, tree.subsample, tree.feature_fraction) == (7, 0.9, 0.85)


@pytest.mark.parametrize("uses_tree", [False, True])
def test_feature_selection_uses_the_mixed_predictors_tree_setting(uses_tree):
    estimator = iLTMRegressor(
        checkpoint=None, device="cpu", preprocessing="none", tree_embedding=not uses_tree,
        corr_select_k=4, corr_select_k_non_tree=2,
    )
    X = np.random.default_rng(5).normal(size=(20, 6))
    actual, _, _ = estimator._preprocess_fitting_data(
        X, np.arange(20), is_classification=False, tree_embedding=uses_tree,
    )
    assert actual.shape[1] == (4 if uses_tree else 2)


@pytest.mark.parametrize("estimator_class", [iLTMRegressor, iLTMClassifier])
def test_disabled_retrieval_ignores_conflicting_learning_flags(estimator_class):
    estimator = estimator_class(
        checkpoint=None, device="cpu", do_retrieval=False,
        retrieval_alpha=0.3, retrieval_temperature=1.8, retrieval_distance="euclidean",
        retrieval_alpha_adaptive=True, retrieval_alpha_finetuning=True,
        retrieval_temperature_finetuning=True,
    )
    assert estimator.do_retrieval is False
    assert clone(estimator).get_params() == estimator.get_params()


@pytest.mark.parametrize("checkpoint", ["rtr", "rtrcb"])
def test_retrieval_checkpoints_still_force_retrieval(local_checkpoint_registry, checkpoint):
    config = sample_hyperparameters(np.random.default_rng(9), [checkpoint])
    config.update(device="cpu", do_retrieval=False)
    assert iLTMRegressor(**config).do_retrieval is True


@pytest.mark.parametrize("estimator_class", [iLTMRegressor, iLTMClassifier])
def test_optimizer_rates_follow_set_params_and_clone(estimator_class):
    estimator = estimator_class(
        checkpoint=None, device="cpu", finetuning_optimizer="AdamW",
        finetuning_lr=8e-4, finetuning_lr_adamw=3e-4, finetuning_lr_lion=1e-3,
    )
    copied = clone(estimator)
    assert copied.get_params() == estimator.get_params()
    assert copied._resolve_finetuning_lr() == 3e-4
    copied.set_params(finetuning_optimizer="LION")
    assert copied._resolve_finetuning_lr() == 1e-3
    copied.set_params(finetuning_lr_lion=None)
    assert copied._resolve_finetuning_lr() == 8e-4
    copied.set_params(finetuning_optimizer="sgd")
    assert copied._resolve_finetuning_lr() == 8e-4
    assert copied.finetuning_lr == estimator.finetuning_lr == 8e-4


@pytest.mark.parametrize("optimizer", ["adamw", "lion"])
def test_scheduler_validation_uses_only_active_rate(optimizer):
    other = "lion" if optimizer == "adamw" else "adamw"
    parameters = {
        "checkpoint": None, "device": "cpu", "finetuning_optimizer": optimizer,
        "finetuning_lr": 1e-8, "scheduler_min_lr": 1e-5,
        f"finetuning_lr_{optimizer}": 2e-4,
        f"finetuning_lr_{other}": 1e-9,
    }
    estimator = iLTMRegressor(**parameters)
    assert estimator._resolve_finetuning_lr() == 2e-4
    parameters[f"finetuning_lr_{optimizer}"] = 1e-6
    with pytest.raises(ValueError, match="effective finetuning_lr"):
        iLTMRegressor(**parameters)
    estimator.set_params(**{f"finetuning_lr_{optimizer}": 1e-6})
    # Validate again at use time after set_params, before accessing a model.
    with pytest.raises(ValueError, match="effective finetuning_lr"):
        estimator._generate_predictor(torch.ones(4, 2), torch.ones(4), n_outputs=1)


def test_fixed_portfolio_keeps_its_learning_rates(local_checkpoint_registry):
    for config in get_hyperparameter_configs():
        estimator = iLTMRegressor(**{**config, "device": "cpu"})
        assert estimator._resolve_finetuning_lr() == config["finetuning_lr"]


@pytest.mark.parametrize("estimator_class", [iLTMRegressor, iLTMClassifier])
@pytest.mark.parametrize("optimizer", ["adamw", "lion"])
def test_sampled_config_fits_with_active_optimizer_rate(
    monkeypatch, tmp_path, estimator_class, optimizer,
):
    # Use a small real checkpoint to exercise training and prediction offline.
    architecture = dict(n_dims=8, hn_n_layers=3, hn_hidden_size=16, rf_size=32,
                        n_classes_limit=5, main_n_layers=3)
    torch.manual_seed(73)
    path = tmp_path / "small.pth"
    torch.save(iLTM(**architecture).state_dict(), path)
    config = sample_hyperparameters(np.random.default_rng(7))
    config.update(
        architecture, checkpoint=str(path), checkpoint_mix=None, device="cpu",
        n_ensemble=1, batch_size=16, finetuning_max_steps=1, finetuning_batch_size=16,
        finetuning_optimizer=optimizer, tree_embedding=False, do_retrieval=False,
        preprocessing="none", corr_select_k=0, adaptive_memory=False,
        auto_stop_on_low_cpu_memory=False,
    )
    received_rates = []
    fine_tune = interface.fine_tune_main_network

    def record_rate(*args, **kwargs):
        received_rates.append(kwargs["finetuning_lr"])
        return fine_tune(*args, **kwargs)

    monkeypatch.setattr(interface, "fine_tune_main_network", record_rate)
    estimator = estimator_class(**config)
    estimator._model_cache = {}
    X = np.random.default_rng(11).normal(size=(40, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) if estimator_class is iLTMClassifier else X[:, 0]
    estimator.fit(X, y)
    assert received_rates == [config[f"finetuning_lr_{optimizer}"]]
    prediction = estimator.predict(X[:5])
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()
    if estimator_class is iLTMClassifier:
        probabilities = estimator.predict_proba(X[:5])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
