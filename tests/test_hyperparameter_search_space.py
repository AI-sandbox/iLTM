"""
Tests for the hyperparameter search space module.

These tests focus on:

- Structural correctness of the search space specification.
- Correct behavior of `sample_hyperparameters`.
- Basic integration that sampled configs can be used with iLTM models.
"""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score

from iltm import (
    get_hyperparameter_search_space,
    sample_hyperparameters,
    iLTMRegressor,
    iLTMClassifier,
    AVAILABLE_CHECKPOINTS,
)


VALID_TYPES = {"constant", "categorical", "float_uniform", "log_uniform"}


class TestSearchSpaceDefinition:
    """Tests for the search space specification returned by get_hyperparameter_search_space."""

    def test_space_structure_and_types(self):
        """Search space is a dict of spec dicts with valid types and required fields."""
        space = get_hyperparameter_search_space()

        assert isinstance(space, dict)
        # Sanity check that some expected keys exist
        for required in ["checkpoint", "device", "n_ensemble", "finetuning"]:
            assert required in space

        for name, spec in space.items():
            assert isinstance(spec, dict), f"Spec for {name} is not a dict"
            assert "type" in spec, f"Spec for {name} missing 'type'"
            kind = spec["type"]
            assert kind in VALID_TYPES, f"Unknown type {kind!r} for {name}"

            if kind == "constant":
                assert "value" in spec, f"Constant spec for {name} missing 'value'"

            elif kind == "categorical":
                assert "choices" in spec, f"Categorical spec for {name} missing 'choices'"
                choices = spec["choices"]
                assert isinstance(choices, (list, tuple)) and len(choices) > 0
                if "probs" in spec:
                    probs = spec["probs"]
                    assert len(probs) == len(choices)
                    # Probabilities should be non negative and sum to 1 within tolerance
                    assert all(p >= 0 for p in probs)
                    assert pytest.approx(sum(probs), rel=1e-6) == 1.0

            elif kind in {"float_uniform", "log_uniform"}:
                assert "low" in spec and "high" in spec, f"{kind} spec for {name} missing bounds"
                assert spec["low"] <= spec["high"], f"low > high for {name}"

    def test_space_is_deterministic(self):
        """Calling get_hyperparameter_search_space twice returns identical structures."""
        s1 = get_hyperparameter_search_space()
        s2 = get_hyperparameter_search_space()
        assert s1 == s2

    def test_space_respects_available_checkpoints(self):
        """available_checkpoints argument is reflected in the checkpoint spec."""
        custom = ["xgbrconcat", "cbrconcat"]
        space = get_hyperparameter_search_space(available_checkpoints=custom)
        assert space["checkpoint"]["choices"] == custom


class TestSampleHyperparameters:
    """Tests for the sampling helper that draws configs from the space."""

    def test_sample_returns_valid_config(self):
        """sample_hyperparameters produces a config dict aligned with the space."""
        space = get_hyperparameter_search_space()
        rng = np.random.default_rng(seed=0)
        cfg = sample_hyperparameters(rng)

        assert isinstance(cfg, dict)
        # Keys in config and space should match exactly
        assert set(cfg.keys()) == set(space.keys())

        # Some basic sanity checks
        assert cfg["checkpoint"] in AVAILABLE_CHECKPOINTS
        assert cfg["device"] == "cuda:0"
        assert cfg["finetuning"] is True

    def test_sample_respects_available_checkpoints(self):
        """sample_hyperparameters respects the available_checkpoints argument."""
        rng = np.random.default_rng(seed=0)
        custom = ["xgbrconcat", "cbrconcat"]
        cfg = sample_hyperparameters(rng, available_checkpoints=custom)
        assert cfg["checkpoint"] in custom

    def test_sample_reproducible_with_seed(self):
        """Same seed produces identical configurations."""
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)
        cfg1 = sample_hyperparameters(rng1)
        cfg2 = sample_hyperparameters(rng2)
        assert cfg1 == cfg2

    def test_sample_differs_with_different_seeds(self):
        """Different seeds produce different configurations (with very high probability)."""
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=123)
        cfg1 = sample_hyperparameters(rng1)
        cfg2 = sample_hyperparameters(rng2)
        assert cfg1 != cfg2


class TestSampledConfigParameterRanges:
    """Test that sampled configurations fall inside the intended ranges."""

    def test_parameter_ranges_valid(self):
        """Sampled configs lie inside the ranges implied by the search space."""
        rng = np.random.default_rng(seed=42)

        for _ in range(10):
            config = sample_hyperparameters(rng)

            # Check ensemble size
            assert config["n_ensemble"] in [4, 8, 12, 16, 32, 64]

            # Check batch size
            assert config["batch_size"] in [2048, 4096]

            # Finetuning parameters
            assert config["finetuning"] is True
            assert config["finetuning_dropout"] in [0.0, 0.15]
            assert config["finetuning_max_steps"] in [2048, 4096]
            assert config["finetuning_batch_size"] in [64, 128, 256, 512, 1024, 2048, 4096]
            assert 1e-4 <= config["finetuning_lr"] <= 3e-3
            assert 0.5 <= config["gradient_clip_norm"] <= 1.5
            assert config["finetuning_optimizer"] in ["adamw", "lion"]

            # Tree parameters
            assert config["tree_data_split"] in ["dynamic", "all"]
            assert config["tree_for_each_predictor"] is True
            assert config["tree_n_estimators"] in [100, 125, 150, 200, 300]
            assert 1e-3 <= config["tree_lr"] <= 1.0
            assert config["tree_max_depth"] in [4, 5, 6]
            assert config["tree_min_samples_leaf"] in [1, 2, 4, 8, 12, 16]
            assert 0.5 <= config["tree_subsample"] <= 1.0
            assert 0.6 <= config["tree_feature_fraction"] <= 1.0
            assert config["tree_gamma"] in [0.0, 0.05, 0.1, 0.25, 0.5]
            assert config["tree_l2_leaf_reg"] in [0.1, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 5]
            assert 0.1 <= config["tree_bagging_temperature"] <= 1.0

            # Retrieval parameters
            assert isinstance(config["do_retrieval"], bool)
            assert 0.0 <= config["retrieval_alpha"] <= 1.0
            assert 1.0 <= config["retrieval_temperature"] <= 2.5
            assert config["retrieval_distance"] in ["cosine", "euclidean"]
            assert config["retrieval_alpha_finetuning"] is False
            assert config["retrieval_temperature_finetuning"] is False

            # Other parameters
            assert config["device"] == "cuda:0"
            assert config["pca_sampling"] == "zeropad"
            assert 1e-7 <= config["scheduler_min_lr"] <= 3e-4
            assert isinstance(config["clip_predictions"], bool)
            assert config["corr_select_k"] in [
                0,
                1,
                2,
                5,
                10,
                20,
                50,
                100,
                200,
                300,
                400,
                512,
                1024,
                2048,
                4096,
            ]


class TestSampledConfigWorksWithModels:
    """
    Basic integration tests: a sampled configuration can be used
    to construct and train iLTMRegressor and iLTMClassifier.
    """

    def test_config_works_with_regressor(self, tiny_regression_data):
        X, y = tiny_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
        )

        rng = np.random.default_rng(seed=42)
        config = sample_hyperparameters(rng)

        # Override with small values for fast testing
        config.update(
            {
                "n_ensemble": 2,
                "finetuning_max_steps": 10,
                "device": "cpu",  # Use CPU for tests
            }
        )

        reg = iLTMRegressor(**config)
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape[0] == X_test.shape[0]

        # Very light performance sanity check
        r2 = r2_score(y_test, y_pred)
        assert np.isfinite(r2)

    def test_config_works_with_classifier(self, tiny_classification_data):
        X, y = tiny_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
        )

        rng = np.random.default_rng(seed=42)
        config = sample_hyperparameters(rng)

        # Override with small values for fast testing
        config.update(
            {
                "n_ensemble": 2,
                "finetuning_max_steps": 10,
                "device": "cpu",  # Use CPU for tests
            }
        )

        clf = iLTMClassifier(**config)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
        # Assuming binary labels in tiny_classification_data
        assert set(np.unique(y_pred)).issubset({0, 1})

        y_proba = clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba[:, 1])
        assert auc > 0.5, f"AUC score {auc} should be > 0.5"
