"""
Tests for the hyperparameter search space module.

These tests focus on:

- Structural correctness of the search space specification.
- Correct behavior of `sample_hyperparameters`.
- Basic integration that sampled configs can be used with iLTM models.
"""

import inspect
import json

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score

from iltm import (
    get_hyperparameter_configs,
    get_hyperparameter_search_space,
    sample_hyperparameters,
    iLTMRegressor,
    iLTMClassifier,
    AVAILABLE_CHECKPOINTS,
)
from iltm.model_checkpoints import resolve_checkpoint_mix


VALID_TYPES = {"constant", "categorical", "float_uniform", "log_uniform"}
COMMON_TREE_PARAMETERS = {
    "tree_data_split",
    "tree_for_each_predictor",
    "tree_n_estimators",
    "tree_lr",
    "tree_max_depth",
    "tree_l2_leaf_reg",
    "tree_min_samples_leaf",
    "tree_subsample",
    "tree_feature_fraction",
}
XGBOOST_PARAMETERS = {
    "tree_gamma",
}
CATBOOST_PARAMETERS = {"tree_bagging_temperature"}
RETRIEVAL_PARAMETERS = {
    "retrieval_alpha",
    "retrieval_temperature",
    "retrieval_distance",
}


class TestHyperparameterConfigs:
    def test_portfolio_is_complete_name_free_and_unique(self):
        configs = get_hyperparameter_configs()

        assert len(configs) == 25
        assert all("name" not in config for config in configs)
        assert len({json.dumps(config, sort_keys=True) for config in configs}) == 25

    def test_portfolio_pairs_unmixed_configs_with_checkpoint_mix_twins(self):
        configs = get_hyperparameter_configs()

        for single, mixed in zip(configs[:6], configs[6:12], strict=True):
            expected = dict(single)
            expected["checkpoint_mix"] = mixed["checkpoint_mix"]
            assert mixed == expected
            assert len(mixed["checkpoint_mix"]) == mixed["n_ensemble"]

        assert sum("checkpoint_mix" in config for config in configs) == 15

    def test_portfolio_configs_are_valid_estimator_parameters(self):
        valid_parameters = set(inspect.signature(iLTMRegressor).parameters)

        for config in get_hyperparameter_configs():
            assert set(config) <= valid_parameters
            assert config["scheduler_min_lr"] <= config["finetuning_lr"]

    def test_portfolio_returns_defensive_copies(self):
        first = get_hyperparameter_configs()
        first[0]["checkpoint"] = "changed"
        first[6]["checkpoint_mix"].append("changed")

        second = get_hyperparameter_configs()
        assert second[0]["checkpoint"] == "xgbrconcat"
        assert len(second[6]["checkpoint_mix"]) == second[6]["n_ensemble"]


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
            assert set(spec) <= {"type", "value", "choices", "probs", "low", "high"}
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
        assert "probs" not in space["checkpoint"]

    def test_default_checkpoint_weights_follow_unmixed_portfolio(self):
        spec = get_hyperparameter_search_space()["checkpoint"]
        probabilities = dict(zip(spec["choices"], spec["probs"]))

        assert probabilities["cbrconcat"] == pytest.approx(0.70)
        assert probabilities["xgbrconcat"] == pytest.approx(0.20)
        assert probabilities["rtr"] == pytest.approx(0.10)
        assert set(probabilities) == {"cbrconcat", "xgbrconcat", "rtr"}
        assert sum(probabilities.values()) == pytest.approx(1.0)

    def test_regression_prediction_clipping_favors_enabled(self):
        space = get_hyperparameter_search_space()

        assert space["clip_predictions"] == {
            "type": "categorical",
            "choices": [False, True],
            "probs": [0.3, 0.7],
        }

    def test_time_aware_ranges_and_controls(self):
        space = get_hyperparameter_search_space()

        assert space["n_ensemble"] == {"type": "categorical", "choices": [8, 12]}
        assert space["tree_lr"]["type"] == "float_uniform"
        assert space["finetuning_max_steps"] == {"type": "constant", "value": 2048}
        assert space["finetuning_batch_size"]["choices"] == [1024, 2048]
        assert space["max_train_batches_per_epoch"]["value"] == 128
        assert space["finetuning_subset_frac"]["value"] is None
        assert space["finetuning_subset_max_samples"]["value"] == 100_000
        assert space["val_max_samples"]["value"] == 25_000

    def test_discrete_tree_and_feature_choices_match_portfolio_support(self):
        space = get_hyperparameter_search_space()
        defaults = inspect.signature(iLTMRegressor).parameters
        for name in (
            "tree_n_estimators", "tree_max_depth", "tree_min_samples_leaf",
            "tree_l2_leaf_reg", "tree_subsample", "tree_feature_fraction",
            "tree_gamma", "corr_select_k",
        ):
            portfolio_values = {
                config.get(name, defaults[name].default)
                for config in get_hyperparameter_configs()
            }
            if name in {"tree_subsample", "tree_feature_fraction"}:
                portfolio_values = {round(value, 2) for value in portfolio_values}
            assert set(space[name]["choices"]) == portfolio_values
        assert "corr_select_k_non_tree" not in space
        assert not any(name.startswith("xgb_") for name in space)

    def test_optimizer_rate_ranges_cover_the_portfolio(self):
        space = get_hyperparameter_search_space()
        assert "finetuning_lr" not in space
        for config in get_hyperparameter_configs():
            spec = space[f"finetuning_lr_{config['finetuning_optimizer']}"]
            assert spec["type"] == "log_uniform"
            assert spec["low"] <= config["finetuning_lr"] <= spec["high"]

    def test_mix_preserves_each_portfolio_recipe(self):
        spec = get_hyperparameter_search_space()["checkpoint_mix"]
        assert spec == {"type": "categorical", "choices": [None, "default"], "probs": [0.4, 0.6]}
        recipe = resolve_checkpoint_mix("default")
        for config in get_hyperparameter_configs():
            if "checkpoint_mix" in config:
                assert list(recipe[:config["n_ensemble"]]) == config["checkpoint_mix"]

    @pytest.mark.parametrize("checkpoints", [["cbrconcat"], ["cbrconcat", "xgbrconcat"], ["rtr"]])
    def test_incomplete_checkpoint_list_disables_mix(self, checkpoints):
        assert get_hyperparameter_search_space(checkpoints)["checkpoint_mix"] == {
            "type": "constant", "value": None,
        }

    def test_mix_uses_supplied_checkpoint_paths(self):
        paths = [f"/custom/checkpoints/{name}.pth" for name in ("cbrconcat", "xgbrconcat", "rtr")]
        spec = get_hyperparameter_search_space(paths)["checkpoint_mix"]
        assert spec["probs"] == [0.4, 0.6]
        assert set(spec["choices"][1]) == set(paths)
        expected = resolve_checkpoint_mix("default")
        assert spec["choices"][1] == tuple(f"/custom/checkpoints/{name}.pth" for name in expected)

    def test_empty_checkpoint_list_is_rejected(self):
        with pytest.raises(ValueError, match="at least one checkpoint"):
            get_hyperparameter_search_space([])

    @pytest.mark.parametrize("requested", [False, True])
    def test_classifier_ignores_regression_prediction_clipping(self, requested):
        classifier = iLTMClassifier(
            checkpoint=None,
            device="cpu",
            clip_predictions=requested,
        )

        assert classifier.clip_predictions is False
        assert classifier.get_params()["clip_predictions"] is False


class TestSampleHyperparameters:
    """Tests for the sampling helper that draws configs from the space."""

    def test_sample_returns_valid_config(self):
        """sample_hyperparameters produces a config dict aligned with the space."""
        space = get_hyperparameter_search_space()
        rng = np.random.default_rng(seed=0)
        cfg = sample_hyperparameters(rng)

        assert isinstance(cfg, dict)
        assert set(cfg) == set(space)

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

    @pytest.mark.parametrize("name", AVAILABLE_CHECKPOINTS)
    def test_sample_accepts_relocated_checkpoint_paths(self, name):
        checkpoint = f"/tmp/relocated/{name}.pth"
        config = sample_hyperparameters(
            np.random.default_rng(0),
            available_checkpoints=[checkpoint],
        )

        assert config["checkpoint"] == checkpoint
        assert set(config) == set(get_hyperparameter_search_space())
        named_config = sample_hyperparameters(np.random.default_rng(0), [name])
        config.pop("checkpoint")
        named_config.pop("checkpoint")
        assert config == named_config

    def test_other_settings_are_sampled_independently_of_checkpoint(self):
        # Changing the checkpoint does not change any of the other draws.
        for seed in range(20):
            tree = sample_hyperparameters(np.random.default_rng(seed), ["xgbrconcat"])
            non_tree = sample_hyperparameters(np.random.default_rng(seed), ["r128bn"])
            tree.pop("checkpoint")
            non_tree.pop("checkpoint")
            assert tree == non_tree

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

    def test_structured_choices_survive_serialization_and_are_copied(self):
        from iltm.utils import sample_from_spec

        paths = [f"/custom/{name}.pth" for name in ("cbrconcat", "xgbrconcat", "rtr")]
        space = json.loads(json.dumps(get_hyperparameter_search_space(paths)))
        spec = space["checkpoint_mix"]
        spec["probs"] = [0.0, 1.0]
        sampled = sample_from_spec(np.random.default_rng(0), spec)
        assert sampled == spec["choices"][1]
        sampled[0] = "changed"
        assert spec["choices"][1][0] == "/custom/cbrconcat.pth"
        assert resolve_checkpoint_mix("default")[0] == "cbrconcat"


class TestSearchSpaceSampling:
    def test_default_choices_are_plain_scalars(self):
        space = get_hyperparameter_search_space()
        assert json.loads(json.dumps(space)) == space
        for spec in space.values():
            if spec["type"] == "categorical":
                assert all(value is None or type(value) in (str, bool, int, float)
                           for value in spec["choices"])

    @pytest.mark.parametrize("name", [
        "n_ensemble", "batch_size", "finetuning_batch_size", "tree_l2_leaf_reg",
    ])
    def test_mild_preferences_use_uniform_choices(self, name):
        assert "probs" not in get_hyperparameter_search_space()[name]

    @pytest.mark.parametrize("name,probs", [
        ("checkpoint_mix", [0.4, 0.6]),
        ("finetuning_dropout", [0.6, 0.4]),
        ("finetuning_optimizer", [0.4, 0.6]),
        ("tree_n_estimators", [0.1, 0.25, 0.15, 0.5]),
        ("tree_max_depth", [0.3, 0.7]),
        ("clip_predictions", [0.3, 0.7]),
    ])
    def test_weighted_choices_use_rounded_probabilities(self, name, probs):
        assert get_hyperparameter_search_space()[name]["probs"] == probs

    @pytest.mark.parametrize("checkpoint", AVAILABLE_CHECKPOINTS)
    def test_every_checkpoint_samples_the_same_keys(self, checkpoint):
        config = sample_hyperparameters(np.random.default_rng(0), [checkpoint])
        assert set(config) == set(get_hyperparameter_search_space())
        assert COMMON_TREE_PARAMETERS | XGBOOST_PARAMETERS | CATBOOST_PARAMETERS <= set(config)
        assert RETRIEVAL_PARAMETERS <= set(config)
        assert config["do_retrieval"] is True

    def test_checkpoint_subset_only_changes_checkpoint_and_mix_distributions(self):
        space = get_hyperparameter_search_space()
        subset = get_hyperparameter_search_space(["xgb", "rtr"])
        space.pop("checkpoint")
        subset.pop("checkpoint")
        space.pop("checkpoint_mix")
        subset.pop("checkpoint_mix")
        assert space == subset

    def test_can_sample_with_a_generic_adapter(self):
        rng = np.random.default_rng(17)
        config = {}
        for name, spec in get_hyperparameter_search_space().items():
            if spec["type"] == "constant":
                config[name] = spec["value"]
            elif spec["type"] == "categorical":
                config[name] = spec["choices"][rng.choice(len(spec["choices"]), p=spec.get("probs"))]
            elif spec["type"] == "float_uniform":
                config[name] = rng.uniform(spec["low"], spec["high"])
            elif spec["type"] == "log_uniform":
                config[name] = np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"])))
        assert config == sample_hyperparameters(np.random.default_rng(17))
        for estimator_class in (iLTMRegressor, iLTMClassifier):
            assert set(config) <= set(inspect.signature(estimator_class).parameters)


class TestSampledConfigParameterRanges:
    """Test that sampled configurations fall inside the intended ranges."""

    def test_parameter_ranges_valid(self):
        """Sampled configs lie inside the ranges implied by the search space."""
        rng = np.random.default_rng(seed=42)

        for _ in range(10):
            config = sample_hyperparameters(rng)

            # Check ensemble size
            assert config["n_ensemble"] in [8, 12]

            # Check batch size
            assert config["batch_size"] in [2048, 4096]

            # Finetuning parameters
            assert config["finetuning"] is True
            assert config["finetuning_dropout"] in [0.0, 0.15]
            assert config["finetuning_max_steps"] == 2048
            assert config["finetuning_batch_size"] in [1024, 2048]
            assert 1.7e-4 <= config["finetuning_lr_adamw"] <= 5.5e-4
            assert 2e-4 <= config["finetuning_lr_lion"] <= 1.5e-3
            assert 0.5 <= config["gradient_clip_norm"] <= 1.5
            assert config["finetuning_optimizer"] in ["adamw", "lion"]
            assert config["max_train_batches_per_epoch"] == 128
            assert config["finetuning_subset_frac"] is None
            assert config["finetuning_subset_max_samples"] == 100_000
            assert config["val_max_samples"] == 25_000

            assert config["do_retrieval"] is True
            assert config["retrieval_alpha_finetuning"] is False
            assert config["retrieval_alpha_adaptive"] is True
            assert config["retrieval_temperature_finetuning"] is False
            assert config["retrieval_alpha"] == 0.75
            assert 1.0 <= config["retrieval_temperature"] <= 2.0
            assert config["retrieval_distance"] == "cosine"

            assert config["tree_data_split"] == "all"
            assert config["tree_for_each_predictor"] is True
            assert config["tree_n_estimators"] in [100, 125, 150, 200]
            assert 5e-2 <= config["tree_lr"] <= 0.7
            assert config["tree_max_depth"] in [4, 5]
            assert config["tree_l2_leaf_reg"] in [0.75, 1, 1.25, 2, 2.5, 3]
            assert config["tree_min_samples_leaf"] in [1, 4, 12]
            assert config["tree_subsample"] in [0.56, 0.63, 0.71]
            assert config["tree_feature_fraction"] in [0.65, 0.81, 0.99]
            assert config["tree_gamma"] in [0.0, 0.5]
            assert config["tree_bagging_temperature"] is None

            # Other parameters
            assert config["device"] == "cuda:0"
            assert config["pca_sampling"] == "zeropad"
            assert 1e-7 <= config["scheduler_min_lr"] <= 2e-6
            assert isinstance(config["clip_predictions"], bool)
            assert config["corr_select_k"] in [0, 300, 512, 1024, 2048]


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
