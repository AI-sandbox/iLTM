"""
Hyperparameter search space for iLTM.

This module exposes a description of the recommended search
space and a helper to draw random configurations from it.

High level API
--------------

- get_hyperparameter_search_space(available_checkpoints=None)

    Returns a dictionary that describes the distribution of every hyperparameter
    that iLTM recommends tuning. The returned structure is deliberately simple
    so it can be re-expressed with any hyperparameter optimization library or
    custom search procedure.

- sample_hyperparameters(rng, available_checkpoints=None)

    Draws a single random configuration from the same space using NumPy.
    This is intended for quick baselines, smoke tests, or simple schedulers
    that accept externally sampled configurations. For more advanced or
    adaptive search strategies you will usually want your tuning framework
    to sample configurations directly using `get_hyperparameter_search_space`.

- get_hyperparameter_configs()

    Returns the recommended fixed portfolio of configurations.

Notes
-----

The selected checkpoint automatically determines several parameters via
`model_checkpoints.py`: `tree_embedding`, `tree_model`,
`concat_tree_with_orig_features`, and `preprocessing`. Tree embedding
parameters in this search space are only used when the selected checkpoint
enables `tree_embedding`.

Device is fixed to `"cuda:0"` because CPU execution is very slow and not
recommended for typical workloads.
"""

from typing import Any

import numpy as np

from .hyperparameter_configs import get_hyperparameter_configs
from .model_checkpoints import AVAILABLE_CHECKPOINTS, restrict_checkpoint_choices
from .utils import sample_from_spec

HyperparamSpec = dict[str, Any]
SearchSpace = dict[str, HyperparamSpec]


def get_hyperparameter_search_space(
    available_checkpoints: list[str] | None = None,
) -> SearchSpace:
    """
    Return the canonical hyperparameter search space for iLTM.

    Parameters
    ----------
    available_checkpoints : list[str] | None, optional
        List of checkpoint names or paths to choose from. If None, uses the
        recommended checkpoint choices.

    Returns
    -------
    SearchSpace
        A dictionary mapping hyperparameter names to small spec dictionaries.
        Each spec describes the recommended distribution for that parameter.

    The specification format is intentionally minimal so that it can be
    re-expressed in any hyperparameter optimization library (e.g.,
    Optuna, Hyperopt, etc.) or custom search procedure.
    """
    space: SearchSpace = {
        "checkpoint": {"type": "categorical", "choices": ["cbrconcat", "xgbrconcat", "rtr"], "probs": [0.7, 0.2, 0.1]},
        "checkpoint_mix": {"type": "categorical", "choices": [None, "default"], "probs": [0.4, 0.6]},
        "device": {"type": "constant", "value": "cuda:0"},
        "n_ensemble": {"type": "categorical", "choices": [8, 12]},
        "batch_size": {"type": "categorical", "choices": [2048, 4096]},
        "finetuning": {"type": "constant", "value": True},
        "finetuning_dropout": {"type": "categorical", "choices": [0.0, 0.15], "probs": [0.6, 0.4]},
        "finetuning_max_steps": {"type": "constant", "value": 2048},
        "finetuning_batch_size": {"type": "categorical", "choices": [1024, 2048]},
        "finetuning_data": {"type": "constant", "value": "entire_dataset"},
        "finetuning_lr_adamw": {"type": "log_uniform", "low": 1.7e-4, "high": 5.5e-4},
        "finetuning_lr_lion": {"type": "log_uniform", "low": 2e-4, "high": 1.5e-3},
        "gradient_clip_norm": {"type": "float_uniform", "low": 0.5, "high": 1.5},
        "finetuning_optimizer": {"type": "categorical", "choices": ["adamw", "lion"], "probs": [0.4, 0.6]},
        "max_train_batches_per_epoch": {"type": "constant", "value": 128},
        "finetuning_subset_frac": {"type": "constant", "value": None},
        "finetuning_subset_max_samples": {"type": "constant", "value": 100_000},
        "val_max_samples": {"type": "constant", "value": 25_000},
        "tree_data_split": {"type": "constant", "value": "all"},
        "tree_for_each_predictor": {"type": "constant", "value": True},
        "tree_n_estimators": {
            "type": "categorical", "choices": [100, 125, 150, 200], "probs": [0.1, 0.25, 0.15, 0.5],
        },
        "tree_lr": {"type": "float_uniform", "low": 5e-2, "high": 0.7},
        "tree_max_depth": {"type": "categorical", "choices": [4, 5], "probs": [0.3, 0.7]},
        "tree_min_samples_leaf": {"type": "categorical", "choices": [1, 4, 12], "probs": [0.10, 0.05, 0.85]},
        "tree_l2_leaf_reg": {"type": "categorical", "choices": [0.75, 1.0, 1.25, 2.0, 2.5, 3.0]},
        "tree_subsample": {"type": "categorical", "choices": [0.56, 0.63, 0.71], "probs": [0.10, 0.85, 0.05]},
        "tree_feature_fraction": {"type": "categorical", "choices": [0.65, 0.81, 0.99], "probs": [0.05, 0.10, 0.85]},
        "tree_gamma": {"type": "categorical", "choices": [0.0, 0.5], "probs": [0.90, 0.10]},
        "tree_bagging_temperature": {"type": "constant", "value": None},
        "do_retrieval": {"type": "constant", "value": True},
        "retrieval_alpha": {"type": "constant", "value": 0.75},
        "retrieval_temperature": {"type": "float_uniform", "low": 1.0, "high": 2.0},
        "retrieval_distance": {"type": "constant", "value": "cosine"},
        "retrieval_alpha_finetuning": {"type": "constant", "value": False},
        "retrieval_alpha_adaptive": {"type": "constant", "value": True},
        "retrieval_temperature_finetuning": {"type": "constant", "value": False},
        "clip_data_value": {"type": "constant", "value": 1_000_000},
        "rf_size": {"type": "constant", "value": 32_768},
        "pca_sampling": {"type": "constant", "value": "zeropad"},
        "scheduler_min_lr": {"type": "log_uniform", "low": 1e-7, "high": 2e-6},
        "clip_predictions": {"type": "categorical", "choices": [False, True], "probs": [0.3, 0.7]},
        "corr_select_k": {"type": "categorical", "choices": [0, 300, 512, 1024, 2048], "probs": [0.60, 0.05, 0.10, 0.15, 0.10]},
    }

    return restrict_checkpoint_choices(space, available_checkpoints)


def sample_hyperparameters(
    rng: np.random.Generator,
    available_checkpoints: list[str] | None = None,
) -> dict[str, Any]:
    """Sample a single random configuration from the recommended space."""
    space = get_hyperparameter_search_space(available_checkpoints)
    return {name: sample_from_spec(rng, spec) for name, spec in space.items()}


__all__ = [
    "AVAILABLE_CHECKPOINTS",
    "HyperparamSpec",
    "SearchSpace",
    "get_hyperparameter_search_space",
    "get_hyperparameter_configs",
    "sample_hyperparameters",
]
