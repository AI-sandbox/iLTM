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

from typing import Dict, Any
import numpy as np


AVAILABLE_CHECKPOINTS = [
    "xgbrconcat",
    "cbrconcat",
    "r128bn",
    "rnobn",
    "xgb",
    "catb",
    "rtr",
    "rtrcb",
]

NON_TREE_CHECKPOINTS = ("r128bn", "rnobn", "rtr")
XGBOOST_CHECKPOINTS = ("xgb", "xgbrconcat")
CATBOOST_CHECKPOINTS = ("catb", "cbrconcat", "rtrcb")
TREE_CHECKPOINTS = XGBOOST_CHECKPOINTS + CATBOOST_CHECKPOINTS
FORCED_RETRIEVAL_CHECKPOINTS = ("rtr", "rtrcb")

COMMON_PARAMETER_NAMES = (
    "device",
    "n_ensemble",
    "batch_size",
    "finetuning",
    "finetuning_dropout",
    "finetuning_max_steps",
    "finetuning_batch_size",
    "finetuning_data",
    "finetuning_lr",
    "gradient_clip_norm",
    "finetuning_optimizer",
    "max_train_batches_per_epoch",
    "finetuning_subset_frac",
    "finetuning_subset_max_samples",
    "val_max_samples",
    "clip_data_value",
    "rf_size",
    "pca_sampling",
    "scheduler_min_lr",
    "clip_predictions",
    "corr_select_k",
    "retrieval_alpha_finetuning",
    "retrieval_temperature_finetuning",
)

COMMON_TREE_PARAMETER_NAMES = (
    "tree_data_split",
    "tree_for_each_predictor",
    "tree_n_estimators",
    "tree_lr",
    "tree_max_depth",
    "tree_min_samples_leaf",
    "tree_l2_leaf_reg",
)

XGBOOST_PARAMETER_NAMES = (
    "tree_subsample",
    "tree_feature_fraction",
    "tree_gamma",
)

CATBOOST_PARAMETER_NAMES = ("tree_bagging_temperature",)

RETRIEVAL_PARAMETER_NAMES = (
    "retrieval_alpha",
    "retrieval_temperature",
    "retrieval_distance",
)


# A single hyperparameter specification and the full search space.
HyperparamSpec = Dict[str, Any]
SearchSpace = Dict[str, HyperparamSpec]


def _rand_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """
    Sample from a log-uniform distribution on [low, high].

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random number generator.
    low : float
        Lower bound (inclusive).
    high : float
        Upper bound (inclusive).

    Returns
    -------
    float
        A value sampled from a log-uniform distribution between low and high.
    """
    log_low = np.log(low)
    log_high = np.log(high)
    return float(np.exp(rng.uniform(log_low, log_high)))


def _checkpoint_family(checkpoint: str | None) -> str | None:
    if checkpoint is None:
        return None
    checkpoint_name = str(checkpoint)
    if checkpoint_name.endswith(".pth"):
        checkpoint_name = checkpoint_name[:-4]
    for family in AVAILABLE_CHECKPOINTS:
        if checkpoint_name.endswith(family):
            return family
    return None


def _uses_non_tree_embedding_checkpoint(checkpoint: str | None) -> bool:
    return _checkpoint_family(checkpoint) in NON_TREE_CHECKPOINTS


def _sample_from_spec(
    rng: np.random.Generator,
    spec: HyperparamSpec,
    *,
    checkpoint: str | None = None,
) -> Any:
    """
    Sample a single value from a HyperparamSpec entry.

    The spec format is intentionally simple:

        {"type": "constant", "value": ...}
        {"type": "categorical", "choices": [...], "probs": [...]}   # probs optional
        {"type": "float_uniform", "low": float, "high": float}
        {"type": "log_uniform",   "low": float, "high": float}
    """
    kind = spec["type"]

    if kind == "constant":
        return spec["value"]

    if kind == "categorical":
        choices = spec["choices"]
        probs = spec.get("probs")
        if _uses_non_tree_embedding_checkpoint(checkpoint):
            choices = spec.get("non_tree_embedding_choices", choices)
            probs = spec.get("non_tree_embedding_probs", probs)
        value = rng.choice(choices, p=probs)
        # Ensure we return plain Python scalars rather than NumPy scalars
        if isinstance(value, np.generic):
            return value.item()
        return value

    if kind == "float_uniform":
        return float(rng.uniform(spec["low"], spec["high"]))

    if kind == "log_uniform":
        return _rand_log_uniform(rng, spec["low"], spec["high"])

    raise ValueError(f"Unknown hyperparameter type {kind!r}")


def get_hyperparameter_search_space(
    available_checkpoints: list[str] | None = None,
) -> SearchSpace:
    """
    Return the canonical hyperparameter search space for iLTM.

    Parameters
    ----------
    available_checkpoints : list[str] | None, optional
        List of checkpoint names to choose from. If None, uses all
        `AVAILABLE_CHECKPOINTS`.

    Returns
    -------
    SearchSpace
        A dictionary mapping hyperparameter names to small spec dictionaries.
        Each spec describes the recommended distribution for that parameter.

    The specification format is intentionally minimal so that it can be
    re-expressed in any hyperparameter optimization library (e.g., 
    Optuna, Hyperopt, etc.) or custom search procedure.

    A ``checkpoints`` field restricts a parameter to those checkpoint choices.
    A ``condition`` field restricts a parameter to configurations where the
    controlling parameter equals the stated value. A
    ``forced_true_checkpoints`` field identifies checkpoints that force the
    controlling Boolean parameter to true.
    """
    if available_checkpoints is None:
        available_checkpoints = list(AVAILABLE_CHECKPOINTS)
    else:
        available_checkpoints = list(available_checkpoints)

    if not available_checkpoints:
        raise ValueError("available_checkpoints must contain at least one checkpoint.")

    tree_checkpoints = [
        checkpoint
        for checkpoint in available_checkpoints
        if _checkpoint_family(checkpoint) in TREE_CHECKPOINTS
    ]
    xgboost_checkpoints = [
        checkpoint
        for checkpoint in available_checkpoints
        if _checkpoint_family(checkpoint) in XGBOOST_CHECKPOINTS
    ]
    catboost_checkpoints = [
        checkpoint
        for checkpoint in available_checkpoints
        if _checkpoint_family(checkpoint) in CATBOOST_CHECKPOINTS
    ]
    forced_retrieval_checkpoints = [
        checkpoint
        for checkpoint in available_checkpoints
        if _checkpoint_family(checkpoint) in FORCED_RETRIEVAL_CHECKPOINTS
    ]

    space: SearchSpace = {
        "checkpoint": {"type": "categorical", "choices": available_checkpoints},
        "device": {"type": "constant", "value": "cuda:0"},
        "n_ensemble": {"type": "categorical", "choices": [4, 8, 12, 16, 32]},
        "batch_size": {"type": "categorical", "choices": [2048, 4096]},
        "finetuning": {"type": "constant", "value": True},
        "finetuning_dropout": {"type": "categorical", "choices": [0.0, 0.15]},
        "finetuning_max_steps": {"type": "categorical", "choices": [2048, 4096]},
        "finetuning_batch_size": {"type": "categorical", "choices": [1024, 2048, 4096]},
        "finetuning_data": {"type": "constant", "value": "entire_dataset"},
        "finetuning_lr": {"type": "log_uniform", "low": 1e-4, "high": 3e-3},
        "gradient_clip_norm": {"type": "float_uniform", "low": 0.5, "high": 1.5},
        "finetuning_optimizer": {"type": "categorical", "choices": ["adamw", "lion"]},
        "max_train_batches_per_epoch": {"type": "constant", "value": 128},
        "finetuning_subset_frac": {"type": "constant", "value": None},
        "finetuning_subset_max_samples": {"type": "constant", "value": 100_000},
        "val_max_samples": {"type": "constant", "value": 25_000},
        "tree_data_split": {
            "type": "categorical",
            "choices": ["dynamic", "all"],
            "checkpoints": tree_checkpoints,
        },
        "tree_for_each_predictor": {
            "type": "constant",
            "value": True,
            "checkpoints": tree_checkpoints,
        },
        "tree_n_estimators": {
            "type": "categorical",
            "choices": [100, 125, 150, 200, 300],
            "checkpoints": tree_checkpoints,
        },
        "tree_lr": {
            "type": "log_uniform",
            "low": 1e-3,
            "high": 1.0,
            "checkpoints": tree_checkpoints,
        },
        "tree_max_depth": {
            "type": "categorical",
            "choices": [4, 5, 6],
            "probs": [0.20, 0.65, 0.15],
            "checkpoints": tree_checkpoints,
        },
        "tree_min_samples_leaf": {
            "type": "categorical",
            "choices": [1, 2, 4, 8, 12, 16, 90],
            "probs": [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10],
            "checkpoints": tree_checkpoints,
        },
        "tree_l2_leaf_reg": {
            "type": "categorical",
            "choices": [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0],
            "checkpoints": tree_checkpoints,
        },
        "tree_subsample": {
            "type": "float_uniform",
            "low": 0.5,
            "high": 1.0,
            "checkpoints": xgboost_checkpoints,
        },
        "tree_feature_fraction": {
            "type": "float_uniform",
            "low": 0.6,
            "high": 1.0,
            "checkpoints": xgboost_checkpoints,
        },
        "tree_gamma": {
            "type": "categorical",
            "choices": [0.0, 0.05, 0.1, 0.25, 0.5, 1.5],
            "probs": [0.54, 0.09, 0.09, 0.09, 0.09, 0.10],
            "checkpoints": xgboost_checkpoints,
        },
        "tree_bagging_temperature": {
            "type": "float_uniform",
            "low": 0.1,
            "high": 1.0,
            "checkpoints": catboost_checkpoints,
        },
        "do_retrieval": {
            "type": "categorical",
            "choices": [True, False],
            "probs": [0.65, 0.35],
            "forced_true_checkpoints": forced_retrieval_checkpoints,
        },
        "retrieval_alpha": {
            "type": "float_uniform",
            "low": 0.0,
            "high": 1.0,
            "condition": {"parameter": "do_retrieval", "value": True},
        },
        "retrieval_temperature": {
            "type": "float_uniform",
            "low": 1.0,
            "high": 2.5,
            "condition": {"parameter": "do_retrieval", "value": True},
        },
        "retrieval_distance": {
            "type": "categorical",
            "choices": ["cosine", "euclidean"],
            "condition": {"parameter": "do_retrieval", "value": True},
        },
        "retrieval_alpha_finetuning": {"type": "constant", "value": False},
        "retrieval_temperature_finetuning": {"type": "constant", "value": False},
        "clip_data_value": {"type": "constant", "value": 1_000_000},
        "rf_size": {"type": "constant", "value": 32_768},
        "pca_sampling": {"type": "constant", "value": "zeropad"},
        "scheduler_min_lr": {"type": "log_uniform", "low": 1e-7, "high": 3e-4},
        "clip_predictions": {
            "type": "categorical",
            "choices": [False, True],
            "probs": [0.3, 0.7],
        },
        "corr_select_k": {
            "type": "categorical",
            "choices": [0, 50, 100, 200, 300, 400, 512, 1024, 2048, 4096],
            "probs": [
                20 / 88,
                5 / 88,
                10 / 88,
                15 / 88,
                15 / 88,
                8 / 88,
                8 / 88,
                3 / 88,
                2 / 88,
                2 / 88,
            ],
            "non_tree_embedding_choices": [
                0,
                5,
                10,
                50,
                100,
                200,
                300,
                400,
                512,
                1024,
                2048,
                4096,
            ],
            "non_tree_embedding_probs": [
                20 / 93,
                2 / 93,
                3 / 93,
                5 / 93,
                10 / 93,
                15 / 93,
                15 / 93,
                8 / 93,
                8 / 93,
                3 / 93,
                2 / 93,
                2 / 93,
            ],
        },
    }

    return space


def _sample_hyperparameters_for_checkpoint(
    rng: np.random.Generator,
    space: SearchSpace,
    checkpoint: str,
) -> Dict[str, Any]:
    checkpoint_family = _checkpoint_family(checkpoint)
    cfg: Dict[str, Any] = {"checkpoint": checkpoint}

    for name in COMMON_PARAMETER_NAMES:
        cfg[name] = _sample_from_spec(
            rng,
            space[name],
            checkpoint=checkpoint,
        )

    if checkpoint_family in TREE_CHECKPOINTS:
        for name in COMMON_TREE_PARAMETER_NAMES:
            cfg[name] = _sample_from_spec(rng, space[name], checkpoint=checkpoint)

        if checkpoint_family in XGBOOST_CHECKPOINTS:
            for name in XGBOOST_PARAMETER_NAMES:
                cfg[name] = _sample_from_spec(rng, space[name], checkpoint=checkpoint)
        elif checkpoint_family in CATBOOST_CHECKPOINTS:
            for name in CATBOOST_PARAMETER_NAMES:
                cfg[name] = _sample_from_spec(rng, space[name], checkpoint=checkpoint)

    if checkpoint_family in FORCED_RETRIEVAL_CHECKPOINTS:
        cfg["do_retrieval"] = True
    else:
        cfg["do_retrieval"] = _sample_from_spec(
            rng,
            space["do_retrieval"],
            checkpoint=checkpoint,
        )

    if cfg["do_retrieval"]:
        for name in RETRIEVAL_PARAMETER_NAMES:
            cfg[name] = _sample_from_spec(rng, space[name], checkpoint=checkpoint)

    return cfg


def sample_hyperparameters(
    rng: np.random.Generator,
    available_checkpoints: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Sample a single random configuration from the recommended space.
    """
    space = get_hyperparameter_search_space(available_checkpoints)
    checkpoint = _sample_from_spec(rng, space["checkpoint"])
    return _sample_hyperparameters_for_checkpoint(rng, space, checkpoint)


__all__ = [
    "AVAILABLE_CHECKPOINTS",
    "HyperparamSpec",
    "SearchSpace",
    "get_hyperparameter_search_space",
    "sample_hyperparameters",
]
