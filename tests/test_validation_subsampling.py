from __future__ import annotations

import numpy as np
import pandas as pd

from iltm import iLTMClassifier, iLTMRegressor


def test_external_validation_control_preserves_full_data():
    X = pd.DataFrame({"value": np.arange(20)})
    y = pd.Series(np.tile([0, 1], 10))
    model = iLTMClassifier(
        checkpoint=None,
        val_max_samples=8,
        val_subsample_before_preprocessing=False,
    )

    X_subset, y_subset = model._subsample_external_validation(X, y)

    assert X_subset is X
    assert y_subset is y


def test_external_validation_subsample_is_deterministic_and_stratified():
    X = pd.DataFrame({"value": np.arange(20)})
    y = pd.Series(np.tile([0, 1], 10))
    model = iLTMClassifier(
        checkpoint=None,
        seed=7,
        val_max_samples=8,
        val_subsample_before_preprocessing=True,
    )

    first_X, first_y = model._subsample_external_validation(X, y)
    second_X, second_y = model._subsample_external_validation(X, y)

    assert first_X.index.tolist() == second_X.index.tolist()
    assert first_y.tolist() == second_y.tolist()
    assert first_y.value_counts().to_dict() == {0: 4, 1: 4}


def test_external_validation_subsample_stratifies_without_validating_remainder():
    X = np.arange(25_003).reshape(-1, 1)
    y = np.arange(25_003) % 10
    model = iLTMClassifier(
        checkpoint=None,
        seed=999,
        val_max_samples=25_000,
        val_subsample_before_preprocessing=True,
    )

    first_X, first_y = model._subsample_external_validation(X, y)
    second_X, second_y = model._subsample_external_validation(X, y)

    assert first_X.shape == (25_000, 1)
    assert first_y.shape == (25_000,)
    np.testing.assert_array_equal(np.bincount(first_y), np.full(10, 2_500))
    np.testing.assert_array_equal(first_X, second_X)
    np.testing.assert_array_equal(first_y, second_y)


def test_external_validation_subsample_supports_numpy_regression():
    X = np.arange(60).reshape(20, 3)
    y = np.linspace(0.0, 1.0, 20)
    model = iLTMRegressor(
        checkpoint=None,
        seed=11,
        val_max_samples=6,
        val_subsample_before_preprocessing=True,
    )

    X_subset, y_subset = model._subsample_external_validation(X, y)

    assert X_subset.shape == (6, 3)
    assert y_subset.shape == (6,)
    np.testing.assert_array_equal(X_subset[:, 0] // 3, np.searchsorted(y, y_subset))
