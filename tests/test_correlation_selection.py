import numpy as np
import pytest

import iltm.inference_interface as inference_interface
from iltm import iLTMRegressor
from iltm.utils import select_top_correlated_features


@pytest.mark.parametrize(
    ("correlations", "expected"),
    [
        ([0.8, -0.9, 0.7], 1),
        ([0.8, 0.9, 0.7], 1),
        ([-0.8, -0.9, -0.7], 1),
    ],
)
def test_select_one_feature_uses_strongest_absolute_correlation(
    correlations,
    expected,
):
    selected = select_top_correlated_features(np.array(correlations), 1)

    np.testing.assert_array_equal(selected, np.array([expected]))


@pytest.mark.parametrize("num_features", [2, 3, 4, 5])
def test_selection_returns_requested_number_of_features(num_features):
    correlations = np.array([0.9, 0.8, -0.7, -0.6, 0.5, -0.4])

    selected = select_top_correlated_features(correlations, num_features)

    assert selected.size == num_features


@pytest.mark.parametrize("n_features", [100, 1_000, 10_000, 20_000])
def test_zero_corr_select_skips_correlations_at_or_below_implicit_cap(
    monkeypatch,
    n_features,
):
    def fail_if_called(X, y):
        raise AssertionError("correlations should not be computed")

    monkeypatch.setattr(
        inference_interface,
        "compute_feature_target_correlations",
        fail_if_called,
    )
    estimator = iLTMRegressor(
        checkpoint=None,
        device="cpu",
        preprocessing="none",
        corr_select_k=0,
        adaptive_memory=False,
    )
    X = np.zeros((4, n_features), dtype=np.float32)
    y = np.arange(4, dtype=np.float32)

    X_out, _, preprocessing = estimator._preprocess_fitting_data(
        X,
        y,
        is_classification=False,
    )

    assert X_out.shape == X.shape
    assert preprocessing["corr_selected_indices"] is None


@pytest.mark.parametrize("corr_select_k", [0, 30_000])
def test_corr_select_never_retains_more_than_implicit_cap(
    monkeypatch,
    corr_select_k,
):
    calls = []

    def record_correlations(X, y):
        calls.append(X.shape)
        return np.ones(X.shape[1], dtype=np.float64)

    monkeypatch.setattr(
        inference_interface,
        "compute_feature_target_correlations",
        record_correlations,
    )
    estimator = iLTMRegressor(
        checkpoint=None,
        device="cpu",
        preprocessing="none",
        corr_select_k=corr_select_k,
        adaptive_memory=False,
    )
    X = np.zeros((4, 20_001), dtype=np.float32)
    y = np.arange(4, dtype=np.float32)

    X_out, _, preprocessing = estimator._preprocess_fitting_data(
        X,
        y,
        is_classification=False,
    )

    assert calls == [X.shape]
    assert X_out.shape == (len(X), 20_000)
    assert preprocessing["corr_selected_indices"].shape == (20_000,)
