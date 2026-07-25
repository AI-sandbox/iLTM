from __future__ import annotations

import warnings

import numpy as np
import pytest

from iltm.realmlp_td_s_preprocessing import RobustScaleSmoothClipTransform


def _reference_statistics(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(X, axis=0)
    quantiles = np.nanquantile(X, [0.25, 0.75], axis=0)
    if np.issubdtype(X.dtype, np.floating) and X.dtype.itemsize < 8:
        quantiles = quantiles.astype(X.dtype, copy=False)
    q25, q75 = quantiles
    iqr = q75 - q25
    width = 0.5 * (np.nanmax(X, axis=0) - np.nanmin(X, axis=0))
    denominator = np.where(np.isfinite(iqr) & (iqr != 0), iqr, width)
    median = np.where(np.isfinite(median), median, 0)
    factors = np.zeros_like(denominator, dtype=np.float64)
    good = np.isfinite(denominator) & (denominator != 0)
    factors[good] = 1 / (denominator[good] + 1e-30)
    return median.astype(np.float32), factors.astype(np.float32)


def _assert_matches_reference(X: np.ndarray) -> None:
    held_out = X[: min(len(X), 3)].copy()
    if np.issubdtype(X.dtype, np.floating):
        held_out = np.concatenate(
            [
                held_out,
                np.full((1, X.shape[1]), np.nan, dtype=X.dtype),
            ],
            axis=0,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected_median, expected_factors = _reference_statistics(X)
        transform = RobustScaleSmoothClipTransform().fit(X.copy())

    np.testing.assert_array_equal(transform._median, expected_median)
    np.testing.assert_array_equal(transform._factors, expected_factors)
    expected = RobustScaleSmoothClipTransform()
    expected._median = expected_median
    expected._factors = expected_factors
    np.testing.assert_array_equal(
        transform.transform(held_out),
        expected.transform(held_out),
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32])
@pytest.mark.parametrize("n_samples", [1, 2, 3, 4, 5, 7, 8, 9, 20])
def test_binary_statistics_match_reference_for_every_zero_count(
    dtype,
    n_samples,
):
    columns = []
    for zero_count in range(n_samples + 1):
        column = np.ones(n_samples)
        column[:zero_count] = 0
        columns.append(column)
    _assert_matches_reference(np.column_stack(columns).astype(dtype))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_binary_statistics_match_reference_with_missing_values(dtype):
    X = np.array(
        [
            [0, 0, 1, np.nan, np.nan, 0],
            [1, 0, 1, np.nan, 0, np.nan],
            [np.nan, 1, 1, np.nan, 1, np.nan],
            [1, np.nan, 1, np.nan, np.nan, 1],
            [0, 1, 1, np.nan, 0, 1],
        ],
        dtype=dtype,
    )
    _assert_matches_reference(X)


@pytest.mark.parametrize("order", ["C", "F"])
def test_mixed_binary_and_continuous_statistics_match_reference(order):
    rng = np.random.default_rng(13)
    X = rng.normal(size=(101, 8)).astype(np.float32)
    X[:, 0] = rng.integers(0, 2, size=len(X))
    X[::7, 0] = np.nan
    X[:, 1] = 0
    X[:, 2] = 1
    X[:, 3] = np.nan
    X[::11, 4] = np.nan
    _assert_matches_reference(np.array(X, order=order))
