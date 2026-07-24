from __future__ import annotations

import warnings

import numpy as np
import pytest

from iltm.realmlp_td_s_preprocessing import RobustScaleSmoothClipTransform


def _legacy_statistics(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(X, axis=0)
    q75 = np.nanquantile(X, 0.75, axis=0)
    q25 = np.nanquantile(X, 0.25, axis=0)
    iqr = q75 - q25
    width = 0.5 * (np.nanmax(X, axis=0) - np.nanmin(X, axis=0))
    denominator = np.where(np.isfinite(iqr) & (iqr != 0), iqr, width)
    median = np.where(np.isfinite(median), median, 0)
    factors = np.zeros_like(denominator, dtype=np.float64)
    good = np.isfinite(denominator) & (denominator != 0)
    factors[good] = 1 / (denominator[good] + 1e-30)
    return median.astype(np.float32), factors.astype(np.float32)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32])
def test_joint_quantiles_match_scalar_reference(dtype):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(100, 7)).astype(dtype)
    X[:, 0] = 3
    if np.issubdtype(dtype, np.floating):
        X[::7, 1] = np.nan
        X[:, 2] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        transform = RobustScaleSmoothClipTransform().fit(X.copy())
        expected_median, expected_factors = _legacy_statistics(X)

    np.testing.assert_array_equal(transform._median, expected_median)
    np.testing.assert_array_equal(transform._factors, expected_factors)


@pytest.mark.parametrize(
    ("storage_dtype", "float_dtype", "bits"),
    [
        (np.uint16, np.float16, [0xBCFD, 0x3537]),
        (np.uint32, np.float32, [0xBD8B393C, 0x3CFD463B]),
    ],
)
def test_joint_quantiles_preserve_rounding(storage_dtype, float_dtype, bits):
    X = np.array(bits, dtype=storage_dtype).view(float_dtype).reshape(2, 1)
    expected_median, expected_factors = _legacy_statistics(X)
    transform = RobustScaleSmoothClipTransform().fit(X)

    np.testing.assert_array_equal(transform._median, expected_median)
    np.testing.assert_array_equal(transform._factors, expected_factors)


def test_joint_quantiles_preserve_empty_axis_behavior():
    transform = RobustScaleSmoothClipTransform().fit(
        np.empty((3, 0), dtype=np.float32)
    )
    assert transform._median.shape == transform._factors.shape == (0,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="zero-size array"):
            RobustScaleSmoothClipTransform().fit(
                np.empty((0, 3), dtype=np.float32)
            )
