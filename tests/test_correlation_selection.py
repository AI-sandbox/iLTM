import numpy as np
import pytest

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
