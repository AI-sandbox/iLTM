import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from iltm.utils import robust_roc_auc_score


@pytest.mark.parametrize("classes", [(0, 1), (0, 2), (1, 2)])
def test_auc_for_two_class_subset_uses_present_score_columns(classes):
    negative_class, positive_class = classes
    y_true = np.array(
        [negative_class, negative_class, positive_class, positive_class]
    )
    y_score = np.full((4, 3), 0.05)
    y_score[:2, negative_class] = 0.90
    y_score[2:, positive_class] = 0.90

    assert robust_roc_auc_score(y_true, y_score) == pytest.approx(1.0)


@pytest.mark.parametrize("classes", [(10, 20), ("cat", "dog")])
def test_two_column_auc_supports_arbitrary_class_labels(classes):
    negative_class, positive_class = classes
    y_true = np.array(
        [negative_class, negative_class, positive_class, positive_class]
    )
    y_score = np.array(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.10, 0.90],
            [0.20, 0.80],
        ]
    )

    assert robust_roc_auc_score(y_true, y_score) == pytest.approx(1.0)


def test_binary_auc_matches_sklearn_for_nonperfect_scores():
    y_true = np.array([10, 20, 10, 20, 10, 20])
    positive_scores = np.array([0.10, 0.40, 0.35, 0.80, 0.70, 0.60])
    y_score = np.column_stack([1.0 - positive_scores, positive_scores])

    expected = roc_auc_score(y_true, positive_scores)
    assert robust_roc_auc_score(y_true, y_score) == pytest.approx(expected)


@pytest.mark.parametrize("multi_class", ["ovo", "ovr"])
def test_full_multiclass_auc_matches_sklearn(multi_class):
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_score = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.30, 0.60],
            [0.40, 0.35, 0.25],
            [0.20, 0.45, 0.35],
            [0.30, 0.20, 0.50],
        ]
    )

    expected = roc_auc_score(y_true, y_score, multi_class=multi_class)
    actual = robust_roc_auc_score(
        y_true,
        y_score,
        multi_class=multi_class,
    )
    assert actual == pytest.approx(expected)


def test_multiclass_subset_auc_filters_and_normalizes_scores():
    y_true = np.array([0, 2, 3, 0, 2, 3])
    y_score = np.array(
        [
            [0.55, 0.35, 0.05, 0.05],
            [0.05, 0.40, 0.45, 0.10],
            [0.05, 0.20, 0.15, 0.60],
            [0.45, 0.20, 0.25, 0.10],
            [0.10, 0.30, 0.50, 0.10],
            [0.10, 0.25, 0.20, 0.45],
        ]
    )
    filtered_scores = y_score[:, [0, 2, 3]]
    filtered_scores /= filtered_scores.sum(axis=1, keepdims=True)

    expected = roc_auc_score(
        y_true,
        filtered_scores,
        multi_class="ovo",
    )
    assert robust_roc_auc_score(y_true, y_score) == pytest.approx(expected)


def test_multiclass_subset_with_zero_present_mass_has_chance_auc():
    y_true = np.array([0, 2, 3, 0, 2, 3])
    y_score = np.zeros((len(y_true), 4))
    y_score[:, 1] = 1.0

    assert robust_roc_auc_score(y_true, y_score) == pytest.approx(0.5)
