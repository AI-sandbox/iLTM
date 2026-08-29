from __future__ import annotations

import pytest

from iltm.inference_interface import _should_reduce_auto_val_checks


@pytest.mark.parametrize("train_rows", [1, 25_000])
def test_small_training_fold_always_keeps_four_checks(train_rows):
    assert not _should_reduce_auto_val_checks(
        train_rows=train_rows,
        predictor_times=[10.0],
        remaining_time=1.0,
        predictors_left=7,
    )


def test_large_training_fold_reduces_when_ensemble_is_projected_to_miss():
    assert _should_reduce_auto_val_checks(
        train_rows=25_001,
        predictor_times=[10.0, 12.0],
        remaining_time=50.0,
        predictors_left=5,
    )


def test_large_training_fold_keeps_four_when_ensemble_is_projected_to_fit():
    assert not _should_reduce_auto_val_checks(
        train_rows=25_001,
        predictor_times=[10.0, 12.0],
        remaining_time=56.0,
        predictors_left=5,
    )


@pytest.mark.parametrize(
    ("predictor_times", "remaining_time", "predictors_left"),
    [([], 10.0, 3), ([10.0], None, 3), ([10.0], 10.0, 0)],
)
def test_auto_policy_waits_for_a_meaningful_projection(
    predictor_times,
    remaining_time,
    predictors_left,
):
    assert not _should_reduce_auto_val_checks(
        train_rows=25_001,
        predictor_times=predictor_times,
        remaining_time=remaining_time,
        predictors_left=predictors_left,
    )
