from __future__ import annotations

import numpy as np
import pandas as pd

import iltm.inference_interface as inference_interface
from iltm import iLTMClassifier
from iltm.realmlp_td_s_preprocessing import (
    CustomOneHotPipeline,
    RealMLPTDSepPipeline,
)


def _mixed_frame():
    return pd.DataFrame(
        {
            "numeric": np.array(
                [1.0, 2.0, np.nan, 8.0],
                dtype=np.float32,
            ),
            "category": ["a", "b", "a", None],
            "integer": [1, 7, 3, 9],
        }
    )


def test_fit_transform_reuses_training_pipeline_output(monkeypatch):
    calls = []
    original_transform = CustomOneHotPipeline.transform

    def recording_transform(self, X, y=None):
        calls.append(len(X))
        return original_transform(self, X, y)

    monkeypatch.setattr(
        CustomOneHotPipeline,
        "transform",
        recording_transform,
    )
    pipeline = RealMLPTDSepPipeline(cat_features=[1])

    pipeline.fit_transform(_mixed_frame())

    assert calls == [4]


def test_fit_transform_matches_fit_then_transform():
    X = _mixed_frame()
    held_out = pd.DataFrame(
        {
            "numeric": np.array([4.0, np.nan], dtype=np.float32),
            "category": ["unseen", "a"],
            "integer": [5, 11],
        }
    )
    reference = RealMLPTDSepPipeline(cat_features=[1])
    reference.fit(X)
    expected_train = reference.transform(X)
    expected_held_out = reference.transform(held_out)

    candidate = RealMLPTDSepPipeline(cat_features=[1])
    actual_train = candidate.fit_transform(X)
    actual_held_out = candidate.transform(held_out)

    assert candidate._cat_dim == reference._cat_dim
    for actual, expected in zip(actual_train, expected_train):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(actual_held_out, expected_held_out):
        np.testing.assert_array_equal(actual, expected)


def test_iltm_preprocessor_uses_fit_transform(monkeypatch):
    calls = []

    class _SpyPipeline:
        def fit(self, X, y=None):
            raise AssertionError("separate fit() should not be called")

        def fit_transform(self, X, y=None):
            calls.append("fit_transform")
            array = np.asarray(X, dtype=np.float32)
            return array, np.empty((len(array), 0), dtype=np.float32)

    monkeypatch.setattr(
        inference_interface,
        "get_realmlp_td_s_pipeline_separated",
        lambda **kwargs: _SpyPipeline(),
    )
    estimator = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        preprocessing="realmlp_td_s_v0",
        corr_select_k=0,
        adaptive_memory=False,
    )

    X_out, y_out, _ = estimator._preprocess_fitting_data(
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.array([0, 1, 0, 1, 0, 1]),
        is_classification=True,
    )

    assert calls == ["fit_transform"]
    assert X_out.shape == (6, 2)
    np.testing.assert_array_equal(
        y_out,
        np.array([0, 1, 0, 1, 0, 1]),
    )
