from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import iltm.inference_interface as inference_interface
from iltm import iLTMClassifier
from iltm.realmlp_td_s_preprocessing import (
    RealMLPTDSepPipeline,
    to_numeric_coerce,
)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(
            {
                "float": np.array([1.5, np.nan, -2], dtype=np.float32),
                "integer": np.array([1, 2, 3], dtype=np.int64),
                "boolean": [True, False, True],
            }
        ),
        pd.DataFrame(
            {
                "integer": pd.array([1, None, 3], dtype="Int64"),
                "float": pd.array([1, None, 4], dtype="Float32"),
                "boolean": pd.array([True, None, False], dtype="boolean"),
            }
        ),
    ],
)
def test_numeric_fast_path_matches_legacy_without_aliasing(frame):
    expected = frame.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    original = frame.copy(deep=True)

    actual = to_numeric_coerce(frame, dtype=np.float32)

    pd.testing.assert_frame_equal(actual, expected)
    actual.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(frame, original)


def test_numeric_fast_path_and_guarded_fallbacks(monkeypatch):
    original = pd.to_numeric
    calls = []

    def recording(values, *args, **kwargs):
        calls.append(values.name)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(pd, "to_numeric", recording)
    dense = pd.DataFrame(np.arange(12, dtype=np.float32).reshape(3, 4))
    to_numeric_coerce(dense, dtype=np.float32)
    assert calls == []

    to_numeric_coerce(dense, dtype=None)
    assert calls == list(dense.columns)
    calls.clear()

    sparse = pd.DataFrame(
        {
            "left": pd.arrays.SparseArray([0.0, 1.0, 0.0]),
            "right": pd.arrays.SparseArray([2.0, 0.0, 3.0]),
        }
    )
    expected = sparse.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    calls.clear()
    actual = to_numeric_coerce(sparse, dtype=np.float32)

    assert calls == ["left", "right"]
    pd.testing.assert_frame_equal(actual, expected)


def test_float32_pipeline_matches_legacy_output():
    X = pd.DataFrame(
        {
            "numeric": np.array([1.0, np.nan, 4.0], dtype=np.float32),
            "category": ["a", "b", "a"],
        }
    )
    legacy = RealMLPTDSepPipeline(cat_features=[1])
    expected = legacy.fit_transform(X)
    candidate = RealMLPTDSepPipeline(cat_features=[1], output_dtype=np.float32)
    actual = candidate.fit_transform(X)

    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_array_equal(actual_part, expected_part)
        assert actual_part.dtype == np.float32


def test_concat_tree_preprocessing_requests_float32(monkeypatch):
    captured = {}

    class _Pipeline:
        def fit_transform(self, X, y=None):
            array = np.asarray(X, dtype=np.float32)
            return array, np.empty((len(array), 0), dtype=np.float32)

    def make_pipeline(**kwargs):
        captured.update(kwargs)
        return _Pipeline()

    monkeypatch.setattr(
        inference_interface,
        "get_realmlp_td_s_pipeline_separated",
        make_pipeline,
    )
    estimator = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        preprocessing="realmlp_td_s_v0",
        tree_embedding=True,
        concat_tree_with_orig_features=True,
        corr_select_k=0,
        adaptive_memory=False,
    )

    estimator._preprocess_fitting_data(
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.array([0, 1, 0, 1, 0, 1]),
        is_classification=True,
    )

    assert captured["output_dtype"] is np.float32
