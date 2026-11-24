from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from iltm.tree_embedding import TreeEmbedding


def _make_tree(**kwargs):
    return TreeEmbedding(
        cat_features=[], n_jobs=1, n_estimators=12, max_depth=3, **kwargs
    )


def _legacy_dmatrix(X, y=None):
    return xgb.DMatrix(X, y, enable_categorical=True, nthread=1)


def _assert_same_matrix(actual, expected):
    assert actual.num_row() == expected.num_row()
    assert actual.num_col() == expected.num_col()
    assert actual.feature_names == expected.feature_names
    assert actual.feature_types == expected.feature_types
    np.testing.assert_array_equal(actual.get_label(), expected.get_label())
    actual_data, expected_data = actual.get_data(), expected.get_data()
    np.testing.assert_array_equal(actual_data.indptr, expected_data.indptr)
    np.testing.assert_array_equal(actual_data.indices, expected_data.indices)
    np.testing.assert_array_equal(
        actual_data.data.view(np.uint32), expected_data.data.view(np.uint32)
    )


@pytest.fixture
def tree():
    return _make_tree(tree_model="XGBoost_hist", task_type="regression", device="cpu")


@pytest.fixture
def matrix_inputs(monkeypatch):
    inputs = []
    constructor = xgb.DMatrix

    def record_input(data, *args, **kwargs):
        inputs.append(data)
        return constructor(data, *args, **kwargs)

    monkeypatch.setattr(xgb, "DMatrix", record_input)
    return inputs


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("n_rows", "columns", "order"),
    [
        pytest.param(0, pd.Index(["x", "y"]), "C", id="zero-rows"),
        pytest.param(1, pd.Index(["feature"]), "C", id="one-cell"),
        pytest.param(
            7,
            pd.Index([37, "unicode 雪", "with spaces", "missing"]),
            "C",
            id="mixed-labels",
        ),
        pytest.param(17, pd.RangeIndex(15_154), "F", id="wide-column-major"),
    ],
)
def test_native_float_arrays_match_pandas(
    tree, matrix_inputs, dtype, n_rows, columns, order
):
    rng = np.random.default_rng(42)
    values = np.array(rng.normal(size=(n_rows, len(columns))), dtype=dtype, order=order)
    if n_rows:
        values[:, -1] = np.nan
    if n_rows >= 7:
        values[:7, 0] = [
            np.inf,
            -np.inf,
            1e20,
            -1e20,
            -0.0,
            np.finfo(dtype).tiny,
            np.nan,
        ]
    X = pd.DataFrame(values, columns=columns, index=np.arange(n_rows)[::-1] // 2)
    sanitized = tree._handle_inf_and_value_too_large(X)
    labels = pd.Series(np.arange(n_rows, dtype=np.float32), index=X.index)

    expected = _legacy_dmatrix(sanitized, labels)
    actual = tree._make_xgboost_dmatrix(sanitized, labels)

    assert isinstance(matrix_inputs[-1], np.ndarray)
    _assert_same_matrix(actual, expected)


@pytest.mark.parametrize(
    "X",
    [
        pytest.param(
            pd.DataFrame({"x": pd.array([1.0, None, -1.0], dtype="Float32")}),
            id="nullable-float",
        ),
        pytest.param(
            pd.DataFrame({"sparse": pd.arrays.SparseArray([0.0, 1.0, np.nan])}),
            id="sparse-float",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "float32": np.array([np.nan, -0.0, 1e-30], dtype=np.float32),
                    "float64": [1.0, np.nan, -1e20],
                    "uint64": np.array([2**63 + 1, 1, 0], dtype=np.uint64),
                    "nullable_int": pd.array([1, None, -2], dtype="Int64"),
                    "bool": [True, False, True],
                    "nullable_bool": pd.array([True, None, False], dtype="boolean"),
                    "category": pd.Categorical(
                        ["雪", None, "a"], categories=["a", "雪", "unseen"]
                    ),
                }
            ),
            id="mixed-dtypes",
        ),
        pytest.param(
            pd.DataFrame(
                [[1.0, np.nan], [2.0, 3.0]],
                columns=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]),
            ),
            id="multiindex-columns",
        ),
    ],
)
def test_other_schema_uses_pandas_without_changing_output(tree, matrix_inputs, X):
    expected = _legacy_dmatrix(X)
    actual = tree._make_xgboost_dmatrix(X)

    assert matrix_inputs[-1] is X
    _assert_same_matrix(actual, expected)


@pytest.mark.parametrize(
    "X",
    [
        pytest.param(
            pd.DataFrame([[1.0, 2.0]], columns=[1, "1"]),
            id="duplicate-stringified-labels",
        ),
        pytest.param(pd.DataFrame({"bad<name": [1.0]}), id="invalid-feature-name"),
        pytest.param(
            pd.DataFrame({"datetime": pd.to_datetime(["2020-01-01"])}),
            id="unsupported-dtype",
        ),
    ],
)
def test_invalid_inputs_raise_same_error(tree, X):
    with pytest.raises(ValueError) as expected:
        _legacy_dmatrix(X)
    with pytest.raises(type(expected.value)) as actual:
        tree._make_xgboost_dmatrix(X)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("tree_model", ["XGBoost_hist", "XGBoost_approx"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_fitted_tree_and_leaf_embeddings_match_pandas(
    monkeypatch, tree_model, device, task_type
):
    if device != "cpu":
        import torch

        if not torch.cuda.is_available() or not xgb.build_info().get("USE_CUDA", False):
            pytest.skip("Requires XGBoost with CUDA and an available GPU")

    rng = np.random.default_rng(21)
    X = pd.DataFrame(
        rng.normal(size=(257, 5)).astype(np.float32), columns=["a", "b", "c", "d", "e"]
    )
    y = 2 * X["a"].to_numpy() + X["b"].to_numpy()
    if task_type == "classification":
        y = np.digitize(y, [-1.0, 1.0])
    X.loc[[0, 1, 2, 3], "d"] = np.array(
        [np.nan, np.inf, -np.inf, 1e20], dtype=np.float32
    )

    actual = _make_tree(tree_model=tree_model, task_type=task_type, device=device)
    actual.fit_tree(X.iloc[:129], y[:129], eval_set=(X.iloc[129:], y[129:]))
    if device == "cpu":
        reference = _make_tree(tree_model=tree_model, task_type=task_type, device="cpu")
        monkeypatch.setattr(reference, "_make_xgboost_dmatrix", _legacy_dmatrix)
        reference.fit_tree(X.iloc[:129], y[:129], eval_set=(X.iloc[129:], y[129:]))
        assert actual.model.best_iteration == reference.model.best_iteration
        np.testing.assert_array_equal(actual.transform(X), reference.transform(X))
        np.testing.assert_array_equal(
            actual.get_tree_predictions(X), reference.get_tree_predictions(X)
        )
        actual.device = (
            "cuda:0"  # Exercise padding with a real CPU booster even without CUDA.
        )

    # Compare both input routes on the same booster, including GPU padding boundaries.
    inputs = [X.iloc[:n] for n in [1, 17, 127, 128, 129, len(X)]]
    fast = [actual.transform(batch) for batch in inputs]
    monkeypatch.setattr(actual, "_make_xgboost_dmatrix", _legacy_dmatrix)
    for batch, embeddings in zip(inputs, fast, strict=True):
        np.testing.assert_array_equal(embeddings, actual.transform(batch))
