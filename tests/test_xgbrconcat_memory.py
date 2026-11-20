from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import iltm.utils as iltm_utils
from iltm import iLTMClassifier


def _concat_classifier(*, optimized: bool, corr_select_k: int = 3):
    return iLTMClassifier(
        checkpoint=None,
        device="cpu",
        tree_embedding=optimized,
        concat_tree_with_orig_features=optimized,
        cat_features=[1],
        preprocessing="realmlp_td_s_v0",
        corr_select_k=corr_select_k,
    )


def _mixed_original_and_embeddings():
    original = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            "category": ["a", "b", "a", "c", "b", "c"],
        },
        index=[9, 2, 7, 4, 11, 3],
    )
    embeddings = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
        ],
        dtype=np.int8,
    )
    target = np.array([0, 1, 0, 1, 1, 0])
    return original, embeddings, target


def test_tree_concat_preserves_column_dtypes_index_and_values():
    original, embeddings, _ = _mixed_original_and_embeddings()
    combined = iLTMClassifier._concatenate_original_and_tree_features(
        original, embeddings
    )
    assert isinstance(combined, pd.DataFrame)
    assert combined.index.equals(original.index)
    assert combined.columns.equals(pd.RangeIndex(5))
    assert combined.dtypes.iloc[0] == original.dtypes.iloc[0]
    assert combined.dtypes.iloc[1] == original.dtypes.iloc[1]
    assert all(dtype == np.dtype(np.int8) for dtype in combined.dtypes.iloc[2:])
    pd.testing.assert_frame_equal(
        combined.iloc[:, :2], original.set_axis(range(2), axis=1)
    )
    np.testing.assert_array_equal(combined.iloc[:, 2:].to_numpy(), embeddings)


def test_xgbrconcat_fitting_output_matches_legacy_object_concat():
    original, embeddings, target = _mixed_original_and_embeddings()
    optimized = _concat_classifier(optimized=True)
    legacy = _concat_classifier(optimized=False)
    combined = optimized._concatenate_original_and_tree_features(original, embeddings)
    optimized_x, optimized_y, optimized_preprocessor = (
        optimized._preprocess_fitting_data(combined, target, is_classification=True)
    )
    legacy_x, legacy_y, legacy_preprocessor = legacy._preprocess_fitting_data(
        np.concatenate([original, embeddings], axis=1),
        target,
        is_classification=True,
    )
    assert optimized_x.dtype == np.float32
    np.testing.assert_array_equal(optimized_y, legacy_y)
    np.testing.assert_array_equal(
        optimized_preprocessor["corr_selected_indices"],
        legacy_preprocessor["corr_selected_indices"],
    )
    np.testing.assert_allclose(optimized_x, legacy_x, rtol=1e-6, atol=1e-7)
    column_transformer = optimized_preprocessor["pipeline"]._pipe.named_steps["one_hot"]
    intermediate = column_transformer.transform(combined)
    assert intermediate.dtype == np.float32
    predicted_x = optimized._preprocess_test_data(combined, optimized_preprocessor)
    assert predicted_x.dtype == torch.float32
    np.testing.assert_allclose(predicted_x.numpy(), optimized_x, rtol=0, atol=0)


@pytest.mark.parametrize("tree_for_each_predictor", [False, True])
@pytest.mark.parametrize(
    ("preprocessing", "concat"),
    [("realmlp_td_s_v0", True), ("minimal", True), ("realmlp_td_s_v0", False)],
)
def test_tree_preprocessing_uses_categories_from_retained_columns(
    monkeypatch, tree_for_each_predictor, preprocessing, concat,
):
    X = pd.DataFrame(np.random.default_rng(19).normal(size=(8, 6150)).astype(np.float32))
    X[1] = [0, 1] * 4
    X[6148] = pd.Series(["a", "b", pd.NA, "b"] * 2, dtype="string")
    X[6149] = pd.Categorical(["c", "d"] * 4)
    y = np.arange(len(X)) % 2
    original_categories = [1, 6148, 6149]
    model = iLTMClassifier(
        checkpoint=None, device="cpu", cat_features=original_categories,
        preprocessing=preprocessing, tree_embedding=True, concat_tree_with_orig_features=concat,
        tree_for_each_predictor=tree_for_each_predictor, tree_data_split="all",
        n_ensemble=2 if tree_for_each_predictor else 1,
        retrieval_alpha_adaptive=False, auto_stop_on_low_cpu_memory=False,
    )
    model.checkpoint = "unused"

    def make_tree(*, tree_model, seed):
        if seed == model.seed:
            retained = np.r_[2:6145, 6148]
        else:
            retained = np.r_[:6143, 6149]
        return SimpleNamespace(
            orig_feature_indices_to_keep_=retained,
            fit_tree=lambda *args, **kwargs: None,
            transform=lambda X: np.ones((len(X), 2), dtype=np.int8),
        )

    fit_inputs = []
    predict_inputs = []

    def generate_predictor(X, y, *, X_val, **kwargs):
        fit_inputs.append(X_val)
        torch.testing.assert_close(X_val, X[: len(X_val)].float())
        return {"timed_out": False}

    def forward(predictor, X, *, n_outputs, **kwargs):
        predict_inputs.append(X)
        return torch.zeros((len(X), n_outputs))

    monkeypatch.setattr(model, "_create_tree_embedding", make_tree)
    monkeypatch.setattr(
        model, "_initialize_model",
        lambda: SimpleNamespace(pca_sampling="zeropad", n_dims=2, clip_data_value=1.0),
    )
    monkeypatch.setattr(model, "_generate_predictor", generate_predictor)
    monkeypatch.setattr(model, "_forward_pass_predictor", forward)

    model.fit(X, y, eval_set=(X.iloc[:4], y[:4]))
    expected = [[6143], [1, 6143]] if tree_for_each_predictor else [[6143]]
    if not concat:
        expected = [[] for _ in expected]
    for prep, indices in zip(model.preprocessors_, expected, strict=True):
        actual = prep["cat_features"] if preprocessing == "minimal" else prep["pipeline"].cat_features
        assert actual == indices
    assert model.cat_features == original_categories
    assert model.predict_proba(X.iloc[:4]).shape == (4, 2)
    for predicted, fitted in zip(predict_inputs, fit_inputs, strict=True):
        torch.testing.assert_close(predicted, fitted)


def test_minimal_concat_keeps_all_missing_original_columns():
    model = iLTMClassifier(checkpoint=None, device="cpu", preprocessing="minimal")
    original = pd.DataFrame({
        "category": pd.Series([pd.NA] * 3, dtype="string"),
        "value": pd.Series([pd.NA] * 3, dtype="Float64"),
    })
    leaves = np.array([[1], [0], [1]], dtype=np.int8)
    inputs = model._concatenate_original_and_tree_features(original, leaves)
    fitted, _, prep = model._preprocess_fitting_data(
        inputs, np.array([1, 0, 1]), is_classification=True, cat_features=[0],
    )

    held_out = inputs.copy()
    held_out.iloc[1, :2] = ["unseen", 7.0]
    predicted = model._preprocess_test_data(held_out, prep).numpy()
    assert predicted.shape == fitted.shape == (3, 3)
    np.testing.assert_allclose(predicted[[0, 2]], fitted[[0, 2]], rtol=1e-6)
    assert predicted[1, 1] == 7.0
    assert np.isfinite(predicted).all()


def test_predict_once_uses_dtype_preserving_tree_concat(monkeypatch):
    original, embeddings, _ = _mixed_original_and_embeddings()
    classifier = _concat_classifier(optimized=True, corr_select_k=0)
    classifier.tree_for_each_predictor = False

    class FakeTree:
        n_orig_features_to_keep_ = None

        def transform(self, X):
            assert X is original
            return embeddings

    classifier.tr_ = FakeTree()
    classifier.preprocessors_ = [{"corr_selected_indices": None}]
    captured = {}

    def capture_preprocessing(X, preprocessor):
        captured["X"] = X
        captured["preprocessor"] = preprocessor
        return torch.zeros((len(X), 1))

    monkeypatch.setattr(classifier, "_preprocess_test_data", capture_preprocessing)
    result = classifier._preprocess_for_predict_once(original)
    assert result.shape == (len(original), 1)
    assert captured["preprocessor"] is classifier.preprocessors_[0]
    assert isinstance(captured["X"], pd.DataFrame)
    assert captured["X"].dtypes.iloc[-1] == np.dtype(np.int8)
    assert captured["X"].index.equals(original.index)


def test_per_predictor_inference_uses_dtype_preserving_tree_concat(monkeypatch):
    original, embeddings, _ = _mixed_original_and_embeddings()
    classifier = _concat_classifier(optimized=True, corr_select_k=0)
    classifier.inference_chunk_rows = 4
    classifier.predictors_ = [{"predictor": 0}]
    classifier.preprocessors_ = [{"corr_selected_indices": None}]
    captured_batches = []

    class FakeTree:
        n_orig_features_to_keep_ = None

        def transform(self, X):
            return embeddings[: len(X)]

    classifier.tr_ = [FakeTree()]

    def capture_preprocessing(X, preprocessor):
        captured_batches.append(X)
        return torch.zeros((len(X), 1))

    def fake_forward(predictor, X, *, n_outputs, **kwargs):
        return torch.zeros((len(X), n_outputs))

    monkeypatch.setattr(classifier, "_preprocess_test_data", capture_preprocessing)
    monkeypatch.setattr(classifier, "_forward_pass_predictor", fake_forward)
    output = classifier._predict_ensemble(original, n_outputs=2)
    assert output.shape == (len(original), 2)
    assert [len(batch) for batch in captured_batches] == [4, 2]
    assert all(isinstance(batch, pd.DataFrame) for batch in captured_batches)
    assert all(batch.dtypes.iloc[-1] == np.dtype(np.int8) for batch in captured_batches)


def test_selected_preprocessed_columns_do_not_materialize_full_width_matrix():
    x_num = np.arange(600, dtype=np.float32).reshape(3, 200)
    x_cat = np.arange(900, dtype=np.float32).reshape(3, 300)
    selected_indices = np.array([0, 199, 200, 499])
    selected = iLTMClassifier._combine_preprocessed_columns(
        x_num, x_cat, selected_indices
    )
    expected = np.concatenate([x_num, x_cat], axis=1)[:, selected_indices]
    np.testing.assert_array_equal(selected, expected)
    assert selected.shape == (3, len(selected_indices))
    assert selected.nbytes == 3 * len(selected_indices) * np.dtype(np.float32).itemsize


def test_dtype_standardization_does_not_scan_numeric_tree_columns(monkeypatch):
    frame = pd.DataFrame(
        {
            "float": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "leaf_1": np.array([0, 1, 0], dtype=np.int8),
            "leaf_2": np.array([1, 0, 1], dtype=np.int8),
            "mixed": pd.Series(["a", 2, "b"], dtype=object),
            "flag": [True, False, True],
        }
    )
    original_apply = pd.Series.apply
    scanned_columns = []

    def recording_apply(series, *args, **kwargs):
        scanned_columns.append(series.name)
        return original_apply(series, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "apply", recording_apply)
    standardized = iltm_utils.standardize_column_dtypes(frame)
    assert scanned_columns == ["mixed"]
    assert standardized["float"].dtype == np.float32
    assert standardized["leaf_1"].dtype == np.int8
    assert standardized["leaf_2"].dtype == np.int8
    assert standardized["mixed"].tolist() == ["a", "2", "b"]
    assert isinstance(standardized["flag"].dtype, pd.CategoricalDtype)


def test_correlations_match_full_width_reference_with_bounded_float64_cast(
    monkeypatch,
):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(101, 13)).astype(np.float32)
    X[:, 4] = 1.0
    y = rng.normal(size=101).astype(np.float32)
    X64 = np.asarray(X, dtype=float)
    y64 = np.asarray(y, dtype=float)
    X_centered = X64 - X64.mean(axis=0, keepdims=True)
    y_centered = y64 - y64.mean()
    denominator = np.sqrt((X_centered**2).sum(axis=0)) * np.sqrt((y_centered**2).sum())
    denominator[denominator == 0] = np.inf
    expected = np.nan_to_num(
        np.clip(
            (X_centered * y_centered[:, None]).sum(axis=0) / denominator,
            -1.0,
            1.0,
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    monkeypatch.setattr(
        iltm_utils,
        "_CORRELATION_WORKING_SET_BYTES",
        X.shape[0] * np.dtype(float).itemsize * 3,
    )
    original_asarray = np.asarray
    float64_cast_widths = []

    def recording_asarray(value, dtype=None, *args, **kwargs):
        array = original_asarray(value, dtype=dtype, *args, **kwargs)
        if dtype is float and array.ndim == 2:
            float64_cast_widths.append(array.shape[1])
        return array

    monkeypatch.setattr(iltm_utils.np, "asarray", recording_asarray)
    actual = iltm_utils.compute_feature_target_correlations(X, y)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(
        iltm_utils.select_top_correlated_features(actual, 6),
        iltm_utils.select_top_correlated_features(expected, 6),
    )
    assert float64_cast_widths
    assert max(float64_cast_widths) <= 3


def test_correlations_do_not_mutate_float64_input():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(37, 11))
    y = rng.normal(size=37)
    expected = X.copy()

    iltm_utils.compute_feature_target_correlations(X, y)

    np.testing.assert_array_equal(X, expected)
