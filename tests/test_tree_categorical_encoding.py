from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from iltm import iLTMClassifier
from iltm.tree_embedding import TreeEmbedding
from iltm.utils import detect_object_string_columns


def test_categorical_detection_handles_pandas_string_dtype():
    X = pd.DataFrame({
        "string": pd.Series(["alpha", "beta", None], dtype="string"),
        "object": pd.Series(["alpha", "beta", None], dtype=object),
        "mixed": pd.Series(["alpha", 1, None], dtype=object),
    })

    assert detect_object_string_columns(X) == [0, 1]


@pytest.mark.parametrize("X", [
    np.arange(6).reshape(2, 3),
    pd.DataFrame({"signed": [-1, 2], "unsigned": np.array([0, 1], dtype=np.uint32), "float": [np.nan, 0.5]}),
])
def test_numeric_detection_preserves_explicit_categories(X):
    assert detect_object_string_columns(X) == []
    assert detect_object_string_columns(X, cat_features=[2, 0, 2]) == [0, 2]


@pytest.mark.parametrize("checkpoint_name", ["cbrconcat", "rtr"])
def test_pandas_category_dtype_reaches_checkpoint_preprocessing(monkeypatch, tmp_path, checkpoint_name):
    checkpoint = tmp_path / f"{checkpoint_name}.pth"
    checkpoint.touch()
    estimator = iLTMClassifier(
        checkpoint=str(checkpoint), device="cpu", n_jobs=2, n_ensemble=1,
        cat_features=[2], tree_n_estimators=2, retrieval_alpha_adaptive=False,
        auto_stop_on_low_cpu_memory=False,
    )
    monkeypatch.setattr(
        estimator, "_initialize_model",
        lambda: SimpleNamespace(pca_sampling="zeropad", n_dims=2, clip_data_value=1.0),
    )
    monkeypatch.setattr(estimator, "_generate_predictor", lambda *args, **kwargs: {"timed_out": False})
    X = pd.DataFrame({
        "text": pd.Categorical(["a", "b"] * 4),
        "code": pd.Categorical([0.25, 0.5] * 4),
        "specified": [0, 0, 1, 1] * 2,
        "number": np.arange(8),
    })
    y = np.arange(len(X)) % 2

    def forward(predictor, X, *, n_outputs, **kwargs):
        assert torch.all(X[0, -3:-1] != X[1, -3:-1])
        return torch.zeros((len(X), n_outputs))

    monkeypatch.setattr(estimator, "_forward_pass_predictor", forward)
    estimator.fit(X, y)

    assert estimator.cat_features_ == [0, 1, 2]
    assert estimator.preprocessors_[0]["pipeline"].cat_features == [0, 1, 2]
    assert estimator.predict_proba(X).shape == (len(X), 2)


def test_refit_redetects_categories_from_constructor_settings(monkeypatch):
    estimator = iLTMClassifier(
        checkpoint=None, device="cpu", n_jobs=2, n_ensemble=1, cat_features=[0],
        tree_model="XGBoost_hist", tree_embedding=True, tree_data_split="all",
        tree_n_estimators=2, concat_tree_with_orig_features=True,
        retrieval_alpha_adaptive=False, auto_stop_on_low_cpu_memory=False,
    )
    estimator.checkpoint = "unused"
    monkeypatch.setattr(
        estimator, "_initialize_model",
        lambda: SimpleNamespace(pca_sampling="zeropad", n_dims=2, clip_data_value=1.0),
    )
    monkeypatch.setattr(estimator, "_generate_predictor", lambda *args, **kwargs: {"timed_out": False})
    X = pd.DataFrame({"specified": [0, 1] * 4, "detected": ["a", "b"] * 4}, dtype=object)
    y = np.arange(len(X)) % 2

    estimator.fit(X, y)
    assert estimator.cat_features_ == [0, 1]
    assert estimator.tr_[0].cat_features == [0, 1]
    assert estimator.preprocessors_[0]["pipeline"].cat_features == [0, 1]
    assert estimator.get_params()["cat_features"] == [0]

    estimator.fit(X[["specified"]], y)
    assert estimator.cat_features_ == [0]
    assert estimator.tr_[0].cat_features == [0]
    assert estimator.preprocessors_[0]["pipeline"].cat_features == [0]


def test_tree_validation_uses_training_categorical_encoder():
    tree = TreeEmbedding(
        tree_model="XGBoost_hist",
        cat_features=[0],
        task_type="classification",
        device="cpu",
    )
    X_train = pd.DataFrame({"category": ["alpha", "beta", "alpha"], "value": [1, 2, 3]})
    X_val = pd.DataFrame({"category": ["beta", "unseen"], "value": [4, 5]})

    encoded_train = tree._handle_categorical_features(X_train, fit=True)
    training_categories = tree.encoders["category"].categories_[0].copy()
    encoded_val = tree._handle_categorical_features(X_val, fit=False)

    np.testing.assert_array_equal(training_categories, np.array(["alpha", "beta"], dtype=object))
    np.testing.assert_array_equal(tree.encoders["category"].categories_[0], training_categories)
    np.testing.assert_array_equal(encoded_train["category"].to_numpy(), np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(encoded_val["category"].to_numpy(), np.array([1.0, -1.0]))


def test_xgboost_fit_with_eval_preserves_training_categorical_encoder():
    tree = TreeEmbedding(
        tree_model="XGBoost_hist",
        cat_features=[0],
        task_type="classification",
        device="cpu",
        n_estimators=2,
        max_depth=2,
        min_samples_leaf=1,
        select_best_model=True,
    )
    X_train = pd.DataFrame(
        {
            "category": ["alpha", "beta", "alpha", "beta", "alpha", "beta"],
            "value": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    y_train = np.array([0, 1, 0, 1, 0, 1])
    X_val = pd.DataFrame({"category": ["beta", "unseen"], "value": [6.0, 7.0]})
    y_val = np.array([1, 0])

    tree.fit_tree(X_train, y_train, eval_set=(X_val, y_val))

    np.testing.assert_array_equal(
        tree.encoders["category"].categories_[0], np.array(["alpha", "beta"], dtype=object)
    )
    assert tree.transform(X_val).shape[0] == len(X_val)


def test_tree_prediction_does_not_refit_categorical_encoder():
    tree = TreeEmbedding(
        tree_model="GB",
        cat_features=[0],
        task_type="classification",
        device="cpu",
    )
    X_train = pd.DataFrame({"category": ["alpha", "beta"], "value": [1, 2]})
    X_test = pd.DataFrame({"category": ["beta", "unseen"], "value": [3, 4]})
    tree._handle_categorical_features(X_train, fit=True)
    training_categories = tree.encoders["category"].categories_[0].copy()

    class RecordingModel:
        def predict(self, X):
            np.testing.assert_array_equal(X["category"].to_numpy(), np.array([1.0, -1.0]))
            return np.array([1, 0])

    tree.model = RecordingModel()
    predictions = tree.get_tree_predictions(X_test)

    np.testing.assert_array_equal(predictions, np.array([1, 0]))
    np.testing.assert_array_equal(tree.encoders["category"].categories_[0], training_categories)
