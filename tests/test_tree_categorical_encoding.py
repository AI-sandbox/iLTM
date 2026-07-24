import numpy as np
import pandas as pd

from iltm.tree_embedding import TreeEmbedding


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
