import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from iltm import iLTMClassifier, iLTMRegressor


class TestRetrievalClassification:
    
    def test_binary_classification_with_retrieval(self, small_classification_data):
        X, y = small_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=2,
            finetuning_max_steps=10,
            checkpoint="xgbrconcat",
            device="cpu",
            do_retrieval=True,
            retrieval_alpha=0.5,
            retrieval_temperature=1.0,
            retrieval_distance="cosine"
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        
        assert y_pred.shape[0] == X_test.shape[0]
        assert set(y_pred).issubset(set([0, 1]))
        assert y_proba.shape == (X_test.shape[0], 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)
        assert accuracy_score(y_test, y_pred) > 0.3
    
    def test_multiclass_with_retrieval(self, small_multiclass_data):
        X, y = small_multiclass_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=2,
            finetuning_max_steps=10,
            checkpoint="cbrconcat",
            device="cpu",
            do_retrieval=True,
            retrieval_alpha=0.7,
            retrieval_temperature=2.0,
            retrieval_distance="euclidean"
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        
        assert y_pred.shape[0] == X_test.shape[0]
        assert set(y_pred).issubset(set([0, 1, 2]))
        assert y_proba.shape == (X_test.shape[0], 3)
        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)
    
    def test_retrieval_vs_no_retrieval(self, tiny_classification_data):
        X, y = tiny_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf_no_retrieval = iLTMClassifier(
            n_ensemble=1,
            finetuning_max_steps=5,
            checkpoint="xgbrconcat",
            device="cpu",
            do_retrieval=False
        )
        clf_no_retrieval.fit(X_train, y_train)
        y_pred_no_retrieval = clf_no_retrieval.predict(X_test)
        
        clf_with_retrieval = iLTMClassifier(
            n_ensemble=1,
            finetuning_max_steps=5,
            checkpoint="xgbrconcat",
            device="cpu",
            do_retrieval=True,
            retrieval_alpha=0.5
        )
        clf_with_retrieval.fit(X_train, y_train)
        y_pred_with_retrieval = clf_with_retrieval.predict(X_test)
        
        assert y_pred_no_retrieval.shape[0] == X_test.shape[0]
        assert y_pred_with_retrieval.shape[0] == X_test.shape[0]
    
    def test_retrieval_alpha_values(self, tiny_classification_data):
        X, y = tiny_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            clf = iLTMClassifier(
                n_ensemble=1,
                finetuning_max_steps=5,
                checkpoint="cbrconcat",
                device="cpu",
                do_retrieval=True,
                retrieval_alpha=alpha
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            assert y_pred.shape[0] == X_test.shape[0]
    
    def test_retrieval_distance_types(self, tiny_classification_data):
        X, y = tiny_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        for distance in ["cosine", "euclidean"]:
            clf = iLTMClassifier(
                n_ensemble=1,
                finetuning_max_steps=5,
                checkpoint="xgbrconcat",
                device="cpu",
                do_retrieval=True,
                retrieval_distance=distance
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            assert y_pred.shape[0] == X_test.shape[0]


class TestRetrievalRegression:
    
    def test_regression_with_retrieval(self, small_regression_data):
        X, y = small_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = iLTMRegressor(
            n_ensemble=2,
            finetuning_max_steps=10,
            checkpoint="xgbrconcat",
            device="cpu",
            do_retrieval=True,
            retrieval_alpha=0.5,
            retrieval_temperature=1.0,
            retrieval_distance="cosine"
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        
        assert y_pred.shape[0] == X_test.shape[0]
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.dtype in [np.float32, np.float64]
        mse = mean_squared_error(y_test, y_pred)
        assert mse < np.var(y_test) * 3
    
    def test_regression_retrieval_vs_no_retrieval(self, tiny_regression_data):
        X, y = tiny_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg_no_retrieval = iLTMRegressor(
            n_ensemble=1,
            finetuning_max_steps=5,
            checkpoint="cbrconcat",
            device="cpu",
            do_retrieval=False
        )
        reg_no_retrieval.fit(X_train, y_train)
        y_pred_no_retrieval = reg_no_retrieval.predict(X_test)
        
        reg_with_retrieval = iLTMRegressor(
            n_ensemble=1,
            finetuning_max_steps=5,
            checkpoint="cbrconcat",
            device="cpu",
            do_retrieval=True,
            retrieval_alpha=0.5
        )
        reg_with_retrieval.fit(X_train, y_train)
        y_pred_with_retrieval = reg_with_retrieval.predict(X_test)
        
        assert y_pred_no_retrieval.shape[0] == X_test.shape[0]
        assert y_pred_with_retrieval.shape[0] == X_test.shape[0]
    
    def test_regression_retrieval_parameters(self, tiny_regression_data):
        X, y = tiny_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        for temperature in [0.5, 1.0, 2.0]:
            reg = iLTMRegressor(
                n_ensemble=1,
                finetuning_max_steps=5,
                checkpoint="xgbrconcat",
                device="cpu",
                do_retrieval=True,
                retrieval_temperature=temperature
            )
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            assert y_pred.shape[0] == X_test.shape[0]
    
    def test_regression_with_eval_set_and_retrieval(self, small_regression_data):
        X, y = small_regression_data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        reg = iLTMRegressor(
            n_ensemble=2,
            finetuning_max_steps=10,
            checkpoint="cbrconcat",
            device="cpu",
            do_retrieval=True,
            retrieval_alpha=0.6
        )
        reg.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]

