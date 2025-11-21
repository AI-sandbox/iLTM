import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from iltm import iLTMClassifier


class TestiLTMClassifierBasic:
    
    def test_binary_classification_fit_predict(self, small_classification_data):
        X, y = small_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=2,
            finetuning_max_steps=10,
            checkpoint="xgbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        assert y_pred.shape[0] == X_test.shape[0]
        assert set(y_pred).issubset(set([0, 1]))
        assert accuracy_score(y_test, y_pred) > 0.4
    
    def test_binary_classification_predict_proba(self, small_classification_data):
        X, y = small_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=2,
            finetuning_max_steps=10,
            checkpoint="cbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train)
        
        y_proba = clf.predict_proba(X_test)
        
        assert y_proba.shape == (X_test.shape[0], 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(y_proba >= 0) and np.all(y_proba <= 1)
    
    def test_multiclass_classification(self, small_multiclass_data):
        X, y = small_multiclass_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=2,
            finetuning_max_steps=10,
            checkpoint="cbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        
        assert y_pred.shape[0] == X_test.shape[0]
        assert y_proba.shape == (X_test.shape[0], 3)
        assert set(y_pred).issubset(set([0, 1, 2]))
        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)
    
    def test_with_eval_set(self, small_classification_data):
        X, y = small_classification_data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=2,
            finetuning_max_steps=10,
            checkpoint="xgbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        y_pred = clf.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_pandas_dataframe_input(self, small_classification_data):
        X, y = small_classification_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y, name="target")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=2,
            finetuning_max_steps=10,
            checkpoint="cbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_voting_soft(self, tiny_classification_data):
        X, y = tiny_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=2,
            finetuning_max_steps=5,
            voting="soft",
            checkpoint="xgbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_voting_hard(self, tiny_classification_data):
        X, y = tiny_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=2,
            finetuning_max_steps=5,
            voting="hard",
            checkpoint="cbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]


class TestiLTMClassifierPreprocessing:
    
    def test_minimal_preprocessing(self, tiny_classification_data):
        X, y = tiny_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=1,
            finetuning_max_steps=5,
            preprocessing="minimal",
            checkpoint="cbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_no_preprocessing(self, tiny_classification_data):
        X, y = tiny_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=1,
            finetuning_max_steps=5,
            preprocessing="none",
            checkpoint="xgbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]


class TestiLTMClassifierOptions:
    
    def test_no_finetuning(self, tiny_classification_data):
        X, y = tiny_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=1,
            finetuning=False,
            checkpoint="xgbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_fit_with_time_limit(self, tiny_classification_data):
        X, y = tiny_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=5,
            finetuning_max_steps=10,
            checkpoint="cbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train, fit_max_time=30.0)
        
        assert len(clf.predictors_) >= 1
        
        y_pred = clf.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_classes_attribute(self, small_classification_data):
        X, y = small_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = iLTMClassifier(
            n_ensemble=1,
            finetuning_max_steps=5,
            checkpoint="xgbrconcat",
            device="cpu"
        )
        clf.fit(X_train, y_train)
        
        assert hasattr(clf, 'classes_')
        assert len(clf.classes_) == 2
        assert set(clf.classes_) == set([0, 1])

