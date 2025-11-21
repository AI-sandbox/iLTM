import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from iltm import iLTMRegressor


class TestiLTMRegressorBasic:
    
    def test_regression_fit_predict(self, small_regression_data):
        X, y = small_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = iLTMRegressor(
            n_ensemble=2,
            finetuning_max_steps=10,
            checkpoint="xgbrconcat",
            device="cpu"
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        
        assert y_pred.shape[0] == X_test.shape[0]
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.dtype in [np.float32, np.float64]
        
        mse = mean_squared_error(y_test, y_pred)
        assert mse < np.var(y_test) * 2
    
    def test_with_eval_set(self, small_regression_data):
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
            device="cpu"
        )
        reg.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_pandas_dataframe_input(self, small_regression_data):
        X, y = small_regression_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y, name="target")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42
        )
        
        reg = iLTMRegressor(
            n_ensemble=2,
            finetuning_max_steps=10,
            checkpoint="cbrconcat",
            device="cpu"
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_clip_predictions(self, tiny_regression_data):
        X, y = tiny_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = iLTMRegressor(
            n_ensemble=1,
            finetuning_max_steps=5,
            clip_predictions=True,
            checkpoint="cbrconcat",
            device="cpu"
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        
        train_min, train_max = y_train.min(), y_train.max()
        margin = (train_max - train_min) * 0.5
        
        assert y_pred.min() >= train_min - margin
        assert y_pred.max() <= train_max + margin
    
    def test_normalize_predictions(self, tiny_regression_data):
        X, y = tiny_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = iLTMRegressor(
            n_ensemble=1,
            finetuning_max_steps=5,
            normalize_predictions=True,
            checkpoint="xgbrconcat",
            device="cpu"
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]


class TestiLTMRegressorPreprocessing:
    
    def test_minimal_preprocessing(self, tiny_regression_data):
        X, y = tiny_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = iLTMRegressor(
            n_ensemble=1,
            finetuning_max_steps=5,
            preprocessing="minimal",
            checkpoint="cbrconcat",
            device="cpu"
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_no_preprocessing(self, tiny_regression_data):
        X, y = tiny_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = iLTMRegressor(
            n_ensemble=1,
            finetuning_max_steps=5,
            preprocessing="none",
            checkpoint="xgbrconcat",
            device="cpu"
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]


class TestiLTMRegressorOptions:
    
    def test_no_finetuning(self, tiny_regression_data):
        X, y = tiny_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = iLTMRegressor(
            n_ensemble=1,
            finetuning=False,
            checkpoint="xgbrconcat",
            device="cpu"
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_fit_with_time_limit(self, tiny_regression_data):
        X, y = tiny_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = iLTMRegressor(
            n_ensemble=5,
            finetuning_max_steps=10,
            checkpoint="cbrconcat",
            device="cpu"
        )
        reg.fit(X_train, y_train, fit_max_time=30.0)
        
        assert len(reg.predictors_) >= 1
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_target_normalization_attributes(self, small_regression_data):
        X, y = small_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = iLTMRegressor(
            n_ensemble=1,
            finetuning_max_steps=5,
            checkpoint="cbrconcat",
            device="cpu"
        )
        reg.fit(X_train, y_train)
        
        assert hasattr(reg, '_y_mean')
        assert hasattr(reg, '_y_std')
        assert isinstance(reg._y_mean, float)
        assert isinstance(reg._y_std, float)
        assert reg._y_std > 0

