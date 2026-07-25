import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from iltm import iLTMRegressor


def _mock_calibrated_regressor(monkeypatch, raw_predict):
    reg = iLTMRegressor(
        checkpoint=None,
        device="cpu",
        finetuning=False,
        clip_predictions=False,
    )

    def fit_common(*args, **kwargs):
        reg.normalize_predictions_ = True
        reg.clip_predictions_ = False
        reg.predictors_ = [{}]
        reg.preprocessors_ = [{}]
        return reg

    monkeypatch.setattr(reg, "_fit_common", fit_common)
    monkeypatch.setattr(reg, "_predict_ensemble", raw_predict)
    return reg


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

    def test_prediction_calibration_is_batch_invariant(self, monkeypatch):
        def raw_predict(X, **kwargs):
            return torch.as_tensor(
                np.asarray(X)[:, 0],
                dtype=torch.float32,
            )

        reg = _mock_calibrated_regressor(monkeypatch, raw_predict)

        reg.fit(
            np.array([[100.0], [101.0], [102.0], [103.0]]),
            np.array([10.0, 12.0, 14.0, 16.0]),
            eval_set=(
                np.array([[0.0], [1.0], [2.0]]),
                np.array([10.0, 12.0, 14.0]),
            ),
        )
        X_test = np.array([[3.0], [4.0]])

        batch_predictions = reg.predict(X_test)
        separate_predictions = np.concatenate(
            [reg.predict(X_test[i : i + 1]) for i in range(len(X_test))]
        )

        np.testing.assert_allclose(batch_predictions, np.array([16.0, 18.0]))
        np.testing.assert_array_equal(
            separate_predictions,
            batch_predictions,
        )

    def test_prediction_calibration_uses_training_data_without_eval_set(
        self,
        monkeypatch,
    ):
        reg = _mock_calibrated_regressor(
            monkeypatch,
            lambda X, **kwargs: torch.as_tensor(
                np.asarray(X)[:, 0],
                dtype=torch.float32,
            ),
        )

        reg.fit(
            np.array([[0.0], [1.0], [2.0], [3.0]]),
            np.array([10.0, 12.0, 14.0, 16.0]),
        )

        np.testing.assert_allclose(
            reg.predict(np.array([[4.0], [5.0]])),
            np.array([18.0, 20.0]),
        )

    def test_constant_prediction_calibration_uses_target_mean(
        self,
        monkeypatch,
    ):
        reg = _mock_calibrated_regressor(
            monkeypatch,
            lambda X, **kwargs: torch.full(
                (len(X),),
                3.0,
                dtype=torch.float32,
            ),
        )

        reg.fit(
            np.array([[0.0], [1.0], [2.0]]),
            np.array([10.0, 12.0, 14.0]),
        )
        predictions = reg.predict(np.array([[3.0], [4.0]]))

        assert np.isfinite(predictions).all()
        np.testing.assert_allclose(predictions, np.array([12.0, 12.0]))

    def test_prediction_calibration_caps_dataframe_rows(self, monkeypatch):
        rows_seen = []

        def raw_predict(X, **kwargs):
            rows_seen.append(len(X))
            return torch.as_tensor(
                np.array(X)[:, 0],
                dtype=torch.float32,
            )

        reg = _mock_calibrated_regressor(monkeypatch, raw_predict)
        X_calibration = pd.DataFrame({"value": np.arange(5000)})

        reg.fit(
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            eval_set=(
                X_calibration,
                2.0 * X_calibration["value"].to_numpy(),
            ),
        )

        assert rows_seen == [4096]

    def test_prediction_calibration_failure_invalidates_fit(self, monkeypatch):
        def fail_prediction(*args, **kwargs):
            raise RuntimeError("synthetic calibration failure")

        reg = _mock_calibrated_regressor(monkeypatch, fail_prediction)

        with pytest.raises(RuntimeError, match="synthetic calibration failure"):
            reg.fit(
                np.array([[0.0], [1.0]]),
                np.array([0.0, 1.0]),
            )

        assert not reg.__sklearn_is_fitted__()
        assert reg.predictors_ == []
        assert reg.preprocessors_ == []
    
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
