import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import clone

from iltm import iLTMClassifier, iLTMRegressor


class TestSklearnAPI:
    
    def test_classifier_clone(self, tiny_classification_data):
        X, y = tiny_classification_data
        
        clf1 = iLTMClassifier(
            n_ensemble=1,
            finetuning_max_steps=5,
            checkpoint="xgbrconcat",
            device="cpu"
        )
        clf2 = clone(clf1)
        
        assert clf2.n_ensemble == clf1.n_ensemble
        assert clf2.finetuning_max_steps == clf1.finetuning_max_steps
        assert clf2 is not clf1
    
    def test_regressor_clone(self, tiny_regression_data):
        X, y = tiny_regression_data
        
        reg1 = iLTMRegressor(
            n_ensemble=1,
            finetuning_max_steps=5,
            checkpoint="cbrconcat",
            device="cpu"
        )
        reg2 = clone(reg1)
        
        assert reg2.n_ensemble == reg1.n_ensemble
        assert reg2.finetuning_max_steps == reg1.finetuning_max_steps
        assert reg2 is not reg1
    
    def test_classifier_get_params(self):
        clf = iLTMClassifier(
            n_ensemble=3,
            finetuning_max_steps=100,
            checkpoint="cbrconcat",
            device="cpu"
        )
        params = clf.get_params()
        
        assert 'n_ensemble' in params
        assert 'finetuning_max_steps' in params
        assert params['n_ensemble'] == 3
        assert params['finetuning_max_steps'] == 100
    
    def test_regressor_get_params(self):
        reg = iLTMRegressor(
            n_ensemble=3,
            finetuning_max_steps=100,
            checkpoint="cbrconcat",
            device="cpu"
        )
        params = reg.get_params()
        
        assert 'n_ensemble' in params
        assert 'finetuning_max_steps' in params
        assert params['n_ensemble'] == 3
        assert params['finetuning_max_steps'] == 100
    
    def test_classifier_set_params(self):
        clf = iLTMClassifier(
            n_ensemble=1,
            checkpoint="cbrconcat",
            device="cpu"
        )
        clf.set_params(n_ensemble=5, finetuning_max_steps=200)
        
        assert clf.n_ensemble == 5
        assert clf.finetuning_max_steps == 200
    
    def test_regressor_set_params(self):
        reg = iLTMRegressor(
            n_ensemble=1,
            checkpoint="cbrconcat",
            device="cpu"
        )
        reg.set_params(n_ensemble=5, finetuning_max_steps=200)
        
        assert reg.n_ensemble == 5
        assert reg.finetuning_max_steps == 200

