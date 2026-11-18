import contextlib
import os
import pickle
import random

import numpy as np
import pytest
import torch

import iltm.inference_interface as interface
from iltm import iLTMClassifier, iLTMRegressor
from iltm.iltm_model import iLTM
from iltm.utils import preserve_random_state, seed_everything


ARCHITECTURE = dict(
    n_dims=4, hn_n_layers=3, hn_hidden_size=8, rf_size=12,
    n_classes_limit=5, main_n_layers=3,
)


def snapshot():
    return (
        random.getstate(), pickle.dumps(np.random.get_state()),
        torch.get_rng_state().clone(), torch.cuda.get_rng_state_all(),
        os.environ.get('PYTHONHASHSEED'),
    )


def assert_restored(expected):
    actual = snapshot()
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert len(actual[3]) == len(expected[3])
    assert all(torch.equal(a, b) for a, b in zip(actual[3], expected[3]))
    assert actual[4] == expected[4]


@pytest.mark.parametrize('hash_seed', [None, '', '123'])
@pytest.mark.parametrize('fail', [False, True])
def test_context_restores_rngs_and_environment_even_on_error(monkeypatch, hash_seed, fail):
    if hash_seed is None:
        monkeypatch.delenv('PYTHONHASHSEED', raising=False)
    else:
        monkeypatch.setenv('PYTHONHASHSEED', hash_seed)
    before = snapshot()
    error_context = pytest.raises(ValueError, match='test failure') if fail else contextlib.nullcontext()
    with error_context:
        with preserve_random_state():
            seed_everything(19)
            random.random()
            np.random.uniform(size=3)
            torch.rand(3)
            if torch.cuda.is_available():
                torch.rand(3, device='cuda')
            if fail:
                raise ValueError('test failure')
    assert_restored(before)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / 'small.pth'
    with preserve_random_state():
        torch.manual_seed(19)
        torch.save(iLTM(**ARCHITECTURE).state_dict(), path)
    return path


def make_estimator(cls, checkpoint, device, finetuning):
    return cls(
        checkpoint=str(checkpoint), device=device, n_ensemble=2,
        batch_size=16, preprocessing='none', pca_sampling='zeropad',
        tree_embedding=False, adaptive_memory=False,
        auto_stop_on_low_cpu_memory=False, inference_storage_dtype='float32',
        finetuning=finetuning, finetuning_max_steps=2,
        finetuning_batch_size=8, finetuning_dropout=0.3,
        retrieval_alpha_adaptive=True, **ARCHITECTURE,
    )


@pytest.mark.parametrize('cls', [iLTMClassifier, iLTMRegressor])
@pytest.mark.parametrize('finetuning', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_public_calls_preserve_rngs_and_match_original_fit(
    monkeypatch, checkpoint, cls, finetuning, device,
):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 6)).astype(np.float32)
    y = np.arange(len(X)) % 2 if cls is iLTMClassifier else rng.normal(size=len(X)).astype(np.float32)
    eval_set = (X[:8], y[:8])

    # Run the original entry points to verify that the learning computation
    # remains identical, including initialization, low-rank PCA and dropout.
    with monkeypatch.context() as original:
        original.setattr(interface._iLTMBase, '__init__', interface._iLTMBase.__init__.__wrapped__)
        original.setattr(cls, 'fit', cls.fit.__wrapped__)
        original.setattr(cls, '_model_cache', {})
        baseline = make_estimator(cls, checkpoint, device, finetuning)
        baseline.fit(X, y, eval_set=eval_set)
        expected = baseline.predict_proba(X[:7]) if cls is iLTMClassifier else baseline.predict(X[:7])

    monkeypatch.setattr(cls, '_model_cache', {})
    before = snapshot()
    model = make_estimator(cls, checkpoint, device, finetuning)
    assert_restored(before)
    assert model.fit(X, y, eval_set=eval_set) is model
    assert_restored(before)
    if cls is iLTMClassifier:
        np.testing.assert_array_equal(model.predict_proba(X[:7]), expected)
        assert_restored(before)
        model.voting = 'hard'
    else:
        np.testing.assert_array_equal(model.predict(X[:7]), expected)
    model.predict(X[:7])
    assert_restored(before)


@pytest.mark.parametrize('cls', [iLTMClassifier, iLTMRegressor])
def test_failed_fit_preserves_caller_state(monkeypatch, checkpoint, cls):
    model = make_estimator(cls, checkpoint, 'cpu', True)

    def fail(**kwargs):
        raise ValueError('test failure')

    monkeypatch.setattr(interface, 'fine_tune_main_network', fail)
    before = snapshot()
    with pytest.raises(ValueError, match='test failure'):
        model.fit(np.arange(96, dtype=np.float32).reshape(16, 6), np.arange(16) % 2)
    assert_restored(before)
