import contextlib
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import random

import numpy as np
import pytest
import torch

import iltm.inference_interface as interface
from iltm import iLTMClassifier, iLTMRegressor
from iltm.iltm_model import iLTM
from iltm._random import dropout, fit_random_generators, numpy_random_state, torch_generator
from iltm.utils import TorchPCA, _BatchedTensorSubset


ARCHITECTURE = dict(
    n_dims=4, hn_n_layers=3, hn_hidden_size=8, rf_size=12,
    n_classes_limit=5, main_n_layers=3,
)


def snapshot():
    return (
        random.getstate(), pickle.dumps(np.random.get_state()),
        torch.get_rng_state().clone(), torch.cuda.get_rng_state_all(),
        os.environ.get('PYTHONHASHSEED'), torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
    )


def assert_restored(expected):
    actual = snapshot()
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert len(actual[3]) == len(expected[3])
    assert all(torch.equal(a, b) for a, b in zip(actual[3], expected[3]))
    assert actual[4:] == expected[4:]


@pytest.mark.parametrize('hash_seed', [None, '', '123'])
@pytest.mark.parametrize('fail', [False, True])
def test_context_restores_rngs_and_leaves_hash_seed_alone(monkeypatch, hash_seed, fail):
    if hash_seed is None:
        monkeypatch.delenv('PYTHONHASHSEED', raising=False)
    else:
        monkeypatch.setenv('PYTHONHASHSEED', hash_seed)
    before = snapshot()
    error_context = pytest.raises(ValueError, match='test failure') if fail else contextlib.nullcontext()
    with error_context:
        with fit_random_generators(19):
            assert os.environ.get('PYTHONHASHSEED') == hash_seed
            numpy_random_state().uniform(size=3)
            torch.rand(3, generator=torch_generator())
            if torch.cuda.is_available():
                torch.rand(3, device='cuda', generator=torch_generator('cuda'))
            if fail:
                raise ValueError('test failure')
    assert_restored(before)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / 'small.pth'
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(19)
        torch.save(iLTM(**ARCHITECTURE).state_dict(), path)
    return path


def make_estimator(cls, checkpoint, device, finetuning):
    return cls(
        checkpoint=str(checkpoint), device=device, n_ensemble=2,
        batch_size=16, preprocessing='none', pca_sampling='zeropad', pca_fit='lowrank',
        tree_embedding=False, adaptive_memory=False,
        auto_stop_on_low_cpu_memory=False, inference_storage_dtype='float32',
        finetuning=finetuning, finetuning_max_steps=2,
        finetuning_batch_size=8, finetuning_dropout=0.3,
        retrieval_alpha_adaptive=True, **ARCHITECTURE,
    )


@pytest.mark.parametrize('cls', [iLTMClassifier, iLTMRegressor])
@pytest.mark.parametrize('finetuning', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_public_calls_preserve_rngs_and_match_cached_fit(
    monkeypatch, checkpoint, cls, finetuning, device,
):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 6)).astype(np.float32)
    y = np.arange(len(X)) % 2 if cls is iLTMClassifier else rng.normal(size=len(X)).astype(np.float32)
    eval_set = (X[:8], y[:8])

    monkeypatch.setattr(cls, '_model_cache', {})
    before = snapshot()
    first = make_estimator(cls, checkpoint, device, finetuning)
    assert_restored(before)
    first.fit(X, y, eval_set=eval_set)
    assert_restored(before)
    expected = first.predict_proba(X[:7]) if cls is iLTMClassifier else first.predict(X[:7])
    assert_restored(before)
    assert cls._model_cache
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


def test_concurrent_draws_follow_the_callers_sequence(monkeypatch, checkpoint):
    model = make_estimator(iLTMClassifier, checkpoint, 'cpu', True)
    python_rng = random.Random()
    python_rng.setstate(random.getstate())
    numpy_rng = np.random.RandomState()
    numpy_rng.set_state(np.random.get_state())
    torch_rng = torch.Generator().set_state(torch.get_rng_state())
    expected = [
        (python_rng.random(), numpy_rng.random_sample(), torch.rand((), generator=torch_rng).item())
        for _ in range(3)
    ]

    def draw():
        return random.random(), np.random.random_sample(), torch.rand(()).item()

    tune = model._auto_tune_for_memory
    with ThreadPoolExecutor(max_workers=1) as other_thread:
        during = []

        def draw_during_fit():
            during.append(other_thread.submit(draw).result())
            return tune()

        monkeypatch.setattr(model, '_auto_tune_for_memory', draw_during_fit)
        first = other_thread.submit(draw).result()
        X = np.arange(96, dtype=np.float32).reshape(16, 6)
        model.fit(X, np.arange(16) % 2)
        last = other_thread.submit(draw).result()
    assert [first, *during, last] == expected


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_training_draws_advance_and_repeat(monkeypatch, checkpoint, device):
    X = np.random.default_rng(7).normal(size=(32, 6)).astype(np.float32)
    y = np.arange(len(X)) % 2
    fetch_batch = _BatchedTensorSubset.__getitems__
    monkeypatch.setattr(iLTMClassifier, '_model_cache', {})

    def fit(seed):
        batches, masks = [], []

        def record_batch(dataset, indices):
            batches.append(tuple(dataset.indices[torch.as_tensor(indices)].tolist()))
            return fetch_batch(dataset, indices)

        def record_dropout(x, p, training):
            out = dropout(x, p=p, training=training)
            if training and p > 0 and x.requires_grad:
                # The derivative gives the actual mask, even for zero-valued inputs.
                mask, = torch.autograd.grad(out, x, torch.ones_like(out), retain_graph=True)
                masks.append(mask.detach().ne(0).cpu().numpy().tobytes())
            return out

        model = make_estimator(iLTMClassifier, checkpoint, device, True)
        model.set_params(seed=seed, finetuning_data='bootstrap')
        with monkeypatch.context() as trace:
            trace.setattr(_BatchedTensorSubset, '__getitems__', record_batch)
            trace.setattr('iltm.utils.dropout', record_dropout)
            model.fit(X, y)
        return batches, masks

    first = fit(19)
    assert first == fit(19)
    different = fit(23)
    for sequence, other in zip(first, different):
        assert sequence != other
        assert len(set(sequence)) > 1


@pytest.mark.parametrize("shape", [(31, 9), (9, 31)])
def test_local_lowrank_pca_matches_reference_projection(shape):
    X = torch.randn(*shape, dtype=torch.float64, generator=torch.Generator().manual_seed(42))
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(19)
        U, S, V = torch.svd_lowrank(X, q=4, M=X.mean(0))
    pca = TorchPCA(n_components=4, fit="lowrank")
    before = snapshot()
    with fit_random_generators(19):
        scores = pca.fit_transform(X)
    torch.testing.assert_close(scores @ pca.components_, (U * S) @ V.mT)
    assert_restored(before)
