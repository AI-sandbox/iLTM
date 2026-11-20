from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
import torch
from sklearn.base import clone

from iltm import iLTMClassifier, iLTMRegressor


@pytest.fixture
def matmul():
    backend = torch.backends.cuda.matmul
    if not hasattr(backend, "fp32_precision"):
        pytest.skip("PyTorch lacks the CUDA matmul precision API")
    with torch.backends.flags(fp32_precision="none"):
        cuda_precision = torch.backends.cudnn.fp32_precision
        torch.backends.cudnn.fp32_precision = "none"
        previous = backend.fp32_precision
        try:
            yield backend
        finally:
            torch.backends.fp32_precision = "none"
            backend.fp32_precision = previous
            torch.backends.cudnn.fp32_precision = cuda_precision


@pytest.mark.parametrize("cls", [iLTMClassifier, iLTMRegressor])
@pytest.mark.parametrize(
    ("device", "allow_tf32", "caller_precision", "expected"),
    [
        ("cuda:0", True, "none", "tf32"),
        ("cuda:0", False, "tf32", "ieee"),
        ("cuda:0", None, "tf32", "tf32"),
        ("cpu", True, "ieee", "ieee"),
    ],
)
def test_tf32_is_scoped_to_fit_and_predict(monkeypatch, matmul, cls, device, allow_tf32, caller_precision, expected):
    matmul.fp32_precision = caller_precision
    model = clone(cls(checkpoint=None, device=device, allow_tf32=allow_tf32, retrieval_alpha_adaptive=False))
    assert model.allow_tf32 is allow_tf32
    assert matmul.fp32_precision == caller_precision
    X = np.zeros((2, 1), dtype=np.float32)
    y = np.array([0, 1])

    def fit_common(*args, **kwargs):
        assert matmul.fp32_precision == expected
        model.normalize_predictions_ = False
        model.clip_predictions_ = False
        return model

    monkeypatch.setattr(model, "_fit_common", fit_common)
    assert model.fit(X, y) is model
    assert matmul.fp32_precision == caller_precision

    def predict_ensemble(*args, **kwargs):
        assert matmul.fp32_precision == expected
        return torch.zeros((len(X), 2) if cls is iLTMClassifier else len(X))

    methods = [model.predict, model.predict_proba] if cls is iLTMClassifier else [model.predict]
    monkeypatch.setattr(model, "_predict_ensemble", predict_ensemble)
    for predict in methods:
        assert len(predict(X)) == len(X)
        assert matmul.fp32_precision == caller_precision

    def fail_prediction(*args, **kwargs):
        assert matmul.fp32_precision == expected
        raise ValueError("prediction failed")

    monkeypatch.setattr(model, "_predict_ensemble", fail_prediction)
    for predict in methods:
        with pytest.raises(ValueError, match="prediction failed"):
            predict(X)
        assert matmul.fp32_precision == caller_precision


def test_tf32_requires_precision_api_only_when_requested_on_cuda(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda, "matmul", object())
    Xt, yt = np.zeros((2, 1), dtype=np.float32), np.array([0, 1])
    model = iLTMClassifier(checkpoint=None, device="cuda:0")
    monkeypatch.setattr(model, "_fit_common", lambda *args, **kwargs: model)
    assert model.fit(Xt, yt) is model

    model.set_params(allow_tf32=True)
    with pytest.raises(RuntimeError, match="requires PyTorch 2.9 or newer"):
        model.fit(Xt, yt)


@pytest.mark.parametrize("parent_name", ["generic", "cuda"])
@pytest.mark.parametrize("override", ["none", "ieee"])
def test_fit_preserves_precision_inheritance(monkeypatch, matmul, parent_name, override):
    parent = torch.backends if parent_name == "generic" else torch.backends.cudnn
    parent.fp32_precision = "ieee"
    matmul.fp32_precision = override
    model = iLTMClassifier(checkpoint=None, device="cuda:0", allow_tf32=True, retrieval_alpha_adaptive=False)

    def fit_common(*args, **kwargs):
        assert matmul.fp32_precision == "tf32"
        assert parent.fp32_precision == "ieee"
        return model

    monkeypatch.setattr(model, "_fit_common", fit_common)
    model.fit(np.zeros((2, 1), dtype=np.float32), np.array([0, 1]))
    assert matmul.fp32_precision == "ieee"
    parent.fp32_precision = "tf32"
    assert matmul.fp32_precision == ("tf32" if override == "none" else "ieee")
    if parent_name == "generic":
        assert torch.backends.cudnn.fp32_precision == "tf32"


def test_inheriting_fit_waits_for_tf32_override(monkeypatch, matmul):
    matmul.fp32_precision = "ieee"
    X, y = np.zeros((2, 1), dtype=np.float32), np.array([0, 1])
    explicit = iLTMClassifier(checkpoint=None, device="cuda:0", allow_tf32=True, retrieval_alpha_adaptive=False)
    inheriting = iLTMClassifier(checkpoint=None, device="cuda:0", retrieval_alpha_adaptive=False)
    override_active = Event()
    release_override = Event()
    inheriting_started = Event()
    inheriting_entered = Event()

    def fit_explicit(*args, **kwargs):
        assert matmul.fp32_precision == "tf32"
        override_active.set()
        assert release_override.wait(timeout=5)
        return explicit

    def fit_inheriting(*args, **kwargs):
        inheriting_entered.set()
        assert matmul.fp32_precision == "ieee"
        return inheriting

    def run_inheriting():
        inheriting_started.set()
        return inheriting.fit(X, y)

    monkeypatch.setattr(explicit, "_fit_common", fit_explicit)
    monkeypatch.setattr(inheriting, "_fit_common", fit_inheriting)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(explicit.fit, X, y)
        try:
            assert override_active.wait(timeout=5)
            second = executor.submit(run_inheriting)
            assert inheriting_started.wait(timeout=5)
            assert not inheriting_entered.wait(timeout=0.2)
        finally:
            release_override.set()
        assert first.result(timeout=5) is explicit
        assert second.result(timeout=5) is inheriting
    assert matmul.fp32_precision == "ieee"
