"""Random streams shared by the operations in one estimator fit."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import torch
from torch.nn import functional as F


class _FitGenerators:
    def __init__(self, seed: int, rf_storage_dtype: torch.dtype | None):
        self.rf_storage_dtype = rf_storage_dtype
        self.seed = seed
        self.numpy = np.random.RandomState(seed)  # scikit-learn accepts RandomState
        self.by_device: dict[torch.device, torch.Generator] = {}


# Keeping the streams in the call context also keeps them out of cached models
# and serialized predictors. Standalone torch utilities retain their usual RNG.
_active: ContextVar[_FitGenerators | None] = ContextVar("iltm_fit_generators", default=None)


@contextmanager
def fit_random_generators(seed: int, *, rf_storage_dtype: torch.dtype | None = None):
    token = _active.set(_FitGenerators(seed, rf_storage_dtype))
    try:
        yield
    finally:
        _active.reset(token)


def rf_storage_dtype() -> torch.dtype | None:
    generators = _active.get()
    return generators.rf_storage_dtype if generators is not None else None


def numpy_random_state() -> np.random.RandomState | None:
    generators = _active.get()
    return generators.numpy if generators is not None else None


def torch_generator(device: torch.device | str = "cpu") -> torch.Generator | None:
    generators = _active.get()
    if generators is None:
        return None
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device not in generators.by_device:
        generators.by_device[device] = torch.Generator(device=device).manual_seed(generators.seed)
    return generators.by_device[device]


def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training or not 0.0 < p < 1.0:
        return F.dropout(x, p=p, training=training)
    generator = torch_generator(x.device)
    if generator is None:
        return F.dropout(x, p=p, training=True)
    mask = torch.empty_like(x, dtype=torch.bool).bernoulli_(1.0 - p, generator=generator)
    return (x * mask).div_(1.0 - p)
