from __future__ import annotations

import hashlib
from typing import NamedTuple

import torch
from torch import nn

from ._random import rf_storage_dtype, torch_generator


def _initialize(weight: torch.Tensor, generator: torch.Generator | None) -> None:
    nn.init.kaiming_normal_(
        weight, mode="fan_out", nonlinearity="relu", generator=generator
    )


def _weight_digest(weight: torch.Tensor, dtype: torch.dtype) -> bytes:
    digest = hashlib.sha256()
    # Bound the temporary allocation when hashing in inference storage precision.
    for block in weight.detach().view(-1).split(4 * 1024 * 1024):
        digest.update(block.to(dtype).view(torch.uint8).numpy())
    return digest.digest()


class _RFInitialization(NamedTuple):
    rng_state: bytes
    init_dtype: torch.dtype
    storage_dtype: torch.dtype
    digest: bytes


class RandomFeatureLinear(nn.Linear):
    """Store an unchanged CPU random projection by its initialization RNG state."""

    @classmethod
    def initialize(cls, in_features: int, out_features: int) -> RandomFeatureLinear:
        layer = cls(in_features, out_features, bias=False, device="meta").to_empty(
            device="cpu"
        )
        weight = layer.weight.requires_grad_(False)
        generator = torch_generator()
        state = generator.get_state() if generator is not None else None
        _initialize(weight, generator)
        dtype = rf_storage_dtype() or weight.dtype
        if state is not None and weight.numel() * dtype.itemsize > state.numel() + 128:
            layer._initial_rf = _RFInitialization(
                state.numpy().tobytes(),
                weight.dtype,
                dtype,
                _weight_digest(weight, dtype),
            )
        return layer

    def __getstate__(self):
        state = super().__getstate__()
        initial = state.pop("_initial_rf", None)
        weight = self.weight
        if (
            initial is not None
            and weight.device.type == "cpu"
            and weight.dtype == initial.storage_dtype
            and weight.shape == (self.out_features, self.in_features)
            and weight.is_contiguous()
            and _weight_digest(weight, weight.dtype) == initial.digest
        ):
            state["_initial_rf"] = initial
            state["_rf_requires_grad"] = weight.requires_grad
            state["_rf_runtime"] = (
                str(torch.__version__),
                torch.backends.cpu.get_cpu_capability(),
            )
            state["_parameters"] = {**state["_parameters"], "weight": None}
        return state

    def __setstate__(self, state):
        requires_grad = state.pop("_rf_requires_grad", None)
        runtime = state.pop("_rf_runtime", None)
        super().__setstate__(state)
        if requires_grad is None:
            return
        rng_state, init_dtype, dtype, expected_digest = self._initial_rf
        generator = torch.Generator(device="cpu")
        generator.set_state(torch.frombuffer(bytearray(rng_state), dtype=torch.uint8))
        weight = torch.empty(
            (self.out_features, self.in_features), device="cpu", dtype=init_dtype
        )
        _initialize(weight, generator)
        weight = weight.to(dtype)
        if _weight_digest(weight, dtype) != expected_digest:
            raise RuntimeError(
                "This PyTorch runtime did not reproduce the saved iLTM random feature weights. "
                f"Load with the original PyTorch version and CPU capability ({runtime[0]}, {runtime[1]})."
            )
        self.weight = nn.Parameter(weight, requires_grad=requires_grad)
