from __future__ import annotations

import pytest
import torch
from iltm import utils


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("indices", [[6, 2, 2, 0, 4], [], [0, 1, 2, 3, 4, 5, 6]])
def test_context_batch_gather_preserves_rows_ids_layout_and_rng(device_name, regression, indices):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device_name)
    X = torch.arange(84, dtype=torch.float32).reshape(12, 7).T
    targets = torch.arange(14)[::2]
    if regression:
        targets = targets.to(torch.float64) / 3
    original_ids = torch.tensor([140, 73, 73, 109, 3, 140, 28])
    source = (X, targets, original_ids)
    context = tuple(tensor.to(device) for tensor in source)
    positions = torch.tensor(indices, dtype=torch.long)
    if device_name == "cuda":
        positions = positions.pin_memory()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device_name == "cuda" else None
    actual = utils._gather_tensor_batch(context, positions, device)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    for reference, observed, full_context in zip(source, actual, context, strict=True):
        expected = reference.index_select(0, torch.tensor(indices, dtype=torch.long)).to(device)
        assert torch.equal(observed, expected)
        assert observed.shape == expected.shape
        assert observed.dtype == expected.dtype
        assert observed.device == expected.device
        assert observed.is_contiguous()
        if observed.numel():
            assert observed.untyped_storage().data_ptr() != full_context.untyped_storage().data_ptr()
    assert torch.equal(actual[-1], original_ids[positions.cpu()].to(device))


@pytest.mark.parametrize("pinned", [False, True])
def test_context_gather_waits_for_all_copies_before_return(monkeypatch, pinned):
    events = []
    transferred_indices = object()

    class Indices:
        def is_pinned(self):
            return pinned

        def to(self, device, *, non_blocking):
            events.append(("copy_indices", device.type, non_blocking))
            return transferred_indices

    class ContextTensor:
        def __init__(self, name):
            self.name = name

        def index_select(self, dim, indices):
            assert dim == 0
            assert indices is transferred_indices
            events.append(("gather", self.name))
            return self

    class Stream:
        def synchronize(self):
            events.append("synchronize")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    tensors = tuple(ContextTensor(name) for name in ["features", "targets", "original_ids"])
    assert utils._gather_tensor_batch(tensors, Indices(), torch.device("cuda")) == tensors
    assert events == [("copy_indices", "cuda", pinned)] + [
        ("gather", name) for name in ["features", "targets", "original_ids"]
    ] + ["synchronize"]
