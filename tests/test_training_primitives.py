from __future__ import annotations

import pytest
import torch
from iltm.utils import (
    _BatchedTensorDataset,
    _BatchedTensorSubset,
    _clip_grad_norm,
    _keep_tensor_batch,
    _move_tensor_batch,
    sample_data,
)
from torch.utils.data import DataLoader, Subset, TensorDataset


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_vectorized_subset_matches_scalar_loading_and_rng(shuffle, batch_size):
    X = torch.arange(84, dtype=torch.float32).reshape(12, 7).T
    y = torch.arange(7)
    indices = torch.tensor([4, 1, 4, -1, 0])
    reference = Subset(TensorDataset(X, y, y + 100), indices)
    candidate = _BatchedTensorSubset(_BatchedTensorDataset(X, y, y + 100), indices)
    torch.manual_seed(177)
    expected = list(DataLoader(reference, batch_size=batch_size, shuffle=shuffle))
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(177)
    actual = list(DataLoader(candidate, batch_size=batch_size, shuffle=shuffle, collate_fn=_keep_tensor_batch))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for expected_batch, actual_batch in zip(expected, actual, strict=True):
        for expected_tensor, actual_tensor in zip(expected_batch, actual_batch, strict=True):
            assert torch.equal(expected_tensor, actual_tensor)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("return_rest", [False, True])
def test_sampling_preserves_selected_rows_returned_indices_and_rng(device_name, as_dict, return_rest):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    X = torch.arange(161, dtype=torch.float32).reshape(7, 23).T.to(device_name)
    y = torch.arange(len(X), device=device_name)
    inputs = {"x_num": X, "x_cat": X[:, :2].long()} if as_dict else X
    torch.manual_seed(177)
    indices = torch.randperm(len(y))[:9]
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(177)
    actual = sample_data(
        inputs,
        y,
        to_meta_model=False,
        batch_size=9,
        return_rest=return_rest,
        return_sampled_indices=not return_rest,
    )
    assert torch.equal(torch.get_rng_state(), expected_rng)
    actual_X, actual_y = actual[:2]
    expected_tensors = inputs if as_dict else {"X": inputs}
    actual_tensors = actual_X if as_dict else {"X": actual_X}
    for key, tensor in expected_tensors.items():
        assert torch.equal(actual_tensors[key], tensor[indices])
    assert torch.equal(actual_y, y[indices])
    if return_rest:
        mask = torch.ones(len(y), dtype=torch.bool)
        mask[indices] = False
        rest_X, rest_y = actual[2:]
        rest_tensors = rest_X if as_dict else {"X": rest_X}
        for key, tensor in expected_tensors.items():
            assert torch.equal(rest_tensors[key], tensor[mask])
        assert torch.equal(rest_y, y[mask])
    else:
        assert torch.equal(actual[2], indices)
        assert actual[2].device.type == "cpu"


@pytest.mark.parametrize("sampling", ["repeat", "bootstrap", "zeropad"])
def test_sampling_extension_preserves_values_and_rng(sampling):
    X = torch.arange(45, dtype=torch.float32).reshape(9, 5)
    y = torch.arange(len(X))
    torch.manual_seed(181)
    indices = torch.randperm(len(y))[:4]
    expected_X, expected_y = X[indices], y[indices]
    if sampling == "repeat":
        expected_X = torch.repeat_interleave(expected_X, 2, dim=0)
        expected_y = torch.repeat_interleave(expected_y, 2, dim=0)
    elif sampling == "bootstrap":
        extra = torch.randint(0, 4, (3,))
        expected_X = torch.cat((expected_X, expected_X[extra]), dim=0)
        expected_y = torch.cat((expected_y, expected_y[extra]), dim=0)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(181)
    actual_X, actual_y = sample_data(X, y, batch_size=4, min_samples=7, pca_sampling=sampling)
    assert torch.equal(expected_X, actual_X)
    assert torch.equal(expected_y, actual_y)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("scenario", ["float32", "float64", "mixed", "zeros", "nan", "inf", "missing"])
def test_grouped_gradient_scaling_preserves_norm_gradients_and_rng(device_name, scenario):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(37)
    parameters, copies = [], []
    for index in range(5):
        dtype = torch.float64 if scenario == "float64" or (scenario == "mixed" and index % 2) else torch.float32
        value = torch.randn((17, 7), dtype=dtype, device=device_name).T
        parameter = torch.nn.Parameter(value)
        parameter.grad = torch.zeros_like(parameter) if scenario == "zeros" else torch.randn_like(parameter)
        if scenario in ("nan", "inf"):
            parameter.grad[0, 0] = float(scenario)
        if scenario == "missing" and index % 2:
            parameter.grad = None
        copy = torch.nn.Parameter(value.clone())
        copy.grad = parameter.grad.clone() if parameter.grad is not None else None
        parameters.append(parameter)
        copies.append(copy)
    rng_before = torch.get_rng_state().clone()
    expected_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=False)
    actual_norm = _clip_grad_norm(copies, 1.0)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0, equal_nan=True)
    for parameter, copy in zip(parameters, copies, strict=True):
        if parameter.grad is None:
            assert copy.grad is None
        else:
            torch.testing.assert_close(copy.grad, parameter.grad, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
def test_grouped_transfer_preserves_synchronous_return(cuda, pinned, monkeypatch):
    events = []

    class FakeTensor:
        def __init__(self, name):
            self.name = name

        def is_pinned(self):
            return pinned

        def to(self, device, **kwargs):
            events.append((self.name, kwargs))
            return self

    class FakeStream:
        def synchronize(self):
            events.append("synchronize")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: FakeStream())
    tensors = (FakeTensor("inputs"), FakeTensor("targets"), FakeTensor("ids"))
    device = torch.device("cuda" if cuda else "cpu")
    assert _move_tensor_batch(tensors, device) == tensors
    if cuda and pinned:
        assert events[-1] == "synchronize"
        assert events[:-1] == [(name, {"non_blocking": True}) for name in ("inputs", "targets", "ids")]
    else:
        assert events == [(name, {}) for name in ("inputs", "targets", "ids")]


def test_cpu_transfer_preserves_values_and_tensor_identity():
    batch = (torch.randn(7, 3), torch.arange(7), torch.arange(7) + 100)
    actual = _move_tensor_batch(batch, torch.device("cpu"))
    assert all(expected is observed for expected, observed in zip(batch, actual, strict=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_grouped_cuda_transfer_matches_blocking_copy():
    tensors = (torch.randn(3, 7).T.pin_memory(), torch.arange(7).pin_memory(), torch.arange(7).pin_memory())
    device = torch.device("cuda")
    expected = tuple(tensor.to(device) for tensor in tensors)
    actual = _move_tensor_batch(tensors, device)
    for before, after in zip(expected, actual, strict=True):
        assert torch.equal(before, after)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_fine_tuning_preserves_gradient_updates_and_best_validation_state(device_name, monkeypatch):
    from iltm import utils

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    class ConstantRegressor(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))
            self.initial_transformations_finetuning = True

        def forward(self, X, *args, **kwargs):
            return self.bias * torch.ones(len(X), device=self.bias.device)

        def get_main_network_parts(self):
            return {"bias": self.bias.detach().clone()}

    device = torch.device(device_name)
    X = torch.zeros(7, 3)
    y = torch.full((7,), 2.0)
    expected = ConstantRegressor().to(device)
    optimizer = torch.optim.SGD([{"params": list(expected.parameters()), "lr": 0.1, "weight_decay": 0.01}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)
    criterion = torch.nn.MSELoss()
    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(expected(X.to(device)), y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0, foreach=False)
        optimizer.step()
        scheduler.step()
    monkeypatch.setattr(utils, "MainNetworkTrainable", ConstantRegressor)
    actual = utils.fine_tune_main_network(
        cfg={},
        X=X,
        y=y,
        n_classes=1,
        rf=None,
        pca=None,
        main_network=[],
        norm=None,
        device=device,
        max_epochs=3,
        batch_size=len(X),
        finetuning_optimizer="sgd",
        finetuning_lr=0.1,
        X_val=X,
        y_val=y,
        val_max_samples=None,
        early_stopping_mode="epoch",
    )
    assert actual["timed_out"] is False
    assert torch.equal(actual["bias"], expected.bias.detach())
