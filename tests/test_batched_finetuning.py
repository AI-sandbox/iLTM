from __future__ import annotations

import logging

import iltm.utils as iltm_utils
import pytest
import torch
from iltm.utils import _BatchedTensorDataset, _keep_tensor_batch
from torch.utils.data import DataLoader, Subset, TensorDataset


@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_batched_loading_preserves_values_order_and_rng(subset, shuffle, batch_size):
    X = torch.arange(84, dtype=torch.float32).reshape(12, 7).T
    y = torch.arange(7)
    row_ids = y + 101
    reference = TensorDataset(X, y, row_ids)
    candidate = _BatchedTensorDataset(X, y, row_ids)
    assert candidate.tensors[0] is X
    if subset:
        indices = torch.tensor([4, 1, 4, 6, 0])
        reference = Subset(reference, indices)
        candidate = Subset(candidate, indices)
    torch.manual_seed(177)
    expected = list(DataLoader(reference, batch_size=batch_size, shuffle=shuffle))
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(177)
    actual = list(DataLoader(candidate, batch_size=batch_size, shuffle=shuffle, collate_fn=_keep_tensor_batch))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        for actual_tensor, expected_tensor in zip(actual_batch, expected_batch, strict=True):
            assert torch.equal(actual_tensor, expected_tensor)
            assert actual_tensor.shape == expected_tensor.shape
            assert actual_tensor.dtype == expected_tensor.dtype


class _SmallModel(torch.nn.Module):
    def __init__(self, cfg, n_classes, *args, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(3, n_classes)
        self.initial_transformations_finetuning = True

    def forward(self, X, *args, **kwargs):
        return self.layer(X)

    def get_main_network_parts(self):
        return {}


def test_validation_loss_preserves_precision_and_sum_of_batch_means(monkeypatch, caplog):
    losses = iter([16777216.0, 1.0, 1.0])
    batch_sizes = []

    class FixedLoss:
        def __call__(self, output, target):
            batch_sizes.append(len(target))
            return torch.tensor(next(losses), dtype=torch.float32)

    monkeypatch.setattr(iltm_utils, "MainNetworkTrainable", _SmallModel)
    monkeypatch.setattr(iltm_utils.nn, "CrossEntropyLoss", FixedLoss)
    caplog.set_level(logging.DEBUG, logger=iltm_utils.logger.name)
    X = torch.arange(57, dtype=torch.float32).reshape(19, 3) / 30
    y = torch.arange(len(X)) % 3
    iltm_utils.fine_tune_main_network(
        cfg={},
        X=X,
        y=y,
        n_classes=3,
        rf=None,
        pca=None,
        main_network=[],
        norm=None,
        device=torch.device("cpu"),
        max_epochs=0,
        batch_size=3,
        X_val=X[:8],
        y_val=y[:8],
        val_max_samples=None,
        classification_val_metric="logloss",
    )
    assert batch_sizes == [3, 3, 2]
    assert "Training done | best LogLoss=16777218.000000" in caplog.messages
