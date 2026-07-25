from __future__ import annotations

import pytest
import torch

import iltm.utils as iltm_utils


class _PlateauModel(torch.nn.Module):
    """Minimal trainable model with a constant validation metric."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.initial_transformations_finetuning = True
        self.training_batches = 0

    def forward(
        self,
        X,
        X_ctxt_superset=None,
        y_ctxt_superset=None,
        training=False,
    ):
        if training:
            self.training_batches += 1
            return self.weight.expand(len(X))
        return self.weight.detach().expand(len(X))

    def get_main_network_parts(self):
        return {
            "rf": None,
            "pca": None,
            "main_network": [],
            "norm": None,
            "training_batches": self.training_batches,
        }


def _run_plateau(
    monkeypatch,
    *,
    cap: int | None,
    patience_epochs: int = 3,
    patience_checks: int | None = None,
    subset_max_samples: int | None = None,
):
    monkeypatch.setattr(iltm_utils, "MainNetworkTrainable", _PlateauModel)
    X = torch.arange(9, dtype=torch.float32).reshape(-1, 1)
    return iltm_utils.fine_tune_main_network(
        cfg={},
        X=X,
        y=torch.zeros(9),
        n_classes=1,
        rf=None,
        pca=None,
        main_network=[],
        norm=None,
        device=torch.device("cpu"),
        max_epochs=20,
        batch_size=1,
        finetuning_optimizer="adamw",
        finetuning_lr=1e-4,
        finetuning_data="entire_dataset",
        finetuning_dropout=0.0,
        X_val=torch.zeros(4, 1),
        y_val=torch.zeros(4),
        finetuning_val_frac=0.0,
        early_stopping_mode="auto",
        patience_epochs=patience_epochs,
        patience_checks=patience_checks,
        val_checks_per_epoch_target=4,
        max_train_batches_per_epoch=cap,
        finetuning_subset_frac=None,
        finetuning_subset_max_samples=subset_max_samples,
        val_max_samples=None,
        min_epochs=0,
        cooldown_checks=0,
        classification_val_metric="auto",
        fit_deadline=None,
    )


@pytest.mark.parametrize(("cap", "expected_batches"), [(5, 16), (3, 10)])
def test_binding_epoch_cap_scales_implicit_patience(
    monkeypatch, cap, expected_batches
):
    result = _run_plateau(monkeypatch, cap=cap)
    assert result["training_batches"] == expected_batches


def test_uncapped_patience_preserves_existing_behavior(monkeypatch):
    result = _run_plateau(monkeypatch, cap=None)
    assert result["training_batches"] == 24


@pytest.mark.parametrize("cap", [3, 5, None])
def test_explicit_check_patience_is_cap_invariant(monkeypatch, cap):
    result = _run_plateau(
        monkeypatch,
        cap=cap,
        patience_epochs=99,
        patience_checks=3,
    )
    assert result["training_batches"] == 6


def test_epoch_subset_keeps_full_training_tensors(monkeypatch):
    captured = []

    class _StopLoaderConstruction(Exception):
        pass

    def capture_loader(dataset, *args, **kwargs):
        captured.append(dataset)
        raise _StopLoaderConstruction

    monkeypatch.setattr(iltm_utils, "DataLoader", capture_loader)

    with pytest.raises(_StopLoaderConstruction):
        _run_plateau(
            monkeypatch,
            cap=None,
            subset_max_samples=4,
        )

    dataset = captured[0]
    assert isinstance(dataset, torch.utils.data.Subset)
    assert len(dataset) == 4
    assert len(dataset.dataset) == 9
