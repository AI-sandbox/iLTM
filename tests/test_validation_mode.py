from __future__ import annotations

import pytest
import torch

import iltm.utils as iltm_utils


class _ModeTrackingModel(torch.nn.Module):
    latest: "_ModeTrackingModel | None" = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.norm = torch.nn.BatchNorm1d(1)
        self.initial_transformations_finetuning = True
        self.training_batches = 0
        self.training_modes = []
        type(self).latest = self

    def forward(
        self,
        X,
        X_ctxt_superset=None,
        y_ctxt_superset=None,
        training=False,
    ):
        if training:
            self.training_batches += 1
            self.training_modes.append(self.training)
            return self.weight.expand(len(X))
        return self.weight.detach().expand(len(X))

    def get_main_network_parts(self):
        return {
            "rf": None,
            "pca": None,
            "main_network": [],
            "norm": self.norm,
            "training_modes": list(self.training_modes),
            "export_training": self.training,
        }


class _FailingValidationModel(_ModeTrackingModel):
    def forward(self, X, *args, training=False, **kwargs):
        if not training and self.training_batches >= 2:
            raise RuntimeError("intentional validation failure")
        return super().forward(X, *args, training=training, **kwargs)


def _run_finetuning(monkeypatch, model_class):
    monkeypatch.setattr(iltm_utils, "MainNetworkTrainable", model_class)
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
        max_epochs=2,
        batch_size=1,
        finetuning_optimizer="adamw",
        finetuning_lr=1e-4,
        finetuning_data="entire_dataset",
        finetuning_dropout=0.0,
        X_val=torch.zeros(4, 1),
        y_val=torch.zeros(4),
        finetuning_val_frac=0.0,
        early_stopping_mode="auto",
        patience_epochs=1,
        patience_checks=100,
        val_checks_per_epoch_target=4,
        max_train_batches_per_epoch=5,
        finetuning_subset_frac=None,
        finetuning_subset_max_samples=None,
        val_max_samples=None,
        min_epochs=0,
        cooldown_checks=0,
        classification_val_metric="auto",
        fit_deadline=None,
    )


def test_validation_restores_training_and_exports_eval_mode(monkeypatch):
    result = _run_finetuning(monkeypatch, _ModeTrackingModel)

    assert result["training_modes"] == [True] * 10
    assert result["export_training"] is False
    assert result["norm"].training is False


def test_validation_exception_restores_training_mode(monkeypatch):
    with pytest.raises(RuntimeError, match="intentional validation failure"):
        _run_finetuning(monkeypatch, _FailingValidationModel)

    assert _FailingValidationModel.latest is not None
    assert _FailingValidationModel.latest.training is True


def test_no_validation_keeps_finetuned_weights(monkeypatch):
    monkeypatch.setattr(
        iltm_utils,
        "MainNetworkTrainable",
        _ModeTrackingModel,
    )
    X = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    iltm_utils.fine_tune_main_network(
        cfg={},
        X=X,
        y=torch.ones(4),
        n_classes=1,
        rf=None,
        pca=None,
        main_network=[],
        norm=None,
        device=torch.device("cpu"),
        max_epochs=1,
        batch_size=2,
        finetuning_lr=0.1,
        X_val=None,
        y_val=None,
        finetuning_val_frac=0.0,
        finetuning_subset_max_samples=None,
        val_max_samples=None,
    )

    assert _ModeTrackingModel.latest is not None
    assert _ModeTrackingModel.latest.training_batches == 2
    assert _ModeTrackingModel.latest.weight.item() > 0.05
