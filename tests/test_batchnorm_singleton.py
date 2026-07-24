from __future__ import annotations

import pytest
import torch


class _IdentityPCA:
    def transform(self, X):
        return X


class _RecordingBatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.calls: list[tuple[int, bool]] = []

    def forward(self, X):
        self.calls.append((len(X), self.training))
        return super().forward(X)


class _RaisingBatchNorm1d(torch.nn.BatchNorm1d):
    def forward(self, X):
        assert not self.training
        raise RuntimeError("intentional singleton normalization failure")


class _BatchNormTaskModel(torch.nn.Module):
    def __init__(self, cfg, n_classes, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        self.initial_transformations_finetuning = True
        self.rf = torch.nn.Identity()
        self.pca = _IdentityPCA()
        self.norm = torch.nn.BatchNorm1d(2)
        self.head = torch.nn.Linear(2, n_classes if n_classes > 1 else 1)
        self.training_batch_sizes = []
        self.training_rows = []
        self.norm_modes_after_training_forward = []

    def forward(
        self,
        X,
        X_ctxt_superset=None,
        y_ctxt_superset=None,
        training=False,
    ):
        from iltm.utils import transform_data_for_main_network

        if training:
            self.training_batch_sizes.append(len(X))
            self.training_rows.extend(int(value) for value in X[:, 0])
        X = transform_data_for_main_network(
            X,
            self.cfg,
            X.device,
            self.rf,
            self.pca,
            self.norm,
            training_finetuning=training,
        )
        if training:
            self.norm_modes_after_training_forward.append(self.norm.training)
        outputs = self.head(X)
        return outputs if self.n_classes > 1 else outputs.squeeze(1)

    def get_main_network_parts(self):
        return {
            "norm": self.norm,
            "export_training": self.training,
            "training_batch_sizes": list(self.training_batch_sizes),
            "training_rows": list(self.training_rows),
            "norm_modes_after_training_forward": list(
                self.norm_modes_after_training_forward
            ),
        }


def _cfg():
    return {
        "pca_sampling": "repeat",
        "n_dims": 2,
        "clip_data_value": 100.0,
    }


@pytest.mark.parametrize(
    "n_classes",
    [1, 2, 3],
    ids=["regression", "binary", "multiclass"],
)
def test_finetuning_processes_final_singleton_for_each_task(
    monkeypatch, n_classes
):
    import iltm.utils as iltm_utils

    monkeypatch.setattr(
        iltm_utils,
        "MainNetworkTrainable",
        _BatchNormTaskModel,
    )
    X = torch.column_stack(
        [
            torch.arange(5, dtype=torch.float32),
            torch.arange(10, 15, dtype=torch.float32),
        ]
    )
    if n_classes == 1:
        y = torch.arange(5, dtype=torch.float32)
        y_val = torch.arange(4, dtype=torch.float32)
    else:
        y = torch.arange(5) % n_classes
        y_val = torch.arange(4) % n_classes
    X_val = torch.column_stack(
        [
            torch.arange(4, dtype=torch.float32),
            torch.arange(20, 24, dtype=torch.float32),
        ]
    )

    result = iltm_utils.fine_tune_main_network(
        cfg=_cfg(),
        X=X,
        y=y,
        n_classes=n_classes,
        rf=None,
        pca=None,
        main_network=[],
        norm=None,
        device=torch.device("cpu"),
        max_epochs=1,
        batch_size=2,
        finetuning_optimizer="adamw",
        finetuning_lr=1e-4,
        finetuning_data="entire_dataset",
        X_val=X_val,
        y_val=y_val,
        early_stopping_mode="epoch",
        patience_epochs=10,
        finetuning_subset_max_samples=None,
        val_max_samples=None,
    )

    assert sorted(result["training_batch_sizes"]) == [1, 2, 2]
    assert sorted(result["training_rows"]) == list(range(5))
    assert result["norm_modes_after_training_forward"] == [True, True, True]
    assert result["export_training"] is False
    assert result["norm"].training is False


def test_singleton_norm_mode_is_restored_when_normalization_raises():
    from iltm.utils import transform_data_for_main_network

    norm = _RaisingBatchNorm1d(2)
    norm.train()
    with pytest.raises(
        RuntimeError,
        match="intentional singleton normalization failure",
    ):
        transform_data_for_main_network(
            torch.ones(1, 2),
            _cfg(),
            torch.device("cpu"),
            torch.nn.Identity(),
            _IdentityPCA(),
            norm,
            training_finetuning=True,
        )
    assert norm.training is True


def test_singleton_uses_running_stats_and_keeps_affine_gradients():
    from iltm.utils import transform_data_for_main_network

    norm = torch.nn.BatchNorm1d(2)
    norm.running_mean.copy_(torch.tensor([1.0, 2.0]))
    norm.running_var.copy_(torch.tensor([4.0, 9.0]))
    norm.weight.data.copy_(torch.tensor([2.0, 3.0]))
    norm.bias.data.copy_(torch.tensor([-1.0, 1.0]))
    norm.train()
    X = torch.tensor([[5.0, 8.0]])
    expected = (
        (X - norm.running_mean) / torch.sqrt(norm.running_var + norm.eps)
    ) * norm.weight + norm.bias
    running_mean = norm.running_mean.clone()
    running_var = norm.running_var.clone()
    batches_tracked = norm.num_batches_tracked.clone()

    output = transform_data_for_main_network(
        X,
        _cfg(),
        torch.device("cpu"),
        torch.nn.Identity(),
        _IdentityPCA(),
        norm,
        training_finetuning=True,
    )
    output.sum().backward()

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(norm.running_mean, running_mean)
    torch.testing.assert_close(norm.running_var, running_var)
    torch.testing.assert_close(norm.num_batches_tracked, batches_tracked)
    assert norm.weight.grad is not None
    assert norm.bias.grad is not None
    assert norm.training is True


def test_retrieval_uses_eval_for_singleton_query_and_train_for_context():
    from iltm.utils import full_main_forward

    norm = _RecordingBatchNorm1d(2)
    norm.train()
    main_network = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)]
    )
    query = torch.tensor([[100.0, 200.0]])
    context = torch.tensor(
        [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]
    )
    targets = torch.tensor([0.0, 1.0, 2.0, 3.0])

    outputs = full_main_forward(
        query,
        n_classes=1,
        batch_size=4,
        model_cfg=_cfg(),
        rf=torch.nn.Identity(),
        pca=_IdentityPCA(),
        norm=norm,
        main_network=main_network,
        device=torch.device("cpu"),
        use_amp=False,
        do_retrieval=True,
        X_ctxt_superset=context,
        y_ctxt_superset=targets,
        retrieval_alpha=0.5,
        retrieval_temperature=1.0,
        retrieval_distance="cosine",
        training_finetuning=True,
    )

    assert outputs.shape == (1,)
    assert torch.isfinite(outputs).all()
    assert norm.calls == [(1, False), (4, True)]
    assert norm.training is True
    assert norm.num_batches_tracked.item() == 1
    torch.testing.assert_close(norm.running_mean, 0.1 * context.mean(dim=0))
