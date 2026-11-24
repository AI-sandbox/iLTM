from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from iltm import utils
from iltm._random import fit_random_generators, torch_generator


class _RecordingModel(torch.nn.Module):
    def __init__(self, cfg, n_classes, *args, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(5, n_classes)
        self.initial_transformations_finetuning = True
        self.training_ids = []
        self.validation_ids = []
        self.context_ids = None

    def forward(self, inputs, *args, training=False, **kwargs):
        row_ids = inputs[:, 0].long()
        if training:
            self.training_ids.append(row_ids.detach().cpu())
            if "query_ids" in kwargs:
                assert torch.equal(kwargs["query_ids"], row_ids)
                if self.context_ids is None:
                    self.context_ids = kwargs["context_ids_superset"].detach().cpu().clone()
        else:
            self.validation_ids.append(row_ids.detach().cpu())
        return self.layer(inputs)

    def get_main_network_parts(self):
        return {
            "training_ids": torch.cat(self.training_ids),
            "validation_ids": torch.cat(self.validation_ids) if self.validation_ids else torch.empty(0),
            "context_ids": self.context_ids,
        }


def _data(n_rows):
    X = torch.arange(n_rows * 5, dtype=torch.float32).reshape(n_rows, 5) / 100
    X[:, 0] = torch.arange(n_rows)
    return X, torch.arange(n_rows) % 3


def _seed():
    torch.manual_seed(181)
    np.random.seed(182)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("validation", ["internal", "external"])
def test_training_context_preserves_global_ids_after_split_bootstrap_and_subsampling(
    device_name, bootstrap, validation, monkeypatch
):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    X, y = _data(21)
    borrowed_contexts = []
    original_gather = utils._gather_tensor_batch

    def record_gather(tensors, indices, device):
        borrowed_contexts.append(tuple(tensor.data_ptr() for tensor in tensors))
        return original_gather(tensors, indices, device)

    monkeypatch.setattr(utils, "MainNetworkTrainable", _RecordingModel)
    monkeypatch.setattr(utils, "_gather_tensor_batch", record_gather)
    _seed()
    out = utils.fine_tune_main_network(
        cfg={},
        X=X,
        y=y,
        n_classes=3,
        rf=None,
        pca=None,
        main_network=[],
        norm=None,
        device=torch.device(device_name),
        max_epochs=3,
        batch_size=4,
        do_retrieval=True,
        finetuning_data="bootstrap" if bootstrap else "entire_dataset",
        finetuning_subset_max_samples=7,
        finetuning_subset_frac=0.75,
        X_val=X[:5] if validation == "external" else None,
        y_val=y[:5] if validation == "external" else None,
        finetuning_val_frac=0.2,
        val_max_samples=None,
        early_stopping_mode="epoch",
    )
    assert out["timed_out"] is False
    assert len(out["training_ids"]) == 21
    assert set(out["training_ids"].tolist()).issubset(set(out["context_ids"].tolist()))
    if bootstrap:
        assert len(torch.unique(out["context_ids"])) < len(out["context_ids"])
    else:
        assert len(torch.unique(out["context_ids"])) == len(out["context_ids"])
    if validation == "internal":
        assert set(out["training_ids"].tolist()).isdisjoint(set(out["validation_ids"].tolist()))
    if device_name == "cuda":
        assert len(borrowed_contexts) == 6
        assert len(set(borrowed_contexts)) == 1
    else:
        assert borrowed_contexts == []


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("n_rows", [11, 65536, 65537])
@pytest.mark.parametrize("retrieval", [False, True])
def test_training_context_borrowing_requires_complete_cuda_context(device_name, n_rows, retrieval, monkeypatch):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    X, y = _data(n_rows)
    gathered_sizes = []
    original_gather = utils._gather_tensor_batch

    def record_gather(tensors, indices, device):
        gathered_sizes.append(tuple(len(tensor) for tensor in tensors))
        return original_gather(tensors, indices, device)

    monkeypatch.setattr(utils, "MainNetworkTrainable", _RecordingModel)
    monkeypatch.setattr(utils, "_gather_tensor_batch", record_gather)
    _seed()
    out = utils.fine_tune_main_network(
        cfg={},
        X=X,
        y=y,
        n_classes=3,
        rf=None,
        pca=None,
        main_network=[],
        norm=None,
        device=torch.device(device_name),
        max_epochs=1,
        batch_size=4,
        do_retrieval=retrieval,
        finetuning_val_frac=0,
        max_train_batches_per_epoch=1,
    )
    assert out["timed_out"] is False
    if device_name == "cuda" and retrieval and n_rows <= 65536:
        assert gathered_sizes == [(n_rows, n_rows, n_rows)]
    else:
        assert gathered_sizes == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_borrowed_context_keeps_cpu_sampler_order_and_rng(monkeypatch):
    X, y = _data(21)
    monkeypatch.setattr(utils, "MainNetworkTrainable", _RecordingModel)
    results = []
    for device_name in ("cpu", "cuda"):
        _seed()
        out = utils.fine_tune_main_network(
            cfg={},
            X=X,
            y=y,
            n_classes=3,
            rf=None,
            pca=None,
            main_network=[],
            norm=None,
            device=torch.device(device_name),
            max_epochs=3,
            batch_size=4,
            do_retrieval=True,
            finetuning_data="bootstrap",
            finetuning_subset_max_samples=7,
            finetuning_val_frac=0,
        )
        results.append((out, torch.get_rng_state().clone(), np.random.get_state()))
    expected, actual = results
    assert torch.equal(expected[0]["training_ids"], actual[0]["training_ids"])
    assert torch.equal(expected[0]["context_ids"], actual[0]["context_ids"])
    assert torch.equal(expected[1], actual[1])
    assert expected[2][0] == actual[2][0]
    assert np.array_equal(expected[2][1], actual[2][1])
    assert expected[2][2:] == actual[2][2:]


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "n_rows,batch_size,subset_frac,bootstrap,reuses_projection",
    [
        (1, 8, None, False, True),
        (9, 12, None, False, True),
        (9, 9, None, True, True),
        (9, 8, None, False, False),
        (9, 12, 0.5, False, False),
    ],
)
def test_single_batch_projection_reuse(
    device_name,
    n_rows,
    batch_size,
    subset_frac,
    bootstrap,
    reuses_projection,
    monkeypatch,
):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device_name)
    model_class = utils.MainNetworkTrainable

    def run(*, reuse):
        _seed()
        X = torch.randn(n_rows + 3, 5)
        y = torch.arange(len(X)) % 3
        rf = torch.nn.Sequential(torch.nn.Linear(5, 8, bias=False), torch.nn.ReLU())
        pca = SimpleNamespace(mean_=torch.randn(8), components_=torch.randn(4, 8))
        norm = utils.SimpleBatchNorm1dFixed(6)
        norm.fit(torch.randn(12, 6))
        layers = [
            utils.Linear(0.1 * torch.randn(6, n_outputs), torch.zeros(n_outputs))
            for n_outputs in (6, 3)
        ]
        outputs = []
        projection_calls = []
        projection_grads = []
        norm_states = []

        class RecordingModel(model_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.rf.register_forward_hook(
                    lambda *_: projection_calls.append(self.training)
                )
                self.rf[0].weight.register_hook(
                    lambda grad: projection_grads.append(grad.detach().cpu().clone())
                )

            def forward(self, *args, **kwargs):
                if not reuse:
                    kwargs.pop("query_context_positions", None)
                result = super().forward(*args, **kwargs)
                outputs.append(result.detach().cpu())
                norm_states.append(
                    {
                        name: value.detach().cpu().clone()
                        for name, value in self.norm.named_buffers()
                    }
                )
                return result

        monkeypatch.setattr(utils, "MainNetworkTrainable", RecordingModel)
        with fit_random_generators(421):
            result = utils.fine_tune_main_network(
                cfg={"n_dims": 6, "pca_sampling": "zeropad", "clip_data_value": 2.5},
                X=X[:n_rows],
                y=y[:n_rows],
                X_val=X[n_rows:],
                y_val=y[n_rows:],
                n_classes=3,
                rf=rf,
                pca=pca,
                main_network=layers,
                norm=norm,
                device=device,
                max_epochs=2,
                batch_size=batch_size,
                finetuning_optimizer="lion",
                finetuning_dropout=0.15,
                do_retrieval=True,
                finetuning_data="bootstrap" if bootstrap else "entire_dataset",
                finetuning_subset_frac=subset_frac,
            )
            rng_states = [
                torch_generator().get_state(),
                torch_generator(device).get_state(),
            ]
        assert not result["timed_out"]
        return outputs, projection_calls, projection_grads, norm_states, rng_states

    expected = run(reuse=False)
    actual = run(reuse=True)
    outputs, calls, gradients, norm_states, rng_states = actual
    rtol, atol = 2e-5, 2e-6
    torch.testing.assert_close(outputs, expected[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(gradients, expected[2], rtol=rtol, atol=atol)
    torch.testing.assert_close(norm_states, expected[3], rtol=rtol, atol=atol)
    torch.testing.assert_close(rng_states, expected[4], rtol=0, atol=0)
    assert sum(calls) == len(gradients) * (1 if reuses_projection else 2)
    assert calls.count(False) == expected[1].count(False)
