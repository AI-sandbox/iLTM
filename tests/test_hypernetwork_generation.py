from __future__ import annotations

from types import MethodType

import iltm.iltm_model as model_module
import pytest
import torch
from iltm.iltm_model import HypernetworkBlock


def reference_class_means(self, X, y, n_classes):
    if n_classes == 1:
        return torch.mean(X, dim=0, keepdim=True)
    return torch.cat(
        [
            torch.mean(X[y == label] if torch.sum(y == label) > 0 else X, dim=0, keepdim=True)
            for label in range(n_classes)
        ],
        dim=0,
    )


def reference_last_layer(self, out, dataset_representation, y, n_classes):
    data = torch.cat((out, dataset_representation), dim=1)
    per_sample = self.get_main_weights(data, self.hypernetworks[-1])
    if n_classes == 1:
        weights = torch.mean(per_sample, dim=0, keepdim=True)
        means = torch.mean(out, dim=0, keepdim=True)
    else:
        weights, means = [], []
        for label in range(n_classes):
            present = torch.sum(y == label) > 0
            weights.append(torch.mean(per_sample[y == label] if present else per_sample, dim=0, keepdim=True))
            means.append(torch.mean(out[y == label] if present else out, dim=0, keepdim=True))
        weights, means = torch.cat(weights), torch.cat(means)
    weights[:, :-1] = weights[:, :-1] + means
    weights = weights.transpose(0, 1)
    _, layer = model_module.forward_linear_layer(out, weights, n_classes)
    return layer


def restore_baseline_methods(block):
    block._calculate_class_means = MethodType(reference_class_means, block)
    block._forward_last_hypernetwork_module = MethodType(reference_last_layer, block)


@pytest.mark.parametrize("n_classes", [1, 2, 5])
@pytest.mark.parametrize("missing_class", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_generated_weights_outputs_gradients_and_rng_match(n_classes, missing_class, training):
    torch.manual_seed(51)
    reference = HypernetworkBlock(
        n_classes_limit=8,
        hn_n_layers=3,
        hn_hidden_size=16,
        main_n_layers=3,
        n_dims=8,
        bottleneck_size=0,
        hyper_dropout=0.2,
    )
    actual = HypernetworkBlock(
        n_classes_limit=8,
        hn_n_layers=3,
        hn_hidden_size=16,
        main_n_layers=3,
        n_dims=8,
        bottleneck_size=0,
        hyper_dropout=0.2,
    )
    actual.load_state_dict(reference.state_dict())
    restore_baseline_methods(reference)
    X = torch.randn(17, 8)
    if n_classes == 1:
        y = torch.randn(17)
    else:
        effective_classes = n_classes - 1 if missing_class else n_classes
        y = torch.arange(17) % effective_classes
    results = []
    for block in (reference, actual):
        x = X.clone().requires_grad_()
        torch.manual_seed(195)
        generated = block(x, y, n_classes, training=training)
        rng_after = torch.get_rng_state()
        outputs = generated[-1](x)
        loss = outputs.square().sum()
        loss.backward()
        results.append((generated, rng_after, outputs.detach(), x.grad, dict(block.named_parameters())))
    expected, observed = results
    assert torch.equal(expected[1], observed[1])
    assert torch.equal(expected[2], observed[2])
    assert torch.equal(expected[3], observed[3])
    for before, after in zip(expected[0], observed[0], strict=True):
        assert torch.equal(before.m, after.m)
        assert torch.equal(before.b, after.b)
        assert before.m.stride() == after.m.stride()
        assert before.b.stride() == after.b.stride()
    for name, parameter in expected[4].items():
        actual_gradient = observed[4][name].grad
        if parameter.grad is None:
            assert actual_gradient is None
        else:
            assert torch.equal(parameter.grad, actual_gradient), name


def test_final_layer_generation_skips_unused_matrix_multiply(monkeypatch):
    torch.manual_seed(9)
    block = HypernetworkBlock(8, 3, 16, 3, 8, 0)
    out = torch.randn(12, 8)
    dataset_representation = torch.randn(12, 8 * 3 + 8)
    labels = torch.arange(12) % 3

    def unexpected_forward(*args, **kwargs):
        raise AssertionError("Final generated layer output is unused during generation.")

    monkeypatch.setattr(model_module, "forward_linear_layer", unexpected_forward)
    layer = block._forward_last_hypernetwork_module(out, dataset_representation, labels, 3)
    assert layer.m.shape == (8, 3)
    assert layer.b.shape == (3,)
