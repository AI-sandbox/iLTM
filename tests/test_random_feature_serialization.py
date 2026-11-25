import pickle

import pytest
import torch
from torch import nn

import iltm._random_features as rf
from iltm._random import fit_random_generators, torch_generator


@pytest.mark.parametrize("dtype", [None, torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("checkpoint", ["frozen", "restored", "trained"])
def test_projection_pickle(dtype, checkpoint):
    with fit_random_generators(71, rf_storage_dtype=dtype):
        torch.rand(17, generator=torch_generator())
        layer = rf.RandomFeatureLinear.initialize(65, 257)
        next_draw = torch.rand(5, generator=torch_generator())
    # Match the original draw, including its position in the ensemble's RNG stream.
    generator = torch.Generator().manual_seed(71)
    torch.rand(17, generator=generator)
    expected = torch.empty(257, 65)
    nn.init.kaiming_normal_(
        expected, mode="fan_out", nonlinearity="relu", generator=generator
    )
    assert torch.equal(expected, layer.weight)
    assert torch.equal(torch.rand(5, generator=generator), next_draw)
    if checkpoint != "frozen":
        with torch.no_grad():
            layer.weight[-1, -1] += 1
        if checkpoint == "restored":
            layer.load_state_dict({"weight": expected})
    layer.to(dtype=dtype)
    layer.requires_grad_(True)
    ambient_state = torch.get_rng_state()
    weight = layer.weight
    compact = checkpoint != "trained"
    for _ in range(2):
        state = layer.__getstate__()
        assert (state["_parameters"]["weight"] is None) == compact
        payload = pickle.dumps(layer)
        if compact:
            assert len(payload) < weight.nbytes // 2
        else:
            assert len(payload) >= weight.nbytes
        layer = pickle.loads(payload)
        assert torch.equal(
            weight.detach().view(torch.uint8), layer.weight.detach().view(torch.uint8)
        )
        assert layer.weight.requires_grad
    assert torch.equal(torch.get_rng_state(), ambient_state)


def test_tiny_and_global_rng_projections_save_weights():
    with fit_random_generators(2):
        tiny = rf.RandomFeatureLinear.initialize(2, 2)
    with torch.random.fork_rng(devices=[]):
        standalone = rf.RandomFeatureLinear.initialize(65, 257)
    for layer in (tiny, standalone):
        assert layer.__getstate__()["_parameters"]["weight"] is layer.weight


def test_independent_members_and_incompatible_generator(monkeypatch):
    with fit_random_generators(2):
        layers = [rf.RandomFeatureLinear.initialize(65, 257) for _ in range(2)]
    assert not torch.equal(layers[0].weight, layers[1].weight)
    saved = pickle.dumps(layers)
    restored = pickle.loads(saved)
    for original, loaded in zip(layers, restored):
        assert torch.equal(original.weight, loaded.weight)
    monkeypatch.setattr(rf, "_initialize", lambda weight, generator: weight.zero_())
    with pytest.raises(RuntimeError, match="did not reproduce"):
        pickle.loads(saved)
