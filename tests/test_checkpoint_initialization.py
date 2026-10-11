from __future__ import annotations

import iltm.inference_interface as interface
import pytest
import torch
from iltm.iltm_model import iLTM

ARCHITECTURE = {
    "n_dims": 8,
    "hn_n_layers": 3,
    "hn_hidden_size": 16,
    "rf_size": 32,
    "n_classes_limit": 5,
    "main_n_layers": 3,
}


def baseline_initialize(estimator):
    model = iLTM(
        n_dims=estimator.n_dims,
        hn_n_layers=estimator.hn_n_layers,
        hn_hidden_size=estimator.hn_hidden_size,
        clip_data_value=estimator.clip_data_value,
        rf_size=estimator.rf_size,
        n_classes_limit=estimator.n_classes_limit,
        dim_exp_type=estimator.dim_exp_type,
        bottleneck_size=estimator.bottleneck_size,
        main_n_layers=estimator.main_n_layers,
        pca_fit=estimator.pca_fit,
        pca_svd_driver=estimator.pca_svd_driver,
        hyper_dropout=estimator.hyper_dropout,
        pca_sampling=estimator.pca_sampling,
    ).to(estimator.device)
    state = torch.load(estimator.model_path, map_location=estimator.device, weights_only=True)
    state = {
        "hypernetwork_block." + key if key.startswith(("hypernetworks.", "hn_emb_to_weights.")) else key: value
        for key, value in state.items()
    }
    if state:
        try:
            model.load_state_dict(state)
        except RuntimeError as error:
            raise RuntimeError(f"Failed to load the model from {estimator.model_path}") from error
    return model.eval()


def estimator(path, *, device="cpu"):
    result = interface.iLTMClassifier(checkpoint=str(path), device=device, **ARCHITECTURE)
    result._model_cache = {}
    return result


def checkpoint(tmp_path, case):
    torch.manual_seed(73)
    state = iLTM(**ARCHITECTURE).state_dict()
    if case == "float64":
        state = {key: value.to(torch.float64) for key, value in state.items()}
    elif case == "noncontiguous":
        state = {key: value.t().contiguous().t() if value.ndim == 2 else value for key, value in state.items()}
    elif case == "legacy":
        state = {key.removeprefix("hypernetwork_block."): value for key, value in state.items()}
    elif case == "duplicate_alias_values":
        for key, value in state.items():
            state[key] = value.clone()
        state["hypernetwork_block.hypernetworks.0.2.weight"].add_(1)
    elif case == "unexpected_alias":
        state["hypernetwork_block.hn_emb_to_weights.1.weight"] = state["hypernetwork_block.hn_emb_to_weights.0.weight"]
    elif case in ("padded_offset", "oversized_storage"):
        key = "hypernetwork_block.hypernetworks.0.0.weight"
        value = state[key]
        offset = 13 if case == "padded_offset" else 0
        padded = torch.cat([torch.zeros(offset), value.flatten(), torch.zeros(17)])
        state[key] = padded[offset : offset + value.numel()].view(value.shape)
    elif case == "empty":
        state = {}
    path = tmp_path / f"{case}.pth"
    torch.save(state, path)
    return path


@pytest.mark.parametrize(
    "case",
    [
        "standard",
        "legacy",
        "float64",
        "noncontiguous",
        "empty",
        "duplicate_alias_values",
        "unexpected_alias",
        "padded_offset",
        "oversized_storage",
    ],
)
def test_initialization_preserves_weights_rng_and_aliases(tmp_path, case):
    path = checkpoint(tmp_path, case)
    results = []
    for initialize in (baseline_initialize, lambda est: est._initialize_model()):
        torch.manual_seed(103)
        model = initialize(estimator(path))
        results.append((model, torch.get_rng_state()))
    original, optimized = results
    assert torch.equal(original[1], optimized[1])
    assert original[0].training is optimized[0].training is False
    for key, value in original[0].state_dict().items():
        assert torch.equal(value, optimized[0].state_dict()[key]), key
        assert value.stride() == optimized[0].state_dict()[key].stride()
        assert value.dtype == optimized[0].state_dict()[key].dtype
        assert value.storage_offset() == optimized[0].state_dict()[key].storage_offset()
        assert value.untyped_storage().nbytes() == optimized[0].state_dict()[key].untyped_storage().nbytes()
    for model, _ in results:
        blocks = model.hypernetwork_block.hypernetworks
        assert blocks[0][2] is blocks[1][2] is blocks[2][2]
        assert all(parameter.requires_grad for parameter in model.parameters())
        weights = model.hypernetwork_block.hn_emb_to_weights
        assert weights[0].weight.untyped_storage().data_ptr() != weights[1].weight.untyped_storage().data_ptr()
    torch.manual_seed(319)
    X = torch.randn(20, ARCHITECTURE["n_dims"])
    y = torch.arange(20) % 3
    with torch.no_grad():
        reference = original[0].hypernetwork_block(X, y, 3)
        actual = optimized[0].hypernetwork_block(X, y, 3)
    for reference_layer, actual_layer in zip(reference, actual, strict=True):
        assert torch.equal(reference_layer.m, actual_layer.m)
        assert torch.equal(reference_layer.b, actual_layer.b)


@pytest.mark.parametrize(
    ("case", "expected_transfers"),
    [
        ("standard", 0),
        ("legacy", 0),
        ("float64", 1),
        ("noncontiguous", 1),
        ("empty", 1),
        ("duplicate_alias_values", 1),
        ("unexpected_alias", 1),
        ("padded_offset", 1),
        ("oversized_storage", 1),
    ],
)
def test_assign_avoids_transferring_unused_initialization(tmp_path, monkeypatch, case, expected_transfers):
    path = checkpoint(tmp_path, case)
    calls = []
    original_to = iLTM.to

    def traced_to(model, *args, **kwargs):
        calls.append(model)
        return original_to(model, *args, **kwargs)

    monkeypatch.setattr(iLTM, "to", traced_to)
    estimator(path)._initialize_model()
    assert len(calls) == expected_transfers


def test_cache_hit_preserves_rng_and_model_identity(tmp_path):
    est = estimator(checkpoint(tmp_path, "standard"))
    model = est._initialize_model()
    rng_before = torch.get_rng_state()
    assert est._initialize_model() is model
    assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_nondefault_dtype_conversion_is_preserved(tmp_path, default_dtype):
    path = checkpoint(tmp_path, "standard")
    saved_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        torch.manual_seed(719)
        reference = baseline_initialize(estimator(path))
        expected_rng = torch.get_rng_state()
        torch.manual_seed(719)
        actual = estimator(path)._initialize_model()
        assert torch.equal(expected_rng, torch.get_rng_state())
        for key, value in reference.state_dict().items():
            assert value.dtype == actual.state_dict()[key].dtype == default_dtype
            assert torch.equal(value, actual.state_dict()[key])
    finally:
        torch.set_default_dtype(saved_dtype)


def test_invalid_checkpoint_reports_same_load_error(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"unexpected": torch.ones(3)}, path)
    for initialize in (baseline_initialize, lambda est: est._initialize_model()):
        with pytest.raises(RuntimeError, match="Failed to load the model"):
            initialize(estimator(path))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("legacy", [False, True])
def test_only_cuda_zip_checkpoint_loads_use_mmap(tmp_path, monkeypatch, device, legacy):
    path = checkpoint(tmp_path, "standard")
    original_load = torch.load
    if legacy:
        state = original_load(path, weights_only=True)
        torch.save(state, path, _use_new_zipfile_serialization=False)
    calls = []

    def traced_load(*args, **kwargs):
        calls.append(kwargs.copy())
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", traced_load)
    monkeypatch.setattr(iLTM, "to", lambda model, *args, **kwargs: model)
    estimator(path, device=device)._initialize_model()
    assert len(calls) == 1
    assert calls[0].get("mmap", False) is (device.startswith("cuda") and not legacy)
    assert str(calls[0]["map_location"]) == device


@pytest.mark.parametrize("failure", [OSError("Mapping unavailable"), RuntimeError("unable to mmap checkpoint")])
def test_mapping_filesystem_error_retries_without_mapping(tmp_path, monkeypatch, failure):
    path = checkpoint(tmp_path, "standard")
    original_load = torch.load
    calls = []

    def traced_load(*args, **kwargs):
        calls.append(kwargs.copy())
        if kwargs.get("mmap"):
            raise failure
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", traced_load)
    monkeypatch.setattr(iLTM, "to", lambda model, *args, **kwargs: model)
    estimator(path, device="cuda:0")._initialize_model()
    assert len(calls) == 2
    assert calls[0]["mmap"] is True
    assert "mmap" not in calls[1]
    assert all(call["weights_only"] is True for call in calls)


def test_non_mapping_load_error_is_preserved(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "standard")
    calls = []
    failure = RuntimeError("Corrupt checkpoint data")

    def traced_load(*args, **kwargs):
        calls.append(kwargs)
        raise failure

    monkeypatch.setattr(torch, "load", traced_load)
    with pytest.raises(RuntimeError, match="Corrupt checkpoint data") as caught:
        estimator(path, device="cuda:0")._initialize_model()
    assert caught.value is failure
    assert len(calls) == 1


def test_meta_checkpoint_parameter_retains_copy_load_failure(tmp_path):
    path = checkpoint(tmp_path, "standard")
    state = torch.load(path, weights_only=True)
    key = "hypernetwork_block.hypernetworks.0.0.weight"
    state[key] = state[key].to("meta")
    torch.save(state, path)
    for initialize in (baseline_initialize, lambda est: est._initialize_model()):
        with pytest.raises(RuntimeError, match="Failed to load the model"):
            initialize(estimator(path))


def test_tensor_subclass_checkpoint_retains_ordinary_model_parameters(tmp_path, monkeypatch):
    class CheckpointTensor(torch.Tensor):
        pass

    path = checkpoint(tmp_path, "standard")
    state = torch.load(path, weights_only=True)
    key = "hypernetwork_block.hypernetworks.0.0.weight"
    state[key] = state[key].as_subclass(CheckpointTensor)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: state)
    reference = baseline_initialize(estimator(path))
    actual = estimator(path)._initialize_model()
    for name, parameter in actual.named_parameters():
        assert type(parameter) is torch.nn.Parameter
        torch.testing.assert_close(parameter, dict(reference.named_parameters())[name], rtol=0, atol=0)
