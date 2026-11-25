import pickle

import pytest
import torch

from iltm.utils import PCAModule, TorchPCA


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("layout", ["strided", "contiguous_prefix", "offset", "compact"])
def test_pca_pickle_preserves_projection_and_layout(monkeypatch, trainable, layout):
    rng = torch.Generator().manual_seed(17)
    rows, width, kept = 128, 256, 9
    full = torch.randn(rows, width, generator=rng)
    if layout == "strided":
        basis = full.T.contiguous().T[:kept]
    elif layout == "contiguous_prefix":
        basis = full[:kept]
    elif layout == "offset":
        basis = full.T.contiguous().T[2:2 + kept]
    else:
        basis = full[:kept].clone()
    pca = TorchPCA(n_components=kept)
    pca.mean_ = torch.randn(width, generator=rng)
    pca.components_ = basis
    model = PCAModule(pca) if trainable else pca
    inputs = torch.randn(17, width, generator=rng)
    expected = model.transform(inputs).detach()

    def weight(obj):
        return obj.pca_components.weight if trainable else obj.components_

    original = weight(model)
    with monkeypatch.context() as patch:
        old_getstate = torch.nn.Module.__getstate__ if trainable else object.__getstate__
        patch.setattr(type(model), "__getstate__", old_getstate)
        legacy = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    for obj in (model, pickle.loads(legacy)):
        saved = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        restored = pickle.loads(saved)
        actual = weight(restored)
        assert actual.stride() == original.stride()
        assert actual.storage_offset() == original.storage_offset()
        assert type(actual) is type(original)
        assert actual.dtype == original.dtype
        assert actual.requires_grad == original.requires_grad
        assert torch.equal(
            actual.detach().contiguous().view(torch.int32), original.detach().contiguous().view(torch.int32)
        )
        assert torch.equal(restored.transform(inputs).detach().view(torch.int32), expected.view(torch.int32))
        if layout != "compact":
            assert len(legacy) - len(saved) >= (rows - kept) * width * full.element_size() - 512
        else:
            assert abs(len(legacy) - len(saved)) < 512
    assert weight(model) is original
    assert original.untyped_storage().data_ptr() == basis.untyped_storage().data_ptr()


def test_unfitted_pca_pickle():
    restored = pickle.loads(pickle.dumps(TorchPCA(n_components=3)))
    assert restored.n_components == 3
    assert not hasattr(restored, "components_")
