import pytest
import torch

from iltm import iLTMClassifier


@pytest.mark.parametrize("dictionary_input", [False, True])
def test_stratified_bootstrap_sampling(monkeypatch, dictionary_input):
    estimator = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        stratify_sampling=True,
        batch_size=4,
        n_dims=8,
    )
    row_ids = torch.arange(12, dtype=torch.float32)
    y = torch.tensor([0] * 8 + [1] * 2 + [2] * 2)
    if dictionary_input:
        X = {
            "x_num": torch.stack([row_ids, row_ids + 10], dim=1),
            "x_cat": row_ids.reshape(-1, 1),
        }
    else:
        X = torch.stack([row_ids, row_ids + 10], dim=1)

    bootstrap_upper_bounds = []

    def balanced_bootstrap_indices(low, high, size):
        assert low == 0
        bootstrap_upper_bounds.append(high)
        return torch.arange(size[0]) % high

    monkeypatch.setattr(torch, "randint", balanced_bootstrap_indices)

    X_sampled, y_sampled, feature_indices = estimator._sample_data(
        X,
        y,
        pca_sampling="bootstrap",
    )

    sampled_row_ids = (
        X_sampled["x_num"][:, 0] if dictionary_input else X_sampled[:, 0]
    ).long()
    assert y_sampled.shape == (estimator.n_dims,)
    assert torch.equal(y_sampled, y[sampled_row_ids])
    assert bootstrap_upper_bounds == [3]
    assert sorted(torch.bincount(y_sampled).tolist()) == [2, 3, 3]
    assert feature_indices is None
