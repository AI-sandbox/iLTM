import pytest
import torch

import iltm.utils as utils


def _identity_forward(X, *args, **kwargs):
    return X, X


def _kwargs():
    return {
        "n_classes": 3,
        "batch_size": 3,
        "distance_type": "cosine",
        "temperature": 1.5,
        "model_cfg": {},
        "rf": None,
        "pca": None,
        "norm": None,
        "main_network": None,
        "device": torch.device("cpu"),
    }


def test_training_retrieval_excludes_matching_row_ids(monkeypatch):
    monkeypatch.setattr(
        utils,
        "forward_main_network_with_preprocessing",
        _identity_forward,
    )
    context = torch.eye(3)
    targets = torch.arange(3)
    ids = torch.arange(3)

    torch.manual_seed(0)
    probabilities = utils.retrieval(
        context,
        targets,
        context,
        query_ids=ids,
        context_ids_superset=ids,
        **_kwargs(),
    ).exp()

    torch.testing.assert_close(
        probabilities.diag(),
        torch.full((3,), 1e-12),
        rtol=1e-5,
        atol=1e-13,
    )
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(3),
    )


def test_inference_retrieval_is_unchanged_without_row_ids(monkeypatch):
    monkeypatch.setattr(
        utils,
        "forward_main_network_with_preprocessing",
        _identity_forward,
    )
    context = torch.eye(3)
    targets = torch.arange(3)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])

    torch.manual_seed(0)
    actual = utils.retrieval(context, targets, queries, **_kwargs()).exp()
    expected = torch.softmax(
        torch.nn.functional.normalize(queries, dim=-1)
        @ torch.nn.functional.normalize(context, dim=-1).T
        / 1.5,
        dim=-1,
    )

    torch.testing.assert_close(actual, expected)


def test_all_self_context_falls_back_to_main_output(monkeypatch):
    monkeypatch.setattr(
        utils,
        "forward_main_network_with_preprocessing",
        _identity_forward,
    )
    query = torch.tensor([[0.2, 0.3, 0.5]])
    context = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    targets = torch.tensor([0, 1])

    actual = utils.full_main_forward(
        query,
        n_classes=3,
        batch_size=2,
        model_cfg={},
        rf=None,
        pca=None,
        norm=None,
        main_network=None,
        device=torch.device("cpu"),
        use_amp=False,
        do_retrieval=True,
        X_ctxt_superset=context,
        y_ctxt_superset=targets,
        retrieval_alpha=0.75,
        query_ids=torch.tensor([4]),
        context_ids_superset=torch.tensor([4, 4]),
    )

    torch.testing.assert_close(actual, query)


def test_row_id_arguments_must_be_paired(monkeypatch):
    monkeypatch.setattr(
        utils,
        "forward_main_network_with_preprocessing",
        _identity_forward,
    )

    with torch.no_grad(), pytest.raises(ValueError, match="must be provided together"):
        utils.retrieval(
            torch.eye(3),
            torch.arange(3),
            torch.eye(3),
            query_ids=torch.arange(3),
            **_kwargs(),
        )


def test_sampled_indices_require_unexpanded_data():
    with pytest.raises(ValueError, match="to_meta_model must be False"):
        utils.sample_data(
            torch.eye(3),
            torch.arange(3),
            return_sampled_indices=True,
        )
